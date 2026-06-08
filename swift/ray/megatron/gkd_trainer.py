# Copyright (c) ModelScope Contributors. All rights reserved.
"""Driver-side GKD trainer for Ray-based Megatron training."""
from __future__ import annotations

import copy
import random
import ray
import torch
from contextlib import contextmanager
from typing import List

from swift.infer_engine.protocol import RequestConfig, RolloutOutput
from swift.rlhf_trainers.gkd_loss import DataSource, TeacherOutput, build_opsd_teacher_data
from swift.rlhf_trainers.utils import (build_teacher_infer_request, parse_prompt_logprobs,
                                       replace_assistant_response_with_ids)
from swift.utils import get_logger
from .base_trainer import BaseRayTrainer
from .driver_utils import extract_iteration

logger = get_logger()


class GKDTrainer(BaseRayTrainer):

    def _prepare_state(self) -> None:
        info = self._data_info
        args = info['_driver_args']
        template = info['template']
        self.args = args
        self.template = template
        self.device = torch.device('cpu')

        self.global_batch_size = int(args.global_batch_size)
        self.temperature = args.temperature
        self.beta = args.beta
        self.sft_alpha = args.sft_alpha
        self.gkd_logits_topk = args.gkd_logits_topk
        # GKD on-policy schedule: each step is on-policy (student generates) with
        # probability ``lmbda``; otherwise off-policy (distill on dataset responses).
        self.lmbda = args.lmbda
        self.seq_kd = args.seq_kd
        self._data_source_rng = random.Random(getattr(args, 'seed', 42))

        # steps_per_generation>1: one generation (one data_source) feeds spg training steps,
        # reducing weight-sync / generation frequency. Mirrors the non-ray GKD trainer.
        gen_bs = getattr(args, 'generation_batch_size', None)
        spg = getattr(args, 'steps_per_generation', None)
        if gen_bs is not None:
            self._steps_per_generation = max(int(gen_bs) // self.global_batch_size, 1)
        elif spg is not None:
            self._steps_per_generation = int(spg)
        else:
            self._steps_per_generation = 1
        # GKD generates exactly one completion per prompt (on-policy student generation),
        # so num_generations is always 1 here regardless of the (GRPO-oriented) default.
        info['num_generations'] = 1
        self._padding_to = info.get('_padding_to')
        # Prefer the resolved local snapshot dir (teacher_model_dir) over the raw model id
        # (teacher_model); bridge.load_weights needs a real path to locate safetensors.
        self._teacher_model_dir = getattr(args, 'teacher_model_dir', None) or args.teacher_model
        self._teacher_model_server = args.teacher_model_server
        # Self-distillation: teacher_model == student model. The worker scores the teacher
        # via disable_adapter() (LoRA) and loads NO separate teacher. The driver's own args
        # may lack tuner_type (it lives in the train group), so detect self-distillation by
        # teacher_model == model directly and force _teacher_model_dir=None to skip
        # init_teacher_model (which would otherwise fail to load a non-existent teacher).
        self._teacher_use_disable_adapter = getattr(args, '_teacher_use_disable_adapter', False)
        if not self._teacher_use_disable_adapter and args.teacher_model is not None \
                and args.teacher_model == args.model:
            self._teacher_use_disable_adapter = True
        if self._teacher_use_disable_adapter:
            self._teacher_model_dir = None

        if self._teacher_model_server and not self.teacher_replicas:
            raise NotImplementedError('teacher_model_server is not yet supported in the Ray pipeline. '
                                      'Use teacher_model (colocated) or teacher replicas (teacher.gpus > 0) instead.')

        vp_size = getattr(args, 'virtual_pipeline_model_parallel_size', None)
        assert vp_size is None or vp_size == 1, \
            'Ray GKD does not support VPP (virtual_pipeline_model_parallel_size > 1).'

        # truncation_strategy='delete': resample prompts whose encode fails (over max_length).
        self.truncation_strategy = getattr(args, 'truncation_strategy', None)
        self.max_completion_length = args.max_completion_length
        self._max_resample_rounds = getattr(args, 'max_resample_times', 10) or 10
        self._needs_resample_iterator = self.truncation_strategy == 'delete'

    def _train_loop(self, tg, train_iters, iteration):
        ckpt = self.ckpt_manager
        spg = self._steps_per_generation

        # Initialize colocated teacher if configured
        if self._teacher_model_dir and not self._teacher_model_server:
            tg.execute('init_teacher_model', self._teacher_model_dir)
            logger.info('Colocated teacher model initialized from %s', self._teacher_model_dir)

        while iteration < train_iters:
            # One generation (a single data_source) feeds ``spg`` training steps.
            prompt_batch = next(self._data_iter)
            if self.truncation_strategy == 'delete':
                prompt_batch = self._resample_failed_prompts(prompt_batch)
            data_source = self._determine_data_source()

            if data_source == DataSource.STUDENT:
                # On-policy: sync the latest weights to the rollout engine and generate.
                ckpt.sync_weights(merge_and_sync=True)
                with self._generation_context(tg, ckpt):
                    rollout_batch = self._expand_for_generation(prompt_batch)
                    completions = self._generate(rollout_batch)
                    rollout_with_outputs = self._postprocess_rollout(rollout_batch, completions)
                source_items = rollout_with_outputs
            else:
                # Off-policy (lmbda<1): distill on the dataset's ground-truth responses,
                # no generation and no weight sync to the rollout engine.
                rollout_with_outputs = None
                source_items = list(prompt_batch)

            self._maybe_log_completions(rollout_with_outputs, gen_step=iteration)

            # Split one generation into ``spg`` chunks; each chunk is one training step
            # (same data_source). spg=1 degenerates to a single chunk == the whole batch.
            n = len(source_items)
            chunk_size = max(n // spg, 1)
            for step_idx in range(spg):
                if iteration >= train_iters:
                    break
                chunk = source_items[step_idx * chunk_size:(step_idx + 1) * chunk_size]
                if not chunk:
                    break
                samples = self._encode_rollout_batch(chunk)

                if self.teacher_replicas:
                    if rollout_with_outputs is None:
                        raise NotImplementedError('Teacher replicas currently require on-policy generation (lmbda=1). '
                                                  'Use a colocated teacher_model for lmbda<1 (off-policy) training.')
                    self._fetch_teacher_from_replicas(chunk, samples)
                elif self._teacher_use_disable_adapter or (self._teacher_model_dir and not self._teacher_model_server):
                    # Teacher outputs are cached worker-local and injected at train_step via
                    # _inject_cached_teacher_logits (required for CP per-rank slice alignment).
                    tg.compute_teacher_logits(samples)

                # Tag each sample with the step's data_source so the worker can gate the SFT
                # loss in loss_func (SFT only on dataset responses, not on-policy generations).
                for s in samples:
                    s['data_source'] = data_source
                results = tg.train_step(
                    samples, extra_metrics={'on_policy': 1.0 if data_source == DataSource.STUDENT else 0.0})
                iteration = extract_iteration(results)

        return iteration

    def _determine_data_source(self):
        """Pick the data source for this step (GKD on/off-policy schedule).

        With probability ``lmbda`` the step is on-policy (the student generates the
        response); otherwise it is off-policy and we distill on the dataset's
        ground-truth response. ``seq_kd`` (teacher-generated responses) is not
        implemented in the Ray pipeline and falls back to dataset responses.
        """
        if self._data_source_rng.random() < self.lmbda:
            return DataSource.STUDENT
        if self.seq_kd:
            logger.warning_once('seq_kd=True but teacher generation is not implemented in Ray GKD; '
                                'using dataset responses for off-policy steps.')
        return DataSource.DATASET

    def _expand_for_generation(self, prompt_batch):
        from swift.utils import remove_response
        expanded = []
        for item in prompt_batch:
            item_copy = copy.deepcopy(item)
            if 'messages' in item_copy:
                remove_response(item_copy['messages'])
            expanded.append(item_copy)
        return expanded

    def _generate(self, batch) -> List[RolloutOutput]:
        args = self.args
        request_config = RequestConfig(
            n=1,
            max_tokens=args.max_completion_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            stop=args.stop_words or None,
            return_details=True,
        )
        completions = self._distribute_to_replicas(list(batch), request_config)
        return [RolloutOutput(response=resp) for resp in completions]

    def _postprocess_rollout(self, rollout_batch, outputs):
        from swift.utils import remove_response
        merged = []
        for inp, output in zip(rollout_batch, outputs):
            item = dict(inp)
            if output is None:
                merged.append(item)
                continue
            response = output.response
            choice = response.choices[0]
            messages = copy.deepcopy(item.get('messages') or [])
            remove_response(messages)
            messages.append({'role': 'assistant', 'content': choice.message.content or ''})
            item['messages'] = messages
            item['response_token_ids'] = choice.token_ids or []
            item['finish_reason'] = choice.finish_reason or 'stop'
            item['is_truncated'] = item['finish_reason'] == 'length'
            item['add_eos'] = False
            merged.append(item)
        return merged

    @contextmanager
    def _extended_max_length(self):
        """Temporarily extend template.max_length by max_completion_length so the
        prompt+response (token-in-token-out) encodes without truncation. Mirrors the
        non-ray trainer's _template_context."""
        template = self.template
        original = template.max_length
        template.max_length = original + self.args.max_completion_length
        try:
            yield
        finally:
            template.max_length = original

    def _encode_rollout_batch(self, rollout_batch):
        """Encode rollout samples for the training workers.

        Token-in-token-out: replace the assistant text with the on-policy
        ``response_token_ids`` before encoding, so the student is trained on the
        exact tokens that were generated (and that the teacher scores). Re-encoding
        the decoded text would re-tokenize the response and break student/teacher
        alignment. Mirrors the non-ray trainer's ``_encode_batch``.
        """
        template = self.template
        samples = []
        with self._extended_max_length():
            for orig_item in rollout_batch:
                item = orig_item
                if item.get('response_token_ids'):
                    item = dict(item)
                    item['messages'] = replace_assistant_response_with_ids(
                        copy.deepcopy(item['messages']), item['response_token_ids'])
                encoded = template.encode(item, return_length=True)
                sample = {'encoded': encoded}
                # OPSD: if the dataset row carries a `teacher_prompt`, also encode the
                # teacher-prompt view (same on-policy response, token-in-token-out) so the
                # colocated teacher can be scored on its own sequence. No-op otherwise.
                opsd_encoded = self._encode_opsd_teacher(orig_item, template)
                if opsd_encoded is not None:
                    sample['opsd_teacher_encoded'] = opsd_encoded
                samples.append(sample)
        return samples

    @staticmethod
    def _encode_opsd_teacher(item, template):
        """Encode the OPSD teacher view of a rollout item, or None if not OPSD.

        Mirrors the non-ray Megatron GKD: replace the last user turn with
        ``teacher_prompt`` (keeping the on-policy response), then re-apply the
        token-in-token-out response so teacher and student score identical tokens.
        """
        opsd_list = build_opsd_teacher_data([item])  # None unless item has teacher_prompt
        if not opsd_list:
            return None
        opsd_item = opsd_list[0]
        if opsd_item.get('response_token_ids'):
            opsd_item['messages'] = replace_assistant_response_with_ids(
                copy.deepcopy(opsd_item['messages']), opsd_item['response_token_ids'])
        return template.encode(opsd_item, return_length=True)

    def _fetch_teacher_from_replicas(self, rollout_with_outputs, samples):
        """Fetch teacher logprobs from Ray teacher replicas (token-in-token-out).

        OPSD: when an item carries a ``teacher_prompt``, the teacher must score its OWN
        (teacher_prompt + same on-policy response) sequence. We build the request from that
        teacher view and attach ``opsd_teacher_labels`` (from the teacher-side encoding) so
        ``extract_active`` aligns the shared response by mask rather than by position.
        """
        topk = self.gkd_logits_topk
        assert topk is not None, 'gkd_logits_topk must be set when using teacher replicas'

        requests = []
        teacher_encodeds = []  # teacher-side encoded (OPSD) or None (non-OPSD)
        for item, sample in zip(rollout_with_outputs, samples):
            # Reuse the teacher-prompt encoding already produced by _encode_rollout_batch
            # (avoids a second tokenize/template pass); only the vLLM request is rebuilt.
            opsd_encoded = sample.get('opsd_teacher_encoded')
            if opsd_encoded is not None:
                opsd_item = build_opsd_teacher_data([item])[0]
                if opsd_item.get('response_token_ids'):
                    opsd_item['messages'] = replace_assistant_response_with_ids(
                        copy.deepcopy(opsd_item['messages']), opsd_item['response_token_ids'])
                requests.append(build_teacher_infer_request(opsd_item))
                teacher_encodeds.append(opsd_encoded)
            else:
                requests.append(build_teacher_infer_request(item))
                teacher_encodeds.append(None)
        request_config = RequestConfig(prompt_logprobs=topk, max_tokens=1, temperature=0.0)

        replicas = self.teacher_replicas
        n = len(replicas)
        chunk_size = (len(requests) + n - 1) // n
        refs = []
        for i, replica in enumerate(replicas):
            shard = requests[i * chunk_size:(i + 1) * chunk_size]
            if not shard:
                continue
            refs.append(replica.generate(shard, request_config))
        parts = ray.get(refs)
        responses = []
        for p in parts:
            responses.extend(p)

        for sample, response, t_encoded in zip(samples, responses, teacher_encodeds):
            parsed = parse_prompt_logprobs(response, topk=topk)
            encoded = t_encoded if t_encoded is not None else sample['encoded']
            opsd_labels = t_encoded.get('labels') if t_encoded is not None else None
            sample['teacher_output'] = self._build_per_sample_teacher_output(parsed, encoded, topk, opsd_labels)

    @staticmethod
    def _build_per_sample_teacher_output(parsed, encoded, topk, opsd_teacher_labels=None):
        """Build a per-sample TeacherOutput from parsed prompt logprobs.

        For OPSD, ``encoded`` is the teacher-side encoding and ``opsd_teacher_labels`` are
        its labels; together they let ``extract_active`` mask-align the shared response.
        """
        lps, ixs = parsed
        input_ids = encoded['input_ids']
        seq_len = len(input_ids) if isinstance(input_ids, list) else input_ids.shape[-1]

        parsed_len = len(lps)
        topk_logprobs = torch.full((seq_len, topk), float('-inf'), dtype=torch.float32)
        topk_indices = torch.zeros(seq_len, topk, dtype=torch.long)
        length = min(parsed_len, seq_len)
        if length > 0:
            topk_logprobs[:length] = torch.tensor(lps[:length], dtype=torch.float32)
            topk_indices[:length] = torch.tensor(ixs[:length], dtype=torch.long)

        kwargs = dict(topk_logprobs=topk_logprobs.unsqueeze(0), topk_indices=topk_indices.unsqueeze(0))
        if opsd_teacher_labels is not None:
            labels = opsd_teacher_labels
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.long)
            kwargs['opsd_teacher_labels'] = labels.unsqueeze(0) if labels.dim() == 1 else labels
        return TeacherOutput(**kwargs)

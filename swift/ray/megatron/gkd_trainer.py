# Copyright (c) ModelScope Contributors. All rights reserved.
"""Driver-side GKD trainer for Ray-based Megatron training."""
from __future__ import annotations

import copy
import random
import ray
import torch
from typing import List

from swift.infer_engine.protocol import RequestConfig, RolloutOutput
from swift.rlhf_trainers.gkd_loss import DataSource, TeacherOutput, build_opsd_teacher_data
from swift.rlhf_trainers.utils import (build_teacher_infer_request, parse_prompt_logprobs,
                                       replace_assistant_response_with_ids)
from swift.utils import get_logger
from .base_trainer import BaseRayTrainer
from .driver_utils import extract_iteration, extract_train_metrics

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

        self._steps_per_generation = 1
        # GKD generates exactly one completion per prompt (on-policy student generation),
        # so num_generations is always 1 here regardless of the (GRPO-oriented) default.
        info['num_generations'] = 1
        self._padding_to = info.get('_padding_to')
        # Prefer the resolved local snapshot dir (teacher_model_dir) over the raw model id
        # (teacher_model); bridge.load_weights needs a real path to locate safetensors.
        self._teacher_model_dir = getattr(args, 'teacher_model_dir', None) or args.teacher_model
        self._teacher_model_server = args.teacher_model_server

        if self._teacher_model_server and not self.teacher_replicas:
            raise NotImplementedError('teacher_model_server is not yet supported in the Ray pipeline. '
                                      'Use teacher_model (colocated) or teacher replicas (teacher.gpus > 0) instead.')

    def _train_loop(self, tg, train_iters, iteration):
        ckpt = self.ckpt_manager

        # Initialize colocated teacher if configured
        if self._teacher_model_dir and not self._teacher_model_server:
            tg.execute('init_teacher_model', self._teacher_model_dir)
            logger.info('Colocated teacher model initialized from %s', self._teacher_model_dir)

        while iteration < train_iters:
            prompt_batch = next(self._data_iter)
            data_source = self._determine_data_source()

            if data_source == DataSource.STUDENT:
                # On-policy: sync the latest weights to the rollout engine and generate.
                ckpt.sync_weights(merge_and_sync=True)
                with self._generation_context(tg, ckpt):
                    rollout_batch = self._expand_for_generation(prompt_batch)
                    completions = self._generate(rollout_batch)
                    rollout_with_outputs = self._postprocess_rollout(rollout_batch, completions)
                samples = self._encode_rollout_batch(rollout_with_outputs)
            else:
                # Off-policy (lmbda<1): distill on the dataset's ground-truth responses,
                # no generation. Dataset items already carry the assistant response, so
                # _encode_rollout_batch encodes it as-is (no token-in-token-out swap) and
                # no weight sync to the rollout engine is needed.
                rollout_with_outputs = None
                samples = self._encode_rollout_batch(prompt_batch)

            if self.teacher_replicas:
                if rollout_with_outputs is None:
                    raise NotImplementedError('Teacher replicas currently require on-policy generation (lmbda=1). '
                                              'Use a colocated teacher_model for lmbda<1 (off-policy) training.')
                self._fetch_teacher_from_replicas(rollout_with_outputs, samples)
            elif self._teacher_model_dir and not self._teacher_model_server:
                teacher_outputs = tg.compute_teacher_logits(samples)
                # In full_logits mode, compute_teacher_logits returns [] (logits cached on
                # worker GPUs); _inject_cached_teacher_logits handles injection at train_step.
                if teacher_outputs:
                    for sample, t_out in zip(samples, teacher_outputs):
                        sample['teacher_output'] = t_out

            self._maybe_log_completions(rollout_with_outputs, gen_step=iteration)
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

    def _encode_rollout_batch(self, rollout_batch):
        """Encode rollout samples for the training workers.

        Token-in-token-out: replace the assistant text with the on-policy
        ``response_token_ids`` before encoding, so the student is trained on the
        exact tokens that were generated (and that the teacher scores). Re-encoding
        the decoded text would re-tokenize the response and break student/teacher
        alignment. Mirrors the non-ray trainer's ``_encode_batch``.
        """
        template = self.template
        original_max_length = template.max_length
        samples = []
        try:
            template.max_length = original_max_length + self.args.max_completion_length
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
        finally:
            template.max_length = original_max_length
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
        """Fetch teacher logprobs from Ray teacher replicas (token-in-token-out)."""
        topk = self.gkd_logits_topk
        assert topk is not None, 'gkd_logits_topk must be set when using teacher replicas'

        # OPSD requires the teacher to score its own teacher_prompt and to set
        # opsd_teacher_labels for mask-based alignment. Standalone replicas only score the
        # student prompt and cannot produce those labels, which would silently misalign the
        # JSD. Fail loudly instead; use a colocated teacher_model for OPSD.
        if any(item.get('teacher_prompt') for item in rollout_with_outputs):
            raise NotImplementedError('OPSD (teacher_prompt) is not supported with standalone teacher replicas; '
                                      'use a colocated teacher_model for OPSD.')

        requests = [build_teacher_infer_request(item) for item in rollout_with_outputs]
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

        for sample, response in zip(samples, responses):
            parsed = parse_prompt_logprobs(response, topk=topk)
            teacher_output = self._build_per_sample_teacher_output(parsed, sample['encoded'], topk)
            sample['teacher_output'] = teacher_output

    @staticmethod
    def _build_per_sample_teacher_output(parsed, encoded, topk):
        """Build a per-sample TeacherOutput from parsed prompt logprobs."""
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

        return TeacherOutput(
            topk_logprobs=topk_logprobs.unsqueeze(0),
            topk_indices=topk_indices.unsqueeze(0),
        )

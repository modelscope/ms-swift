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
from swift.rl_core.data import GKDSample
from swift.rlhf_trainers.gkd_loss import DataSource, TeacherOutput
from swift.rlhf_trainers.utils import (get_non_thinking_prefix_ids, parse_prompt_logprobs,
                                       replace_assistant_response_with_ids)
from swift.utils import get_logger, remove_response
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
        self._teacher_model_dir = getattr(args, 'teacher_model_dir', None) or args.teacher_model
        self._teacher_model_server = args.teacher_model_server
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
        self._max_resample_rounds = getattr(args, 'max_resample_times', 3)
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
                    gkd_samples = self._postprocess_rollout(rollout_batch, completions)
                source_items = gkd_samples
            else:
                # Off-policy (lmbda<1): distill on the dataset's ground-truth responses,
                # no generation and no weight sync to the rollout engine.
                gkd_samples = [GKDSample.from_row(item) for item in prompt_batch]
                source_items = gkd_samples

            self._maybe_log_completions(gkd_samples if data_source == DataSource.STUDENT else None, gen_step=iteration)

            # Split one generation into ``spg`` chunks; each chunk is one training step
            # (same data_source). spg=1 degenerates to a single chunk == the whole batch.
            # n == global_batch_size * spg: the driver dataloader uses drop_last=True + a
            # cyclic iterator (see _setup_dataloader) and GKD uses num_generations=1, so n is
            # always an exact multiple of spg and the spg chunks tile source_items with no
            # remainder. (max(., 1) only guards the impossible spg > n case.)
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
                    if data_source != DataSource.STUDENT:
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
                results = tg.train_step(samples)
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
        """Convert prompt dicts to GKDSample and strip response for generation."""
        samples = [GKDSample.from_row(item) for item in prompt_batch]
        for s in samples:
            remove_response(s.messages)
        return samples

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

    def _postprocess_rollout(self, samples, outputs):
        """Merge rollout outputs back onto GKDSample (deepcopy to match HF path)."""
        results = []
        for sample, output in zip(samples, outputs):
            if output is not None:
                sample = copy.deepcopy(sample)
                sample.apply_rollout_output(rollout_output=output)
            results.append(sample)
        return results

    @contextmanager
    def _extended_max_length(self):
        """Temporarily extend template.max_length by max_completion_length so the
        prompt+response (token-in-token-out) encodes without truncation.
        """
        template = self.template
        original = template.max_length
        template.max_length = original + self.args.max_completion_length
        try:
            yield
        finally:
            template.max_length = original

    def _encode_rollout_batch(self, samples_or_dicts):
        """Encode samples into per-sample worker payloads.

        Handles both GKDSample (on-policy) and dict (off-policy) inputs.
        Deep-copies messages to avoid mutating the sample's history (driver may
        reuse samples across steps_per_generation).
        For OPSD, also encodes the teacher view (teacher_prompt + same response).
        """
        samples = [GKDSample.from_row(d) if isinstance(d, dict) else d for d in samples_or_dicts]
        template = self.template
        non_thinking_prefix_ids = get_non_thinking_prefix_ids(template)
        result = []
        with self._extended_max_length():
            for s in samples:
                # Student encode (deep-copy messages to avoid in-place mutation)
                item = s.to_template_dict()
                if item.get('messages') is not None:
                    item['messages'] = [m.copy() for m in item['messages']]
                if s.response_token_ids:
                    item['messages'] = replace_assistant_response_with_ids(
                        item['messages'], s.response_token_ids, non_thinking_prefix_ids=non_thinking_prefix_ids)
                encoded = template.encode(item, return_length=True)
                payload = {'encoded': encoded}

                # OPSD: encode teacher view (teacher_prompt + same response tokens)
                if s.build_teacher_view():
                    teacher_item = s.to_teacher_template_dict()
                    if teacher_item.get('messages') is not None:
                        teacher_item['messages'] = [m.copy() for m in teacher_item['messages']]
                    if teacher_item.get('response_token_ids'):
                        teacher_item['messages'] = replace_assistant_response_with_ids(
                            teacher_item['messages'],
                            teacher_item['response_token_ids'],
                            non_thinking_prefix_ids=non_thinking_prefix_ids)
                    payload['opsd_teacher_encoded'] = template.encode(teacher_item, return_length=True)
                result.append(payload)
        return result

    def _fetch_teacher_from_replicas(self, gkd_samples: List[GKDSample], samples):
        """Fetch teacher logprobs from Ray teacher replicas.

        Uses to_infer_request() + teacher_messages replacement (OPSD) to build
        unified RolloutInferRequest objects, matching HF GKD's _build_teacher_requests.
        """
        topk = self.gkd_logits_topk
        assert topk is not None, 'gkd_logits_topk must be set when using teacher replicas'

        requests = []
        teacher_encodeds = []  # teacher-side encoded (OPSD) or None (non-OPSD)
        for s, sample in zip(gkd_samples, samples):
            req = s.to_infer_request()
            opsd_encoded = sample.get('opsd_teacher_encoded')
            if s.teacher_messages:
                req.messages = s.teacher_messages
                teacher_encodeds.append(opsd_encoded)
            else:
                teacher_encodeds.append(None)
            requests.append(req)

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
    def _build_per_sample_teacher_output(parsed, encoded, topk, labels=None):
        """Build a per-sample TeacherOutput from parsed prompt logprobs.

        For OPSD, ``encoded`` is the teacher-side encoding and ``labels`` are
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
        if labels is not None:
            t_labels = labels
            if not isinstance(t_labels, torch.Tensor):
                t_labels = torch.tensor(t_labels, dtype=torch.long)
            kwargs['labels'] = t_labels.unsqueeze(0) if t_labels.dim() == 1 else t_labels
        return TeacherOutput(**kwargs)

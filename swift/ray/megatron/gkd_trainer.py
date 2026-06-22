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
from swift.rlhf_trainers.utils import get_non_thinking_prefix_ids, parse_prompt_logprobs
from swift.utils import get_logger, remove_response
from .base_trainer import BaseRayTrainer
from .driver_utils import extract_iteration
from .worker_group import DPDispatchedDict

logger = get_logger()


class GKDTrainer(BaseRayTrainer):

    def _prepare_state(self) -> None:
        super()._prepare_state()
        args = self.args

        self.sft_alpha = args.sft_alpha
        self.gkd_logits_topk = args.gkd_logits_topk
        # GKD on-policy schedule: each step is on-policy (student generates) with
        # probability ``lmbda``; otherwise off-policy (distill on dataset responses).
        self.lmbda = args.lmbda
        self._data_source_rng = random.Random(getattr(args, 'seed', 42))

        # GKD generates exactly one completion per prompt (on-policy student generation),
        # so num_generations is always 1 here regardless of the (GRPO-oriented) default.
        self._data_info['num_generations'] = 1
        self._teacher_model_dir = getattr(args, 'teacher_model_dir', None) or args.teacher_model
        self._teacher_model_server = args.teacher_model_server
        self._teacher_use_disable_adapter = args._teacher_use_disable_adapter
        if self._teacher_use_disable_adapter:
            self._teacher_model_dir = None

        if self._teacher_model_server and not self.teacher_replicas:
            raise NotImplementedError('teacher_model_server is not yet supported in the Ray pipeline. '
                                      'Use teacher_model (colocated) or teacher replicas (teacher.gpus > 0) instead.')

        vp_size = getattr(args, 'virtual_pipeline_model_parallel_size', None)
        assert vp_size is None or vp_size == 1, \
            'Ray GKD does not support VPP (virtual_pipeline_model_parallel_size > 1).'

        # truncation_strategy='delete': resample prompts whose encode fails (over max_length).
        self.truncation_strategy = args.truncation_strategy
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
                prompt_batch = self._resample_failed_prompts(prompt_batch, strip_response=False)
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

                use_colocated_teacher = self._teacher_use_disable_adapter or (self._teacher_model_dir
                                                                              and not self._teacher_model_server)
                if self.teacher_replicas:
                    if data_source != DataSource.STUDENT:
                        raise NotImplementedError('Teacher replicas currently require on-policy generation (lmbda=1). '
                                                  'Use a colocated teacher_model for lmbda<1 (off-policy) training.')
                    self._fetch_teacher_from_replicas(chunk, samples)

                # Driver collates the student (and, for the colocated path, the teacher view)
                # micro-batches; the worker only runs prepare_batch (PP/CP slice) + forward.
                dispatch = self._collate_for_workers_gkd(tg, samples, data_source, with_teacher=use_colocated_teacher)
                if use_colocated_teacher:
                    # Teacher forwards on the worker (CP slicing keeps each rank's shard
                    # aligned) and caches per-micro-batch; train_step attaches the cache.
                    tg.compute_teacher_logits(dispatch)
                results = tg.train_step(dispatch)
                iteration = extract_iteration(results)

        return iteration

    def _determine_data_source(self):
        """Pick the data source for this step (GKD on/off-policy schedule).

        With probability ``lmbda`` the step is on-policy (the student generates the
        response); otherwise it is off-policy and we distill on the dataset's
        ground-truth response.
        """
        if self._data_source_rng.random() < self.lmbda:
            return DataSource.STUDENT
        return DataSource.DATASET

    def _expand_for_generation(self, prompt_batch):
        """Convert prompt dicts to GKDSample and strip response for generation."""
        samples = [GKDSample.from_row(item) for item in prompt_batch]
        for s in samples:
            remove_response(s.messages)
        return samples

    def _generate(self, batch: List[GKDSample]) -> List[RolloutOutput]:
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
        # Convert samples to RolloutInferRequest at the engine boundary
        # (same pattern as GRPO Ray trainer).
        requests = [s.to_infer_request() for s in batch]
        completions = self._distribute_to_replicas(requests, request_config)
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

    def _encode_rollout_batch(self, samples: List[GKDSample]):
        """Encode samples into per-sample worker payloads.

        Uses the shared ``encode_gkd_samples`` helper (same as HF / Megatron GKD)
        so OPSD logic is fully encapsulated. Returns per-sample payloads with
        ``encoded`` (student) and optionally ``teacher_encoded`` (OPSD teacher view).
        """
        from swift.rlhf_trainers.gkd_helpers import encode_gkd_samples
        template = self.template
        with self._extended_max_length():
            student_list, teacher_list, has_opsd = encode_gkd_samples(samples, template)
        result = []
        for i in range(len(samples)):
            payload = {'encoded': student_list[i]}
            if has_opsd:
                payload['teacher_encoded'] = teacher_list[i]
            result.append(payload)
        return result

    def _collate_for_workers_gkd(self, tg, samples: List[dict], data_source, *, with_teacher: bool):
        """Driver-side GKD collate: ``List[payload-dict]`` -> ``{dp_rank: [model_inputs]}``.

        Mirrors the non-Ray GKD ``_encode_samples`` (data_collator on the rank, teacher
        forward later via prepare_batch). Each micro-batch ``model_inputs`` carries:
        - student forward tensors (``template.data_collator`` of ``encoded``),
        - ``data_source`` (SFT gating in loss_func),
        - ``teacher_model_inputs`` (colocated path): the collated teacher VIEW for the
          worker teacher forward — OPSD uses ``teacher_encoded``, else ``encoded``,
        - ``teacher_output`` (teacher-replicas path): the per-sample TeacherOutputs
          (already on each sample) collated into one batched TeacherOutput.
        """
        from swift.megatron.utils import get_padding_to
        from .megatron_worker import MegatronWorker

        template = self.template
        padding_to = self._padding_to if self._padding_to is not None else get_padding_to(self.args)
        dp_size = tg.dp_size
        mbs = int(self.args.micro_batch_size)
        n = len(samples)
        if n % dp_size != 0:
            raise ValueError(f'_collate_for_workers_gkd: batch size {n} not divisible by dp_size {dp_size}.')
        shard_size = n // dp_size

        dispatch = DPDispatchedDict()
        for dp_rank in range(dp_size):
            shard = samples[dp_rank * shard_size:(dp_rank + 1) * shard_size]
            micro_batches = []
            for i in range(0, len(shard), mbs):
                chunk = shard[i:i + mbs]
                model_inputs = template.data_collator([s['encoded'] for s in chunk], padding_to=padding_to)
                model_inputs['data_source'] = data_source
                if with_teacher:
                    has_opsd = chunk[0].get('teacher_encoded') is not None
                    key = 'teacher_encoded' if has_opsd else 'encoded'
                    model_inputs['teacher_model_inputs'] = template.data_collator(
                        batch=[s[key] for s in chunk], padding_to=padding_to)
                elif chunk[0].get('teacher_output') is not None:
                    # Teacher-replicas path: per-sample TeacherOutputs collated on the driver
                    # (pure tensor ops). The teacher seq length differs from the student under
                    # OPSD, so align by mask (is_opsd) rather than padding to the student length.
                    if getattr(self.args, 'context_parallel_size', 1) > 1:
                        raise ValueError('Standalone teacher replicas (teacher.gpus > 0) do not support '
                                         'context_parallel_size > 1: per-sample teacher token-logprobs are built '
                                         'from raw sequence lengths and cannot be CP-sharded to align with the '
                                         'student. Use a colocated teacher_model for CP>1.')
                    has_opsd = any(s.get('teacher_encoded') is not None for s in chunk)
                    model_inputs['teacher_output'] = MegatronWorker._collate_teacher_outputs(
                        [s['teacher_output'] for s in chunk],
                        self.device,
                        padding_free=template.padding_free,
                        target_seq_len=model_inputs['labels'].shape[-1],
                        is_opsd=has_opsd)
                micro_batches.append(model_inputs)
            dispatch[dp_rank] = micro_batches
        return dispatch

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
            teacher_encoded = sample.get('teacher_encoded')
            if s.teacher_messages:
                req.messages = s.teacher_messages
                teacher_encodeds.append(teacher_encoded)
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
            teacher_labels = t_encoded.get('labels') if t_encoded is not None else None
            sample['teacher_output'] = self._build_per_sample_teacher_output(parsed, encoded, topk, teacher_labels)

    @staticmethod
    def _build_per_sample_teacher_output(parsed, encoded, topk, labels=None):
        """Build a per-sample TeacherOutput from parsed prompt logprobs.

        For OPSD, ``encoded`` is the teacher-side encoding and ``labels`` are
        its labels; together they let ``extract_active`` mask-align the shared response.
        For non-OPSD, teacher and student share the same encoding, so we fall
        back to ``encoded['labels']`` when ``labels`` is not provided.
        """
        if labels is None:
            labels = encoded.get('labels')
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

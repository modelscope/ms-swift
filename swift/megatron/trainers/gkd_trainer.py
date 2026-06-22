# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import random
import torch
import torch.nn.functional as F
from contextlib import contextmanager
from functools import partial
from mcore_bridge import set_random_seed
from megatron.core import mpu
from megatron.core.rerun_state_machine import RerunDataIterator
from transformers import AutoConfig
from transformers.utils import ContextManagers
from typing import Dict, List, Optional

from swift.infer_engine.protocol import RequestConfig, RolloutInferRequest
from swift.megatron.arguments import MegatronArguments
from swift.megatron.model import get_mcore_model
from swift.rl_core.data import GKDSample
from swift.rlhf_trainers.gkd_helpers import assemble_teacher_output, build_teacher_requests, encode_gkd_samples
from swift.rlhf_trainers.gkd_loss import DataSource, TeacherOutput, gkd_loss
from swift.rlhf_trainers.utils import parse_prompt_logprobs
from swift.rlhf_trainers.vllm_client import VLLMInferClient
from swift.template import Template
from swift.utils import get_logger, is_last_rank, to_device
from ..utils import forward_step_helper, get_padding_to
from .gkd_utils import cp_reduce, tp_gather_topk, vocab_parallel_topk
from .rlhf_mixin import MegatronRLHFTrainer
from .rollout_mixin import MegatronRolloutMixin
from .utils import gather_object, load_megatron_model_to_gpu, offload_megatron_model_to_cpu
from .vocab_parallel_utils import vocab_parallel_kl_div, vocab_parallel_log_softmax

logger = get_logger()


class MegatronGKDTrainer(MegatronRolloutMixin, MegatronRLHFTrainer):

    sample_cls = GKDSample

    def __init__(self, args: MegatronArguments, template, **kwargs):
        self.vllm_client = kwargs.pop('vllm_client', None)

        # GKD-specific parameters
        self.beta = args.beta  # JSD interpolation coefficient
        self.temperature = args.temperature
        self.lmbda = args.lmbda  # On-policy probability
        self.offload_teacher_model = args.offload_teacher_model  # Offload teacher to CPU
        self.teacher_model_server = getattr(args, 'teacher_model_server', None)
        self.use_teacher_api = self.teacher_model_server is not None
        self._is_self_distillation = (args.teacher_model is None and self.teacher_model_server is None)
        self._teacher_use_disable_adapter = getattr(args, '_teacher_use_disable_adapter', False)
        if self._teacher_use_disable_adapter:
            logger.info('Self-distillation mode: using disable_adapter() for fixed teacher (no extra model)')
        self.sft_alpha = getattr(args, 'sft_alpha', 0.0)  # Weight for SFT loss

        # GKD top-k logits configuration
        self.gkd_logits_topk = getattr(args, 'gkd_logits_topk', None)

        self.use_vllm = getattr(args, 'use_vllm', False)
        self.steps_per_generation = args.steps_per_generation
        self.generation_batch_size = args.generation_batch_size
        super().__init__(args, template)

        if self.use_teacher_api:
            if is_last_rank():
                self.teacher_client = VLLMInferClient(base_urls=[self.teacher_model_server])
            else:
                self.teacher_client = None
            logger.info(f'Using teacher model API for logprobs, top_logprobs={self.gkd_logits_topk}')

        # Get device for data processing
        self.device = torch.cuda.current_device()

        # Initialize vLLM rollout engine if on-policy generation is enabled
        self._init_rollout_engine()

        # Truncation strategy for handling sequences that exceed max_length
        self.truncation_strategy = args.truncation_strategy
        self.max_completion_length = args.max_completion_length

        self.resample_data_iterator = None
        self._buffered_inputs = None

        self._prepare_logging()

    def train(self, train_dataset, val_dataset):
        if self.truncation_strategy == 'delete':
            self.resample_data_iterator = self._init_resample_data_iterator(train_dataset)
        super().train(train_dataset, val_dataset)

    def prepare_model(self):
        super().prepare_model()
        if self.use_teacher_api or self._is_self_distillation:
            if self._is_self_distillation:
                logger.info('Self-distillation mode: using student model as teacher (no separate teacher loaded)')
            else:
                logger.info('Skipping local teacher model loading - using external API for teacher logprobs')
            return
        args = self.args
        vp_size = getattr(args, 'virtual_pipeline_model_parallel_size')
        assert vp_size is None or vp_size == 1, 'GKD currently does not support VPP.'
        self.teacher_hf_config = AutoConfig.from_pretrained(args.teacher_model_dir, trust_remote_code=True)
        self.teacher_models = get_mcore_model(args, self.teacher_hf_config)
        self.teacher_config = self.teacher_models[0].config
        if not args.use_cpu_initialization:
            # same as wrap_model in megatron_lm_utils.py
            for teacher_model in self.teacher_models:
                teacher_model.cuda(torch.cuda.current_device())
        for teacher_model in self.teacher_models:
            teacher_model.requires_grad_(False)
            teacher_model.eval()
        self.teacher_config.bridge.load_weights(self.teacher_models, args.teacher_model_dir)

        # Offload teacher models to CPU if enabled
        if self.offload_teacher_model:
            self._offload_teacher_models()
            logger.info('Teacher models offloaded to CPU to save GPU memory')

    def _offload_teacher_models(self):
        """Offload teacher models to CPU to save GPU memory."""
        if self.teacher_models and not self.use_teacher_api:
            offload_megatron_model_to_cpu(self.teacher_models)

    def _load_teacher_models_to_gpu(self):
        """Load teacher models back to GPU."""
        if self.teacher_models and not self.use_teacher_api:
            load_megatron_model_to_gpu(self.teacher_models, load_grad=False)

    @contextmanager
    def load_teacher_model_context(self):
        """Context manager to load teacher models for forward pass and optionally offload after.

        When offload_teacher_model is enabled:
        - Load teacher models to GPU before forward pass
        - Offload teacher models to CPU after forward pass

        This saves GPU memory during the training step.
        """
        if not self.offload_teacher_model:
            yield
            return

        self._load_teacher_models_to_gpu()
        try:
            yield
        finally:
            self._offload_teacher_models()

    @contextmanager
    def _template_context(self, template: Template, max_length: Optional[int] = None):
        """Context manager to temporarily modify max_length constraint from template."""
        original_max_length = template.max_length
        template.max_length = max_length
        try:
            yield
        finally:
            template.max_length = original_max_length

    def _build_teacher_requests(self, samples: List[GKDSample]):
        if not self.use_teacher_api:
            return []
        return build_teacher_requests(samples)

    def _encode_samples(self, samples: List[GKDSample]) -> Dict[str, torch.Tensor]:
        template = self.template
        args = self.args

        with self._template_context(template):
            student_encoded_list, teacher_encoded_list, has_opsd = encode_gkd_samples(samples, template)

        padding_to = get_padding_to(args)
        encoded_batch = to_device(template.data_collator(student_encoded_list, padding_to=padding_to), self.device)
        if has_opsd:
            teacher_model_inputs = to_device(
                template.data_collator(teacher_encoded_list, padding_to=padding_to), self.device)
        else:
            teacher_model_inputs = encoded_batch.copy()
        encoded_batch['teacher_model_inputs'] = teacher_model_inputs
        return encoded_batch

    def _get_random_num(self) -> float:
        """Generate a deterministic random number consistent across all processes.

        Uses an isolated Random instance with seed based on args.seed + step counter

        Returns:
            float: A random number in the range [0.0, 1.0).
        """
        seed = int(getattr(self.args, 'seed', 0))
        seed += int(self._step)
        rng = random.Random(seed)
        return rng.random()

    def _determine_data_source(self) -> DataSource:
        """Determine data source for current step based on GKD algorithm.

        GKD training mode selection logic:
        1. With probability lmbda: On-Policy (student generates)
        2. Otherwise: Off-Policy (use dataset responses)

        Returns:
            DataSource enum indicating which source to use.
        """
        random_num = self._get_random_num()

        if random_num < self.lmbda:
            # Mode 1: On-Policy learning, student model generates responses
            if self.use_vllm:
                return DataSource.STUDENT
            else:
                # If vLLM not enabled, fall back to dataset
                logger.warning_once('On-policy mode triggered but use_vllm=False. '
                                    'Falling back to dataset responses. Enable vLLM for on-policy generation.')
                return DataSource.DATASET
        else:
            # Mode 2: Off-Policy learning, use dataset responses
            return DataSource.DATASET

    def _init_resample_data_iterator(self, train_dataset):
        """Initialize an independent data iterator for resampling.

        Uses a different seed (args.seed + 1) to avoid overlapping with training samples.

        Args:
            train_dataset: The training dataset to create the resample iterator from.

        Returns:
            The resample data iterator (first element of the iterator tuple).
        """
        args = self.args
        resample_seed = getattr(args, 'seed', 42) + 1
        try:
            set_random_seed(
                resample_seed,
                args.data_parallel_random_init,
                args.te_rng_tracker,
            )
            resample_data_iterator = self._prepare_data_iterator(train_dataset, use_origin_cyclic=True)[0]
        finally:
            set_random_seed(
                args.seed,
                args.data_parallel_random_init,
                args.te_rng_tracker,
            )
        return resample_data_iterator

    def resample_encode_failed_inputs(self, inputs: List[Dict], max_resample_rounds: int = 10) -> List[Dict]:
        """Attempt to encode each input. If encoding fails, resample until we have enough valid samples.

        Args:
            inputs: List of input data samples
            max_resample_rounds: Maximum number of resample rounds

        Returns:
            List of successfully encoded input samples with the same length as inputs
        """
        template = self.template
        required_count = len(inputs)
        valid_samples = []
        pending_samples = list(inputs)

        for _ in range(max_resample_rounds + 1):
            still_needed = required_count - len(valid_samples)
            if still_needed <= 0:
                break

            while len(pending_samples) < still_needed:
                pending_samples.extend(next(self.resample_data_iterator))

            while pending_samples and len(valid_samples) < required_count:
                data = pending_samples.pop(0)
                try:
                    template.encode(data)
                    valid_samples.append(data)
                except Exception as e:
                    logger.info(f'Encoding failed for one sample; will resample. {e}')

        if len(valid_samples) < required_count:
            raise RuntimeError(
                f'Failed to collect {required_count} valid samples after {max_resample_rounds} resample rounds. '
                f'Only collected {len(valid_samples)} valid samples. '
                'Consider increasing `max_length` or adjusting the `truncation_strategy`.')

        return valid_samples[:required_count]

    def _fetch_teacher_parsed_logprobs(self, requests: List[RolloutInferRequest]):
        """Fetch teacher logprobs from the teacher API server.

        Args:
            requests: List of RolloutInferRequest (from _build_teacher_requests).
        """
        rollout_group = self._get_rollout_group()
        rollout_rank = torch.distributed.get_rank(group=rollout_group)
        contribution = list(requests) if rollout_rank == 0 else []

        world_size = torch.distributed.get_world_size()
        all_contributions = [None] * world_size
        torch.distributed.all_gather_object(all_contributions, contribution)

        if self.is_main_process:
            flat_global = []
            for c in all_contributions:
                if c:
                    flat_global.extend(c)
            request_config = RequestConfig(prompt_logprobs=self.gkd_logits_topk, max_tokens=1, temperature=0.0)
            responses = self.teacher_client.infer(flat_global, request_config=request_config, use_tqdm=False)
            parsed_global = [parse_prompt_logprobs(r, topk=self.gkd_logits_topk) for r in responses]
        else:
            parsed_global = None

        obj_list = [parsed_global]
        torch.distributed.broadcast_object_list(obj_list, src=world_size - 1)
        parsed_global = obj_list[0]

        n = len(requests)
        dp_rank = mpu.get_data_parallel_rank()
        return parsed_global[dp_rank * n:(dp_rank + 1) * n]

    def _assemble_teacher_outputs(self, encoded_batches: List[Dict]) -> None:
        for encoded_batch in encoded_batches:
            parsed = encoded_batch.pop('_teacher_parsed')
            teacher_model_inputs = encoded_batch['teacher_model_inputs']
            teacher_out = assemble_teacher_output(
                parsed,
                teacher_model_inputs=teacher_model_inputs,
                topk=self.gkd_logits_topk,
                template_padding_free=self.template.padding_free,
                device=self.device,
            )
            if teacher_out.labels is not None:
                teacher_out.labels = torch.roll(teacher_out.labels, shifts=-1, dims=-1)
            encoded_batch['teacher_output'] = teacher_out

    def _compute_teacher_logits(self, encoded_batches: List[Dict], vp_stage: Optional[int] = None) -> None:
        if self.use_teacher_api:
            self._assemble_teacher_outputs(encoded_batches)
            return
        if self._is_self_distillation:
            # Self-distillation teacher == current student weights. Computing it here (at batch
            # preparation, once per steps_per_generation cycle) would reuse stale weights across the
            # cycle's train steps. Defer to _replace_data_iterator so each train step recomputes the
            # teacher with up-to-date student weights (weights are constant within a train step).
            return
        self._compute_teacher_logits_local(encoded_batches, vp_stage)

    def _compute_teacher_logits_local(self, encoded_batches: List[Dict], vp_stage: Optional[int] = None) -> None:
        """Compute teacher_output for each micro-batch via a local forward.

        Handles both a separate fixed teacher and self-distillation (teacher == current student
        weights). For self-distillation the caller is responsible for invoking this per train step
        so the weights are current.
        """
        topk = self.gkd_logits_topk
        if self._is_self_distillation:
            teacher_model = self.unwrapped_models[vp_stage or 0]
            adapter_contexts = []
            if self._teacher_use_disable_adapter:
                adapter_contexts = [m.disable_adapter() for m in self.peft_models]
            outer_context = ContextManagers(adapter_contexts)
        else:
            teacher_model = self.teacher_models[vp_stage or 0]
            outer_context = self.load_teacher_model_context()

        with torch.no_grad(), outer_context:
            for encoded_batch in encoded_batches:
                teacher_model_inputs = encoded_batch['teacher_model_inputs']
                teacher_batch = {
                    k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in teacher_model_inputs.items()
                }
                teacher_data = self._prepare_batch(teacher_batch, vp_stage)
                teacher_data.pop('loss_scale', None)
                teacher_labels = teacher_data.pop('labels', None)
                teacher_logits = forward_step_helper(teacher_model, teacher_data)
                if teacher_logits is not None:
                    teacher_logits = teacher_logits.detach()

                if topk is not None and teacher_logits is not None:
                    topk_logits, topk_indices = vocab_parallel_topk(teacher_logits, k=topk)
                    teacher_out = TeacherOutput(topk_logprobs=topk_logits, topk_indices=topk_indices)
                else:
                    teacher_out = TeacherOutput(full_logits=teacher_logits)

                teacher_out.labels = teacher_labels
                encoded_batch['teacher_output'] = teacher_out

    def _generate_and_score_completions(self, inputs: List[Dict]) -> List[Dict]:
        """Unified rollout → teacher → encode pipeline (mirrors Megatron GRPO).

        Stages: determine data source → to_samples → (student) generate → teacher
        requests/logprobs → encode micro-batches → teacher logits. Returns the flat
        list of encoded micro-batches (length == total microbatches).
        """
        data_source = self._determine_data_source()

        # Convert to samples (resample operates on dict, to_samples after)
        samples = self.to_samples(inputs)

        if data_source == DataSource.STUDENT:
            local_batch = self._get_local_rollout_batch(samples)
            local_batch = self._generate_completions(local_batch)
            samples = self._gather_rollout_results(local_batch)
            self._log_completions_from_samples(samples)
        elif data_source == DataSource.TEACHER:
            logger.warning_once('Teacher mode triggered but teacher generation is not implemented in Megatron GKD yet. '
                                'Falling back to dataset responses.')

        # Teacher API: build requests from samples, fetch logprobs
        local_parsed = None
        if self.use_teacher_api:
            teacher_requests = self._build_teacher_requests(samples)
            if teacher_requests:
                local_parsed = self._fetch_teacher_parsed_logprobs(teacher_requests)

        # Encode micro-batches
        total_microbatches = self.args.num_microbatches * self.steps_per_generation
        micro_batch_size = len(samples) // total_microbatches
        assert micro_batch_size == self.args.micro_batch_size
        all_encoded_batches = []
        for i in range(total_microbatches):
            start_idx = i * micro_batch_size
            end_idx = start_idx + micro_batch_size
            sample_slice = samples[start_idx:end_idx]
            encoded_batch = self._encode_samples(sample_slice)
            encoded_batch['data_source'] = data_source
            if local_parsed is not None:
                encoded_batch['_teacher_parsed'] = local_parsed[start_idx:end_idx]
            all_encoded_batches.append(encoded_batch)
        self._compute_teacher_logits(all_encoded_batches)
        return all_encoded_batches

    def _replace_data_iterator(self, data_iterator):
        num_microbatches = self.args.num_microbatches
        steps_per_generation = self.steps_per_generation

        if self._step % steps_per_generation == 0:
            total_microbatches = num_microbatches * steps_per_generation
            global_batch = []
            for _ in range(total_microbatches):
                raw_batch = next(data_iterator)
                if self.truncation_strategy == 'delete' and self.resample_data_iterator is not None:
                    raw_batch = self.resample_encode_failed_inputs(raw_batch)
                global_batch.extend(raw_batch)

            all_encoded_batches = self._generate_and_score_completions(global_batch)
            self._buffered_inputs = [
                all_encoded_batches[i * num_microbatches:(i + 1) * num_microbatches]
                for i in range(steps_per_generation)
            ]

        step_idx = self._step % steps_per_generation
        encoded_batches = self._buffered_inputs[step_idx]

        # Self-distillation teacher == current student weights. Recompute per train step (weights are
        # constant within a step) instead of once per generation cycle, so it tracks student updates
        # across steps_per_generation. Runs outside the pipeline schedule, so PP > 1 is supported.
        if self._is_self_distillation:
            self._compute_teacher_logits_local(encoded_batches)

        self._step += 1

        return RerunDataIterator(iter(encoded_batches))

    def loss_func(self,
                  output_tensor: torch.Tensor,
                  *,
                  labels: torch.Tensor,
                  teacher_output: TeacherOutput,
                  data_source: DataSource = DataSource.DATASET):
        """Compute GKD loss (JSD + optional SFT loss)."""
        student_logits = output_tensor

        jsd_total, jsd_num_valid = gkd_loss(
            student_logits,
            teacher_output,
            labels,
            self.beta,
            self.temperature,
            gather_fn=tp_gather_topk,
            log_softmax_fn=vocab_parallel_log_softmax,
            kl_div_fn=vocab_parallel_kl_div)
        jsd_loss_val = cp_reduce(jsd_total, jsd_num_valid, cp_size=self.args.context_parallel_size)

        loss = jsd_loss_val

        # Add SFT loss if enabled (skip for student-generated responses)
        sft_loss = None
        if self.sft_alpha > 0 and data_source != DataSource.STUDENT:
            args = self.args
            logits_sbv = student_logits.transpose(0, 1).contiguous()
            model = self.unwrapped_models[0]
            if hasattr(model, 'language_model'):
                model = model.language_model
            per_token_loss = model.compute_language_model_loss(labels, logits_sbv)
            loss_mask = labels != -100
            sft_loss_sum = (per_token_loss * loss_mask).sum()
            sft_loss_count = loss_mask.sum().float()

            # All-reduce across CP group for correct averaging
            if args.context_parallel_size > 1:
                sft_stats = torch.stack([sft_loss_sum, sft_loss_count])
                torch.distributed.all_reduce(
                    sft_stats, op=torch.distributed.ReduceOp.SUM, group=mpu.get_context_parallel_group())
                sft_loss_sum, sft_loss_count = sft_stats[0], sft_stats[1]

            sft_loss = sft_loss_sum / sft_loss_count

            loss = loss + self.sft_alpha * sft_loss

        metric = {'loss': loss.detach().clone()}
        if sft_loss is not None:
            metric['jsd_loss'] = jsd_loss_val.detach().clone()
            metric['sft_loss'] = sft_loss.detach().clone()
        metric = self._all_reduce_metric(metric)

        loss = loss / mpu.get_context_parallel_world_size()

        # Flush completion logs at generation cycle boundaries.
        if (self._step - 1) % self.steps_per_generation == 0:
            self._flush_log_completions()

        return loss, metric

    def forward_step(self, data_iterator, model):
        unwrapped_model = model.module.module
        input_tensor = unwrapped_model.get_input_tensor()
        vp_stage = unwrapped_model.vp_stage

        data = next(data_iterator)
        data_source = data.pop('data_source', DataSource.DATASET)
        teacher_output = data.pop('teacher_output')
        data.pop('teacher_model_inputs', None)  # consumed by _compute_teacher_logits; not needed for student forward
        data = self._prepare_batch(data, vp_stage)

        data.pop('loss_scale', None)
        labels = data.pop('labels', None)

        if input_tensor is not None:
            unwrapped_model.set_input_tensor(input_tensor)
        student_output = model(**data)

        return student_output, partial(
            self.loss_func,
            labels=labels,
            teacher_output=teacher_output,
            data_source=data_source,
        )

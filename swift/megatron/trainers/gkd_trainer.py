# Copyright (c) ModelScope Contributors. All rights reserved.
import random
import torch
import torch.nn.functional as F
from contextlib import contextmanager
from enum import Enum
from functools import partial
from mcore_bridge import set_random_seed
from megatron.core import mpu
from megatron.core.rerun_state_machine import RerunDataIterator
from transformers import AutoConfig
from transformers.utils import ContextManagers
from typing import Dict, List, Optional

from swift.megatron.arguments import MegatronArguments
from swift.megatron.model import get_mcore_model
from swift.rlhf_trainers.gkd_trainer import TeacherOutput
from swift.template import Template
from swift.utils import get_logger, to_device
from ..utils import forward_step_helper, get_padding_to
from .rlhf_mixin import MegatronRLHFTrainer
from .rollout_mixin import MegatronRolloutMixin
from .utils import load_megatron_model_to_gpu, offload_megatron_model_to_cpu
from .vocab_parallel_utils import vocab_parallel_kl_div, vocab_parallel_log_softmax

logger = get_logger()


class DataSource(str, Enum):
    """Data source for GKD training."""
    DATASET = 'dataset'  # Offline: use responses from dataset
    STUDENT = 'student'  # On-policy: use student-generated responses
    TEACHER = 'teacher'  # Sequential KD: use teacher-generated responses


class MegatronGKDTrainer(MegatronRolloutMixin, MegatronRLHFTrainer):

    def __init__(self, args: MegatronArguments, template, **kwargs):
        self.vllm_client = kwargs.pop('vllm_client', None)

        # GKD-specific parameters
        self.beta = args.beta  # JSD interpolation coefficient
        self.temperature = args.temperature
        self.lmbda = args.lmbda  # On-policy probability
        self.seq_kd = args.seq_kd  # Sequential KD: use teacher-generated responses
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

        if self.use_teacher_api:
            logger.info(f'Using teacher model API for logprobs, top_logprobs={self.gkd_logits_topk}')

        self.use_vllm = getattr(args, 'use_vllm', False)
        super().__init__(args, template)

        # Get device for data processing
        self.device = torch.cuda.current_device()

        # Initialize vLLM rollout engine if on-policy generation is enabled
        self._init_rollout_engine()

        # Truncation strategy for handling sequences that exceed max_length
        self.truncation_strategy = args.truncation_strategy
        self.max_completion_length = args.max_completion_length

        self.resample_data_iterator = None

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

    def _build_opsd_teacher_data(self, inputs: List[Dict]) -> Optional[List[Dict]]:
        """Build teacher data for OPSD by replacing the last user message with teacher_prompt."""
        if not all('teacher_prompt' in data and data['teacher_prompt'] for data in inputs):
            return None
        teacher_data = []
        for data in inputs:
            teacher_item = {k: v for k, v in data.items() if k != 'teacher_prompt'}
            messages = [dict(m) for m in data.get('messages', [])]
            for msg in reversed(messages):
                if msg['role'] == 'user':
                    msg['content'] = data['teacher_prompt']
                    break
            teacher_item['messages'] = messages
            teacher_data.append(teacher_item)
        return teacher_data

    def _encode_batch(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Encode a batch of raw data into model inputs."""
        template = self.template
        args = self.args
        max_length = template.max_length + self.max_completion_length
        with self._template_context(template, max_length=max_length):
            encoded_list = [template.encode(data, return_length=True) for data in batch]
            padding_to = get_padding_to(args)
            encoded_batch = to_device(template.data_collator(encoded_list, padding_to=padding_to), self.device)

        encoded_batch['num_samples'] = len(batch)
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
        2. If seq_kd=True and not on-policy: Sequential KD (teacher generates)
        3. Otherwise: Off-Policy (use dataset responses)

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
        elif self.seq_kd:
            # Mode 2: Sequential KD, teacher model generates responses
            # Note: Teacher generation is not implemented yet, use dataset
            logger.warning_once('seq_kd=True but teacher generation is not implemented in Megatron GKD yet. '
                                'Falling back to dataset responses.')
            return DataSource.DATASET
        else:
            # Mode 3: Off-Policy learning, use dataset responses
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

    def _compute_teacher_logits(self, encoded_batches: List[Dict], vp_stage: Optional[int] = None) -> None:
        if self.use_teacher_api:
            self._compute_teacher_logits_from_api(encoded_batches)
        else:
            self._compute_teacher_logits_local(encoded_batches, vp_stage)

    def _compute_teacher_logits_local(self, encoded_batches: List[Dict], vp_stage: Optional[int] = None) -> None:
        topk = self.gkd_logits_topk

        if self._is_self_distillation:
            teacher_model = self.unwrapped_models[0]
            adapter_contexts = []
            if self._teacher_use_disable_adapter:
                adapter_contexts = [m.disable_adapter() for m in self.peft_models]
            outer_context = ContextManagers(adapter_contexts)
        else:
            teacher_model = self.teacher_models[vp_stage or 0]
            outer_context = self.load_teacher_model_context()

        with torch.no_grad(), outer_context:
            for encoded_batch in encoded_batches:
                opsd_batch = encoded_batch.get('opsd_teacher_batch')
                source = opsd_batch if opsd_batch is not None else encoded_batch
                teacher_batch = {
                    k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in source.items() if k not in ('data_source', 'opsd_teacher_batch', 'teacher_output')
                }
                teacher_data = self._prepare_batch(teacher_batch)
                teacher_data.pop('loss_scale', None)
                opsd_teacher_labels = teacher_data.pop('labels', None)
                if opsd_batch is None:
                    opsd_teacher_labels = None
                teacher_logits = forward_step_helper(teacher_model, teacher_data)
                if teacher_logits is not None:
                    teacher_logits = teacher_logits.detach()

                if topk is not None and teacher_logits is not None:
                    topk_logits, topk_indices = self._vocab_parallel_topk(teacher_logits, k=topk)
                    teacher_out = TeacherOutput(topk_logprobs=topk_logits, topk_indices=topk_indices)
                else:
                    teacher_out = TeacherOutput(full_logits=teacher_logits)

                teacher_out.opsd_teacher_labels = opsd_teacher_labels
                encoded_batch['teacher_output'] = teacher_out

    def _compute_teacher_logits_from_api(self, encoded_batches: List[Dict]) -> None:
        from swift.rlhf_trainers.gkd_trainer import fetch_teacher_logprobs
        topk = self.gkd_logits_topk
        # One API call per DP group: rank 0 fetches, others receive via broadcast.
        rollout_group = self._get_rollout_group()
        rollout_rank = torch.distributed.get_rank(group=rollout_group)
        rollout_src = torch.distributed.get_global_rank(rollout_group, 0)

        for encoded_batch in encoded_batches:
            opsd_batch = encoded_batch.get('opsd_teacher_batch')
            source = opsd_batch if opsd_batch is not None else encoded_batch
            input_ids = source['input_ids']
            device = input_ids.device

            if rollout_rank == 0:
                teacher_logprobs, teacher_indices = fetch_teacher_logprobs(
                    self.teacher_model_server, input_ids.tolist(), topk=topk)
                teacher_logprobs = F.pad(teacher_logprobs, (0, 0, 0, 1), value=float('-inf'))
                teacher_indices = F.pad(teacher_indices, (0, 0, 0, 1), value=0)
                teacher_logprobs = teacher_logprobs.to(device)
                teacher_indices = teacher_indices.to(device)
            else:
                bs, seq_len = input_ids.shape
                teacher_logprobs = torch.empty(bs, seq_len, topk, dtype=torch.float32, device=device)
                teacher_indices = torch.empty(bs, seq_len, topk, dtype=torch.long, device=device)

            torch.distributed.broadcast(teacher_logprobs, src=rollout_src, group=rollout_group)
            torch.distributed.broadcast(teacher_indices, src=rollout_src, group=rollout_group)

            opsd_teacher_labels = opsd_batch.get('labels') if opsd_batch is not None else None
            if opsd_teacher_labels is not None:
                opsd_teacher_labels = torch.roll(opsd_teacher_labels, shifts=-1, dims=-1)
            encoded_batch['teacher_output'] = TeacherOutput(
                topk_logprobs=teacher_logprobs,
                topk_indices=teacher_indices,
                opsd_teacher_labels=opsd_teacher_labels,
            )

    def _replace_data_iterator(self, data_iterator):
        num_microbatches = self.args.num_microbatches

        # Determine data source once for the entire global batch
        data_source = self._determine_data_source()

        # Collect all micro-batches into a global batch
        global_batch = []
        for _ in range(num_microbatches):
            raw_batch = next(data_iterator)

            # Resample for encoding failed data when truncation_strategy is 'delete'
            if self.truncation_strategy == 'delete' and self.resample_data_iterator is not None:
                raw_batch = self.resample_encode_failed_inputs(raw_batch)

            global_batch.extend(raw_batch)

        # On-policy mode: generate completions for the entire global batch at once
        if data_source == DataSource.STUDENT:
            local_batch = self._get_local_rollout_batch(global_batch)
            local_batch = self._generate_completions(local_batch)
            global_batch = self._gather_rollout_results(local_batch)
        elif data_source == DataSource.TEACHER:
            logger.warning_once('Teacher mode triggered but teacher generation is not implemented in Megatron GKD yet. '
                                'Falling back to dataset responses.')

        # Build OPSD teacher data if teacher_prompt is present
        teacher_global_batch = self._build_opsd_teacher_data(global_batch)

        # Split global batch back into micro-batches for encoding
        encoded_batches = []
        micro_batch_size = len(global_batch) // num_microbatches
        for i in range(num_microbatches):
            start_idx = i * micro_batch_size
            end_idx = start_idx + micro_batch_size
            raw_batch = global_batch[start_idx:end_idx]
            encoded_batch = self._encode_batch(raw_batch)
            encoded_batch['data_source'] = data_source
            if teacher_global_batch is not None:
                encoded_batch['opsd_teacher_batch'] = self._encode_batch(teacher_global_batch[start_idx:end_idx])
            encoded_batches.append(encoded_batch)

        self._compute_teacher_logits(encoded_batches)

        # Increment step counter (used for deterministic random and weight sync)
        self._step += 1

        return RerunDataIterator(iter(encoded_batches))

    def _align_vocab_size(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> tuple:
        """Align vocab size between student and teacher logits.

        When student and teacher have different vocab sizes, pad the smaller one
        and copy logits from the larger one for the extra tokens.

        Args:
            student_logits: Student logits [..., student_vocab_size]
            teacher_logits: Teacher logits [..., teacher_vocab_size]

        Returns:
            Tuple of aligned (student_logits, teacher_logits)
        """
        stu_vocab = student_logits.shape[-1]
        tea_vocab = teacher_logits.shape[-1]

        if stu_vocab == tea_vocab:
            return student_logits, teacher_logits

        if stu_vocab < tea_vocab:
            # Pad student logits and copy teacher's extra vocab logits
            student_logits = F.pad(student_logits, (0, tea_vocab - stu_vocab), 'constant', 0)
            student_logits[..., stu_vocab:] = teacher_logits[..., stu_vocab:]
        else:
            # Pad teacher logits and copy student's extra vocab logits
            teacher_logits = F.pad(teacher_logits, (0, stu_vocab - tea_vocab), 'constant', 0)
            teacher_logits[..., tea_vocab:] = student_logits[..., tea_vocab:]

        return student_logits, teacher_logits

    def generalized_jsd_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        beta: float = 0.5,
        chunk_size: int = 512,
        teacher_topk_logprobs: Optional[torch.Tensor] = None,
        teacher_topk_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        args = self.args
        if labels is not None:
            mask = labels != -100
            local_num_valid = mask.sum()
        else:
            mask = None
            local_num_valid = torch.tensor(
                student_logits.shape[0] * student_logits.shape[1], dtype=torch.long, device=student_logits.device)
        num_valid = local_num_valid.float()

        # All-reduce num_valid across CP group for correct averaging
        if args.context_parallel_size > 1:
            torch.distributed.all_reduce(
                num_valid, op=torch.distributed.ReduceOp.SUM, group=mpu.get_context_parallel_group())

        if num_valid == 0:
            return (student_logits.sum() * 0).reshape(())

        # Top-k mode: direct computation without vocab parallel
        if teacher_topk_logprobs is not None and teacher_topk_indices is not None:
            if mask is None:
                mask = torch.ones(student_logits.shape[:2], dtype=torch.bool, device=student_logits.device)
            total_loss = self._jsd_topk(student_logits, teacher_topk_logprobs, teacher_topk_indices, mask, beta)
            if args.context_parallel_size > 1:
                torch.distributed.all_reduce(
                    total_loss, op=torch.distributed.ReduceOp.SUM, group=mpu.get_context_parallel_group())
            return total_loss / num_valid

        # Full vocabulary mode (original code)
        # Align vocab size between student and teacher
        student_logits, teacher_logits = self._align_vocab_size(student_logits, teacher_logits)

        if mask is not None:
            student_logits_masked = student_logits[mask]
            teacher_logits_masked = teacher_logits[mask]
        else:
            student_logits_masked = student_logits.view(-1, student_logits.size(-1))
            teacher_logits_masked = teacher_logits.view(-1, teacher_logits.size(-1))
        del student_logits, teacher_logits
        student_logits_masked.div_(self.temperature)
        teacher_logits_masked.div_(self.temperature)

        # Use local count for iteration, global count for averaging
        local_num_valid_int = local_num_valid.item()
        total_loss = student_logits_masked.new_zeros(())

        if beta != 0 and beta != 1:
            beta_t = torch.tensor(beta, dtype=student_logits_masked.dtype, device=student_logits_masked.device)
            log_beta = torch.log(beta_t)
            log_1_minus_beta = torch.log1p(-beta_t)
        else:
            beta_t = log_beta = log_1_minus_beta = None

        for start_idx in range(0, local_num_valid_int, chunk_size):
            end_idx = min(start_idx + chunk_size, local_num_valid_int)
            s_chunk = student_logits_masked[start_idx:end_idx]
            t_chunk = teacher_logits_masked[start_idx:end_idx]

            s_log_probs = vocab_parallel_log_softmax(s_chunk)
            t_log_probs = vocab_parallel_log_softmax(t_chunk)
            del s_chunk, t_chunk

            if beta == 0:
                jsd_chunk = vocab_parallel_kl_div(s_log_probs, t_log_probs)
            elif beta == 1:
                jsd_chunk = vocab_parallel_kl_div(t_log_probs, s_log_probs)
            else:
                mixture_log_probs = torch.logsumexp(
                    torch.stack([s_log_probs + log_1_minus_beta, t_log_probs + log_beta]),
                    dim=0,
                )
                kl_teacher = vocab_parallel_kl_div(mixture_log_probs, t_log_probs)
                kl_student = vocab_parallel_kl_div(mixture_log_probs, s_log_probs)
                del mixture_log_probs
                jsd_chunk = beta_t * kl_teacher + (1 - beta_t) * kl_student
                del kl_teacher, kl_student

            total_loss = total_loss + jsd_chunk.sum()
            del jsd_chunk, s_log_probs, t_log_probs

        del student_logits_masked, teacher_logits_masked

        # All-reduce total_loss across CP group for correct sum
        if args.context_parallel_size > 1:
            torch.distributed.all_reduce(
                total_loss, op=torch.distributed.ReduceOp.SUM, group=mpu.get_context_parallel_group())

        return total_loss / num_valid

    def _vocab_parallel_topk(self, logits: torch.Tensor, k: int) -> tuple:
        """Select global top-k from vocab-parallel sharded logits.

        When TP == 1, this is a plain torch.topk.
        When TP > 1, each rank holds a partition. We select local top-k on each
        rank, all-gather them, pick the global top-k, and return global indices
        (in the full vocab space) with corresponding logits.

        Returns:
            (topk_values, topk_indices) with global vocab indices.
        """
        tp_size = mpu.get_tensor_model_parallel_world_size()
        if tp_size == 1:
            return torch.topk(logits, k=k, dim=-1)

        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_group = mpu.get_tensor_model_parallel_group()
        partition_vocab_size = logits.shape[-1]

        local_topk_vals, local_topk_ids = torch.topk(logits, k=k, dim=-1)
        # Convert local indices to global vocab space
        local_topk_ids = local_topk_ids + tp_rank * partition_vocab_size

        # All-gather across TP ranks: each rank contributes k candidates
        gathered_vals = [torch.empty_like(local_topk_vals) for _ in range(tp_size)]
        gathered_ids = [torch.empty_like(local_topk_ids) for _ in range(tp_size)]
        torch.distributed.all_gather(gathered_vals, local_topk_vals, group=tp_group)
        torch.distributed.all_gather(gathered_ids, local_topk_ids, group=tp_group)

        # Concatenate: [..., tp_size * k] then pick global top-k
        all_vals = torch.cat(gathered_vals, dim=-1)
        all_ids = torch.cat(gathered_ids, dim=-1)
        global_topk_vals, sel = torch.topk(all_vals, k=k, dim=-1)
        global_topk_ids = torch.gather(all_ids, dim=-1, index=sel)

        return global_topk_vals, global_topk_ids

    def _tp_gather_topk(self, logits: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Gather logits at top-k indices with TP-aware vocab partitioning.

        When TP > 1, indices are global vocab IDs. Each rank gathers within its
        local partition (filling out-of-range positions with -inf) and all-reduces
        via MAX so every rank gets the correct value.

        When TP == 1, this is a plain torch.gather.
        """
        tp_size = mpu.get_tensor_model_parallel_world_size()
        if tp_size == 1:
            return torch.gather(logits, dim=-1, index=indices)

        tp_rank = mpu.get_tensor_model_parallel_rank()
        partition_vocab_size = logits.shape[-1]
        vocab_start = tp_rank * partition_vocab_size

        in_range = (indices >= vocab_start) & (indices < vocab_start + partition_vocab_size)
        local_indices = (indices - vocab_start).clamp(0, partition_vocab_size - 1)
        gathered = torch.gather(logits, dim=-1, index=local_indices)
        gathered = gathered.masked_fill(~in_range, float('-inf'))

        gathered_for_reduce = gathered.detach()
        torch.distributed.all_reduce(
            gathered_for_reduce, op=torch.distributed.ReduceOp.MAX, group=mpu.get_tensor_model_parallel_group())
        return torch.where(in_range, gathered, gathered_for_reduce)

    def _jsd_topk(self, student_logits, teacher_topk_logprobs, teacher_topk_indices, mask, beta):
        """Compute JSD on teacher's top-k distribution.

        teacher_topk_indices are always global vocab IDs (from both local teacher
        via _vocab_parallel_topk and API teacher). _tp_gather_topk handles
        the TP-safe gathering of student logits at those global positions.
        """
        s_topk = self._tp_gather_topk(student_logits, teacher_topk_indices)
        s_topk.div_(self.temperature)
        t_topk = teacher_topk_logprobs / self.temperature

        s_topk_masked = s_topk[mask]
        t_topk_masked = t_topk[mask]

        if s_topk_masked.numel() == 0:
            return student_logits.new_zeros(())

        t_log_p = F.log_softmax(t_topk_masked, dim=-1)
        s_log_p = F.log_softmax(s_topk_masked, dim=-1)
        t_p = torch.exp(t_log_p)

        if beta == 0:
            jsd = (t_p * (t_log_p - s_log_p)).sum(dim=-1)
        elif beta == 1:
            s_p = torch.exp(s_log_p)
            jsd = (s_p * (s_log_p - t_log_p)).sum(dim=-1)
        else:
            s_p = torch.exp(s_log_p)
            m_log_p = torch.log(beta * t_p + (1 - beta) * s_p + 1e-10)
            jsd = beta * (t_p * (t_log_p - m_log_p)).sum(-1) + (1 - beta) * (s_p * (s_log_p - m_log_p)).sum(-1)

        return jsd.sum()

    def loss_func(self,
                  output_tensor: torch.Tensor,
                  *,
                  labels: torch.Tensor,
                  teacher_output: TeacherOutput,
                  data_source: DataSource = DataSource.DATASET):
        """Compute GKD loss (JSD + optional SFT loss)."""
        student_logits = output_tensor
        teacher_output.validate()

        opsd_teacher_labels = teacher_output.opsd_teacher_labels
        if opsd_teacher_labels is not None:
            student_mask = labels != -100
            teacher_mask = opsd_teacher_labels != -100
            assert student_mask.sum() == teacher_mask.sum(), (
                f'OPSD label count mismatch: student={student_mask.sum().item()}, '
                f'teacher={teacher_mask.sum().item()}. '
                'Student and teacher must share the same response tokens.')
            s_logits = student_logits[student_mask][None]
            if teacher_output.is_topk_mode:
                t_logits = None
                topk_logprobs = teacher_output.topk_logprobs[teacher_mask][None]
                topk_indices = teacher_output.topk_indices[teacher_mask][None]
            else:
                t_logits = teacher_output.full_logits[teacher_mask][None]
                topk_logprobs = None
                topk_indices = None
            jsd_loss = self.generalized_jsd_loss(
                student_logits=s_logits,
                teacher_logits=t_logits,
                beta=self.beta,
                teacher_topk_logprobs=topk_logprobs,
                teacher_topk_indices=topk_indices,
            )
        else:
            jsd_loss = self.generalized_jsd_loss(
                student_logits=student_logits,
                teacher_logits=teacher_output.full_logits,
                labels=labels,
                beta=self.beta,
                teacher_topk_logprobs=teacher_output.topk_logprobs,
                teacher_topk_indices=teacher_output.topk_indices,
            )

        loss = jsd_loss

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
            metric['jsd_loss'] = jsd_loss.detach().clone()
            metric['sft_loss'] = sft_loss.detach().clone()
        metric = self._all_reduce_metric(metric)

        loss = loss / mpu.get_context_parallel_world_size()

        return loss, metric

    def forward_step(self, data_iterator, model):
        unwrapped_model = model.module.module
        input_tensor = unwrapped_model.get_input_tensor()
        vp_stage = unwrapped_model.vp_stage

        data = next(data_iterator)
        data_source = data.pop('data_source', DataSource.DATASET)
        teacher_output = data.pop('teacher_output', TeacherOutput())
        data.pop('opsd_teacher_batch', None)
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

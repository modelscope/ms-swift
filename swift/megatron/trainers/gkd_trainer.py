# Copyright (c) ModelScope Contributors. All rights reserved.
import random
import torch
import torch.nn.functional as F
from contextlib import contextmanager
from enum import Enum
from functools import partial
from megatron.core import mpu
from megatron.core.rerun_state_machine import RerunDataIterator
from typing import Dict, List, Optional

from swift.megatron.arguments import MegatronArguments
from swift.megatron.model import get_mcore_model
from swift.megatron.utils import set_random_seed
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
        if args.teacher_model:
            self.teacher_bridge = args.megatron_model_meta.bridge_cls(args, attr_prefix='teacher_')
            self.teacher_config = self.teacher_bridge.processor.model_info.config
        self.sft_alpha = getattr(args, 'sft_alpha', 0.0)  # Weight for SFT loss

        # GKD top-k logits configuration
        self.gkd_logits_topk = getattr(args, 'gkd_logits_topk', None)
        # Check use_teacher_api based on args, not client existence
        # (API client is only created on last rank, but all ranks need to know the mode)

        # Validate teacher configuration
        if not self.use_teacher_api:
            assert args.teacher_model is not None, \
                'Teacher model path is required for GKD training (or set teacher_model_server for API mode)'
        else:
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
        if self.use_teacher_api:
            logger.info('Skipping local teacher model loading - using external API for teacher logprobs')
            return
        args = self.args
        vp_size = getattr(args, 'virtual_pipeline_model_parallel_size')
        assert vp_size is None or vp_size == 1, 'GKD currently does not support VPP.'
        orig_model_dir = args.model_dir
        orig_model_type = args.model_type
        args.model_dir = args.teacher_model_dir
        args.model_type = args.teacher_model_type
        try:
            self.teacher_models = get_mcore_model(args, self.teacher_config)
        finally:
            args.model_dir = orig_model_dir
            args.model_type = orig_model_type
        for teacher_model in self.teacher_models:
            teacher_model.requires_grad_(False)
            teacher_model.eval()
        self.teacher_bridge.load_weights(self.teacher_models, args.teacher_model)

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
        teacher_model = self.teacher_models[vp_stage or 0]
        topk = self.gkd_logits_topk

        for encoded_batch in encoded_batches:
            teacher_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in encoded_batch.items()}
            teacher_batch.pop('data_source', None)
            teacher_data = self._prepare_batch(teacher_batch)
            teacher_data.pop('loss_scale', None)
            teacher_data.pop('labels', None)
            with self.load_teacher_model_context(), torch.no_grad():
                teacher_logits = forward_step_helper(self.args, teacher_model, teacher_data)
                if teacher_logits is not None:
                    teacher_logits = teacher_logits.detach()

            if topk is not None and teacher_logits is not None:
                topk_logits, topk_indices = torch.topk(teacher_logits, k=topk, dim=-1)
                encoded_batch['teacher_api_logprobs'] = topk_logits
                encoded_batch['teacher_api_indices'] = topk_indices
                encoded_batch['teacher_logits'] = None
            else:
                encoded_batch['teacher_logits'] = teacher_logits

    def _compute_teacher_logits_from_api(self, encoded_batches: List[Dict]) -> None:
        """Fetch teacher logprobs from external API service."""
        from swift.rlhf_trainers.gkd_trainer import fetch_teacher_logprobs
        topk = self.gkd_logits_topk
        for encoded_batch in encoded_batches:
            input_ids = encoded_batch['input_ids']
            teacher_logprobs, teacher_indices = fetch_teacher_logprobs(
                self.teacher_model_server, input_ids.tolist(), topk=topk)
            # fetch_teacher_logprobs returns [batch, seq_len-1, topk] (shifted).
            # Pad last position with -inf to match student [batch, seq_len, topk].
            teacher_logprobs = F.pad(teacher_logprobs, (0, 0, 0, 1), value=float('-inf'))
            teacher_indices = F.pad(teacher_indices, (0, 0, 0, 1), value=0)
            encoded_batch['teacher_api_logprobs'] = teacher_logprobs.to(input_ids.device)
            encoded_batch['teacher_api_indices'] = teacher_indices.to(input_ids.device)
            encoded_batch['teacher_logits'] = None

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

        # Split global batch back into micro-batches for encoding
        encoded_batches = []
        micro_batch_size = len(global_batch) // num_microbatches
        for i in range(num_microbatches):
            start_idx = i * micro_batch_size
            end_idx = start_idx + micro_batch_size
            raw_batch = global_batch[start_idx:end_idx]
            encoded_batch = self._encode_batch(raw_batch)
            # Store data_source for conditional SFT loss in loss_func
            encoded_batch['data_source'] = data_source
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
        labels: torch.Tensor,
        beta: float = 0.5,
        chunk_size: int = 512,
        teacher_topk_logprobs: torch.Tensor = None,
        teacher_topk_indices: torch.Tensor = None,
    ) -> torch.Tensor:
        args = self.args
        mask = labels != -100
        local_num_valid = mask.sum()
        num_valid = local_num_valid.float()

        # All-reduce num_valid across CP group for correct averaging
        if args.context_parallel_size > 1:
            torch.distributed.all_reduce(
                num_valid, op=torch.distributed.ReduceOp.SUM, group=mpu.get_context_parallel_group())

        if num_valid == 0:
            return (student_logits.sum() * 0).reshape(())

        # Top-k mode: direct computation without vocab parallel
        if teacher_topk_logprobs is not None and teacher_topk_indices is not None:
            total_loss = self._jsd_topk(student_logits, teacher_topk_logprobs, teacher_topk_indices, mask, beta)
            if args.context_parallel_size > 1:
                torch.distributed.all_reduce(
                    total_loss, op=torch.distributed.ReduceOp.SUM, group=mpu.get_context_parallel_group())
            return total_loss / num_valid

        # Full vocabulary mode (original code)
        # Align vocab size between student and teacher
        student_logits, teacher_logits = self._align_vocab_size(student_logits, teacher_logits)

        student_logits_masked = student_logits[mask]
        teacher_logits_masked = teacher_logits[mask]
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

    def _jsd_topk(self, student_logits, teacher_topk_logprobs, teacher_topk_indices, mask, beta):
        """Compute JSD on teacher's top-k distribution.

        Both local and API teacher are handled uniformly: gather student logits at
        teacher's top-k indices, scale by 1/T, and log_softmax over top-k subset.
        By shift-invariance of log_softmax, this gives identical results whether
        teacher_topk_logprobs contains raw logits (local) or raw logprobs (API).

        """
        s_topk = torch.gather(student_logits, dim=-1, index=teacher_topk_indices)
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
                  teacher_logits: torch.Tensor = None,
                  teacher_api_logprobs: torch.Tensor = None,
                  teacher_api_indices: torch.Tensor = None,
                  data_source: DataSource = DataSource.DATASET):
        """Compute GKD loss (JSD + optional SFT loss).

        Args:
            output_tensor: Student model logits [batch, seq_len, vocab_size]
            labels: Token labels for masking [batch, seq_len]
            teacher_logits: Teacher model logits [batch, seq_len, vocab_size] (for local teacher)
            teacher_api_logprobs: Teacher log probabilities [batch, seq_len, topk] (for API mode)
            teacher_api_indices: Teacher token indices [batch, seq_len, topk] (for API mode)
            data_source: Data source (STUDENT/TEACHER/DATASET) for conditional SFT loss
        """
        student_logits = output_tensor

        jsd_loss = self.generalized_jsd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            beta=self.beta,
            teacher_topk_logprobs=teacher_api_logprobs,
            teacher_topk_indices=teacher_api_indices,
        )

        loss = jsd_loss

        # Add SFT loss if enabled (skip for student-generated responses)
        sft_loss = None
        if self.sft_alpha > 0 and data_source != DataSource.STUDENT:
            args = self.args
            logits_sbv = student_logits.transpose(0, 1).contiguous()
            per_token_loss = self.unwrapped_models[0].compute_language_model_loss(labels, logits_sbv)

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
        teacher_logits = data.pop('teacher_logits', None)
        teacher_api_logprobs = data.pop('teacher_api_logprobs', None)
        teacher_api_indices = data.pop('teacher_api_indices', None)
        data = self._prepare_batch(data, vp_stage)

        data.pop('loss_scale', None)
        labels = data.pop('labels', None)

        if input_tensor is not None:
            unwrapped_model.set_input_tensor(input_tensor)
        student_output = model(**data)

        return student_output, partial(
            self.loss_func,
            labels=labels,
            teacher_logits=teacher_logits,
            teacher_api_logprobs=teacher_api_logprobs,
            teacher_api_indices=teacher_api_indices,
            data_source=data_source,
        )

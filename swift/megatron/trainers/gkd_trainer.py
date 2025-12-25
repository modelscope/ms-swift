# Copyright (c) Alibaba, Inc. and its affiliates.
import random
from contextlib import contextmanager
from enum import Enum
from functools import partial
from typing import Any, Dict, List, Literal, Optional

import torch
import torch.nn.functional as F
from megatron.core import mpu
from megatron.core.rerun_state_machine import RerunDataIterator
from megatron.training import get_args, get_model, get_timers
from megatron.training.utils import unwrap_model
from transformers import AutoConfig

from swift.llm import Template, get_model_info_meta, to_device
from swift.utils import get_logger
from ..argument import MegatronArguments
from ..model import get_megatron_model_meta
from ..utils import convert_hf_config, forward_step_helper, get_padding_to
from .rlhf_mixin import MegatronRLHFTrainer
from .rollout_mixin import MegatronRolloutMixin
from .utils import get_swift_datasets_provider, load_megatron_model_to_gpu, offload_megatron_model_to_cpu

logger = get_logger()


class DataSource(str, Enum):
    """Data source for GKD training."""
    DATASET = 'dataset'  # Off-policy: use responses from dataset
    STUDENT = 'student'  # On-policy: use student-generated responses
    TEACHER = 'teacher'  # Sequential KD: use teacher-generated responses


class MegatronGKDTrainer(MegatronRolloutMixin, MegatronRLHFTrainer):

    def __init__(self, args: MegatronArguments, template, **kwargs):
        self.vllm_client = kwargs.pop('vllm_client', None)
        super().__init__(args, template)

        # GKD-specific parameters
        self.beta = args.beta  # JSD interpolation coefficient
        self.temperature = args.temperature
        self.lmbda = args.lmbda  # On-policy probability
        self.seq_kd = args.seq_kd  # Sequential KD: use teacher-generated responses
        self.offload_teacher_model = args.offload_teacher_model  # Offload teacher to CPU
        assert args.teacher_model is not None, 'Teacher model path is required for GKD training'
        self.use_vllm = getattr(args, 'use_vllm', False)

        # Get device for data processing
        self.device = torch.cuda.current_device()

        # Initialize vLLM rollout engine if on-policy generation is enabled
        if self.use_vllm and self.lmbda > 0:
            self._init_rollout_engine()
            logger.info(f'GKD trainer initialized with on-policy generation (vLLM mode: {args.vllm_mode})')
        else:
            logger.info('GKD trainer initialized with off-policy training (dataset responses)')

        # Teacher models will be loaded in setup_model_and_optimizer
        # Using the same parallel parameters (PP/TP/CP/EP) as student model
        self.teacher_models = []

        # Teacher model config for temporary args override during forward
        # When teacher and student have different architecture, we need to override args temporarily
        self._teacher_megatron_config: Optional[Dict] = None  # Will be set in _load_teacher_model

        # Truncation strategy for handling sequences that exceed max_length
        self.truncation_strategy = args.truncation_strategy
        self.max_completion_length = args.max_completion_length

        # Resample iterator will be initialized lazily
        self.resample_data_iterator = None
        self._train_valid_test_dataset_provider = None

    def train(self, train_dataset, val_dataset, data_collator):
        """Override train to initialize resample iterator for truncation_strategy='delete'."""
        # Store dataset provider for lazy resample iterator initialization
        if self.truncation_strategy == 'delete':
            self._train_valid_test_dataset_provider = get_swift_datasets_provider(train_dataset, val_dataset)
            self._train_valid_test_dataset_provider.is_distributed = True
        super().train(train_dataset, val_dataset, data_collator)

    def setup_model_and_optimizer(self, model_provider_func, model_type, *_args, **kwargs):
        """Setup model and optimizer, including teacher model.

        Teacher model uses the same parallel parameters (PP/TP/CP/EP) as student model,
        """
        # Get teacher model path from Swift args
        teacher_model_path = self.args.teacher_model
        logger.info(f'Loading teacher model from: {teacher_model_path}')

        # Load teacher model with same parallel config as student
        self._load_teacher_model(teacher_model_path, model_type, model_provider_func)

        return super().setup_model_and_optimizer(model_provider_func, model_type, *_args, **kwargs)

    def _load_teacher_model(self, teacher_model_path: str, model_type, model_provider_func):
        """Load teacher model with the same parallel parameters (PP/TP/CP/EP) as student model.

        Teacher and student may have the same model_type (e.g., both are 'qwen2_5') but different
        architectures (e.g., Qwen2.5-3B vs Qwen2.5-7B with different hidden_size and num_layers).
        Therefore, we ALWAYS need to use teacher's config to create the model, not student's.

        Process:
        1. Get teacher model info and config
        2. Temporarily modify global Megatron args with teacher's config
        3. Create teacher model using get_model() (respects PP/TP/CP/EP settings from command line)
        4. Load teacher weights
        5. Restore original student config

        Args:
            teacher_model_path: Path to teacher model
            model_type: Megatron model type enum
            model_provider_func: Model provider function (not used, kept for API compatibility)
        """
        megatron_args = get_args()

        # Get teacher model info
        teacher_model_info, _ = get_model_info_meta(
            teacher_model_path,
            model_type=getattr(self.args, 'teacher_model_type', None),
            model_revision=getattr(self.args, 'teacher_model_revision', None),
            use_hf=self.args.use_hf,
            hub_token=self.args.hub_token)
        teacher_model_type = teacher_model_info.model_type

        # Get teacher's HF config and convert to Megatron config
        teacher_config = AutoConfig.from_pretrained(teacher_model_info.model_dir, trust_remote_code=True)
        teacher_megatron_model_meta = get_megatron_model_meta(teacher_model_type)
        if teacher_megatron_model_meta is None:
            raise ValueError(f'Teacher model type "{teacher_model_type}" is not supported in Megatron. '
                             f'Teacher model path: {teacher_model_path}')

        teacher_megatron_config = convert_hf_config(teacher_config)

        # Store teacher config for temporary args override during forward
        self._teacher_megatron_config = teacher_megatron_config

        logger.info(f'Loading teacher model: type={teacher_model_type}, '
                    f'hidden_size={teacher_megatron_config.get("hidden_size")}, '
                    f'num_layers={teacher_megatron_config.get("num_layers")}')

        # Store original student model config from Megatron global args
        # We need to override these with teacher's config temporarily
        essential_keys = {'hf_model_type', 'model_dir'}
        keys_to_override = set(teacher_megatron_config.keys()) | essential_keys

        original_config = {}
        for key in keys_to_override:
            if hasattr(megatron_args, key):
                original_config[key] = getattr(megatron_args, key)

        # Apply teacher config to global Megatron args
        for key, value in teacher_megatron_config.items():
            setattr(megatron_args, key, value)
        megatron_args.hf_model_type = teacher_model_type
        megatron_args.model_dir = teacher_model_info.model_dir

        try:
            # Use get_model() to create teacher with same parallel config (PP/TP/CP/EP) as student
            # but with teacher's model architecture (hidden_size, num_layers, etc.)
            teacher_models = get_model(teacher_megatron_model_meta.model_provider, model_type, wrap_with_ddp=False)

            # Create bridge for teacher model (for weight loading)
            teacher_bridge = teacher_megatron_model_meta.bridge_cls()

            # Load teacher weights
            for m in teacher_models:
                m = unwrap_model(m)
                teacher_bridge.load_weights(m, teacher_model_info.model_dir)

            logger.info(f'Teacher model loaded successfully with PP={megatron_args.pipeline_model_parallel_size}, '
                        f'TP={megatron_args.tensor_model_parallel_size}')

        finally:
            # Restore original student config to Megatron global args
            for key, value in original_config.items():
                setattr(megatron_args, key, value)

        self.teacher_models = teacher_models

        # Offload teacher models to CPU if enabled
        if self.offload_teacher_model:
            self._offload_teacher_models()
            logger.info('Teacher models offloaded to CPU to save GPU memory')

    @contextmanager
    def _teacher_args_context(self):
        """Context manager to temporarily override Megatron args with teacher's config.

        This is necessary for forward_step_helper to use correct hidden_size, num_layers, etc.
        when performing PP communication for teacher model.
        """
        megatron_args = get_args()

        # Save original values and override with teacher config
        original_values = {}
        for key, value in self._teacher_megatron_config.items():
            if hasattr(megatron_args, key):
                original_values[key] = getattr(megatron_args, key)
                setattr(megatron_args, key, value)

        try:
            yield
        finally:
            # Restore original values
            for key, value in original_values.items():
                setattr(megatron_args, key, value)

    def _offload_teacher_models(self):
        """Offload teacher models to CPU to save GPU memory."""
        if self.teacher_models:
            offload_megatron_model_to_cpu(self.teacher_models)

    def _load_teacher_models_to_gpu(self):
        """Load teacher models back to GPU."""
        if self.teacher_models:
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
        args = get_args()
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
        to ensure:
        1. Thread-safety (doesn't interfere with global random state)
        2. Cross-process consistency (all ranks get the same random number)
        3. Reproducibility (same seed + step = same random number)

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

    def _init_resample_data_iterator(self):
        """Initialize an independent data iterator for dynamic resampling (lazy initialization).

        Uses a different seed (args.seed + 1) to avoid overlapping with training samples.

        Returns:
            train_data_iterator: Independent data iterator with different random seed
        """
        from megatron.training.training import build_train_valid_test_data_iterators
        from megatron.training.initialize import _set_random_seed
        args = get_args()

        resample_seed = getattr(args, 'seed', 42) + 1
        try:
            _set_random_seed(
                resample_seed,
                args.data_parallel_random_init,
                args.te_rng_tracker,
                args.inference_rng_tracker,
                use_cudagraphable_rng=args.enable_cuda_graph,
            )
            resample_data_iterator, _, _ = build_train_valid_test_data_iterators(
                self._train_valid_test_dataset_provider)
        finally:
            _set_random_seed(
                args.seed,
                args.data_parallel_random_init,
                args.te_rng_tracker,
                args.inference_rng_tracker,
                use_cudagraphable_rng=args.enable_cuda_graph,
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

        # Lazy initialization of resample_data_iterator
        if self.resample_data_iterator is None:
            self.resample_data_iterator = self._init_resample_data_iterator()

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

    def _get_num_microbatches(self) -> int:
        """Get the number of microbatches for the current training step."""
        from megatron.core.num_microbatches_calculator import get_num_microbatches
        return get_num_microbatches()

    def _compute_teacher_logits(self, encoded_batches: List[Dict]) -> None:
        # Prepare batches for teacher forward (apply PP/CP transformations)
        for encoded_batch in encoded_batches:
            # Deep copy to avoid modifying original batch
            teacher_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in encoded_batch.items()}
            teacher_data = self._prepare_batch(teacher_batch)
            teacher_data.pop('loss_scale', None)
            # Remove labels so returns logits instead of loss
            teacher_data.pop('labels', None)

            # Teacher forward with args override for correct hidden_size
            with self.load_teacher_model_context(), self._teacher_args_context(), torch.no_grad():
                teacher_model = self.teacher_models[0]
                teacher_logits = forward_step_helper(teacher_model, teacher_data)
                if teacher_logits is not None:
                    teacher_logits = teacher_logits.detach()
            encoded_batch['teacher_logits'] = teacher_logits

    def _replace_data_iterator(self, data_iterator, model):
        num_microbatches = self._get_num_microbatches()

        # Determine data source once for the entire global batch
        data_source = self._determine_data_source()

        # Collect all micro-batches into a global batch
        global_batch = []
        for _ in range(num_microbatches):
            raw_batch = next(data_iterator)

            # Resample for encoding failed data when truncation_strategy is 'delete'
            if self.truncation_strategy == 'delete' and self._train_valid_test_dataset_provider is not None:
                raw_batch = self.resample_encode_failed_inputs(raw_batch)

            global_batch.extend(raw_batch)

        # On-policy mode: generate completions for the entire global batch at once
        # This avoids multiple wake/sleep/offload cycles and maximizes vLLM KV cache efficiency
        if data_source == DataSource.STUDENT:
            # Split global batch within rollout group for distributed generation
            local_batch = self._get_local_rollout_batch(global_batch)

            # Generate completions for local batch
            # NOTE: _generate_completions must be called by ALL ranks because
            # _move_model_to_vllm contains collective operations (all_reduce in export_weights).
            # The actual vLLM inference will handle empty batch gracefully.
            local_batch = self._generate_completions(local_batch)

            # Gather results from all ranks in rollout group
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
    ) -> torch.Tensor:
        """Compute the generalized Jensen-Shannon Divergence loss with chunked computation.

        JSD(p, q) = beta * KL(p || m) + (1 - beta) * KL(q || m)
        where m = beta * p + (1 - beta) * q, p = teacher, q = student

        This implementation uses chunked computation to reduce peak memory usage,
        which is critical for large vocab sizes.

        Args:
            student_logits: Student model logits [batch, seq_len, vocab_size]
            teacher_logits: Teacher model logits [batch, seq_len, vocab_size]
            labels: Token labels for masking [batch, seq_len], already shifted
            beta: Interpolation coefficient (0.5 = symmetric JSD)
            chunk_size: Number of tokens to process in each chunk

        Returns:
            Scalar loss value (mean over valid tokens)
        """
        mask = labels != -100
        num_valid = mask.sum()

        if num_valid == 0:
            return (student_logits.sum() * 0).reshape(())

        # Align vocab size between student and teacher
        student_logits, teacher_logits = self._align_vocab_size(student_logits, teacher_logits)

        # Apply temperature scaling and mask
        student_logits_masked = (student_logits / self.temperature)[mask]  # [num_valid_tokens, vocab_size]
        teacher_logits_masked = (teacher_logits / self.temperature)[mask]
        del student_logits, teacher_logits

        num_valid_int = num_valid.item()
        total_loss = student_logits_masked.new_zeros(())

        if beta != 0 and beta != 1:
            beta_t = torch.tensor(beta, dtype=student_logits_masked.dtype, device=student_logits_masked.device)
            log_beta = torch.log(beta_t)
            log_1_minus_beta = torch.log1p(-beta_t)
        else:
            beta_t = log_beta = log_1_minus_beta = None

        for start_idx in range(0, num_valid_int, chunk_size):
            end_idx = min(start_idx + chunk_size, num_valid_int)
            s_chunk = student_logits_masked[start_idx:end_idx]
            t_chunk = teacher_logits_masked[start_idx:end_idx]

            s_log_probs = F.log_softmax(s_chunk, dim=-1)
            t_log_probs = F.log_softmax(t_chunk, dim=-1)
            del s_chunk, t_chunk

            if beta == 0:
                jsd_chunk = F.kl_div(s_log_probs, t_log_probs, reduction='none', log_target=True)
            elif beta == 1:
                jsd_chunk = F.kl_div(t_log_probs, s_log_probs, reduction='none', log_target=True)
            else:
                mixture_log_probs = torch.logsumexp(
                    torch.stack([s_log_probs + log_1_minus_beta, t_log_probs + log_beta]),
                    dim=0,
                )

                kl_teacher = F.kl_div(mixture_log_probs, t_log_probs, reduction='none', log_target=True)
                kl_student = F.kl_div(mixture_log_probs, s_log_probs, reduction='none', log_target=True)
                del mixture_log_probs

                jsd_chunk = beta_t * kl_teacher + (1 - beta_t) * kl_student
                del kl_teacher, kl_student
            total_loss = total_loss + jsd_chunk.sum()
            del jsd_chunk, s_log_probs, t_log_probs

        # Clean up masked logits
        del student_logits_masked, teacher_logits_masked

        return total_loss / num_valid

    def loss_func(self, output_tensor: torch.Tensor, *, labels: torch.Tensor, teacher_logits: torch.Tensor):
        student_logits = output_tensor

        jsd_loss = self.generalized_jsd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            beta=self.beta,
        )

        loss = jsd_loss

        metric = {
            'loss': loss.detach().clone(),
            'jsd_loss': jsd_loss.detach().clone(),
        }
        metric = self._all_reduce_metric(metric)

        loss = loss / mpu.get_context_parallel_world_size()

        return loss, metric

    def forward_step(self, data_iterator, model):

        timers = get_timers()

        unwrapped_model = model.module.module
        input_tensor = unwrapped_model.get_input_tensor()
        vp_stage = unwrapped_model.vp_stage

        timers('batch-generator', log_level=2).start()
        with self.stimer(bdata=True):
            data = self.get_batch(data_iterator, vp_stage)
        timers('batch-generator').stop()

        data.pop('loss_scale', None)
        labels = data.pop('labels', None)
        teacher_logits = data.pop('teacher_logits', None)

        if input_tensor is not None:
            unwrapped_model.set_input_tensor(input_tensor)
        with self.stimer:
            student_output = model(**data)

        return student_output, partial(self.loss_func, labels=labels, teacher_logits=teacher_logits)

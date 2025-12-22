# Copyright (c) Alibaba, Inc. and its affiliates.
import random
from contextlib import contextmanager
from enum import Enum
from functools import partial
from typing import Dict, List, Literal, Optional

import torch
import torch.nn.functional as F
from megatron.core import mpu
from megatron.training import get_args, get_model, get_timers
from megatron.training.utils import unwrap_model
from transformers import AutoConfig

from swift.llm import Template, get_model_info_meta, to_device
from swift.utils import get_logger
from ..argument import MegatronArguments
from ..model import get_megatron_model_meta
from ..utils import convert_hf_config
from .rlhf_mixin import MegatronRLHFTrainer
from .rollout_mixin import MegatronRolloutMixin

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

        self.use_vllm = getattr(args, 'use_vllm', False)

        # Get device for data processing
        self.device = torch.cuda.current_device()

        # Initialize vLLM rollout engine if on-policy generation is enabled
        if self.use_vllm and self.lmbda > 0:
            self._init_rollout_engine()
            logger.info(f'GKD trainer initialized with on-policy generation (vLLM mode: {args.vllm_mode})')
        else:
            logger.info('GKD trainer initialized with off-policy training (dataset responses)')

        # Step counter for weight sync tracking and deterministic random
        self._step = 0

        # Teacher model will be loaded in setup_model_and_optimizer
        self.teacher_models = None

        logger.info(f'GKD params: beta={self.beta}, temperature={self.temperature}, '
                    f'lmbda={self.lmbda}, seq_kd={self.seq_kd}')

    def setup_model_and_optimizer(self, model_provider_func, model_type, *_args, **kwargs):
        """Setup model and optimizer, including teacher model."""
        megatron_args = get_args()

        # Get teacher model path from Swift args (defaults to ref_model if not specified)
        teacher_model_path = (getattr(self.args, 'teacher_model', None) or self.args.model)
        logger.info(f'Loading teacher model from: {teacher_model_path}')

        # Load teacher model
        self._load_teacher_model(teacher_model_path, model_type)

        # Call parent's setup (skip ref_models loading since we use teacher_models)
        # We need to temporarily disable ref_models loading
        original_rlhf_type = megatron_args.rlhf_type
        megatron_args.rlhf_type = 'rm'  # Trick to skip ref_models loading in parent
        try:
            result = super().setup_model_and_optimizer(model_provider_func, model_type, *_args, **kwargs)
        finally:
            megatron_args.rlhf_type = original_rlhf_type

        return result

    def _load_teacher_model(self, teacher_model_path: str, model_type):
        """Load teacher model with config automatically read from model path.

        This method handles both same-architecture and heterogeneous teacher models by:
        1. Getting teacher model info (including model_type) from the model path
        2. Temporarily modifying global Megatron args with teacher's config
        3. Creating and loading the teacher model
        4. Restoring original student config

        Args:
            teacher_model_path: Path to teacher model
            model_type: Megatron model type enum
        """
        megatron_args = get_args()

        # Get teacher model info using Swift's get_model_info_meta
        # Note: model_type=None allows auto-detection from teacher model path
        teacher_model_info, _ = get_model_info_meta(
            teacher_model_path,
            model_type=None,  # Auto-detect from teacher model
            use_hf=self.args.use_hf,
            hub_token=self.args.hub_token)
        teacher_model_type = teacher_model_info.model_type

        # Load teacher's HF config directly (get_model_info_meta doesn't return config)
        teacher_config = AutoConfig.from_pretrained(teacher_model_info.model_dir, trust_remote_code=True)
        logger.info(f'Teacher HF config loaded: hidden_size={teacher_config.hidden_size}, '
                    f'num_hidden_layers={teacher_config.num_hidden_layers}')

        # Get teacher's megatron model meta (may differ from student's)
        teacher_megatron_model_meta = get_megatron_model_meta(teacher_model_type)
        if teacher_megatron_model_meta is None:
            raise ValueError(f'Teacher model type "{teacher_model_type}" is not supported in Megatron. '
                             f'Teacher model path: {teacher_model_path}')

        # Convert HF config to Megatron config
        teacher_megatron_config = convert_hf_config(teacher_config)
        logger.info(f'Teacher model type: {teacher_model_type}')
        logger.info(f'Teacher model config: {teacher_megatron_config}')

        # Debug: Print current student config before modification
        logger.info(f'Student hidden_size before: {megatron_args.hidden_size}')
        logger.info(f'Student num_layers before: {megatron_args.num_layers}')
        logger.info(f'Student padded_vocab_size before: {megatron_args.padded_vocab_size}')

        # Store original student model config from Megatron global args
        original_config = {}
        config_keys = [
            'hidden_size',
            'num_layers',
            'num_attention_heads',
            'ffn_hidden_size',
            'num_query_groups',
            'kv_channels',
            'padded_vocab_size',
            'max_position_embeddings',
            'hf_model_type',
            'model_dir',  # Required for GPTBridge._init_meta_hf_model
        ]
        for key in config_keys:
            if hasattr(megatron_args, key):
                original_config[key] = getattr(megatron_args, key)

        # Apply teacher config to global Megatron args
        for key in config_keys:
            if key in teacher_megatron_config:
                setattr(megatron_args, key, teacher_megatron_config[key])
        # Set hf_model_type so model_provider can find correct megatron_model_meta
        megatron_args.hf_model_type = teacher_model_type
        # Set model_dir so GPTBridge._init_meta_hf_model uses teacher's model dir
        megatron_args.model_dir = teacher_model_info.model_dir

        # Debug: Print teacher config after modification
        logger.info(f'Teacher hidden_size after setting: {megatron_args.hidden_size}')
        logger.info(f'Teacher num_layers after setting: {megatron_args.num_layers}')
        logger.info(f'Teacher padded_vocab_size after setting: {megatron_args.padded_vocab_size}')
        logger.info(f'Teacher model_dir after setting: {megatron_args.model_dir}')

        # Verify that get_args() returns the same object
        verify_args = get_args()
        logger.info(f'Verify: get_args() is megatron_args: {verify_args is megatron_args}')
        logger.info(f'Verify hidden_size from fresh get_args(): {verify_args.hidden_size}')

        try:
            # Create teacher model using teacher's model_provider
            teacher_models = get_model(teacher_megatron_model_meta.model_provider, model_type, wrap_with_ddp=False)

            # Debug: Check teacher model's actual embedding shape
            for i, m in enumerate(teacher_models):
                unwrapped = unwrap_model(m)
                lm_model = getattr(unwrapped, 'language_model', unwrapped)
                if hasattr(lm_model, 'embedding') and hasattr(lm_model.embedding, 'word_embeddings'):
                    embed_weight = lm_model.embedding.word_embeddings.weight
                    logger.info(f'Teacher model [{i}] embedding weight shape: {embed_weight.shape}')
                else:
                    logger.info(f'Teacher model [{i}] does not have embedding.word_embeddings')

            # Create bridge for teacher model (uses megatron_args.model_dir for meta HF model)
            teacher_bridge = teacher_megatron_model_meta.bridge_cls()

            # Debug: Check bridge's args
            logger.info(f'Bridge args.hidden_size: {teacher_bridge.args.hidden_size}')
            logger.info(f'Bridge args.model_dir: {teacher_bridge.args.model_dir}')

            # Load teacher weights
            for m in teacher_models:
                m = unwrap_model(m)
                teacher_bridge.load_weights(m, teacher_model_info.model_dir)
                m.requires_grad_(False).eval()

            self.teacher_models = teacher_models
            logger.info(f'Loaded teacher model: hidden_size={teacher_megatron_config.get("hidden_size")}, '
                        f'num_layers={teacher_megatron_config.get("num_layers")}')

        finally:
            # Restore original student config to Megatron global args
            for key, value in original_config.items():
                setattr(megatron_args, key, value)

    @contextmanager
    def null_ref_context(self):
        """Override to use teacher_models instead of ref_models."""
        yield self.teacher_models

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

        with self._template_context(template):
            encoded_list = [template.encode(data, return_length=True) for data in batch]
            encoded_batch = to_device(template.data_collator(encoded_list), self.device)

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

    def get_batch(self, data_iterator, vp_stage=None):
        """Get batch from data iterator.

        Implements GKD's three training modes:
        1. On-Policy (prob lmbda): student-generated responses
        2. Sequential KD (seq_kd=True): teacher-generated responses
        3. Off-Policy (default): dataset responses
        """
        # Get raw batch from iterator
        raw_batch = next(data_iterator)

        # Determine data source for this step
        data_source = self._determine_data_source()

        # On-policy mode: generate completions using student model via vLLM
        if data_source == DataSource.STUDENT:
            raw_batch = self._generate_completions(raw_batch, step=self._step)

        # Increment step counter (used for deterministic random and weight sync)
        self._step += 1

        # Encode the batch
        encoded_batch = self._encode_batch(raw_batch)

        # Process through parent's _prepare_batch
        return self._prepare_batch(encoded_batch, vp_stage)

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
    ) -> torch.Tensor:
        """Compute the generalized Jensen-Shannon Divergence loss.

        JSD(p, q) = beta * KL(p || m) + (1 - beta) * KL(q || m)
        where m = beta * p + (1 - beta) * q

        Args:
            student_logits: Student model logits [batch, seq_len, vocab_size]
            teacher_logits: Teacher model logits [batch, seq_len, vocab_size]
            labels: Token labels for masking [batch, seq_len]
            beta: Interpolation coefficient (0.5 = symmetric JSD)

        Returns:
            Scalar loss value
        """
        # Get mask from labels
        mask = labels != -100

        # Align vocab size between student and teacher (handles different tokenizers)
        student_logits, teacher_logits = self._align_vocab_size(student_logits, teacher_logits)

        # Apply temperature scaling
        student_logits = student_logits / self.temperature
        teacher_logits = teacher_logits / self.temperature

        # Compute log probabilities (more numerically stable)
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        # Compute probabilities
        student_probs = student_log_probs.exp()
        teacher_probs = teacher_log_probs.exp()

        # Mixture distribution: m = beta * student + (1 - beta) * teacher
        mixture_probs = beta * student_probs + (1 - beta) * teacher_probs
        mixture_log_probs = torch.log(mixture_probs + 1e-10)

        # JSD = beta * KL(student || mixture) + (1 - beta) * KL(teacher || mixture)
        # KL(p || q) = sum(p * (log(p) - log(q)))
        kl_student = (student_probs * (student_log_probs - mixture_log_probs)).sum(dim=-1)
        kl_teacher = (teacher_probs * (teacher_log_probs - mixture_log_probs)).sum(dim=-1)

        jsd = beta * kl_student + (1 - beta) * kl_teacher

        # Apply mask and compute mean
        jsd = jsd * mask
        loss = jsd.sum() / mask.sum().clamp(min=1)

        return loss

    def loss_func(self, output_tensor: torch.Tensor, *, labels: torch.Tensor, teacher_logits: torch.Tensor,
                  packed_seq_params):
        """Compute GKD loss from student and teacher logits.

        Args:
            output_tensor: Student logits [batch, seq_len, vocab_size]
            labels: Token labels for masking
            teacher_logits: Teacher logits [batch, seq_len, vocab_size]
            packed_seq_params: Packed sequence parameters

        Returns:
            Tuple of (loss, metric_dict)
        """
        student_logits = output_tensor
        # teacher_logits is passed via partial, already detached

        # Compute JSD loss
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

        # Fix megatron-lm bug for pipeline parallelism
        loss = loss / mpu.get_context_parallel_world_size()

        return loss, metric

    def forward_step(self, data_iterator, model):
        """Forward step for GKD training.

        1. Get batch data
        2. Forward teacher model WITHOUT labels to get logits
        3. Forward student model WITHOUT labels to get logits
        4. Compute JSD loss in loss_func

        Note: By not passing labels, Megatron models return logits instead of loss.
        We compute JSD loss directly from logits for true knowledge distillation.

        Important: Both teacher and student forward must be executed on ALL ranks
        to maintain NCCL collective operation consistency.
        """
        timers = get_timers()

        # Get the batch
        unwrapped_model = model.module.module
        input_tensor = unwrapped_model.get_input_tensor()
        vp_stage = unwrapped_model.vp_stage

        timers('batch-generator', log_level=2).start()
        with self.stimer(bdata=True):
            data = self.get_batch(data_iterator, vp_stage)
        timers('batch-generator').stop()

        # Store labels for loss computation
        labels = data.pop('labels')
        data.pop('loss_scale', None)

        # Teacher forward - without labels to get logits
        # IMPORTANT: Execute on all ranks to ensure consistent collective operations
        with torch.no_grad():
            teacher_model = self.teacher_models[vp_stage or 0]
            # Clone input tensor if available (for PP intermediate stages)
            teacher_input = input_tensor.detach().clone() if input_tensor is not None else None
            if teacher_input is not None:
                teacher_model.set_input_tensor(teacher_input)
            # Returns [batch, seq_len, vocab] when labels=None (post_process=True)
            # Returns [seq_len, batch, hidden] for intermediate PP stages
            teacher_output = teacher_model(**data)

        # Student forward - without labels to get logits
        if input_tensor is not None:
            unwrapped_model.set_input_tensor(input_tensor)
        with self.stimer:
            student_output = model(**data)

        # For loss_func, we pass teacher logits via partial
        # teacher_output is detached (no gradient) but needs to be on same device
        teacher_logits = teacher_output.detach()

        return student_output, partial(
            self.loss_func,
            labels=labels,
            teacher_logits=teacher_logits,
            packed_seq_params=data.get('packed_seq_params'))

    def custom_log(self, total_loss_dict, mode: Literal['train', 'eval']) -> None:
        """Custom logging for GKD metrics."""
        super().custom_log(total_loss_dict, mode)

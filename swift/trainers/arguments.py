# Copyright (c) Alibaba, Inc. and its affiliates.
import math
import os
import platform
from dataclasses import dataclass, field
from functools import wraps
from typing import List, Literal, Optional, Union

import torch
import torch.utils.checkpoint
from transformers.training_args import TrainingArguments as HfTrainingArguments
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments as HfSeq2SeqTrainingArguments

from swift.utils import get_dist_setting, get_logger, is_liger_available, use_torchacc
from .optimizers.galore import GaLoreConfig

logger = get_logger()


@dataclass
class TrainArgumentsMixin:
    """
    check_model (bool): Flag to check the model is latest. Default is True.
    acc_strategy (Literal['token', 'seq']): Strategy for accumulation. Default is 'token'.
    """
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: Optional[int] = None

    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: Optional[Union[dict, str]] = None
    logging_first_step: bool = True
    logging_steps: int = 5

    weight_decay: float = 0.1
    adam_beta2: float = 0.95
    lr_scheduler_type: str = 'cosine'
    lr_scheduler_kwargs: Optional[Union[dict, str]] = None
    report_to: List[str] = field(default_factory=lambda: ['tensorboard'])
    dataloader_num_workers: Optional[int] = None
    dataloader_prefetch_factor: Optional[int] = None

    # extra
    check_model: bool = True
    acc_strategy: Literal['token', 'seq'] = 'token'
    train_dataloader_shuffle: bool = True

    # torchacc
    metric_warmup_step: Optional[float] = 0
    fsdp_num: int = 1
    acc_steps: int = 1
    use_liger_kernel: bool = False

    # train-eval loop args
    eval_use_evalscope: bool = False
    eval_datasets: List[str] = field(default_factory=list)
    eval_limit: Optional[int] = None
    eval_datasets_args: Optional[Union[str, dict]] = None
    eval_generation_config: Optional[Union[str, dict]] = None

    def _fix_gradient_checkpointing(self):
        # fix use_reentrant
        if hasattr(torch.utils.checkpoint, '_old_checkpoint'):  # avoid double patching
            return
        # Consistent with the default behavior of transformers.
        use_reentrant_ = (
            self.gradient_checkpointing_kwargs.get('use_reentrant', True)
            if self.gradient_checkpointing_kwargs else True)
        _old_checkpoint = torch.utils.checkpoint.checkpoint

        @wraps(_old_checkpoint)
        def _new_checkpoint(*args, use_reentrant=None, **kwargs):
            return _old_checkpoint(*args, use_reentrant=use_reentrant_, **kwargs)

        torch.utils.checkpoint._old_checkpoint = _old_checkpoint
        torch.utils.checkpoint.checkpoint = _new_checkpoint
        try:
            # Fix the old version of transformers.
            import transformers.modeling_utils
            transformers.modeling_utils.checkpoint = _new_checkpoint
        except (ImportError, AttributeError):
            pass

    def _init_liger(self):
        if self.use_liger_kernel:
            assert is_liger_available(), 'use_liger_kernel requires liger_kernels, try `pip install liger-kernel`'

    def __post_init__(self):
        from swift.llm.argument.base_args.model_args import ModelArguments
        if use_torchacc():
            self.dataloader_drop_last = True
        if self.gradient_accumulation_steps is None:
            world_size = get_dist_setting()[2]
            self.gradient_accumulation_steps = max(1, math.ceil(16 / self.per_device_train_batch_size / world_size))
            logger.info(f'Setting args.gradient_accumulation_steps: {self.gradient_accumulation_steps}')
        if self.lr_scheduler_kwargs:
            self.lr_scheduler_kwargs = ModelArguments.parse_to_dict(self.lr_scheduler_kwargs)
        if self.gradient_checkpointing_kwargs:
            self.gradient_checkpointing_kwargs = ModelArguments.parse_to_dict(self.gradient_checkpointing_kwargs)
        self._fix_gradient_checkpointing()
        self._init_liger()
        if self.dataloader_num_workers is None:
            if platform.system() == 'Windows':
                self.dataloader_num_workers = 0
            else:
                self.dataloader_num_workers = 1
            logger.info(f'Setting args.dataloader_num_workers: {self.dataloader_num_workers}')
        if self.dataloader_prefetch_factor is None and self.dataloader_num_workers > 0:
            self.dataloader_prefetch_factor = 10
        if self.eval_use_evalscope:
            try:
                import evalscope
            except ImportError:
                raise ImportError('evalscope is not installed, please install it by `pip install evalscope`')
            self.eval_datasets_args = ModelArguments.parse_to_dict(self.eval_datasets_args)
            self.eval_generation_config = ModelArguments.parse_to_dict(self.eval_generation_config)

        super().__post_init__()


@dataclass
class SwiftArgumentsMixin(TrainArgumentsMixin):
    # Value copied from TrainArguments
    train_type: Optional[str] = None
    optimizer: Optional[str] = None
    local_repo_path: Optional[str] = None
    galore_config: Optional[GaLoreConfig] = None

    def __post_init__(self):
        if hasattr(self, 'output_dir'):
            self.output_dir = os.path.abspath(os.path.expanduser(self.output_dir))
        super().__post_init__()

    @property
    def place_model_on_device(self):
        return False if use_torchacc() else super().place_model_on_device


@dataclass
class GRPOArgumentsMixin:
    epsilon: float = 0.2
    epsilon_high: Optional[float] = None
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.
    # vllm_device, vllm_gpu_memory_utilization, and vllm_max_model_len are defined in HfGRPOConfig.
    num_infer_workers: int = 1
    vllm_max_num_seqs: int = 256
    vllm_enforce_eager: bool = False
    vllm_limit_mm_per_prompt: Optional[Union[dict, str]] = None  # '{"image": 5, "video": 2}'
    vllm_enable_prefix_caching: bool = True
    # reward function args, see details in swift/plugin/orm.py
    # cosine reward, https://arxiv.org/abs/2502.03373
    cosine_min_len_value_wrong: float = -0.5  # r^w_0 in paper, Reward for wrong answers with zero completion length.
    cosine_max_len_value_wrong: float = 0.0  # r^w_L in paper, Reward for wrong answers with max completion length.
    cosine_min_len_value_correct: float = 1.0  # r^c_0 in paper, Reward for correct answers with zero completion length.
    cosine_max_len_value_correct: float = 0.5  # r^c_L in paper, Reward for correct answers with max completion length.
    cosine_max_len: Optional[int] = None  # Lmax in paper, default equal to max_completion_length
    # repetition penalty, https://arxiv.org/abs/2502.03373
    repetition_n_grams: int = 3
    repetition_max_penalty: float = -1.0

    # LMDeploy in GRPO
    use_lmdeploy: bool = False
    lmdeploy_device: Optional[str] = 'auto'
    lmdeploy_session_len: Optional[int] = None
    lmdeploy_cache_max_entry_count: float = 0.8

    async_generate: bool = False
    tensor_parallel_size: int = 1
    sleep_level: int = 0
    move_model_batches: Optional[int] = None
    offload_optimizer: bool = False
    offload_model: bool = False
    gc_collect_after_offload: bool = False
    multi_turn_func: Optional[str] = None

    # mini-batch
    mini_batch_size: Optional[int] = None


@dataclass
class TrainingArguments(SwiftArgumentsMixin, HfTrainingArguments):
    pass


@dataclass
class Seq2SeqTrainingArguments(SwiftArgumentsMixin, HfSeq2SeqTrainingArguments):
    pass

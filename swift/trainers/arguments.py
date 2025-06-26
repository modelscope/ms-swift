# Copyright (c) Alibaba, Inc. and its affiliates.
import math
import os
import platform
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union

from transformers.training_args import TrainingArguments as HfTrainingArguments
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments as HfSeq2SeqTrainingArguments

from swift.utils import get_dist_setting, get_logger, is_liger_available, is_mp, use_torchacc
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
    vit_gradient_checkpointing: Optional[bool] = None
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
    use_liger_kernel: bool = False

    # extra
    check_model: bool = True
    acc_strategy: Literal['token', 'seq'] = 'token'
    train_dataloader_shuffle: bool = True
    max_epochs: Optional[int] = None
    aligner_lr: Optional[float] = None
    vit_lr: Optional[float] = None
    optimizer: Optional[str] = None
    use_logits_to_keep: Optional[bool] = None
    channels: List[str] = None
    ds3_gather_for_generation: bool = True

    # torchacc
    metric_warmup_step: Optional[float] = 0
    fsdp_num: int = 1
    acc_steps: int = 1

    # train-eval loop args
    eval_use_evalscope: bool = False
    eval_datasets: List[str] = field(default_factory=list)
    eval_limit: Optional[int] = None
    eval_datasets_args: Optional[Union[str, dict]] = None
    eval_generation_config: Optional[Union[str, dict]] = None

    @staticmethod
    def _patch_liger_kernel():
        # fix logits_to_keep
        from liger_kernel.transformers.model import loss_utils
        origin_LigerForCausalLMLoss = loss_utils.LigerForCausalLMLoss

        def LigerForCausalLMLoss(hidden_states, *args, **kwargs):
            hidden_states = hidden_states.contiguous()
            return origin_LigerForCausalLMLoss(hidden_states, *args, **kwargs)

        loss_utils.LigerForCausalLMLoss = LigerForCausalLMLoss
        logger.info('Patch liger_kernel successfully.')

    def _init_liger(self):
        if self.use_liger_kernel:
            assert is_liger_available(), 'use_liger_kernel requires liger_kernels, try `pip install liger-kernel`'
            try:
                self._patch_liger_kernel()
            except Exception:
                pass

    def __post_init__(self):
        if is_mp() and self.use_liger_kernel:
            raise ValueError('liger_kernel does not support device_map. '
                             'Please use DDP/DeepSpeed for multi-GPU training.')

        from swift.llm.argument.base_args.model_args import ModelArguments
        if self.optimizer is None and (self.vit_lr is not None or self.aligner_lr is not None):
            self.optimizer = 'multimodal'
        if use_torchacc():
            self.dataloader_drop_last = True
        if self.gradient_accumulation_steps is None:
            world_size = get_dist_setting()[2]
            self.gradient_accumulation_steps = max(1, math.ceil(16 / self.per_device_train_batch_size / world_size))
            logger.info(f'Setting args.gradient_accumulation_steps: {self.gradient_accumulation_steps}')
        if self.lr_scheduler_kwargs:
            self.lr_scheduler_kwargs = ModelArguments.parse_to_dict(self.lr_scheduler_kwargs)
        if self.vit_gradient_checkpointing is None:
            self.vit_gradient_checkpointing = self.gradient_checkpointing
        if self.gradient_checkpointing_kwargs:
            self.gradient_checkpointing_kwargs = ModelArguments.parse_to_dict(self.gradient_checkpointing_kwargs)
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
class RLHFArgumentsMixin:
    # gkd
    sft_alpha: float = 0


@dataclass
class SwiftArgumentsMixin(RLHFArgumentsMixin, TrainArgumentsMixin):
    # Value copied from TrainArguments
    train_type: Optional[str] = None
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
    delta: Optional[float] = None
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.
    num_infer_workers: Optional[int] = None  # deprecated
    # vllm
    vllm_mode: Literal['server', 'colocate'] = 'colocate'
    # internal vllm (colocate)
    vllm_device: Optional[List[str]] = None  # deprecated
    vllm_gpu_memory_utilization: float = 0.9
    vllm_max_model_len: Optional[int] = None
    vllm_max_num_seqs: Optional[int] = None  # deprecated
    vllm_enforce_eager: bool = False
    vllm_limit_mm_per_prompt: Optional[Union[dict, str]] = None  # '{"image": 5, "video": 2}'
    vllm_enable_prefix_caching: bool = True
    vllm_tensor_parallel_size: int = 1

    # external vllm (server)
    vllm_server_base_url: Optional[str] = None
    vllm_server_host: Optional[str] = None
    vllm_server_port: int = 8000
    vllm_server_timeout: float = 240.0
    vllm_client = None  # Not required to set, used for client instantiation

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

    reward_model: Optional[List[str]] = None
    reward_model_plugin: Optional[List[str]] = None

    # sync ref model
    sync_ref_model: bool = False
    ref_model_sync_steps: int = 512
    ref_model_mixup_alpha: float = 0.6

    async_generate: bool = False
    tensor_parallel_size: Optional[int] = None  # deprecated

    sleep_level: int = 0
    move_model_batches: Optional[int] = None
    offload_optimizer: bool = False
    offload_model: bool = False
    gc_collect_after_offload: bool = False

    # multi turn
    multi_turn_func: Optional[str] = None  # deprecated
    multi_turn_scheduler: Optional[str] = None
    max_turns: Optional[int] = None
    completion_length_limit_scope: Literal['total', 'per_round'] = 'per_round'

    # DAPO, https://arxiv.org/abs/2503.14476
    dynamic_sample: bool = False
    max_resample_times: int = 3
    overlong_filter: bool = False
    soft_max_length: Optional[int] = None
    soft_cache_length: Optional[int] = None

    # Dr. GRPO, https://arxiv.org/abs/2503.20783
    scale_rewards: bool = True

    wandb_log_unique_prompts: Optional[bool] = None
    generation_batch_size: Optional[int] = None
    steps_per_generation: Optional[int] = None

    # dataset
    dataset_shuffle: Optional[bool] = True


@dataclass
class TrainingArguments(SwiftArgumentsMixin, HfTrainingArguments):
    pass


@dataclass
class Seq2SeqTrainingArguments(SwiftArgumentsMixin, HfSeq2SeqTrainingArguments):
    pass

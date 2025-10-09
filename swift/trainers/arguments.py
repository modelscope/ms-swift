# Copyright (c) Alibaba, Inc. and its affiliates.
import math
import os
import platform
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union

from transformers.training_args import TrainingArguments as HfTrainingArguments
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments as HfSeq2SeqTrainingArguments

from swift.plugin import loss_mapping
from swift.utils import get_dist_setting, get_logger, is_liger_available, is_mp, json_parse_to_dict
from .optimizers.galore import GaLoreConfig

logger = get_logger()


@dataclass
class TrainArgumentsMixin:
    """
    check_model (bool): Flag to check the model is latest. Default is True.
    acc_strategy (Literal['token', 'seq']): Strategy for accumulation. Default is 'token'.
    optimizer (Optional[str]): Optimizer type to use, define it in the plugin package. Default is None.
    loss_type (Optional[str]): Type of loss function to use. Default is None.
    metric (Optional[str]): Metric to use for evaluation, define it in the plugin package. Default is None.
    """
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: Optional[int] = None
    tuner_backend: Optional[str] = None

    gradient_checkpointing: bool = True
    vit_gradient_checkpointing: Optional[bool] = None
    gradient_checkpointing_kwargs: Optional[Union[dict, str]] = None
    logging_first_step: bool = True
    logging_steps: int = 5
    router_aux_loss_coef: float = 0.
    enable_dft_loss: bool = False  # https://arxiv.org/abs/2508.05629
    enable_channel_loss: bool = False

    weight_decay: float = 0.1
    adam_beta2: float = 0.95
    lr_scheduler_type: str = 'cosine'
    lr_scheduler_kwargs: Optional[Union[dict, str]] = None
    report_to: List[str] = field(default_factory=lambda: ['tensorboard'])
    dataloader_num_workers: Optional[int] = None
    dataloader_persistent_workers: bool = False
    dataloader_prefetch_factor: Optional[int] = None
    use_liger_kernel: bool = False

    # extra
    check_model: bool = True
    acc_strategy: Literal['token', 'seq'] = 'token'
    train_dataloader_shuffle: bool = True
    max_epochs: Optional[int] = None
    aligner_lr: Optional[float] = None
    vit_lr: Optional[float] = None
    use_logits_to_keep: Optional[bool] = None
    ds3_gather_for_generation: bool = True
    resume_only_model: bool = False

    optimizer: Optional[str] = None
    loss_type: Optional[str] = field(default=None, metadata={'help': f'loss_func choices: {list(loss_mapping.keys())}'})
    metric: Optional[str] = None

    # train-eval loop args
    eval_use_evalscope: bool = False
    eval_dataset: List[str] = field(default_factory=list)
    eval_dataset_args: Optional[Union[str, dict]] = None
    eval_limit: Optional[int] = None
    eval_generation_config: Optional[Union[str, dict]] = None
    extra_eval_args: Optional[Union[str, dict]] = None

    # dlrover flash_checkpoint
    use_flash_ckpt: bool = False

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

        if self.optimizer is None and (self.vit_lr is not None or self.aligner_lr is not None):
            self.optimizer = 'multimodal'
        if self.gradient_accumulation_steps is None:
            world_size = get_dist_setting()[2]
            self.gradient_accumulation_steps = max(1, math.ceil(16 / self.per_device_train_batch_size / world_size))
            logger.info(f'Setting args.gradient_accumulation_steps: {self.gradient_accumulation_steps}')
        if self.lr_scheduler_kwargs:
            self.lr_scheduler_kwargs = json_parse_to_dict(self.lr_scheduler_kwargs)
        if self.vit_gradient_checkpointing is None:
            self.vit_gradient_checkpointing = self.gradient_checkpointing
        if self.gradient_checkpointing_kwargs:
            self.gradient_checkpointing_kwargs = json_parse_to_dict(self.gradient_checkpointing_kwargs)
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
            self.eval_dataset_args = json_parse_to_dict(self.eval_dataset_args)
            self.eval_generation_config = json_parse_to_dict(self.eval_generation_config)
            self.extra_eval_args = json_parse_to_dict(self.extra_eval_args)

        super().__post_init__()


@dataclass
class RLHFArgumentsMixin:
    # gkd
    sft_alpha: float = 0
    # chord
    chord_sft_dataset: List[str] = field(default_factory=list)
    chord_sft_per_device_train_batch_size: Optional[int] = None

    chord_enable_phi_function: bool = False
    chord_mu_warmup_steps: Optional[int] = None
    chord_mu_decay_steps: Optional[int] = None
    chord_mu_peak: Optional[float] = None
    chord_mu_valley: Optional[float] = None


@dataclass
class SwiftArgumentsMixin(RLHFArgumentsMixin, TrainArgumentsMixin):
    # Value copied from TrainArguments
    train_type: Optional[str] = None
    local_repo_path: Optional[str] = None
    galore_config: Optional[GaLoreConfig] = None
    padding_side: Optional[str] = None
    padding_free: Optional[bool] = None
    task_type: Optional[str] = None

    def __post_init__(self):
        if hasattr(self, 'output_dir'):
            self.output_dir = os.path.abspath(os.path.expanduser(self.output_dir))
        super().__post_init__()


@dataclass
class VllmArguments:
    """
    VllmArguments is a dataclass that holds the configuration for vllm.

    Args:
        vllm_gpu_memory_utilization (float): GPU memory utilization. Default is 0.9.
        vllm_tensor_parallel_size (int): Tensor parallelism size. Default is 1.
        vllm_pipeline_parallel_size(int): Pipeline parallelism size. Default is 1.
        vllm_max_num_seqs (int): Maximum number of sequences. Default is 256.
        vllm_max_model_len (Optional[int]): Maximum model length. Default is None.
        vllm_disable_custom_all_reduce (bool): Flag to disable custom all-reduce. Default is True.
        vllm_enforce_eager (bool): Flag to enforce eager execution. Default is False.
        vllm_limit_mm_per_prompt (Optional[str]): Limit multimedia per prompt. Default is None.
        vllm_max_lora_rank (int): Maximum LoRA rank. Default is 16.
        vllm_enable_prefix_caching (bool): Flag to enable automatic prefix caching. Default is False.
        vllm_use_async_engine (bool): Whether to use async engine for vLLM. Default is False.
        vllm_quantization (Optional[str]): The quantization method for vLLM. Default is None.
        vllm_data_parallel_size (int): Data parallelism size for vLLM rollout. Default is 1.
    """
    # vllm
    vllm_gpu_memory_utilization: float = 0.9
    vllm_tensor_parallel_size: int = 1
    vllm_pipeline_parallel_size: int = 1
    vllm_enable_expert_parallel: bool = False
    vllm_max_num_seqs: int = 256
    vllm_max_model_len: Optional[int] = None
    vllm_disable_custom_all_reduce: bool = True
    vllm_enforce_eager: bool = False
    vllm_limit_mm_per_prompt: Optional[Union[dict, str]] = None  # '{"image": 5, "video": 2}'
    vllm_max_lora_rank: int = 16
    vllm_enable_prefix_caching: bool = False
    vllm_use_async_engine: bool = False
    vllm_quantization: Optional[str] = None
    vllm_reasoning_parser: Optional[str] = None
    vllm_disable_cascade_attn: bool = False
    # rollout
    vllm_data_parallel_size: int = 1

    # compatibility (will be removed in ms-swift 3.8 and later)
    gpu_memory_utilization: Optional[float] = None
    tensor_parallel_size: Optional[int] = None
    max_model_len: Optional[int] = None
    limit_mm_per_prompt: Optional[Union[dict, str]] = None
    data_parallel_size: Optional[int] = None
    use_async_engine: Optional[bool] = None

    def _handle_compatibility(self):
        if self.gpu_memory_utilization is not None:
            self.vllm_gpu_memory_utilization = self.gpu_memory_utilization
        if self.tensor_parallel_size is not None:
            self.vllm_tensor_parallel_size = self.tensor_parallel_size
        if self.max_model_len is not None:
            self.vllm_max_model_len = self.max_model_len
        if self.limit_mm_per_prompt is not None:
            self.vllm_limit_mm_per_prompt = self.limit_mm_per_prompt
        if self.data_parallel_size is not None:
            self.vllm_data_parallel_size = self.data_parallel_size
        if self.use_async_engine is not None:
            self.vllm_use_async_engine = self.use_async_engine

    def __post_init__(self):
        self._handle_compatibility()
        self.vllm_limit_mm_per_prompt = json_parse_to_dict(self.vllm_limit_mm_per_prompt)

    def get_vllm_engine_kwargs(self):
        adapters = self.adapters
        if hasattr(self, 'adapter_mapping'):
            adapters = adapters + list(self.adapter_mapping.values())
        kwargs = {
            'gpu_memory_utilization': self.vllm_gpu_memory_utilization,
            'tensor_parallel_size': self.vllm_tensor_parallel_size,
            'pipeline_parallel_size': self.vllm_pipeline_parallel_size,
            'enable_expert_parallel': self.vllm_enable_expert_parallel,
            'max_num_seqs': self.vllm_max_num_seqs,
            'max_model_len': self.vllm_max_model_len,
            'disable_custom_all_reduce': self.vllm_disable_custom_all_reduce,
            'enforce_eager': self.vllm_enforce_eager,
            'limit_mm_per_prompt': self.vllm_limit_mm_per_prompt,
            'max_lora_rank': self.vllm_max_lora_rank,
            'enable_lora': len(adapters) > 0,
            'max_loras': max(len(adapters), 1),
            'enable_prefix_caching': self.vllm_enable_prefix_caching,
            'use_async_engine': self.vllm_use_async_engine,
            'quantization': self.vllm_quantization,
            'reasoning_parser': self.vllm_reasoning_parser,
            'disable_cascade_attn': self.vllm_disable_cascade_attn,
            'num_labels': self.num_labels,
        }
        if self.task_type in ('embedding', 'seq_cls') or 'reranker' in self.task_type:
            kwargs['task_type'] = self.task_type

        return kwargs


@dataclass
class GRPOArgumentsMixin(VllmArguments):
    epsilon: float = 0.2
    epsilon_high: Optional[float] = None
    delta: Optional[float] = None
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.
    # vllm
    vllm_mode: Literal['server', 'colocate'] = 'colocate'
    # internal vllm (colocate)
    vllm_enable_prefix_caching: bool = True  # overwrite

    # external vllm (server)
    vllm_server_base_url: Optional[List[str]] = None
    vllm_server_host: Optional[List[str]] = None
    vllm_server_port: List[int] = field(default_factory=lambda: [8000])
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

    sleep_level: int = 0
    move_model_batches: Optional[int] = None
    offload_optimizer: bool = False
    offload_model: bool = False
    gc_collect_after_offload: bool = False  # deprecated

    # multi turn
    multi_turn_func: Optional[str] = None  # deprecated
    multi_turn_scheduler: Optional[str] = None
    max_turns: Optional[int] = None
    completion_length_limit_scope: Literal['total', 'per_round'] = 'per_round'
    vllm_server_pass_dataset: bool = False

    # DAPO, https://arxiv.org/abs/2503.14476
    dynamic_sample: bool = False
    max_resample_times: int = 3
    overlong_filter: bool = False
    soft_max_length: Optional[int] = None
    soft_cache_length: Optional[int] = None

    # Dr. GRPO, https://arxiv.org/abs/2503.20783
    scale_rewards: bool = True

    # entropy
    log_entropy: bool = False
    # Beyond the 80/20 Rule, https://arxiv.org/abs/2506.01939
    top_entropy_quantile: float = 1.0

    # GSPO https://www.arxiv.org/abs/2507.18071
    importance_sampling_level: Literal['token', 'sequence', 'sequence_token'] = 'token'

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

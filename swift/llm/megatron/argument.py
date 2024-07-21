import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

from transformers import AutoConfig

config_mapping = {
    'num_layers': ['num_hidden_layers'],
    'hidden_size': ['hidden_size'],
    'ffn_hidden_size': ['intermediate_size'],
    'num_attention_heads': ['num_attention_heads'],
    'num_query_groups': ['num_key_value_heads'],
    'max_position_embeddings': ['max_position_embeddings'],
    'norm_epsilon': ['rms_norm_eps'],
    'rotary_base': ['rope_theta'],
    'padded_vocab_size': ['vocab_size'],
}


def load_megatron_config(model_dir: str) -> Dict[str, Any]:
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    megatron_config = {}
    for k, value in config_mapping.items():
        for v in value:
            assert hasattr(model_config, v)
            megatron_config[k] = getattr(model_config, v)
    assert getattr(model_config, 'hidden_act') == 'silu'
    megatron_config['swiglu'] = True
    return megatron_config


@dataclass
class ExtraMegatronArguments:
    rotary_base: Optional[int] = None
    padded_vocab_size: Optional[int] = None
    model_series: Optional[str] = None
    # model_type: str = 'qwen2-0_5b'
    # template_type: str = 'qwen'
    # dataset: List[str] = field(default_factory=list)

    target_tensor_model_parallel_size: int = 1
    target_pipeline_model_parallel_size: int = 1
    # target_expert_model_parallel_size: int = 1
    # use_legacy_models: bool = False


@dataclass
class MegatronMixin:
    # model
    num_layers: Optional[int] = None
    hidden_size: Optional[int] = None
    ffn_hidden_size: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_query_groups: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    norm_epsilon: Optional[float] = None
    swiglu: Optional[bool] = None

    # train
    train_iters: Optional[int] = None  # !
    lr_warmup_iters: Optional[int] = None
    eval_iters: Optional[int] = None
    lr_decay_iters: Optional[int] = None
    save: Optional[str] = None
    load: Optional[str] = None
    tensorboard_dir: Optional[str] = None  # !
    log_interval: int = 10
    eval_interval: int = 200
    save_interval: Optional[int] = None

    group_query_attention: Optional[bool] = None
    normalization: Literal['LayerNorm', 'RMSNorm'] = 'RMSNorm'

    position_embedding_type: str = 'rope'
    rotary_percent: float = 1.
    rotary_seq_len_interpolation_factor: int = 1

    no_bias_swiglu_fusion: bool = False
    attention_dropout: float = 0.
    hidden_dropout: float = 0.
    # train
    weight_decay: float = 0.1
    clip_grad: float = 1.
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    micro_batch_size: int = 1
    global_batch_size: int = 16
    recompute_method: Optional[str] = None
    recompute_granularity: Optional[str] = 'selective'
    # train_samples: Optional[int] = None
    no_rope_fusion: bool = True
    use_flash_attn: bool = False

    optimizer: str = 'adam'
    dataloader_type: str = 'cyclic'
    lr: float = 1e-5
    lr_decay_style: str = 'cosine'
    min_lr: int = 1e-6
    fp16: bool = False
    bf16: bool = False
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    seed: int = 1234
    transformer_impl: str = 'transformer_engine'

    disable_bias_linear: bool = True
    add_qkv_bias: bool = True
    sequence_parallel: bool = False

    # sliding_window: Optional[int] = None
    apply_query_key_layer_scaling: bool = False  # fp16
    # distributed_timeout_minutes: int = 10
    num_workers: int = 4

    seq_length: int = 1
    eod_mask_loss: bool = True

    # num_experts: Optional[int] = None
    # expert_model_parallel_size: int = 1
    # moe_router_topk: int = 2
    # enable_shared_expert: bool = False

    log_timers_to_tensorboard: bool = True
    log_batch_size_to_tensorboard: bool = True
    log_validation_ppl_to_tensorboard: bool = True
    log_memory_to_tensorboard: bool = True
    tensorboard_log_interval: int = 1
    tensorboard_queue_size: int = 10
    no_async_tensor_model_parallel_allreduce: bool = False
    untie_embeddings_and_output_weights: bool = True

    # fp8: Optional[bool] = None
    # fp8_amax_compute_algo: str = 'most_recent'
    # fp8_amax_history_len: int = 1
    # max_padding_length: int = 128
    # dataset: str = 'LLama-Pretrain-Raw'
    # patch_tokenizer_type: str = 'Qwen2Tokenizer'
    no_save_optim: Optional[bool] = None
    no_save_rng: Optional[bool] = None
    no_load_optim: Optional[bool] = None
    no_load_rng: Optional[bool] = None
    loss_scale: Optional[float] = None
    use_distributed_optimizer: bool = True
    distributed_backend: str = 'nccl'

    # moe_ffn_hidden_size: Optional[int] = None
    # shared_moe_ffn_hidden_size: Optional[int] = None
    # moe_router_load_balancing_type: str = 'aux_loss'
    # moe_aux_loss_coeff: float = 1e-2
    # use_cpu_initialization: bool = False


@dataclass
class MegatronArguments(ExtraMegatronArguments, MegatronMixin):

    def __post_init__(self):
        if self.group_query_attention is None:
            self.group_query_attention = True if self.num_query_groups > 1 else False
        if self.save_interval is None:
            self.save_interval = self.eval_interval
        if self.lr_decay_iters is None and self.train_iters is not None and self.lr_warmup_iters is not None:
            self.lr_decay_iters = self.train_iters - self.lr_warmup_iters
        if not self.no_async_tensor_model_parallel_allreduce:
            os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'

    def _args_to_argv(self) -> Tuple[List[Any], Dict[str, Any]]:
        new_args = []
        args_dict = asdict(self)
        extra_args = {}
        for k, value in args_dict.items():
            if k in ExtraMegatronArguments.__annotations__:
                extra_args[k] = value
                continue
            if value is None or value is False:
                continue
            new_args.append(f"--{k.replace('_', '-')}")
            if isinstance(value, list):
                new_args += [str(v) for v in value]
            elif value is not True:
                new_args.append(str(value))

        return new_args, extra_args

    def parse_to_megatron(self):
        new_args, extra_args = self._args_to_argv()
        sys._old_argv = sys.argv
        sys.argv = sys.argv[:1] + new_args

        return extra_args
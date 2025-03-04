# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple


@dataclass
class ExtraMegatronArguments:
    padded_vocab_size: Optional[int] = None

    target_tensor_model_parallel_size: int = 1
    target_pipeline_model_parallel_size: int = 1


@dataclass
class MegatronMixin:
    num_layers: Optional[int] = None  #
    hidden_size: Optional[int] = None  #
    ffn_hidden_size: Optional[int] = None  #
    num_attention_heads: Optional[int] = None  #
    num_query_groups: Optional[int] = None  #
    group_query_attention: Optional[bool] = None  #
    max_position_embeddings: Optional[int] = None  #
    norm_epsilon: Optional[float] = None  #
    swiglu: Optional[bool] = None  #
    rotary_base: Optional[int] = None  #
    disable_bias_linear: bool = True  #
    add_qkv_bias: bool = True  #

    train_iters: Optional[int] = None  #
    lr_warmup_iters: Optional[int] = None  #
    eval_iters: Optional[int] = None  #
    lr_decay_iters: Optional[int] = None  #
    save: Optional[str] = None  #
    load: Optional[str] = None
    tensorboard_dir: Optional[str] = None  # !
    log_interval: int = 10  #
    log_throughput: bool = False
    eval_interval: Optional[int] = None  #
    save_interval: int = 500  #

    position_embedding_type: str = 'rope'  #
    rotary_percent: float = 1.  #
    rotary_seq_len_interpolation_factor: int = 1  #
    no_bias_swiglu_fusion: bool = False  #
    attention_dropout: float = 0.  #
    hidden_dropout: float = 0.  #

    optimizer: str = 'adam'
    weight_decay: float = 0.1  #
    clip_grad: float = 1.  #
    adam_beta1: float = 0.9  #
    adam_beta2: float = 0.95  #
    adam_eps: float = 1e-8
    init_method_std: float = 0.01  #
    micro_batch_size: int = 1  #
    global_batch_size: int = 16  #
    recompute_method: Optional[str] = None
    recompute_granularity: Optional[str] = 'selective'
    no_rope_fusion: bool = False
    use_flash_attn: bool = False
    use_cpu_initialization: Optional[bool] = None

    dataloader_type: str = 'cyclic'
    lr: float = 1e-5  #
    lr_decay_style: str = 'cosine'  #
    min_lr: int = 1e-6
    fp16: bool = False
    bf16: bool = False
    tensor_model_parallel_size: int = 1  #
    pipeline_model_parallel_size: int = 1  #
    context_parallel_size: int = 1  #
    seed: int = 42
    sequence_parallel: bool = False
    transformer_impl: str = 'transformer_engine'

    apply_query_key_layer_scaling: bool = False  # fp16
    num_workers: int = 8

    log_timers_to_tensorboard: bool = True  #
    log_batch_size_to_tensorboard: bool = True  #
    log_validation_ppl_to_tensorboard: bool = True  #
    log_memory_to_tensorboard: bool = True  #
    tensorboard_log_interval: int = 1  #
    tensorboard_queue_size: int = 10  #
    untie_embeddings_and_output_weights: bool = True
    seq_length: Optional[int] = None  #

    no_save_optim: bool = False  #
    no_save_rng: bool = False  #
    no_load_optim: bool = False  #
    no_load_rng: bool = False  #
    loss_scale: Optional[float] = None
    use_distributed_optimizer: bool = True
    normalization: Literal['LayerNorm', 'RMSNorm'] = 'RMSNorm'  #
    calculate_per_token_loss: bool = True


@dataclass
class MegatronArguments(ExtraMegatronArguments, MegatronMixin):

    def __post_init__(self):
        if self.group_query_attention is None:
            self.group_query_attention = True if self.num_query_groups > 1 else False
        if self.eval_interval is None:
            self.eval_interval = self.save_interval
        if self.lr_decay_iters is None and self.train_iters is not None and self.lr_warmup_iters is not None:
            self.lr_decay_iters = self.train_iters - self.lr_warmup_iters

    def get_matched_kwargs(args):
        args_dict = asdict(args)
        parameters = inspect.signature(MegatronArguments.__init__).parameters

        for k in list(args_dict.keys()):
            if k not in parameters:
                args_dict.pop(k)
        return args_dict

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

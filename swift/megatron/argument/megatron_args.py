# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch

from swift.llm.argument.base_args import to_abspath

add_prefix_no_list = [
    'bias_swiglu_fusion', 'ropo_fusion', 'gradient_accumulation_fusion', 'save_optim', 'save_rng', 'load_optim',
    'load_rng', 'log_learning_rate_to_tensorboard', 'create_attention_mask_in_dataloader'
]
add_prefix_no_list = set(add_prefix_no_list)


@dataclass
class ExtraMegatronArguments:
    padded_vocab_size: Optional[int] = None
    hf_ckpt_path: Optional[str] = None

    torch_dtype: Optional[torch.dtype] = None
    target_tensor_model_parallel_size: int = 1
    target_pipeline_model_parallel_size: int = 1


@dataclass
class MegatronArguments(ExtraMegatronArguments):
    # training
    micro_batch_size: int = 1
    global_batch_size: int = 16
    recompute_granularity: Literal['selective', 'full'] = 'selective'
    recompute_method: Literal['uniform', 'block'] = None
    recompute_num_layers: Optional[int] = None
    use_cpu_initialization: bool = False
    deterministic_mode: bool = False
    calculate_per_token_loss: bool = True
    train_iters: Optional[int] = None
    train_samples: Optional[int] = None
    log_interval: int = 5
    tensorboard_dir: Optional[str] = None
    bias_swiglu_fusion: bool = True
    ropo_fusion: bool = True
    gradient_accumulation_fusion: bool = True
    cross_entropy_loss_fusion: bool = False
    use_flash_attn: bool = False
    optimizer: Literal['adam', 'sgd'] = 'adam'
    dataloader_type: Literal['single', 'cyclic', 'external'] = 'cyclic'
    sequence_parallel: bool = False
    manual_gc: bool = False
    manual_gc_interval: int = 0

    # learning rate
    lr: float = 1e-5
    lr_decay_style: Literal['cosine', 'linear', 'constant'] = 'cosine'
    # The default is None, which will be set to `train_iters`.
    lr_decay_iters: Optional[int] = None
    lr_decay_samples: Optional[int] = None
    lr_warmup_iters: int = 0
    lr_warmup_samples: int = 0
    min_lr: int = 0

    # regularization
    weight_decay: float = 0.01
    clip_grad: float = 1.
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    sgd_momentum: float = 0.9

    # checkpoint
    save: Optional[str] = None
    save_interval: int = 500
    save_optim: bool = True
    save_rng: bool = True
    load: Optional[str] = None
    load_optim: bool = True
    load_rng: bool = True
    finetune: bool = False
    ckpt_format: Literal['torch', 'torch_dist', 'zarr'] = 'torch_dist'
    auto_detect_ckpt_format: bool = True
    exit_on_missing_checkpoint: bool = True

    # dist
    distributed_backend: Literal['nccl', 'gloo'] = 'nccl'
    use_distributed_optimizer: bool = True
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    tp_comm_overlap: Optional[bool] = None
    overlap_grad_reduce: bool = True
    overlap_param_gather: bool = True
    distributed_timeout_minutes: int = 60

    # model
    num_layers: Optional[int] = None
    hidden_size: Optional[int] = None
    ffn_hidden_size: Optional[int] = None
    num_attention_heads: Optional[int] = None
    group_query_attention: Optional[bool] = None
    num_query_groups: int = 1
    max_position_embeddings: Optional[int] = None
    position_embedding_type: Literal['learned_absolute', 'rope', 'relative', 'none'] = 'rope'
    rotary_base: int = 10000
    rotary_percent: float = 1.
    rotary_seq_len_interpolation_factor: Optional[int] = None
    normalization: Literal['LayerNorm', 'RMSNorm'] = 'RMSNorm'
    norm_epsilon: float = 1e-5
    swiglu: bool = True
    untie_embeddings_and_output_weights: bool = True
    disable_bias_linear: bool = True
    add_qkv_bias: bool = True
    attention_dropout: float = 0.
    hidden_dropout: float = 0.
    make_vocab_size_divisible_by: int = 128
    transformer_impl: Literal['local', 'transformer_engine'] = 'transformer_engine'

    # mixed precision
    fp16: bool = False
    bf16: bool = False
    apply_query_key_layer_scaling: Optional[bool] = None
    attention_softmax_in_fp32: bool = True
    hysteresis: int = 2

    # logging
    log_params_norm: bool = True
    log_throughput: bool = True
    tensorboard_log_interval: int = 1
    tensorboard_queue_size: int = 50
    log_timers_to_tensorboard: bool = True
    log_learning_rate_to_tensorboard: bool = True
    log_validation_ppl_to_tensorboard: bool = True
    log_memory_to_tensorboard: bool = True
    logging_leval: Optional[str] = None

    # evaluate
    eval_iters: int = 100
    eval_interval: Optional[int] = None

    # initialization
    seed: int = 42
    init_method_std: float = 0.02

    # data & tokenizer
    seq_length: Optional[str] = None
    num_workers: int = 4
    eod_mask_loss: bool = False
    create_attention_mask_in_dataloader: bool = False

    def _init_mixed_precision(self):
        if self.torch_dtype == torch.bfloat16:
            self.bf16 = True
        elif self.torch_dtype == torch.float16:
            self.fp16 = True
            if self.apply_query_key_layer_scaling is None:
                self.apply_query_key_layer_scaling = True

    def __post_init__(self):
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
        self.group_query_attention = self.num_query_groups > 1
        if self.eval_interval is None:
            self.eval_interval = self.save_interval
        if self.seq_length is None:
            self.seq_length = self.max_position_embeddings
        if self.tp_comm_overlap is None and self.sequence_parallel:
            self.tp_comm_overlap = True
        if self.tensorboard_dir is None and self.save is not None:
            self.tensorboard_dir = f'{self.save}/runs'
        self.tensorboard_dir = to_abspath(self.tensorboard_dir)

    def _args_to_argv(self) -> Tuple[List[Any], Dict[str, Any]]:
        new_args = []
        args_dict = asdict(self)
        extra_args = {}
        for k, value in args_dict.items():
            if k not in MegatronArguments.__annotations__:
                extra_args[k] = value
                continue
            if k in add_prefix_no_list:
                k = f'no_{k}'
                value = not value
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

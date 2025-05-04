# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
from transformers.utils.versions import require_version

from swift.llm.argument.base_args import to_abspath


@dataclass
class ExtraMegatronArguments:
    padded_vocab_size: Optional[int] = None
    rope_scaling: Optional[Union[dict, str]] = None
    torch_dtype: Optional[torch.dtype] = None
    model_type: Optional[str] = None
    dataloader_persistent_workers: bool = True
    dataloader_prefetch_factor: int = 10


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
    train_iters: Optional[int] = None
    log_interval: int = 5
    tensorboard_dir: Optional[str] = None
    no_masked_softmax_fusion: bool = False
    no_bias_dropout_fusion: bool = False
    no_bias_swiglu_fusion: bool = False
    no_rope_fusion: bool = False
    no_gradient_accumulation_fusion: bool = False
    cross_entropy_loss_fusion: bool = False
    use_flash_attn: bool = False
    optimizer: Literal['adam', 'sgd'] = 'adam'
    dataloader_type: Literal['single', 'cyclic', 'external'] = 'cyclic'
    manual_gc: bool = False
    manual_gc_interval: int = 0

    # learning rate
    lr: float = 1e-5
    lr_decay_style: Literal['cosine', 'linear', 'constant'] = 'cosine'
    # The default is None, which will be set to `train_iters`.
    lr_decay_iters: Optional[int] = None
    lr_warmup_iters: int = 0
    min_lr: float = 0

    # regularization
    weight_decay: float = 0.1
    clip_grad: float = 1.
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    sgd_momentum: float = 0.9

    # checkpoint
    save: Optional[str] = None
    save_interval: int = 500
    no_save_optim: bool = False
    no_save_rng: bool = False
    load: Optional[str] = None
    no_load_optim: bool = False
    no_load_rng: bool = False
    finetune: bool = False
    ckpt_format: Literal['torch', 'torch_dist', 'zarr'] = 'torch_dist'
    no_initialization: bool = True
    auto_detect_ckpt_format: bool = True
    exit_on_missing_checkpoint: bool = True

    # dist
    distributed_backend: Literal['nccl', 'gloo'] = 'nccl'
    use_distributed_optimizer: bool = True
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    sequence_parallel: bool = False
    context_parallel_size: int = 1
    tp_comm_overlap: bool = False
    overlap_grad_reduce: bool = False
    overlap_param_gather: bool = False
    distributed_timeout_minutes: int = 60

    # model
    num_layers: Optional[int] = None
    hidden_size: Optional[int] = None
    ffn_hidden_size: Optional[int] = None
    num_attention_heads: Optional[int] = None
    group_query_attention: Optional[bool] = None
    num_query_groups: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    position_embedding_type: Literal['learned_absolute', 'rope', 'relative', 'none'] = 'rope'
    rotary_base: Optional[int] = None
    rotary_percent: float = 1.
    normalization: Literal['LayerNorm', 'RMSNorm'] = 'RMSNorm'
    norm_epsilon: Optional[float] = None
    swiglu: Optional[bool] = None
    untie_embeddings_and_output_weights: Optional[bool] = None
    disable_bias_linear: Optional[bool] = None
    add_qkv_bias: Optional[bool] = None
    attention_dropout: Optional[float] = None
    hidden_dropout: float = 0.
    kv_channels: Optional[int] = None
    qk_layernorm: Optional[bool] = None
    transformer_impl: Literal['local', 'transformer_engine'] = 'transformer_engine'

    # moe
    num_experts: Optional[int] = None
    moe_ffn_hidden_size: Optional[int] = None
    moe_shared_expert_intermediate_size: Optional[int] = None
    moe_router_topk: Optional[int] = None
    moe_router_pre_softmax: Optional[bool] = None
    moe_aux_loss_coeff: Optional[float] = None

    expert_model_parallel_size: int = 1
    moe_token_dispatcher_type: Literal['allgather', 'alltoall', 'alltoall_seq'] = 'alltoall'
    moe_grouped_gemm: bool = False
    moe_router_load_balancing_type: Literal['aux_loss', 'seq_aux_loss', 'sinkhorn', 'none'] = 'aux_loss'
    moe_z_loss_coeff: Optional[float] = None
    moe_expert_capacity_factor: Optional[float] = None
    moe_shared_expert_overlap: bool = False

    # mixed precision
    fp16: Optional[bool] = None
    bf16: Optional[bool] = None
    apply_query_key_layer_scaling: Optional[bool] = None
    attention_softmax_in_fp32: bool = True

    # logging
    log_params_norm: bool = False
    log_throughput: bool = True
    tensorboard_log_interval: int = 1
    tensorboard_queue_size: int = 50
    log_timers_to_tensorboard: bool = True
    no_log_learning_rate_to_tensorboard: bool = False
    log_validation_ppl_to_tensorboard: bool = True
    log_memory_to_tensorboard: bool = True
    logging_leval: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_exp_name: Optional[str] = None
    wandb_save_dir: Optional[str] = None

    # evaluate
    eval_iters: int = 100
    eval_interval: Optional[int] = None

    # other
    seed: int = 42
    seq_length: Optional[int] = None
    num_workers: int = 4
    no_create_attention_mask_in_dataloader: bool = True

    def _set_default(self):
        if self.num_query_groups is None:
            self.num_query_groups = 1
        if self.norm_epsilon is None:
            self.norm_epsilon = 1e-5
        if self.rotary_base is None:
            self.rotary_base = 10000
        if self.attention_dropout is None:
            self.attention_dropout = 0.
        if self.untie_embeddings_and_output_weights is None:
            self.untie_embeddings_and_output_weights = True
        if self.swiglu is None:
            self.swiglu = True
        if self.add_qkv_bias is None:
            self.add_qkv_bias = True
        if self.disable_bias_linear is None:
            self.disable_bias_linear = True
        if self.moe_router_topk is None:
            self.moe_router_topk = 2
        if self.moe_router_pre_softmax is None:
            self.moe_router_pre_softmax = False
        if self.moe_aux_loss_coeff is None:
            self.moe_aux_loss_coeff = 0.
        if self.qk_layernorm is None:
            self.qk_layernorm = False

    def _init_mixed_precision(self):
        from swift.llm.argument.base_args.model_args import ModelArguments
        ModelArguments._init_mixed_precision(self)
        if self.apply_query_key_layer_scaling is None:
            self.apply_query_key_layer_scaling = self.fp16
        if self.apply_query_key_layer_scaling:
            os.environ['NVTE_APPLY_QK_LAYER_SCALING'] = '1'

    def _init_moe(self):
        if self.moe_shared_expert_intermediate_size == 0:
            self.moe_shared_expert_intermediate_size = None
        if self.moe_ffn_hidden_size is None:
            self.moe_ffn_hidden_size = self.ffn_hidden_size
        else:
            self.ffn_hidden_size = self.moe_ffn_hidden_size

    def __post_init__(self):
        from swift.llm.argument.base_args.model_args import ModelArguments
        if self.use_flash_attn:
            require_version('flash-attn')
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
        self._set_default()
        self.group_query_attention = self.num_query_groups > 1
        if self.rope_scaling is not None:
            self.rope_scaling = ModelArguments.parse_to_dict(self.rope_scaling)
        if self.eval_interval is None:
            self.eval_interval = self.save_interval
        if self.seq_length is None:
            self.seq_length = self.max_position_embeddings
        if self.tensorboard_dir is None and self.save is not None:
            self.tensorboard_dir = f'{self.save}/runs'
        self._init_moe()
        self._init_mixed_precision()

        self.tensorboard_dir = to_abspath(self.tensorboard_dir)

    def _args_to_argv(self) -> Tuple[List[Any], Dict[str, Any]]:
        new_args = []
        args_dict = asdict(self)
        extra_args = {}
        for k, value in args_dict.items():
            if k not in MegatronArguments.__annotations__:
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
        # parameter conflict
        extra_args.pop('loss_scale', None)
        return extra_args

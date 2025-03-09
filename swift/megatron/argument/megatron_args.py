# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from typing import List, Literal, Optional
import os

_add_prefix_no: List[str] = [
    'bias_swiglu_fusion', 'ropo_fusion', 'gradient_accumulation_fusion',
    'save_optim', 'save_rng', 'load_optim', 'load_rng',
    'log_learning_rate_to_tensorboard',
    'create_attention_mask_in_dataloader'
]

@dataclass
class ExtraMegatronArguments:
    padded_vocab_size: Optional[int] = None
    hf_ckpt_path: Optional[str] = None

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
    tp_comm_overlap: bool = True
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
    apply_query_key_layer_scaling: bool = False
    attention_softmax_in_fp32: bool = False
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

    # data
    data_path: List[str] = field(default=list)
    split: str = '99,1,0'
    train_data_path: Optional[str] = field(default=list)
    valid_data_path: Optional[str] = field(default=list)
    test_data_path: Optional[str] = field(default=list)
    seq_length: Optional[str] = None
    num_workers: int = 4
    eod_mask_loss: bool = False
    create_attention_mask_in_dataloader: bool = False

    # tokenizer

    def __post_init__(self):
        self.add_prefix_no = set(_add_prefix_no)
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
        self.group_query_attention = self.num_query_groups > 1
        if self.eval_interval is None:
            self.eval_interval = self.save_interval

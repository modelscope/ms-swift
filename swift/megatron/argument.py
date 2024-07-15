import os
import sys
from swift.utils import subprocess_run
from swift.llm.utils import git_clone_github, is_megatron_available, get_model_tokenizer
import torch.distributed as dist
from functools import wraps
import torch
from dataclasses import dataclass, asdict, field

from typing import Tuple, List, Dict, Any, Optional, Literal


def init_megatron_env() -> None:
    dist.init_process_group(backend='nccl')
    megatron_path = git_clone_github('https://github.com/NVIDIA/Megatron-LM')
    megatron_patch_path = git_clone_github('https://github.com/alibaba/Pai-Megatron-Patch')
    if not is_megatron_available():
        subprocess_run(['pip', 'install', '-e', megatron_path])
    sys.path.append(megatron_patch_path)
    sys.path.append(os.path.join(megatron_patch_path, 'examples', 'qwen2'))
    sys.path.append(megatron_path)



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
    # tie_word_embeddings
    # swiglu
}

def load_megatron_config(model_type: str) -> Dict[str, Any]:


    pass

@dataclass
class ExtraMegatronArguments:
    rotary_base: int
    padded_vocab_size: int
    # model_type: str = 'qwen2-0_5b'
    # template_type: str = 'qwen'
    # dataset: List[str] = field(default_factory=list)

    target_tensor_model_parallel_size: int = 1
    target_pipeline_model_parallel_size: int = 1
    target_expert_model_parallel_size: int = 1
    # use_legacy_models: bool = False


@dataclass
class MegatronArguments(ExtraMegatronArguments):
    # model
    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    num_query_groups: int
    max_position_embeddings: int
    norm_epsilon: float
    untie_embeddings_and_output_weights: bool
    swiglu: bool

    # train
    train_iters: int  # !
    lr_warmup_iters: int
    eval_iters: int
    save_interval: int
    tensorboard_dir: str  # !
    save: str
    load: str
    lr_decay_iters: Optional[int] = None

    group_query_attention: Optional[bool] = None
    normalization: Literal['LayerNorm', 'RMSNorm'] = 'RMSNorm'

    position_embedding_type: str = 'rope'
    rotary_percent: float = 1.
    rotary_seq_len_interpolation_factor: int = 1

    no_bias_swiglu_fusion: bool = False  # ?
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
    train_samples: Optional[int] = None
    log_interval: int = 10

    eval_interval: int = 200
    no_rope_fusion: bool = True
    use_flash_attn: bool = False
    #
    optimizer: str = 'adam'
    dataloader_type: str = 'cyclic'
    lr: float = 1e-5
    lr_decay_style: str = 'cosine'
    min_lr: int = 1e-6
    fp16: bool = False
    bf16: bool = True
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    seed: int = 1234
    transformer_impl: str = 'transformer_engine'

    disable_bias_linear: bool = True
    add_qkv_bias: bool = True

    no_async_tensor_model_parallel_allreduce: bool = False
    sequence_parallel: bool = False
    init_method_std: float = 0.008

    no_save_optim: Optional[bool] = None
    no_save_rng: Optional[bool] = None
    no_load_optim: Optional[bool] = None
    no_load_rng: Optional[bool] = None

    loss_scale: Optional[float] = None

    apply_query_key_layer_scaling: bool = False
    distributed_backend: str = 'nccl'
    distributed_timeout_minutes: int = 10
    use_distributed_optimizer: bool = True

    # train_data_path: List[str] = field(default_factory=list)
    # valid_data_path: List[str] = field(default_factory=list)
    # test_data_path: List[str] = field(default_factory=list)
    seq_length: int = 128
    num_workers: int = 1
    eod_mask_loss: bool = True

    # num_experts: Optional[int] = None
    # expert_model_parallel_size: int = 1
    # moe_router_topk: int = 2
    # enable_shared_expert: bool = False

    log_timers_to_tensorboard: bool = True
    tensorboard_log_interval: int = 1
    log_batch_size_to_tensorboard: bool = True
    log_validation_ppl_to_tensorboard: bool = True
    log_memory_to_tensorboard: bool = True
    tensorboard_queue_size: int = 1

    # fp8: Optional[bool] = None
    # fp8_amax_compute_algo: str = 'most_recent'
    # fp8_amax_history_len: int = 1
    # max_padding_length: int = 128
    # dataset: str = 'LLama-Pretrain-Raw'
    # patch_tokenizer_type: str = 'Qwen2Tokenizer'
    sliding_window: Optional[int] = None

    # moe_ffn_hidden_size: Optional[int] = None
    # shared_moe_ffn_hidden_size: Optional[int] = None
    # moe_router_load_balancing_type: str = 'aux_loss'
    # moe_aux_loss_coeff: float = 1e-2

    use_cpu_initialization: bool = False

    def __post_init__(self):
        self.group_query_attention = True if self.num_query_groups > 1 else False

        assert self.load is not None

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

        def build_tokenizer(args):
            _, tokenizer = get_model_tokenizer('qwen2-7b-instruct', load_model=False)
            args.extra_vocab_size = args.padded_vocab_size - tokenizer.vocab_size
            return tokenizer

        from megatron.training import global_vars
        global_vars.build_tokenizer = build_tokenizer

        from megatron.training import get_args
        from megatron.training import initialize
        _old_initialize_distributed = initialize._initialize_distributed

        @wraps(_old_initialize_distributed)
        def _initialize_distributed():
            args = get_args()
            if dist.is_initialized():
                args.rank, args.local_rank, args.world_size, args.local_world_size = get_dist_setting()
                torch.cuda.set_device(args.local_rank)
            return _old_initialize_distributed()

        initialize._initialize_distributed = _initialize_distributed


        from megatron.training import training
        _old_load_checkpoint = training.load_checkpoint
        @wraps(_old_load_checkpoint)
        def load_checkpoint(model, optimizer, opt_param_scheduler, load_arg='load', strict=False):
            # default: strict=False
            return _old_load_checkpoint(model, optimizer, opt_param_scheduler, load_arg=load_arg, strict=strict)

        training.load_checkpoint = load_checkpoint

        return extra_args

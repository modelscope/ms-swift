import os
# from megatron.training.checkpointing import get_checkpoint_name, get_checkpoint_tracker_filename, read_metadata
# from megatron.training.utils import get_ltor_masks_and_position_ids
import sys
from argparse import Namespace
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from megatron.training import get_args
from megatron.training.initialize import initialize_megatron
from megatron_patch.arguments import get_patch_args
from transformers.modeling_utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME, load_sharded_checkpoint, shard_checkpoint

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
sys.path.append('/mnt/nas2/huangjintao.hjt/work/Pai-Megatron-Patch')


@dataclass
class MegatronArguments:
    # model
    num_layers: int = 24
    hidden_size: int = 896
    ffn_hidden_size: hidden_size = 4864  # intermediate_size
    num_attention_heads: int = 14
    group_query_attention: bool = True
    num_query_groups: int = 2
    max_position_embeddings: int = 131072
    position_embedding_type: str = 'rope'
    use_rotary_position_embeddings: bool = True
    rotary_percent: float = 1.
    rotary_seq_len_interpolation_factor: int = 1
    normalization: str = 'RMSNorm'
    norm_epsilon: float = 1e-6
    swiglu: bool = True
    untie_embeddings_and_output_weights: bool = True
    attention_dropout: float = 0.
    hidden_dropout: float = 0.
    # train
    weight_decay: float = 0.1
    clip_grad: float = 1.
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    micro_batch_size: int = 1
    global_batch_size: int = 8
    recompute_method: Optional[str] = None
    recompute_granularity: Optional[str] = 'selective'
    train_iters: int = 1000
    train_samples: Optional[int] = None
    log_interval: int = 10
    tensorboard_dir: str = 'output/tensorboard'
    save: str = 'output'
    load: Optional[str] = None
    no_rope_fusion: bool = True
    use_flash_attn: bool = False
    disable_bias_linear: bool = True
    add_qkv_bias: bool = True
    optimizer: str = 'adam'
    dataloader_type: str = 'cyclic'
    no_async_tensor_model_parallel_allreduce: bool = False
    sequence_parallel: bool = False
    seed: int = 1234
    init_method_std: float = 0.008
    lr: float = 1e-5
    lr_decay_style: str = 'cosine'
    lr_decay_iters: int = 900
    lr_warmup_iters: int = 100
    min_lr: int = 1e-6
    save_interval: int = 100
    no_save_optim: Optional[bool] = None
    no_save_rng: Optional[bool] = None
    no_load_optim: Optional[bool] = None
    no_load_rng: Optional[bool] = None

    loss_scale: Optional[float] = None

    fp16: bool = False
    bf16: bool = True

    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    apply_query_key_layer_scaling: bool = False
    distributed_backend: str = 'nccl'
    distributed_timeout_minutes: int = 10
    use_distributed_optimizer: bool = True
    eval_iters: int = 10
    eval_interval: int = 200

    train_data_path: List[str] = field(default_factory=list)
    valid_data_path: List[str] = field(default_factory=list)
    test_data_path: List[str] = field(default_factory=list)
    seq_length: int = 128
    num_workers: int = 8
    eod_mask_loss: bool = True

    num_experts: Optional[int] = None
    expert_model_parallel_size: int = 1
    moe_router_topk: int = 2
    enable_shared_expert: bool = False

    log_timers_to_tensorboard: bool = True
    tensorboard_log_interval: int = 1
    log_batch_size_to_tensorboard: bool = True
    log_validation_ppl_to_tensorboard: bool = True
    log_memory_to_tensorboard: bool = True
    tensorboard_queue_size: int = 1

    fp8: Optional[bool] = None
    fp8_amax_compute_algo: str = 'most_recent'
    fp8_amax_history_len: int = 1
    transformer_impl: str = 'transformer_engine'
    max_padding_length: int = 128
    dataset: str = 'LLama-Pretrain-Raw'
    extra_vocab_size: int = 293
    patch_tokenizer_type: str = 'Qwen2Tokenizer'
    sliding_window: Optional[int] = None
    rotary_base: int = 1000000

    moe_ffn_hidden_size: Optional[int] = None
    shared_moe_ffn_hidden_size: Optional[int] = None

    def __post_init__(self):
        assert self.load is not None

    def args_to_argv(self) -> List[Any]:
        new_args = []
        args_dict = asdict(self)
        for k, value in args_dict.items():
            if value is None or value is False:
                continue
            new_args.append(f"--{k.replace('_', '-')}")
            if isinstance(value, list):
                new_args += [str(v) for v in value]
            elif value is not True:
                new_args.append(str(value))
        return new_args

    def get_megatron_args(self) -> Namespace:
        new_args = self.args_to_argv()
        sys._old_argv = sys.argv
        sys.argv = sys.argv[:1] + new_args

        initialize_megatron(extra_args_provider=get_patch_args)
        return get_args()


if __name__ == '__main__':
    args = MegatronArguments(
        load='../../qwen-ckpts/Qwen2-0.5B-hf-to-mcore-te-tp1-pp1',
        train_data_path=['qwen-datasets/alpaca_zh-qwen-train.json'],
        valid_data_path=['qwen-datasets/alpaca_zh-qwen-valid.json'],
        test_data_path=['qwen-datasets/alpaca_zh-qwen-valid.json'],
    )
    megatron_args = args.get_megatron_args()

    print()

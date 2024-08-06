# Copyright (c) Alibaba, Inc. and its affiliates.
import math
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
    'attention_dropout': ['attention_dropout']
}


@dataclass
class ExtraMegatronArguments:
    padded_vocab_size: Optional[int] = None
    model_type: Optional[str] = None

    target_tensor_model_parallel_size: int = 1
    target_pipeline_model_parallel_size: int = 1


@dataclass
class MegatronMixin:
    num_layers: Optional[int] = None
    hidden_size: Optional[int] = None
    ffn_hidden_size: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_query_groups: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    norm_epsilon: Optional[float] = None
    swiglu: Optional[bool] = None
    rotary_base: Optional[int] = None
    group_query_attention: Optional[bool] = None
    disable_bias_linear: bool = True
    add_qkv_bias: bool = True

    train_iters: Optional[int] = None
    lr_warmup_iters: Optional[int] = None
    eval_iters: Optional[int] = None
    lr_decay_iters: Optional[int] = None
    save: Optional[str] = None
    load: Optional[str] = None
    tensorboard_dir: Optional[str] = None  # !
    log_interval: int = 10
    eval_interval: int = 200
    save_interval: Optional[int] = None

    position_embedding_type: str = 'rope'
    rotary_percent: float = 1.
    rotary_seq_len_interpolation_factor: int = 1
    no_bias_swiglu_fusion: bool = False
    attention_dropout: float = 0.
    hidden_dropout: float = 0.

    optimizer: str = 'adam'
    weight_decay: float = 0.1
    clip_grad: float = 1.
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    micro_batch_size: int = 1
    global_batch_size: int = 16
    recompute_method: Optional[str] = None
    recompute_granularity: Optional[str] = 'selective'
    no_rope_fusion: bool = True
    use_flash_attn: bool = False
    use_cpu_initialization: Optional[bool] = None

    dataloader_type: str = 'cyclic'
    lr: float = 1e-5
    lr_decay_style: str = 'cosine'
    min_lr: int = 1e-6
    fp16: bool = False
    bf16: bool = False
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    seed: int = 42
    sequence_parallel: bool = False

    apply_query_key_layer_scaling: bool = False  # fp16
    num_workers: int = 4

    log_timers_to_tensorboard: bool = True
    log_validation_ppl_to_tensorboard: bool = True
    log_memory_to_tensorboard: bool = True
    tensorboard_log_interval: int = 1
    tensorboard_queue_size: int = 10
    no_async_tensor_model_parallel_allreduce: bool = False
    untie_embeddings_and_output_weights: bool = True
    seq_length: int = 1  # not use

    no_save_optim: Optional[bool] = None
    no_save_rng: Optional[bool] = None
    no_load_optim: Optional[bool] = None
    no_load_rng: Optional[bool] = None
    loss_scale: Optional[float] = None
    use_distributed_optimizer: bool = True
    normalization: Literal['LayerNorm', 'RMSNorm'] = 'RMSNorm'


@dataclass
class MegatronArguments(ExtraMegatronArguments, MegatronMixin):

    @staticmethod
    def load_megatron_config(model_dir: str) -> Dict[str, Any]:
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        megatron_config = {}
        for k, value in config_mapping.items():
            for v in value:
                assert hasattr(model_config, v)
                if k == 'rotary_base':
                    megatron_config[k] = int(getattr(model_config, v))
                else:
                    megatron_config[k] = getattr(model_config, v)
        assert getattr(model_config, 'hidden_act') == 'silu'
        megatron_config['swiglu'] = True
        return megatron_config

    @staticmethod
    def from_sft_args(args, train_dataset, val_dataset) -> Dict[str, Any]:
        assert args.optim == 'adamw_torch', 'Currently, only `args.optim="adamw_torch"` is supported.'
        assert args.lr_scheduler_type == 'cosine', 'Currently, only `args.lr_scheduler_type="cosine"` is supported.'

        apply_query_key_layer_scaling = True if args.fp16 else False
        res = {
            'optimizer': 'adam',
            'lr_decay_style': 'cosine',
            'weight_decay': args.weight_decay,
            'clip_grad': args.max_grad_norm,
            'adam_beta1': args.adam_beta1,
            'adam_beta2': args.adam_beta2,
            'adam_eps': args.adam_epsilon,
            'lr': args.learning_rate,
            'min_lr': args.min_lr,
            'fp16': args.fp16,
            'bf16': args.bf16,
            'tensor_model_parallel_size': args.tp,
            'pipeline_model_parallel_size': args.pp,
            'seed': args.seed,
            'load': args.resume_from_checkpoint,
            'save': args.output_dir,
            'tensorboard_dir': args.logging_dir,
            'log_interval': args.logging_steps,
            'eval_interval': args.eval_steps,
            'save_interval': args.save_steps,
            'micro_batch_size': args.batch_size,
            'global_batch_size': args.batch_size * args.gradient_accumulation_steps * args.world_size,
            'sequence_parallel': args.sequence_parallel,
            'apply_query_key_layer_scaling': apply_query_key_layer_scaling,
            'num_workers': args.dataloader_num_workers,
            'use_flash_attn': args.use_flash_attn
        }
        res['train_iters'] = int(math.ceil(len(train_dataset) * args.num_train_epochs / res['global_batch_size']))
        res['eval_iters'] = int(math.ceil(len(val_dataset) / res['global_batch_size']))
        res['lr_warmup_iters'] = (
            args.warmup_steps if args.warmup_steps > 0 else math.ceil(res['train_iters'] * args.warmup_ratio))
        if args.save_only_model:
            res['no_save_optim'] = True
            res['no_save_rng'] = True

        return res

    def __post_init__(self):
        assert self.pipeline_model_parallel_size == 1 and self.target_pipeline_model_parallel_size, (
            'Pipeline model parallel is currently not supported.')
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

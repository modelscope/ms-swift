# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from datetime import timedelta
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import json
import torch
from transformers.utils.versions import require_version

from swift.llm import get_model_info_meta
from swift.utils import get_dist_setting, get_logger, json_parse_to_dict

logger = get_logger()


@dataclass
class RLHFMegatronArgumentsMixin:
    rlhf_type: Literal['dpo', 'kto', 'rm'] = None
    ref_load: Optional[str] = None
    ref_adapter_load: Optional[str] = None

    beta: float = 0.1
    rpo_alpha: Optional[float] = None
    reference_free: bool = False
    label_smoothing: float = 0.
    f_divergence_type: str = 'reverse_kl'
    loss_type: Optional[str] = None

    # kto
    desirable_weight: float = 1.
    undesirable_weight: float = 1.
    calculate_KL: Optional[bool] = None

    # rm
    center_rewards_coefficient: Optional[float] = None

    def _init_kto(self):
        if self.calculate_KL is None:
            # Not all losses require a KL calculation
            self.calculate_KL = True
            if self.loss_type in ['apo_zero_unpaired']:
                self.calculate_KL = False

    def __post_init__(self):
        if self.rlhf_type is None:
            return
        default_loss_type = {'kto': 'kto', 'dpo': 'sigmoid'}
        if self.loss_type is None:
            self.loss_type = default_loss_type.get(self.rlhf_type)
        if self.rlhf_type == 'kto':
            self._init_kto()


@dataclass
class MegatronTunerMixin:
    train_type: Literal['lora', 'full'] = 'full'
    freeze_llm: bool = False
    freeze_vit: bool = True
    freeze_aligner: bool = True
    # full
    freeze_parameters: List[str] = field(default_factory=list)
    freeze_parameters_regex: Optional[str] = None
    freeze_parameters_ratio: float = 0.  # 0 ~ 1
    trainable_parameters: List[str] = field(default_factory=list)
    trainable_parameters_regex: Optional[str] = None
    # lora
    adapter_load: Optional[str] = None
    target_modules: List[str] = field(default_factory=lambda: ['all-linear'])
    target_regex: Optional[str] = None
    modules_to_save: List[str] = field(default_factory=list)

    # lora
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: Literal['none', 'all'] = 'none'
    lora_dtype: Literal['float16', 'bfloat16', 'float32', None] = None
    use_rslora: bool = False

    def __post_init__(self):
        if self.freeze_parameters_ratio > 0 and self.pipeline_model_parallel_size > 1:
            raise ValueError('`freeze_parameters_ratio` is not supported when `pipeline_model_parallel_size` > 1')
        if self.target_regex:
            self.target_modules = self.target_regex


@dataclass
class ExtraMegatronArguments(RLHFMegatronArgumentsMixin, MegatronTunerMixin):
    padded_vocab_size: Optional[int] = None
    initialize_embedding: bool = False
    rope_scaling: Optional[Union[dict, str]] = None
    torch_dtype: Optional[Union[torch.dtype, str]] = None
    padding_free: bool = True
    mlp_padding_free: bool = False
    # mcore-bridge
    model: Optional[str] = None
    model_type: Optional[str] = None
    load_safetensors: bool = False
    save_safetensors: bool = False
    adapters: List[str] = field(default_factory=list)
    ref_model: Optional[str] = None
    ref_adapters: List[str] = field(default_factory=list)
    use_hf: bool = False
    # None: use env var `MODELSCOPE_API_TOKEN`
    hub_token: Optional[str] = field(
        default=None, metadata={'help': 'SDK token can be found in https://modelscope.cn/my/myaccesstoken'})
    merge_lora: Optional[bool] = None
    max_shard_size: str = '5GB'
    # streaming dataloader
    dataloader_persistent_workers: bool = True
    dataloader_prefetch_factor: int = 10

    architectures: Optional[str] = None
    llm_architectures: Optional[str] = None
    max_epochs: Optional[int] = None
    enable_dft_loss: bool = False
    enable_channel_loss: bool = False
    task_type: Literal['causal_lm', 'seq_cls'] = None
    num_labels: Optional[int] = None
    problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] = None

    original_max_position_embeddings: Optional[int] = None
    partial_rotary_factor: Optional[float] = None
    use_shared_expert_gate: Optional[bool] = None

    # visual
    vit_gradient_checkpointing: bool = True
    vit_lr: Optional[float] = None
    aligner_lr: Optional[float] = None
    gradient_checkpointing_kwargs: Optional[Union[dict, str]] = None
    # qwen3_next
    linear_num_value_heads: Optional[int] = None
    linear_num_key_heads: Optional[int] = None
    linear_key_head_dim: Optional[int] = None
    linear_value_head_dim: Optional[int] = None
    linear_conv_kernel_dim: Optional[int] = None
    layer_types: Optional[List[str]] = None
    # qwen3_vl, qwen3_omni
    mrope_interleaved: Optional[bool] = None

    @staticmethod
    def load_args_config(ckpt_dir: Optional[str]) -> Dict[str, Any]:
        res = {}
        if ckpt_dir is None:
            return res
        args_path = os.path.join(ckpt_dir, 'args.json')
        if os.path.exists(args_path):
            with open(args_path, 'r', encoding='utf-8') as f:
                old_args = json.load(f)
            keys = list(f.name for f in fields(MegatronTunerMixin))
            keys += ['load', 'padded_vocab_size', 'task_type', 'num_labels']
            for key in keys:
                old_value = old_args.get(key)
                if old_value is not None:
                    res[key] = old_value
            res.pop('adapter_load', None)
            if res['train_type'] != 'lora':
                res.pop('load', None)
        return res


@dataclass
class MegatronArguments(ExtraMegatronArguments):
    # training
    micro_batch_size: int = 1
    global_batch_size: int = 16
    recompute_granularity: Literal['selective', 'full'] = 'selective'
    recompute_method: Literal['uniform', 'block'] = None
    recompute_num_layers: Optional[int] = None
    recompute_modules: List[str] = field(default_factory=lambda: ['core_attn'])
    use_cpu_initialization: bool = False
    deterministic_mode: bool = False
    train_iters: Optional[int] = None
    log_interval: int = 5
    tensorboard_dir: Optional[str] = None
    no_masked_softmax_fusion: bool = False
    no_bias_dropout_fusion: bool = False
    no_bias_swiglu_fusion: bool = False
    no_rope_fusion: Optional[bool] = None
    no_gradient_accumulation_fusion: bool = False
    cross_entropy_loss_fusion: bool = False
    cross_entropy_fusion_impl: Literal['native', 'te'] = 'native'
    calculate_per_token_loss: Optional[bool] = None
    use_flash_attn: bool = False
    attention_backend: str = 'flash'  # flash, fused, unfused, local, auto
    optimizer: Literal['adam', 'sgd'] = 'adam'
    optimizer_cpu_offload: bool = False
    optimizer_offload_fraction: float = 1.
    use_precision_aware_optimizer: bool = False
    main_grads_dtype: Literal['fp32', 'bf16'] = 'fp32'
    main_params_dtype: Literal['fp32', 'fp16'] = 'fp32'
    exp_avg_dtype: Literal['fp32', 'fp16', 'bf16', 'fp8'] = 'fp32'
    exp_avg_sq_dtype: Literal['fp32', 'fp16', 'bf16', 'fp8'] = 'fp32'
    dataloader_type: Literal['single', 'cyclic', 'external'] = 'cyclic'
    manual_gc: bool = False
    manual_gc_interval: int = 0

    # learning rate
    lr: Optional[float] = None
    lr_decay_style: Literal['cosine', 'linear', 'constant'] = 'cosine'
    # The default is None, which will be set to `train_iters`.
    lr_decay_iters: Optional[int] = None
    lr_warmup_iters: int = 0
    lr_warmup_fraction: Optional[float] = None
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
    save_retain_interval: Optional[int] = None
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
    async_save: bool = False
    use_persistent_ckpt_worker: bool = False
    ckpt_fully_parallel_load: bool = False
    ckpt_assume_constant_structure: bool = False

    # dist
    distributed_backend: Literal['nccl', 'gloo'] = 'nccl'
    local_rank: Optional[int] = None
    use_distributed_optimizer: bool = True
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    decoder_first_pipeline_num_layers: Optional[int] = None
    decoder_last_pipeline_num_layers: Optional[int] = None
    sequence_parallel: bool = False
    context_parallel_size: int = 1
    tp_comm_overlap: bool = False
    overlap_grad_reduce: bool = False
    overlap_param_gather: bool = False
    distributed_timeout_minutes: int = 300000
    num_layers_per_virtual_pipeline_stage: Optional[int] = None
    num_virtual_stages_per_pipeline_rank: Optional[int] = None
    microbatch_group_size_per_virtual_pipeline_stage: Optional[int] = None
    pipeline_model_parallel_layout: Optional[str] = None

    # model
    num_layers: Optional[int] = None
    hidden_size: Optional[int] = None
    ffn_hidden_size: Optional[int] = None
    num_attention_heads: Optional[int] = None
    group_query_attention: Optional[bool] = None
    num_query_groups: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    position_embedding_type: Optional[Literal['learned_absolute', 'rope', 'mrope', 'relative', 'none']] = None
    mrope_section: Optional[List[int]] = None
    rotary_base: Optional[int] = None
    rotary_percent: float = 1.
    rotary_interleaved: Optional[bool] = None
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
    moe_layer_freq: Optional[str] = None
    moe_ffn_hidden_size: Optional[int] = None
    moe_shared_expert_intermediate_size: Optional[int] = None

    moe_router_topk: Optional[int] = None
    moe_router_num_groups: Optional[int] = None
    moe_router_group_topk: Optional[int] = None
    moe_router_pre_softmax: Optional[bool] = None
    moe_router_dtype: Literal['none', 'fp32', 'fp64'] = 'fp32'
    moe_router_score_function: Literal['sigmoid', 'softmax'] = None
    moe_router_bias_update_rate: Optional[float] = None
    moe_router_enable_expert_bias: Optional[bool] = None
    moe_router_topk_scaling_factor: Optional[float] = None
    moe_router_load_balancing_type: Literal['aux_loss', 'seq_aux_loss', 'sinkhorn', 'none'] = None

    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: int = 1
    moe_token_dispatcher_type: Literal['allgather', 'alltoall', 'flex', 'alltoall_seq'] = 'alltoall'
    moe_enable_deepep: bool = False
    moe_grouped_gemm: bool = True
    moe_permute_fusion: bool = False
    moe_aux_loss_coeff: float = 0.
    moe_z_loss_coeff: Optional[float] = None
    moe_shared_expert_overlap: bool = False
    moe_layer_recompute: bool = False
    moe_expert_capacity_factor: Optional[float] = None
    moe_pad_expert_input_to_capacity: bool = False
    moe_token_drop_policy: Literal['probs', 'position'] = 'probs'

    # mla
    multi_latent_attention: Optional[bool] = None
    q_lora_rank: Optional[int] = None
    kv_lora_rank: Optional[int] = None
    qk_head_dim: Optional[int] = None
    qk_pos_emb_head_dim: Optional[int] = None

    # fp8
    fp8_format: Literal['e4m3', 'hybrid'] = None
    fp8_recipe: Literal['tensorwise', 'delayed', 'mxfp8', 'blockwise'] = 'delayed'
    fp8_amax_history_len: int = 1024
    fp8_amax_compute_algo: Literal['most_recent', 'max'] = 'max'
    fp8_param_gather: bool = False

    # mixed precision
    fp16: Optional[bool] = None
    bf16: Optional[bool] = None
    apply_query_key_layer_scaling: Optional[bool] = None
    attention_softmax_in_fp32: bool = True

    # logging
    log_params_norm: bool = False
    log_throughput: bool = False
    tensorboard_log_interval: int = 1
    tensorboard_queue_size: int = 50
    log_timers_to_tensorboard: bool = True
    no_log_learning_rate_to_tensorboard: bool = False
    log_validation_ppl_to_tensorboard: bool = True
    log_memory_to_tensorboard: bool = True
    logging_level: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_exp_name: Optional[str] = None
    wandb_save_dir: Optional[str] = None

    # evaluate
    eval_iters: int = -1
    eval_interval: Optional[int] = None

    # other
    seed: int = 42
    seq_length: Optional[int] = None
    num_workers: int = 4

    # extra_args for megatron
    megatron_extra_kwargs: Optional[Union[dict, str]] = None

    def _set_default(self):
        if self.mlp_padding_free and self.sequence_parallel:
            raise ValueError('mlp_padding_free is not compatible with sequence_parallel.')
        if self.local_rank is None:
            self.local_rank = get_dist_setting()[1]
        if self.lr is None:
            if self.train_type == 'full':
                self.lr = 1e-5
            else:
                self.lr = 1e-4
        if self.num_query_groups is None:
            self.num_query_groups = 1
        if self.norm_epsilon is None:
            self.norm_epsilon = 1e-5
        if self.rotary_base is None:
            self.rotary_base = 10000
        if self.rotary_interleaved is None:
            self.rotary_interleaved = False
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
        if self.qk_layernorm is None:
            self.qk_layernorm = False
        if self.multi_latent_attention is None:
            self.multi_latent_attention = False
        if self.kv_lora_rank is None:
            self.kv_lora_rank = 32
        if self.qk_head_dim is None:
            self.qk_head_dim = 128
        if self.qk_pos_emb_head_dim is None:
            self.qk_pos_emb_head_dim = 64
        if self.task_type is None:
            self.task_type = 'causal_lm'
        if self.calculate_per_token_loss is None:
            self.calculate_per_token_loss = self.task_type == 'causal_lm'
        # moe
        if self.use_shared_expert_gate is None:
            self.use_shared_expert_gate = False
        if self.moe_router_score_function is None:
            self.moe_router_score_function = 'softmax'
        if self.moe_router_topk is None:
            self.moe_router_topk = 2
        if self.moe_router_pre_softmax is None:
            self.moe_router_pre_softmax = False
        if self.moe_router_load_balancing_type is None:
            self.moe_router_load_balancing_type = 'aux_loss'
        if self.moe_router_enable_expert_bias is None:
            self.moe_router_enable_expert_bias = False
        if self.moe_layer_freq is None:
            self.moe_layer_freq = '1'
        if self.mrope_interleaved is None:
            self.mrope_interleaved = False

    def _init_mixed_precision(self):
        from swift.llm.argument.base_args.model_args import ModelArguments
        ModelArguments._init_mixed_precision(self)
        if self.apply_query_key_layer_scaling is None:
            self.apply_query_key_layer_scaling = self.fp16
        if self.apply_query_key_layer_scaling:
            os.environ['NVTE_APPLY_QK_LAYER_SCALING'] = '1'

    def _init_moe(self):
        if self.moe_router_dtype.lower() == 'none':
            self.moe_router_dtype = None
        if self.moe_shared_expert_intermediate_size == 0:
            self.moe_shared_expert_intermediate_size = None
        if self.num_experts is not None:
            if self.moe_ffn_hidden_size is None:
                self.moe_ffn_hidden_size = self.ffn_hidden_size

    @staticmethod
    def _patch_megatron_timeout(distributed_timeout_minutes: int):
        from megatron.core import parallel_state
        create_group_origin = parallel_state.create_group

        def create_group(ranks=None, timeout=None, *args, **kwargs):
            if timeout is None:
                timeout = timedelta(minutes=distributed_timeout_minutes)
            return create_group_origin(ranks, timeout, *args, **kwargs)

        parallel_state.create_group = create_group

    def __post_init__(self):
        require_version('numpy<2.0', 'Please install numpy<2.0 by running: `pip install "numpy<2.0"`.')
        if self.train_type == 'lora':
            if self.num_experts is not None:
                require_version('peft>=0.15')
            else:
                require_version('peft>=0.12')
        RLHFMegatronArgumentsMixin.__post_init__(self)
        MegatronTunerMixin.__post_init__(self)
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
        self._set_default()
        self.model_info, self.model_meta = get_model_info_meta(
            self.model, model_type=self.model_type, use_hf=self.use_hf, hub_token=self.hub_token)
        self.model_type = self.model_info.model_type
        if hasattr(self, 'ddp_timeout'):
            self.distributed_timeout_minutes = self.ddp_timeout // 60
        self._patch_megatron_timeout(self.distributed_timeout_minutes)
        self.group_query_attention = self.num_query_groups > 1
        if self.rope_scaling is not None:
            self.rope_scaling = json_parse_to_dict(self.rope_scaling)
            if 'type' in self.rope_scaling and 'rope_type' not in self.rope_scaling:
                self.rope_scaling['rope_type'] = self.rope_scaling['type']
        if self.task_type != 'causal_lm':
            self.untie_embeddings_and_output_weights = True
        if self.gradient_checkpointing_kwargs is not None:
            self.gradient_checkpointing_kwargs = json_parse_to_dict(self.gradient_checkpointing_kwargs)
        if self.eval_interval is None:
            self.eval_interval = self.save_interval
        if self.seq_length is None:
            self.seq_length = self.max_position_embeddings
        if self.position_embedding_type is None:
            self.position_embedding_type = 'rope'
        if self.merge_lora is None:
            self.merge_lora = self.save_safetensors
        if self.adapters or self.adapter_load or self.ref_adapter_load:
            if self.train_type == 'full':
                self.train_type = 'lora'
                logger.info('Setting args.train_type: lora')
        if self.adapters:
            self._load_adapter_config()
        self._init_moe()
        self._init_mixed_precision()

        self.megatron_extra_kwargs = json_parse_to_dict(self.megatron_extra_kwargs)
        self._init_no_rope_fusion()

    def _init_no_rope_fusion(self):
        if self.no_rope_fusion is not None:
            return
        if self.multi_latent_attention or self.rotary_interleaved:
            # Upgrading transformer_engine requires checking here.
            self.no_rope_fusion = True
        else:
            self.no_rope_fusion = False
        logger.info(f'Setting args.no_rope_fusion: {self.no_rope_fusion}.')

    def _args_to_argv(self) -> Tuple[List[Any], Dict[str, Any]]:
        new_args = []
        args_dict = asdict(self)
        extra_args = {}
        extra_args['model_dir'] = self.model_info.model_dir
        extra_args['is_multimodal'] = self.model_meta.is_multimodal
        # model_type may be overridden by megatron
        extra_args['hf_model_type'] = self.model_type
        megatron_extra_kwargs = args_dict.pop('megatron_extra_kwargs')
        args_dict.update(megatron_extra_kwargs)
        for k, value in args_dict.items():
            if k not in MegatronArguments.__annotations__ and k not in megatron_extra_kwargs:
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

    def _load_adapter_config(self):
        assert len(self.adapters) == 1, 'Currently only support one adapter'
        adapter_path = self.adapters[0]
        adapter_config_path = os.path.join(adapter_path, 'adapter_config.json')
        adapter_config = {}
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
        mapping = {'r': 'lora_rank', 'bias': 'lora_bias'}
        for k in ['lora_alpha', 'lora_dropout', 'use_rslora']:
            mapping[k] = k
        for k, v in adapter_config.items():
            if k not in mapping:
                continue
            k = mapping[k]
            if v != getattr(self, k):
                setattr(self, k, v)
                logger.info(f'Setting {k}: {v}')

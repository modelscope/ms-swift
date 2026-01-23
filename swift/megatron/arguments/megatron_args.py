# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import json
import megatron.core
import torch
from packaging import version
from transformers.utils import is_torch_npu_available
from transformers.utils.versions import require_version

from swift.arguments import ModelArguments
from swift.model import get_model_info_meta
from swift.utils import get_dist_setting, get_logger, json_parse_to_dict

mcore_015 = version.parse(megatron.core.__version__) >= version.parse('0.15.0rc0')
logger = get_logger()
MAX_NPU_EXPERTS_PER_EP = 128


@dataclass
class RLHFMegatronArgumentsMixin:
    rlhf_type: Literal['dpo', 'kto', 'grpo', 'gkd', 'rm'] = None
    ref_load: Optional[str] = None
    ref_adapter_load: Optional[str] = None

    beta: Optional[float] = None
    rpo_alpha: Optional[float] = None
    reference_free: bool = False
    label_smoothing: float = 0.
    f_divergence_type: str = 'reverse_kl'

    # kto
    desirable_weight: float = 1.
    undesirable_weight: float = 1.
    calculate_KL: Optional[bool] = None

    # rm
    center_rewards_coefficient: Optional[float] = None

    # gkd
    teacher_model: Optional[str] = field(default=None)
    teacher_model_type: Optional[str] = field(default=None)
    teacher_model_revision: Optional[str] = field(default=None)
    lmbda: float = 0.5  # On-policy probability: with prob lmbda, use student-generated responses
    seq_kd: bool = False  # Sequential KD: use teacher-generated responses when not on-policy
    offload_teacher_model: bool = False  # Offload teacher model to CPU to save GPU memory
    sft_alpha: float = 0.0  # Weight for SFT loss in GKD (0 = pure JSD, >0 = JSD + sft_alpha * SFT)

    # grpo/gkd
    temperature: float = 0.9  # Temperature for sampling and loss computation

    # grpo
    generation_batch_size: Optional[int] = None
    steps_per_generation: Optional[int] = None
    num_generations: int = 8
    num_generations_eval: Optional[int] = None
    max_completion_length: int = 512
    # GSPO https://arxiv.org/abs/2507.18071
    importance_sampling_level: Literal['token', 'sequence', 'sequence_token'] = 'token'

    # SAPO https://arxiv.org/abs/2511.20347
    # Temperature parameters for soft adaptive gate
    tau_pos: float = 1.0
    tau_neg: float = 1.05

    epsilon: float = 0.2
    epsilon_high: Optional[float] = None
    delta: Optional[float] = None
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.

    use_vllm: bool = True
    vllm_mode: Optional[Literal['server', 'colocate']] = None

    vllm_enable_prefix_caching: bool = True
    vllm_gpu_memory_utilization: float = 0.9
    vllm_tensor_parallel_size: int = 1
    vllm_max_model_len: Optional[int] = None
    vllm_enforce_eager: bool = False
    vllm_limit_mm_per_prompt: Optional[Union[dict, str]] = None  # '{"image": 5, "video": 2}'
    vllm_disable_cascade_attn: bool = False
    vllm_max_num_seqs: Optional[int] = None
    vllm_mm_processor_cache_gb: Optional[float] = None
    vllm_engine_kwargs: Optional[Dict[str, Any]] = None

    sleep_level: Literal[0, 1, 2] = 0
    offload_optimizer: bool = False
    offload_model: bool = False
    offload_bridge: bool = False

    vllm_server_base_url: Optional[List[str]] = None
    vllm_server_host: Optional[List[str]] = None
    vllm_server_port: List[int] = field(default_factory=lambda: [8000])
    vllm_server_timeout: float = 240.0
    vllm_server_group_port: Optional[List[int]] = None

    reward_funcs: List[str] = field(default_factory=list)
    reward_weights: List[float] = None
    # see details in swift/rewards/orm.py
    # cosine reward, https://arxiv.org/abs/2502.03373
    cosine_min_len_value_wrong: float = -0.5  # r^w_0 in paper, Reward for wrong answers with zero completion length.
    cosine_max_len_value_wrong: float = 0.0  # r^w_L in paper, Reward for wrong answers with max completion length.
    cosine_min_len_value_correct: float = 1.0  # r^c_0 in paper, Reward for correct answers with zero completion length.
    cosine_max_len_value_correct: float = 0.5  # r^c_L in paper, Reward for correct answers with max completion length.
    cosine_max_len: Optional[int] = None  # Lmax in paper, default equal to max_completion_length
    # repetition penalty, https://arxiv.org/abs/2502.03373
    repetition_n_grams: int = 3
    repetition_max_penalty: float = -1.0
    # soft_overlong, https://arxiv.org/abs/2503.14476
    soft_max_length: Optional[int] = None
    soft_cache_length: Optional[int] = None
    # DAPO, https://arxiv.org/abs/2503.14476
    dynamic_sample: bool = False
    max_resample_times: int = 3
    overlong_filter: bool = False

    # Dr. GRPO, https://arxiv.org/abs/2503.20783
    # GDPO: normalize each reward function separately
    scale_rewards: Literal['none', 'group', 'batch', 'gdpo'] = 'group'

    # RLOO / REINFORCE++
    advantage_estimator: Literal['grpo', 'rloo', 'reinforce_plus_plus'] = 'grpo'
    kl_in_reward: bool = False

    wandb_log_unique_prompts: Optional[bool] = None
    log_completions: bool = False

    rollout_importance_sampling_mode: Optional[Literal['token_truncate', 'token_mask', 'sequence_truncate',
                                                       'sequence_mask']] = None
    rollout_importance_sampling_threshold: float = 2.0
    log_rollout_offpolicy_metrics: bool = False
    # Off-Policy Sequence Masking: mask out sequences that deviate too much from rollout policy
    # If set, compute mean(rollout_per_token_logps - per_token_logps) per sequence,
    # and mask sequences where this delta > threshold AND advantage < 0
    # Falls back to old_per_token_logps if rollout_per_token_logps is not available
    off_policy_sequence_mask_delta: Optional[float] = None

    # entropy
    log_entropy: bool = False
    # Beyond the 80/20 Rule, https://arxiv.org/abs/2506.01939
    top_entropy_quantile: float = 1.0

    # ───────────────────────────  Not Supported Yet  ───────────────────────────

    # reward model
    reward_model: Optional[List[str]] = None
    reward_model_plugin: Optional[List[str]] = None
    # sync ref model
    sync_ref_model: bool = False
    ref_model_sync_steps: int = 512
    ref_model_mixup_alpha: float = 0.6

    async_generate: bool = False

    move_model_batches: Optional[int] = None

    # multi turn
    multi_turn_scheduler: Optional[str] = None
    max_turns: Optional[int] = None
    completion_length_limit_scope: Literal['total', 'per_round'] = 'per_round'
    vllm_server_pass_dataset: bool = False

    num_iterations: int = 1

    # dataset
    dataset_shuffle: Optional[bool] = True

    def _init_kto(self):
        if self.calculate_KL is None:
            # Not all losses require a KL calculation
            self.calculate_KL = True
            if self.loss_type in ['apo_zero_unpaired']:
                self.calculate_KL = False

    def __post_init__(self):
        if self.rlhf_type is None:
            return
        default_loss_type = {'kto': 'kto', 'dpo': 'sigmoid', 'grpo': 'grpo'}
        default_beta = {'gkd': 0.5, 'grpo': 0.04}
        if self.beta is None:
            self.beta = default_beta.get(self.rlhf_type, 0.1)
        if self.loss_type is None:
            self.loss_type = default_loss_type.get(self.rlhf_type)
        if self.rlhf_type == 'kto':
            self._init_kto()
        if self.rlhf_type == 'grpo':
            assert self.vllm_mode is not None, 'vllm_mode is required for Megatron GRPO'
            self._init_grpo()
            if self.vllm_limit_mm_per_prompt is not None:
                self.vllm_limit_mm_per_prompt = json_parse_to_dict(self.vllm_limit_mm_per_prompt)
            self.vllm_engine_kwargs = json_parse_to_dict(self.vllm_engine_kwargs)

    def _init_grpo(self):

        def _check_not_supported():
            if self.async_generate:
                raise ValueError('async_generate is not supported for Megatron GRPO right now')
            if self.sync_ref_model:
                raise ValueError('sync_ref_model is not supported for Megatron GRPO right now')
            if not self.dataset_shuffle:
                raise ValueError('dataset_shuffle false is not supported for Megatron GRPO')
            if self.multi_turn_scheduler:
                raise ValueError('multi_turn_scheduler is not supported for Megatron GRPO right now')
            if self.num_iterations > 1:
                raise ValueError('num_iterations > 1 is not supported for Megatron GRPO right now')

        def _check_batch_params():
            # Set default values if both are None
            if self.generation_batch_size is None and self.steps_per_generation is None:
                self.steps_per_generation = 1
                self.generation_batch_size = self.global_batch_size * self.steps_per_generation
            # Both configured - error
            elif self.generation_batch_size is not None and self.steps_per_generation is not None:
                raise ValueError("'generation_batch_size' and 'steps_per_generation' cannot be both configured")
            # Only generation_batch_size configured
            elif self.generation_batch_size is not None:
                if self.generation_batch_size % self.global_batch_size != 0:
                    raise ValueError(f'generation_batch_size ({self.generation_batch_size}) '
                                     f'must be divisible by global_batch_size ({self.global_batch_size})')
                self.steps_per_generation = self.generation_batch_size // self.global_batch_size
            # Only steps_per_generation configured
            else:
                self.generation_batch_size = self.global_batch_size * self.steps_per_generation

            world_size = torch.distributed.get_world_size()
            dp_size = world_size // (
                self.pipeline_model_parallel_size * self.tensor_model_parallel_size * self.context_parallel_size)
            num_rollout_prompt = self.generation_batch_size // self.num_generations
            if num_rollout_prompt % dp_size != 0:
                raise ValueError(f'num_rollout_prompt ({num_rollout_prompt}) = generation_batch_size '
                                 f'({self.generation_batch_size}) // num_generations ({self.num_generations}) '
                                 f'must be divisible by dp_size ({dp_size}). '
                                 f'Please adjust generation_batch_size/steps_per_generation/num_generations.')

            per_device_num_rollout_prompt = num_rollout_prompt // dp_size
            assert per_device_num_rollout_prompt >= 1, \
                (f'per_device_num_rollout_prompt ({per_device_num_rollout_prompt}) must be greater than 1, '
                 f'please adjust generation_batch_size/steps_per_generation/num_generations to make it greater than 1')

            if per_device_num_rollout_prompt % self.micro_batch_size != 0:
                raise ValueError(f'Per-device rollout prompt count ({per_device_num_rollout_prompt}) = '
                                 f'(generation_batch_size ({self.generation_batch_size}) // '
                                 f'num_generations ({self.num_generations})) // dp_size ({dp_size}) '
                                 f'must be divisible by micro_batch_size ({self.micro_batch_size}). '
                                 f'Please adjust arguments to satisfy: '
                                 f'(generation_batch_size // num_generations) // dp_size % '
                                 f'micro_batch_size == 0')

            self.per_device_generation_batch_size = self.generation_batch_size // world_size
            assert self.per_device_generation_batch_size >= 1, \
                (f'per_device_generation_batch_size ({self.per_device_generation_batch_size}) must be greater than 1, '
                 f'please adjust generation_batch_size/steps_per_generation/num_generations to make it greater than 1')

        _check_not_supported()
        _check_batch_params()
        self.remove_unused_columns = False
        logger.info(f'Setting args.remove_unused_columns: {self.remove_unused_columns}')
        if self.truncation_strategy is None:
            self.truncation_strategy = 'left'
        assert self.truncation_strategy in ['left', 'delete'
                                            ], ("GRPO requires `truncation_strategy 'left' or 'delete'`, "
                                                f"Current value: `truncation_strategy='{self.truncation_strategy}'`."
                                                )  # noqa
        if self.beta is None:
            self.beta = 0.04  # https://arxiv.org/abs/2402.03300
        if self.async_generate:
            logger.info('Using async mode. This is a approximate version which '
                        'will use the old weights to generate responses to accelerate. '
                        'This will ignore the `CLIP` of advantages, if you found the training '
                        'is unstable, you may consider using --async_generate false.')
        if 'soft_overlong' in self.reward_funcs:
            assert self.soft_cache_length is not None, \
                'The soft_cache_length must be set when using soft overlong rewards.'
            if self.soft_max_length is None:
                self.soft_max_length = self.max_completion_length
                logger.info(f'Auto-configured soft_max_length = max_completion_length {self.max_completion_length}')
        assert self.use_vllm, 'use_vllm must be True for Megatron GRPO'


@dataclass
class MegatronTunerMixin:
    tuner_type: Literal['lora', 'full'] = 'full'
    train_type: Optional[Literal['lora', 'full']] = None  # compat swift3.x
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
        if self.train_type is not None:
            logger.warning('`train_type` is deprecated, please use `tuner_type` instead.')
            self.tuner_type = self.train_type
        if 0 < self.freeze_parameters_ratio < 1 and self.pipeline_model_parallel_size > 1:
            raise ValueError('`freeze_parameters_ratio` is not supported when `pipeline_model_parallel_size` > 1')
        if self.target_regex:
            self.target_modules = self.target_regex


@dataclass
class ExtraMegatronArguments(RLHFMegatronArgumentsMixin, MegatronTunerMixin):
    loss_type: Optional[str] = None  # rlhf / plugins

    check_model: bool = True
    padded_vocab_size: Optional[int] = None
    initialize_embedding: bool = False
    rope_scaling: Optional[Union[dict, str]] = None
    torch_dtype: Optional[Union[torch.dtype, str]] = None
    padding_free: bool = True
    mlp_padding_free: bool = False
    # mcore-bridge
    model: Optional[str] = None
    model_type: Optional[str] = None
    load_safetensors: Optional[bool] = None
    save_safetensors: bool = True
    adapters: List[str] = field(default_factory=list)
    ref_model: Optional[str] = None
    ref_adapters: List[str] = field(default_factory=list)
    use_hf: bool = False
    # None: use env var `MODELSCOPE_API_TOKEN`
    hub_token: Optional[str] = field(
        default=None, metadata={'help': 'SDK token can be found in https://modelscope.cn/my/myaccesstoken'})
    merge_lora: Optional[bool] = None
    max_shard_size: str = '5GB'

    # dataloader
    train_dataloader_shuffle: bool = True
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = True
    dataloader_prefetch_factor: int = 2
    group_by_length: bool = False

    hf_model_type: Optional[str] = None
    llm_model_type: Optional[str] = None
    max_epochs: Optional[int] = None
    enable_dft_loss: bool = False
    enable_channel_loss: bool = False
    task_type: Literal['causal_lm', 'seq_cls', 'embedding', 'generative_reranker'] = None
    num_labels: Optional[int] = None
    problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] = None
    save_strategy: Literal['steps', 'epoch'] = 'steps'

    original_max_position_embeddings: Optional[int] = None
    partial_rotary_factor: Optional[float] = None
    use_shared_expert_gate: Optional[bool] = None

    report_to: Optional[Literal['wandb', 'swanlab']] = None

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
            if res['tuner_type'] != 'lora':
                res.pop('load', None)
        return res


@dataclass
class MegatronArguments(ExtraMegatronArguments):
    # training
    micro_batch_size: int = 1
    global_batch_size: int = 16
    recompute_granularity: Literal['selective', 'full', 'none'] = 'selective'
    recompute_method: Literal['uniform', 'block'] = None
    recompute_num_layers: Optional[int] = None
    recompute_modules: List[str] = field(default_factory=lambda: ['core_attn'])
    use_cpu_initialization: bool = False
    deterministic_mode: bool = False
    train_iters: Optional[int] = None
    log_interval: int = 5
    tensorboard_dir: Optional[str] = None
    no_masked_softmax_fusion: bool = False
    no_bias_dropout_fusion: Optional[bool] = None
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
    account_for_embedding_in_pipeline_split: bool = False
    account_for_loss_in_pipeline_split: bool = False

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
    softmax_type: Optional[Literal['vanilla', 'off-by-one', 'learnable']] = None
    window_size: Optional[str] = None
    window_attn_skip_freq: Optional[str] = None
    max_position_embeddings: Optional[int] = None
    position_embedding_type: Optional[Literal['learned_absolute', 'rope', 'mrope', 'relative', 'none']] = None
    mrope_section: Optional[List[int]] = None
    rotary_base: Optional[int] = None
    rotary_percent: float = 1.
    rotary_interleaved: Optional[bool] = None
    normalization: Literal['LayerNorm', 'RMSNorm'] = 'RMSNorm'
    norm_epsilon: Optional[float] = None
    swiglu: Optional[bool] = None
    quick_geglu: Optional[bool] = None
    activation_func_clamp_value: Optional[float] = None
    glu_linear_offset: Optional[float] = None
    untie_embeddings_and_output_weights: Optional[bool] = None
    disable_bias_linear: Optional[bool] = None
    add_qkv_bias: Optional[bool] = None
    attention_dropout: Optional[float] = None
    hidden_dropout: float = 0.
    kv_channels: Optional[int] = None
    qk_layernorm: Optional[bool] = None
    qk_l2_norm: Optional[bool] = None
    no_rope_freq: Optional[int] = None
    moe_apply_probs_on_input: Optional[bool] = None
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
    moe_router_load_balancing_type: Literal['aux_loss', 'seq_aux_loss', 'global_aux_loss', 'sinkhorn', 'none'] = None

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
    v_head_dim: Optional[int] = None

    # mtp
    mtp_num_layers: Optional[int] = None
    mtp_loss_scaling_factor: float = 0.1

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
    wandb_project: str = 'megatron-swift'
    wandb_exp_name: Optional[str] = None
    wandb_save_dir: Optional[str] = None

    # evaluate
    eval_iters: int = -1
    eval_interval: Optional[int] = None

    # other
    seed: int = 42
    seq_length: Optional[int] = None
    num_workers: int = 4
    no_data_sharding: bool = False

    # extra_args for megatron
    megatron_extra_kwargs: Optional[Union[dict, str]] = None

    def _set_default(self):
        if self.mlp_padding_free and (self.sequence_parallel or self.context_parallel_size > 1):
            raise ValueError('mlp_padding_free is not compatible with sequence parallel or context parallel.')
        if self.local_rank is None:
            self.local_rank = get_dist_setting()[1]
        if self.lr is None:
            if self.tuner_type == 'full':
                self.lr = 1e-5
            else:
                self.lr = 1e-4
        if self.num_query_groups is None:
            self.num_query_groups = 1
        if self.softmax_type is None and mcore_015:
            self.softmax_type = 'vanilla'
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
        if self.quick_geglu is None:
            self.quick_geglu = False
        if self.glu_linear_offset is None and mcore_015:
            self.glu_linear_offset = 0.
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
        if self.v_head_dim is None:
            self.v_head_dim = 128
        if self.task_type is None:
            self.task_type = 'causal_lm'
        if self.calculate_per_token_loss is None:
            self.calculate_per_token_loss = self.task_type == 'causal_lm'
        if self.no_bias_dropout_fusion is None:
            self.no_bias_dropout_fusion = False
        # moe
        MegatronArguments._set_moe_default(self)
        # log
        if self.wandb_exp_name is None:
            self.wandb_exp_name = self.save

    @staticmethod
    def _set_moe_default(self):
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
            self.moe_layer_freq = 1
        if self.mrope_interleaved is None:
            self.mrope_interleaved = False

    def _init_mixed_precision(self):
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
            if is_torch_npu_available() and self.num_experts > MAX_NPU_EXPERTS_PER_EP:
                required_ep = (self.num_experts + MAX_NPU_EXPERTS_PER_EP - 1) // MAX_NPU_EXPERTS_PER_EP
                if self.expert_model_parallel_size < required_ep:
                    logger.warning(f'{">"*20} WARNING {"<"*20}\n'
                                   f'MindSpeed on NPU supports up to {MAX_NPU_EXPERTS_PER_EP} experts per EP group. '
                                   f'num_experts={self.num_experts}, '
                                   f'expert_model_parallel_size={self.expert_model_parallel_size}. '
                                   f'Please set expert_model_parallel_size (EP) to {required_ep} '
                                   f'(num_experts / {MAX_NPU_EXPERTS_PER_EP}) or higher.')

    def __post_init__(self):
        require_version('numpy<2.0', 'Please install numpy<2.0 by running: `pip install "numpy<2.0"`.')
        if self.tuner_type == 'lora':
            if self.num_experts is not None:
                require_version('peft>=0.15')
            else:
                require_version('peft>=0.12')
        RLHFMegatronArgumentsMixin.__post_init__(self)
        MegatronTunerMixin.__post_init__(self)
        os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')
        if self.recompute_granularity == 'none':
            self.recompute_granularity = None
        self._set_default()
        self.model_info, self.model_meta = get_model_info_meta(
            self.model, model_type=self.model_type, use_hf=self.use_hf, hub_token=self.hub_token)
        self.model_type = self.model_info.model_type
        if self.pipeline_model_parallel_size == 1 and (self.decoder_first_pipeline_num_layers is not None
                                                       or self.decoder_last_pipeline_num_layers is not None):
            raise ValueError('pipeline_model_parallel_size must be greater than 1 if you want to set '
                             'decoder_first_pipeline_num_layers or decoder_last_pipeline_num_layers.')
        if hasattr(self, 'ddp_timeout'):
            self.distributed_timeout_minutes = self.ddp_timeout // 60
        self.group_query_attention = self.num_query_groups > 1
        if self.rope_scaling is not None:
            self.rope_scaling = json_parse_to_dict(self.rope_scaling)
            if 'type' in self.rope_scaling and 'rope_type' not in self.rope_scaling:
                self.rope_scaling['rope_type'] = self.rope_scaling['type']
        if self.task_type not in {'causal_lm', 'generative_reranker'}:
            self.untie_embeddings_and_output_weights = True
        if self.gradient_checkpointing_kwargs is not None:
            self.gradient_checkpointing_kwargs = json_parse_to_dict(self.gradient_checkpointing_kwargs)
        if self.save_strategy == 'epoch':
            self.save_interval = 1
            self.eval_interval = 1
        if not self.no_gradient_accumulation_fusion:
            try:
                import apex
            except ImportError:
                logger.warning('apex is not installed, so gradient accumulation fusion is disabled.')
                self.no_gradient_accumulation_fusion = True
        if isinstance(self.ref_adapters, str):
            self.ref_adapters = [self.ref_adapters]
        if self.eval_interval is None:
            self.eval_interval = self.save_interval
        if self.seq_length is None:
            self.seq_length = self.max_position_embeddings
        if self.position_embedding_type is None:
            self.position_embedding_type = 'rope'
        if self.merge_lora is None:
            self.merge_lora = self.save_safetensors
        if self.adapters or self.adapter_load or self.ref_adapter_load:
            if self.tuner_type == 'full':
                self.tuner_type = 'lora'
                logger.info('Setting args.tuner_type: lora')
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

# Copyright (c) ModelScope Contributors. All rights reserved.
import os
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Literal, Optional, Union

import json
import megatron.core
import torch
from megatron.core import mpu
from megatron.core.transformer.enums import AttnBackend
from packaging import version
from transformers.utils.versions import require_version

from swift.arguments import ModelArguments
from swift.megatron.model import get_megatron_model_meta
from swift.megatron.utils import initialize_megatron
from swift.model import get_model_info_meta
from swift.utils import get_dist_setting, get_logger, json_parse_to_dict

mcore_015 = version.parse(megatron.core.__version__) >= version.parse('0.15.0rc0')
logger = get_logger()


@dataclass
class RLHFMegatronArgumentsMixin:
    rlhf_type: Literal['dpo', 'kto', 'grpo', 'gkd', 'rm'] = None
    loss_type: Optional[str] = None  # rlhf / plugins
    mcore_ref_model: Optional[str] = None
    mcore_ref_adapter: Optional[str] = None

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
        if self.truncation_strategy not in {'left', 'delete'}:
            raise ValueError("GRPO requires `truncation_strategy 'left' or 'delete'`, "
                             f"Current value: `truncation_strategy='{self.truncation_strategy}'`.")
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
class MegatronArguments(RLHFMegatronArgumentsMixin, MegatronTunerMixin):
    # training
    micro_batch_size: int = 1
    global_batch_size: int = 16
    recompute_granularity: Literal['selective', 'full', 'none'] = 'selective'
    recompute_method: Literal['uniform', 'block'] = None
    recompute_num_layers: Optional[int] = None
    recompute_modules: List[str] = field(default_factory=lambda: ['core_attn'])
    train_iters: Optional[int] = None
    num_train_epochs: Optional[int] = None

    masked_softmax_fusion: bool = True
    bias_dropout_fusion: bool = True
    bias_activation_fusion: bool = True
    apply_rope_fusion: bool = True
    gradient_accumulation_fusion: bool = True
    cross_entropy_loss_fusion: bool = True
    cross_entropy_fusion_impl: Literal['native', 'te'] = 'native'
    calculate_per_token_loss: Optional[bool] = None
    attention_backend: str = 'flash'  # flash, fused, unfused, local, auto
    optimizer: Literal['adam', 'sgd'] = 'adam'
    optimizer_cpu_offload: bool = False
    optimizer_offload_fraction: float = 1.
    use_precision_aware_optimizer: bool = False
    main_grads_dtype: Literal['fp32', 'bf16'] = 'fp32'
    main_params_dtype: Literal['fp32', 'fp16'] = 'fp32'
    exp_avg_dtype: Literal['fp32', 'fp16', 'bf16', 'fp8'] = 'fp32'
    exp_avg_sq_dtype: Literal['fp32', 'fp16', 'bf16', 'fp8'] = 'fp32'
    manual_gc: bool = False
    manual_gc_interval: int = 0
    manual_gc_eval: bool = True

    # data
    seed: int = 42
    train_dataloader_shuffle: bool = True
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = True
    dataloader_prefetch_factor: int = 2
    data_sharding: bool = False
    group_by_length: bool = False
    te_rng_tracker: bool = False
    data_parallel_random_init: Optional[bool] = False
    padding_free: bool = True
    mlp_padding_free: bool = False

    # learning rate
    lr_warmup_init: float = 0.
    lr: Optional[float] = None
    lr_decay_style: Literal['constant', 'linear', 'cosine', 'inverse-square-root', 'WSD'] = 'cosine'
    # The default is None, which will be set to `train_iters`.
    lr_decay_iters: Optional[int] = None
    lr_warmup_iters: int = 0
    lr_warmup_fraction: Optional[float] = None
    min_lr: float = 0
    # wsd
    lr_wsd_decay_style: Literal['exponential', 'linear', 'cosine', 'minus_sqrt'] = 'exponential'
    lr_wsd_decay_iters: Optional[int] = None

    # regularization
    weight_decay: float = 0.1
    weight_decay_incr_style: Literal['constant', 'linear', 'cosine'] = 'constant'
    start_weight_decay: Optional[float] = None
    end_weight_decay: Optional[float] = None
    clip_grad: float = 1.
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    sgd_momentum: float = 0.9

    # checkpoint
    output_dir: Optional[str] = None
    save_interval: int = 500
    no_save_optim: bool = False
    no_save_rng: bool = False
    mcore_model: Optional[str] = None
    mcore_adapter: Optional[str] = None
    no_load_optim: bool = False
    no_load_rng: bool = False
    finetune: bool = True
    perform_initialization: bool = False
    use_cpu_initialization: bool = False
    async_save: bool = False  # TODO
    use_persistent_ckpt_worker: bool = False
    dist_ckpt_save_pre_mcore_014: bool = False
    dist_ckpt_optim_fully_reshardable: bool = False
    distrib_optim_fully_reshardable_mem_efficient: bool = False

    # dist
    local_rank: Optional[int] = None  # Compatible with DeepSpeed launch
    ddp_timeout: int = 18000000
    ddp_backend: Literal['nccl', 'gloo'] = 'nccl'
    use_distributed_optimizer: bool = True
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    decoder_first_pipeline_num_layers: Optional[int] = None
    decoder_last_pipeline_num_layers: Optional[int] = None
    account_for_embedding_in_pipeline_split: bool = False
    account_for_loss_in_pipeline_split: bool = False
    overlap_p2p_comm: bool = True
    align_param_gather: bool = True

    sequence_parallel: bool = False
    context_parallel_size: int = 1
    tp_comm_overlap: bool = False  # TODO
    overlap_grad_reduce: bool = False  # TODO
    overlap_param_gather: bool = False  # TODO
    virtual_pipeline_model_parallel_size: Optional[int] = None
    microbatch_group_size_per_vp_stage: Optional[int] = None
    pipeline_model_parallel_layout: Optional[str] = None
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: int = 1

    # 'wandb', 'swanlab', 'tensorboard'
    report_to: List[str] = field(default_factory=lambda: ['tensorboard'])
    log_interval: int = 5
    tensorboard_dir: Optional[str] = None
    tensorboard_queue_size: int = 50
    wandb_project: str = 'megatron-swift'
    wandb_exp_name: Optional[str] = None
    swanlab_project: str = 'megatron-swift'
    swanlab_exp_name: Optional[str] = None

    # evaluate
    eval_iters: int = -1
    eval_interval: Optional[int] = None

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
    accumulate_allreduce_grads_in_fp32: bool = False

    # moe
    moe_router_load_balancing_type: Optional[List[str]] = None
    moe_router_dtype: Literal['none', 'fp32', 'fp64'] = 'fp32'
    moe_token_dispatcher_type: Literal['allgather', 'alltoall', 'flex'] = 'alltoall'
    moe_enable_deepep: bool = False
    moe_grouped_gemm: bool = True
    moe_permute_fusion: bool = False
    moe_aux_loss_coeff: float = 0.
    moe_z_loss_coeff: Optional[float] = None
    moe_shared_expert_overlap: bool = False
    moe_layer_recompute: bool = False  # compat mcore 0.12
    moe_expert_capacity_factor: Optional[float] = None
    moe_pad_expert_input_to_capacity: bool = False
    moe_token_drop_policy: Literal['probs', 'position'] = 'probs'

    # mtp
    mtp_num_layers: Optional[int] = None
    mtp_loss_scaling_factor: float = 0.1

    # mcore-bridge
    model: Optional[str] = None
    model_type: Optional[str] = None
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

    # visual
    vit_gradient_checkpointing: Optional[bool] = None
    vit_lr: Optional[float] = None
    aligner_lr: Optional[float] = None
    attn_impl: Optional[str] = None
    gradient_checkpointing_kwargs: Optional[Union[dict, str]] = None

    # other
    check_model: bool = True
    torch_dtype: Optional[Union[torch.dtype, str]] = None
    rope_scaling: Optional[Union[dict, str]] = None
    apply_wd_to_qk_layernorm: bool = False

    enable_dft_loss: bool = False
    enable_channel_loss: bool = False
    task_type: Literal['causal_lm', 'seq_cls', 'embedding', 'generative_reranker'] = None
    num_labels: Optional[int] = None
    problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] = None
    save_strategy: Literal['steps', 'epoch'] = 'steps'
    callbacks: List[str] = field(default_factory=list)

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
            keys += ['mcore_model', 'task_type', 'num_labels']
            for key in keys:
                old_value = old_args.get(key)
                if old_value is not None:
                    res[key] = old_value
            res.pop('mcore_adapter', None)
            if res['tuner_type'] != 'lora':
                res.pop('mcore_model', None)
        return res

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
        if self.task_type is None:
            self.task_type = 'causal_lm'
        if self.calculate_per_token_loss is None:
            self.calculate_per_token_loss = (self.task_type == 'causal_lm' and self.rlhf_type is None)

    def _init_mixed_precision(self):
        ModelArguments._init_mixed_precision(self)
        if self.apply_query_key_layer_scaling is None:
            self.apply_query_key_layer_scaling = self.fp16
        if self.apply_query_key_layer_scaling:
            os.environ['NVTE_APPLY_QK_LAYER_SCALING'] = '1'

    def __post_init__(self):
        require_version('numpy<2.0', 'Please install numpy<2.0 by running: `pip install "numpy<2.0"`.')
        if self.tuner_type == 'lora':
            require_version('peft>=0.15')
        RLHFMegatronArgumentsMixin.__post_init__(self)
        MegatronTunerMixin.__post_init__(self)
        os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')
        if self.recompute_granularity == 'none':
            self.recompute_granularity = None
        if self.recompute_granularity == 'selective' and self.recompute_method is not None:
            raise ValueError('recompute method is not yet supported for selective recomputing granularity')

        self._set_default()
        self._init_vpp_size()
        if self.vit_gradient_checkpointing is None:
            self.vit_gradient_checkpointing = not self.freeze_vit
        if isinstance(self.report_to, str):
            self.report_to = [self.report_to]
        self.model_info, self.model_meta = get_model_info_meta(
            self.model, model_type=self.model_type, use_hf=self.use_hf, hub_token=self.hub_token)

        # Megatron has a model_type parameter with the same name, so we need to avoid conflicts.
        self.model_type = self.model_info.model_type
        self.model_dir = self.model_info.model_dir
        self.is_multimodal = self.model_meta.is_multimodal
        self.megatron_model_meta = get_megatron_model_meta(self.model_type)
        if self.megatron_model_meta is None:
            raise ValueError(f'Model: {self.model} is not supported.')
        self._init_teacher_model()
        if self.apply_wd_to_qk_layernorm and self.model_type not in {'qwen3_next', 'qwen3_5'}:
            raise ValueError('apply_wd_to_qk_layernorm is only supported for qwen3_next and qwen3_5')
        if self.pipeline_model_parallel_size == 1 and (self.decoder_first_pipeline_num_layers is not None
                                                       or self.decoder_last_pipeline_num_layers is not None):
            raise ValueError('pipeline_model_parallel_size must be greater than 1 if you want to set '
                             'decoder_first_pipeline_num_layers or decoder_last_pipeline_num_layers.')
        self.fp8 = self.fp8_format  # compat megatron-lm
        if self.task_type not in {'causal_lm', 'generative_reranker'}:
            self.untie_embeddings_and_output_weights = True
        if self.gradient_checkpointing_kwargs is not None:
            self.gradient_checkpointing_kwargs = json_parse_to_dict(self.gradient_checkpointing_kwargs)
        if self.gradient_accumulation_fusion:
            try:
                import apex
            except ImportError:
                logger.warning('apex is not installed, so gradient accumulation fusion is disabled.')
                self.gradient_accumulation_fusion = False
        self.callbacks += ['print', 'default_flow']
        self.callbacks += self.report_to
        if isinstance(self.ref_adapters, str):
            self.ref_adapters = [self.ref_adapters]
        if self.eval_interval is None:
            self.eval_interval = self.save_interval
        if self.merge_lora is None:
            self.merge_lora = self.save_safetensors
        if self.adapters or self.ref_adapters or self.mcore_adapter or self.mcore_ref_adapter:
            if self.tuner_type == 'full':
                self.tuner_type = 'lora'
                logger.info('Setting args.tuner_type: lora')
        if self.adapters:
            self._load_adapter_config()
        self._init_mixed_precision()
        self._init_multimodal_full()
        self._map_dtype()
        self._init_weigh_decay()
        self.attention_backend = AttnBackend[self.attention_backend]
        if self.sequence_parallel and self.tensor_model_parallel_size <= 1:
            self.sequence_parallel = False
        if self.tp_comm_overlap and not self.sequence_parallel:
            raise ValueError('Tensor parallel communication/GEMM overlap can happen only when '
                             'sequence parallelism is enabled')

        initialize_megatron(self)
        total_model_size = (
            self.tensor_model_parallel_size * self.pipeline_model_parallel_size * self.context_parallel_size)
        # world_size is initialized in initialize_megatron
        self.data_parallel_size = self.world_size // total_model_size
        # Gradient Accumulation
        self.num_microbatches = self.global_batch_size // self.data_parallel_size // self.micro_batch_size

    def _init_teacher_model(self):
        if self.teacher_model is None:
            return
        self.teacher_model_info, self.teacher_model_meta = get_model_info_meta(
            self.teacher_model, model_type=self.teacher_model_type, use_hf=self.use_hf, hub_token=self.hub_token)
        self.teacher_model_type = self.teacher_model_info.model_type
        self.teacher_model_dir = self.teacher_model_info.model_dir
        self.teacher_megatron_model_meta = get_megatron_model_meta(self.teacher_model_type)
        if self.teacher_megatron_model_meta is None:
            raise ValueError(f'Model: {self.teacher_model} is not supported.')

    def _init_vpp_size(self):
        if self.pipeline_model_parallel_layout is not None:
            from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout
            # Parse the input flattened layout to a list and get the vpp size.
            # We will validate the layout more carefully in the TransformerConfig constructor.
            num_stages = PipelineParallelLayerLayout.get_num_stages_from_str(self.pipeline_model_parallel_layout)
            assert num_stages % self.pipeline_model_parallel_size == 0, (
                f'The length of pipeline_model_parallel_layout must be divisible'
                f' by pipeline_model_parallel_size ({num_stages=},'
                f' {self.pipeline_model_parallel_size=})')
            self.virtual_pipeline_model_parallel_size = num_stages // self.pipeline_model_parallel_size
        elif self.virtual_pipeline_model_parallel_size is not None:
            self.virtual_pipeline_model_parallel_size = self.virtual_pipeline_model_parallel_size
        if self.virtual_pipeline_model_parallel_size == 1:
            self.virtual_pipeline_model_parallel_size = None
        if self.virtual_pipeline_model_parallel_size is None:
            self.overlap_p2p_comm = False
            self.align_param_gather = False

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

    def init_iters(self, train_dataset, val_dataset):
        data_parallel_size = mpu.get_data_parallel_world_size()
        step_batch_size = self.micro_batch_size * data_parallel_size
        num_generations = self.num_generations if self.rlhf_type == 'grpo' else 1
        # TODO: Check if it causes duplicate saving at the end.
        if self.save_strategy == 'epoch':
            if hasattr(train_dataset, '__len__'):
                dataset_sample = len(train_dataset) // step_batch_size * step_batch_size * num_generations
                self.save_interval = dataset_sample // self.global_batch_size
                self.eval_interval = self.save_interval
            else:
                raise ValueError('streaming dataset is not supported with `--save_strategy epoch`.')
        if self.num_train_epochs is not None:
            if hasattr(train_dataset, '__len__'):
                dataset_sample = len(train_dataset) // step_batch_size * step_batch_size * num_generations
                self.train_iters = dataset_sample * self.num_train_epochs // self.global_batch_size
            elif self.train_iters is None:
                raise ValueError(
                    'You are using a streaming training dataset. Please explicitly specify `--train_iters`.')
        if self.eval_iters < 0:
            if val_dataset is None:
                self.eval_iters = 0
            elif hasattr(val_dataset, '__len__'):
                dataset_sample = len(val_dataset) // step_batch_size * step_batch_size
                dataset_sample = dataset_sample * num_generations
                self.eval_iters = max(dataset_sample // self.global_batch_size, 1)
            else:
                raise ValueError(
                    'You are using a streaming validation dataset. Please explicitly specify `--eval_iters`.')
            logger.info(f'Setting args.eval_iters: {self.eval_iters}')

        data_parallel_size = mpu.get_data_parallel_world_size()
        step_batch_size = self.micro_batch_size * data_parallel_size
        # To avoid errors caused by the validation set being insufficient to complete a single step.
        if val_dataset is not None and hasattr(val_dataset, '__len__') and len(val_dataset) < step_batch_size:
            val_dataset = None
        if val_dataset is None:
            self.eval_iters = 0

    def _init_multimodal_full(self):
        visual_cls = self.megatron_model_meta.visual_cls
        if self.tuner_type == 'full' and self.is_multimodal and visual_cls is not None:
            vision_tower = [f'visual.{vit}' for vit in getattr(visual_cls, '_vision_tower', [])]
            aligner = [f'visual.{aligner}' for aligner in getattr(visual_cls, '_aligner', [])]
            generator = [f'visual.{generator}' for generator in getattr(visual_cls, '_generator', [])]
            if self.freeze_llm:
                self.freeze_parameters.append('language_model')
            if self.freeze_vit:
                self.freeze_parameters += vision_tower
            if self.freeze_aligner:
                self.freeze_parameters += aligner
            else:
                self.trainable_parameters += aligner
            self.freeze_parameters += generator
            if self.freeze_parameters:
                logger.info(f'freeze_parameters: {self.freeze_parameters}')
            if self.trainable_parameters:
                logger.info(f'additional trainable_parameters: {self.trainable_parameters}')

    def _map_dtype(self):
        dtype_map = {
            'fp32': torch.float32,
            'bf16': torch.bfloat16,
            'fp16': torch.float16,
            'fp8': torch.uint8,
        }
        self.main_grads_dtype = dtype_map[self.main_grads_dtype]
        self.main_params_dtype = dtype_map[self.main_params_dtype]
        self.exp_avg_dtype = dtype_map[self.exp_avg_dtype]
        self.exp_avg_sq_dtype = dtype_map[self.exp_avg_sq_dtype]
        if self.fp16:
            self.torch_dtype = torch.float16
        elif self.bf16:
            self.torch_dtype = torch.bfloat16
            if self.main_grads_dtype == torch.float32:
                self.accumulate_allreduce_grads_in_fp32 = True
        self.params_dtype = self.torch_dtype

    def _init_weigh_decay(self):
        if self.weight_decay_incr_style == 'constant':
            assert self.start_weight_decay is None
            assert self.end_weight_decay is None
            self.start_weight_decay = self.end_weight_decay = self.weight_decay
        else:
            assert self.start_weight_decay is not None
            assert self.end_weight_decay is not None

# Cross-config validation: the one place where rules spanning several Configs are enforced.

from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        DistributedConfig,
        LoggingConfig,
        MegatronConfig,
        ModelConfig,
        MoEConfig,
        QuantizeConfig,
        RLHFConfig,
        TemplateConfig,
        TrainConfig,
        TunerConfig,
    )

logger = logging.getLogger(__name__)


def validate_configs(
    model_config: 'ModelConfig',
    template_config: 'TemplateConfig',
    dataset_config: 'DatasetConfig',
    train_config: 'TrainConfig',
    distributed_config: 'DistributedConfig',
    checkpoint_config: Optional['CheckpointConfig'] = None,
    tuner_config: Optional['TunerConfig'] = None,
    rlhf_config: Optional['RLHFConfig'] = None,
    logging_config: Optional['LoggingConfig'] = None,
    quantize_config: Optional['QuantizeConfig'] = None,
    megatron_config: Optional['MegatronConfig'] = None,
    moe_config: Optional['MoEConfig'] = None,
    *,
    training: bool = True,
) -> None:
    """Validate constraints that span multiple Configs. Raises ValueError on an illegal combination.

    Call this BEFORE building anything heavy (dataset/model)
    """
    from swift.dev.builders.model import is_megatron_backend
    is_megatron = is_megatron_backend(distributed_config)

    _check_group_by_length(dataset_config, template_config)
    _check_lazy_tokenize(dataset_config)
    _check_data_sharding(dataset_config)
    _check_streaming(dataset_config, checkpoint_config)
    _check_backend_specific(model_config, dataset_config, train_config, distributed_config, is_megatron, tuner_config)
    _check_megatron_runtime_configs(megatron_config, moe_config, is_megatron)
    _check_megatron_optimizer(train_config, is_megatron)
    _check_muon(train_config, distributed_config, is_megatron)
    _check_megatron_recompute(train_config, distributed_config, is_megatron)
    _check_megatron_microbatch_schedule(train_config, distributed_config, is_megatron)
    _check_eval_iters(train_config)
    _check_megatron_attn_backend(model_config, template_config, is_megatron)
    _check_mtp(model_config, is_megatron, tuner_config)
    _check_quantization(model_config, distributed_config, is_megatron)
    _check_load_quantization(
        model_config, distributed_config, tuner_config, quantize_config, is_megatron, training=training)
    _check_megatron_fsdp(distributed_config, is_megatron)
    _check_selective_recompute(distributed_config, is_megatron)
    _check_pipeline_decoder_layers(distributed_config, is_megatron)
    _check_tp_comm_overlap(distributed_config, is_megatron)
    _check_sequence_parallel_tp(distributed_config, is_megatron)
    _check_checkpoint_runtime(
        checkpoint_config,
        distributed_config,
        tuner_config,
        rlhf_config,
        is_megatron,
        training=training)
    _check_save_total_limit(checkpoint_config, is_megatron)
    _check_logging(logging_config)
    _check_rlhf_ref_model(model_config, tuner_config, rlhf_config)
    _check_rlhf_advanced(train_config, rlhf_config)
    _check_rlhf_padding_free(template_config, dataset_config, rlhf_config)
    _check_rlhf_sequence_parallel(template_config, rlhf_config)
    # Packing-derived padding_free has already been resolved by process_configs.
    _check_hf_sequence_parallel(model_config, template_config, dataset_config, distributed_config, is_megatron)


def _changed_fields(config) -> list:
    """Return user-visible fields that differ from defaults or were explicit on the CLI."""
    import dataclasses

    if config is None:
        return []
    explicit = getattr(config, '_explicit_fields', None)
    if explicit is not None:
        return sorted(explicit)
    changed = []
    for config_field in dataclasses.fields(config):
        if config_field.default is not dataclasses.MISSING:
            default = config_field.default
        elif config_field.default_factory is not dataclasses.MISSING:
            default = config_field.default_factory()
        else:
            continue
        if getattr(config, config_field.name) != default:
            changed.append(config_field.name)
    return changed


def _check_megatron_runtime_configs(megatron_config: Optional['MegatronConfig'], moe_config: Optional['MoEConfig'],
                                     is_megatron: bool) -> None:
    if not is_megatron:
        changed = _changed_fields(megatron_config) + _changed_fields(moe_config)
        if changed:
            raise ValueError(f'Megatron-only options {changed} cannot be used with the transformers backend.')
        return
    if megatron_config is None:
        return
    unsupported = {
        'apply_dsa_kernel_fusion',
        'csa_dense_mode',
        'sequence_packing_scheduler',
        'use_fused_mhc',
    }
    requested = sorted(unsupported.intersection(_changed_fields(megatron_config)))
    if requested:
        raise NotImplementedError(
            f'{requested} are not exposed by the installed mcore-bridge ModelConfig. '
            'Upgrade mcore-bridge/Megatron-LM or remove these options.')
    if megatron_config.manual_gc_steps < 0:
        raise ValueError('manual_gc_steps must be >= 0.')


def _check_rlhf_advanced(train_config: 'TrainConfig', rlhf_config: Optional['RLHFConfig']) -> None:
    """Validate online-RL features before models, teachers, or rollout workers are created."""
    if rlhf_config is None:
        return
    cfg = rlhf_config
    online_only = bool(cfg.chord_sft_dataset or cfg.advantage_reweight or cfg.sdar_loss_coef > 0 or cfg.dynamic_sample
                       or cfg.sync_ref_model or cfg.multi_turn_scheduler or cfg.use_gym_env)
    if online_only and cfg.rlhf_type != 'grpo':
        raise ValueError('CHORD, RLSD, SDAR, dynamic sampling, reference sync, and multi-turn/gym are GRPO-only.')
    _check_dynamic_sampling(cfg)
    _check_grpo_controls(cfg)
    _check_reference_sync(cfg, train_config)
    _check_chord(cfg)
    _check_self_distillation(cfg, train_config)
    _check_gkd(cfg)
    _check_auxiliary_adapters(cfg)
    _check_multi_turn(cfg)
    if cfg.teacher_model is not None and cfg.teacher_model_server is not None:
        raise ValueError('teacher_model and teacher_model_server are mutually exclusive.')


def _check_multi_turn(cfg: 'RLHFConfig') -> None:
    if cfg.max_turns is not None and cfg.max_turns < 1:
        raise ValueError('max_turns must be >= 1.')
    if cfg.completion_length_limit_scope not in ('total', 'per_round'):
        raise ValueError("completion_length_limit_scope must be 'total' or 'per_round'.")
    if cfg.gym_env is not None and cfg.use_gym_env is False:
        raise ValueError('gym_env cannot be set when use_gym_env=False.')
    if not cfg.multi_turn_scheduler:
        if cfg.use_gym_env:
            raise ValueError('use_gym_env requires a multi_turn_scheduler.')
        return

    _check_multi_turn_registry(cfg)
    if cfg.teacher_model_server:
        raise ValueError('teacher_model_server is not supported with multi-turn GRPO; use a local teacher model.')


def _check_multi_turn_registry(cfg: 'RLHFConfig') -> None:
    from swift.rollout.multi_turn import GYMScheduler, multi_turns
    if cfg.multi_turn_scheduler not in multi_turns:
        raise ValueError(
            f'Unknown multi_turn_scheduler {cfg.multi_turn_scheduler!r}; available: {sorted(multi_turns)}.')
    scheduler_cls = multi_turns[cfg.multi_turn_scheduler]
    if cfg.use_gym_env and not issubclass(scheduler_cls, GYMScheduler):
        raise ValueError('use_gym_env requires a GYMScheduler-compatible multi_turn_scheduler.')
    if issubclass(scheduler_cls, GYMScheduler) and not cfg.use_gym_env:
        raise ValueError('A gym multi_turn_scheduler requires use_gym_env=True.')
    if cfg.multi_turn_scheduler == 'gym_scheduler':
        from swift.rollout.gym_env import envs
        if cfg.gym_env not in envs:
            raise ValueError(f'Unknown gym_env {cfg.gym_env!r}; available: {sorted(envs)}.')


def _check_grpo_controls(cfg: 'RLHFConfig') -> None:
    loss_type = cfg.loss_type[0] if cfg.loss_type else 'grpo'
    grpo_only = bool(
        cfg.log_completions or cfg.num_iterations != 1 or cfg.delta is not None
        or cfg.importance_sampling_level != 'token' or cfg.overlong_filter or cfg.log_entropy
        or cfg.top_entropy_quantile != 1.0 or cfg.rollout_importance_sampling_mode
        or cfg.log_rollout_offpolicy_metrics or cfg.off_policy_sequence_mask_delta is not None
        or loss_type in ('dapo', 'fipo', 'gspo', 'sapo', 'cispo', 'bnpo', 'dr_grpo'))
    if cfg.rlhf_type != 'grpo':
        if grpo_only:
            raise ValueError('GRPO clipping, replay, entropy, FIPO, and completion logging controls are GRPO-only.')
        if cfg.teacher_model_server and cfg.rlhf_type != 'gkd':
            raise ValueError('teacher_model_server is supported only by GRPO and GKD.')
        return
    _check_grpo_loss_type(cfg, loss_type)
    positive = {
        'num_iterations': cfg.num_iterations,
        'rollout_importance_sampling_threshold': cfg.rollout_importance_sampling_threshold,
        'fipo_decay_rate': cfg.fipo_decay_rate,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f'GRPO controls must be > 0: {invalid}.')
    optional_nonnegative = {
        'delta': cfg.delta,
        'off_policy_sequence_mask_delta': cfg.off_policy_sequence_mask_delta,
        'fipo_clip_range': cfg.fipo_clip_range,
        'fipo_safety_threshold': cfg.fipo_safety_threshold,
    }
    invalid = [name for name, value in optional_nonnegative.items() if value is not None and value < 0]
    if invalid:
        raise ValueError(f'GRPO controls must be >= 0 when set: {invalid}.')
    if not 0.0 < cfg.top_entropy_quantile <= 1.0:
        raise ValueError('top_entropy_quantile must be in (0, 1].')


def _check_grpo_loss_type(cfg: 'RLHFConfig', loss_type: str) -> None:
    supported = {'grpo', 'dapo', 'fipo', 'gspo', 'sapo', 'cispo', 'bnpo', 'dr_grpo'}
    if len(cfg.loss_type or []) > 1:
        raise ValueError('GRPO supports exactly one loss_type.')
    if loss_type not in supported:
        raise ValueError(f'Unsupported GRPO loss_type={loss_type!r}; expected one of {sorted(supported)}.')


def _check_dynamic_sampling(cfg: 'RLHFConfig') -> None:
    if not cfg.dynamic_sample:
        return
    if not (cfg.reward_funcs or cfg.reward_model):
        raise ValueError(
            'dynamic_sample requires reward_funcs or reward_model because it filters groups by reward variance.')
    if cfg.num_generations < 2:
        raise ValueError('dynamic_sample requires num_generations >= 2 to measure reward variance.')
    if cfg.max_resample_times < 1:
        raise ValueError('max_resample_times must be >= 1 when dynamic_sample is enabled.')


def _check_reference_sync(cfg: 'RLHFConfig', train_config: 'TrainConfig') -> None:
    if not cfg.sync_ref_model:
        return
    if cfg.beta in (None, 0, 0.0):
        raise ValueError('sync_ref_model requires beta > 0 and an active reference model.')
    if cfg.ref_model_sync_steps < 1:
        raise ValueError('ref_model_sync_steps must be >= 1.')
    if not 0.0 <= cfg.ref_model_mixup_alpha <= 1.0:
        raise ValueError('ref_model_mixup_alpha must be in [0, 1].')
    del train_config


def _check_chord(cfg: 'RLHFConfig') -> None:
    if not cfg.chord_sft_dataset:
        return
    required = {
        'chord_sft_per_device_train_batch_size': cfg.chord_sft_per_device_train_batch_size,
        'chord_mu_warmup_steps': cfg.chord_mu_warmup_steps,
        'chord_mu_decay_steps': cfg.chord_mu_decay_steps,
        'chord_mu_peak': cfg.chord_mu_peak,
        'chord_mu_valley': cfg.chord_mu_valley,
    }
    missing = sorted(name for name, value in required.items() if value is None)
    if missing:
        raise ValueError(f'chord_sft_dataset requires explicit CHORD schedule fields: {missing}.')
    if cfg.chord_sft_per_device_train_batch_size < 1:
        raise ValueError('chord_sft_per_device_train_batch_size must be >= 1.')
    if cfg.chord_mu_warmup_steps < 0 or cfg.chord_mu_decay_steps < 0:
        raise ValueError('CHORD warmup and decay steps must be non-negative.')
    if not 0.0 <= cfg.chord_mu_valley <= cfg.chord_mu_peak <= 1.0:
        raise ValueError('CHORD requires 0 <= chord_mu_valley <= chord_mu_peak <= 1.')


def _check_self_distillation(cfg: 'RLHFConfig', train_config: 'TrainConfig') -> None:
    _check_rlsd(cfg)
    _check_sdar(cfg)
    if (cfg.advantage_reweight == 'rlsd' or cfg.sdar_loss_coef > 0) and train_config.use_liger_kernel:
        raise ValueError('RLSD and SDAR require the unfused per-token loss path; disable use_liger_kernel.')


def _check_rlsd(cfg: 'RLHFConfig') -> None:
    if cfg.advantage_reweight != 'rlsd':
        return
    if not 0.0 <= cfg.rlsd_lambda <= 1.0:
        raise ValueError('rlsd_lambda must be in [0, 1].')
    if cfg.rlsd_reweight_clip_range < 0:
        raise ValueError('rlsd_reweight_clip_range must be >= 0.')
    if cfg.rlsd_lambda_warmup_steps < 0 or cfg.rlsd_lambda_decay_steps < 0:
        raise ValueError('RLSD warmup and decay steps must be non-negative.')
    if not (cfg.reward_funcs or cfg.reward_model):
        raise ValueError('advantage_reweight=rlsd requires reward_funcs or reward_model.')
    if cfg.teacher_model_server:
        raise ValueError('RLSD requires a local or self-distillation teacher, not teacher_model_server.')


def _check_sdar(cfg: 'RLHFConfig') -> None:
    if cfg.sdar_loss_coef <= 0:
        return
    if cfg.sdar_gate_beta <= 0:
        raise ValueError('sdar_gate_beta must be > 0.')
    if cfg.advantage_reweight == 'rlsd':
        raise ValueError('SDAR and RLSD cannot be enabled together.')
    if cfg.teacher_model_server:
        raise ValueError('SDAR requires a local or self-distillation teacher, not teacher_model_server.')


def _check_gkd(cfg: 'RLHFConfig') -> None:
    if cfg.rlhf_type != 'gkd':
        return
    if not 0.0 <= cfg.lmbda <= 1.0:
        raise ValueError('GKD lmbda must be in [0, 1].')
    if cfg.sft_alpha < 0:
        raise ValueError('GKD sft_alpha must be >= 0.')
    if cfg.temperature <= 0:
        raise ValueError('GKD temperature must be > 0.')
    if cfg.gkd_logits_topk is not None and cfg.gkd_logits_topk < 1:
        raise ValueError('GKD gkd_logits_topk must be >= 1 when set.')
    if cfg.teacher_model_server and cfg.gkd_logits_topk is None:
        raise ValueError('GKD teacher_model_server requires gkd_logits_topk >= 1.')
    if cfg.offload_teacher_model and cfg.teacher_model is None:
        raise ValueError('offload_teacher_model requires a distinct local teacher_model.')
    if cfg.teacher_deepspeed and cfg.teacher_model is None:
        raise ValueError('teacher_deepspeed requires a distinct local teacher_model.')


def _check_auxiliary_adapters(cfg: 'RLHFConfig') -> None:
    if len(cfg.ref_adapters) > 1:
        raise ValueError('ref_adapters currently supports one frozen reference adapter.')
    if len(cfg.teacher_adapters) > 1:
        raise ValueError('teacher_adapters currently supports one frozen teacher adapter.')
    if cfg.teacher_adapters and cfg.teacher_model is None:
        raise ValueError('teacher_adapters requires a distinct local teacher_model.')
    if cfg.teacher_adapters and cfg.teacher_model_server:
        raise ValueError('teacher_adapters cannot be combined with teacher_model_server.')
    reward_models = cfg.reward_model or []
    reward_fields = {
        'reward_adapters': cfg.reward_adapters,
        'reward_model_type': cfg.reward_model_type,
        'reward_model_revision': cfg.reward_model_revision,
        'reward_model_plugin': cfg.reward_model_plugin,
        'reward_template': cfg.reward_template,
    }
    for field, values in reward_fields.items():
        if values and not reward_models:
            raise ValueError(f'{field} requires reward_model.')
        if values and len(values) != len(reward_models):
            raise ValueError(f'{field} must contain exactly one value per reward_model.')
    if cfg.rlhf_type != 'grpo' and (cfg.reward_model_plugin or cfg.reward_template):
        raise ValueError('reward_model_plugin and reward_template are supported by GRPO only.')


def _check_logging(logging_config: Optional['LoggingConfig']) -> None:
    if logging_config is None:
        return
    reporters = {name.lower() for name in logging_config.report_to}
    supported = {'none', 'tensorboard', 'wandb', 'swanlab'}
    unknown = reporters - supported
    if unknown:
        raise ValueError(f'Unsupported LoggingConfig.report_to values: {sorted(unknown)}.')
    if 'none' in reporters and len(reporters) > 1:
        raise ValueError('LoggingConfig.report_to cannot combine "none" with an active tracker.')
    if logging_config.logging_strategy == 'steps' and logging_config.logging_steps <= 0:
        raise ValueError('LoggingConfig.logging_steps must be > 0 when logging_strategy="steps".')
    if logging_config.swanlab_notification_method == 'email':
        required = (
            logging_config.swanlab_sender_email,
            logging_config.swanlab_receiver_email,
            logging_config.swanlab_smtp_server,
            logging_config.swanlab_smtp_port,
        )
        if not all(required):
            raise ValueError('SwanLab email notification requires sender_email, receiver_email, smtp_server, and '
                             'smtp_port.')


def _check_muon(train_config: 'TrainConfig', distributed_config: 'DistributedConfig', is_megatron: bool) -> None:
    """Reject muon pairings that Megatron itself refuses, or that would train something else.

    Mirrors legacy megatron_args.py::_check_muon with one deliberate difference: legacy silently sets
    ``use_distributed_optimizer = False`` when muon is selected, and this raises instead. The knob is
    one the user typed, and turning off the distributed optimizer changes both the memory profile and
    what a checkpoint contains -- the same class of hidden downgrade as the padding_free case above.

    The mcore version gate is legacy's too. Checking it here means a CLI mistake fails on the driver;
    it is safe to read on this side because it is a package version rather than a device property, so
    unlike the FP8/Blackwell checks it does not describe hardware this process may not have.
    """
    if 'muon' not in train_config.optimizer:
        return

    if not is_megatron:
        raise ValueError(f'TrainConfig.optimizer={train_config.optimizer!r} is a Megatron optimizer, but the active '
                         'backend is transformers. Use TrainConfig.optim for the transformers path, or switch '
                         'DistributedConfig.backend.')

    from swift.dev.naming import mcore_version_at_least
    if not mcore_version_at_least('0.16'):
        raise ValueError(f'TrainConfig.optimizer={train_config.optimizer!r} requires megatron-core>=0.16, which is '
                         'where the muon implementation lands.')

    if train_config.optimizer == 'muon':
        # Plain muon orthogonalises whole parameters, so it needs each one gathered before the step;
        # both overlaps hand it a shard instead. megatron asserts the same pairing.
        for attr in ('overlap_grad_reduce', 'overlap_param_gather'):
            if getattr(distributed_config, attr):
                raise ValueError(
                    f"optimizer='muon' is incompatible with DistributedConfig.{attr}=True: muon computes its "
                    'update from the whole parameter, which an overlapped reduce/gather has not finished '
                    f"assembling. Use optimizer='dist_muon', which is sharded, or set {attr}=False.")

    if distributed_config.use_distributed_optimizer:
        raise ValueError(f'TrainConfig.optimizer={train_config.optimizer!r} does not support '
                         'DistributedConfig.use_distributed_optimizer=True; muon maintains its own state layout. '
                         'legacy turned the distributed optimizer off silently here -- set it to False explicitly, '
                         'so the memory profile and checkpoint contents of the run are not a surprise.')


def _check_megatron_fsdp(distributed_config: 'DistributedConfig', is_megatron: bool) -> None:
    """Reject Megatron-FSDP pairings that megatron itself rejects, or that silently do nothing.

    Duplicated on purpose with MegatronStrategy._check_fsdp, which is the authority: that one runs in
    the process that builds the model, so it also covers cookbook users who never touch dev's config
    layer. Checking here as well means a CLI typo fails on the driver, before ranks are spawned.
    """
    if not distributed_config.use_megatron_fsdp:
        return

    if not is_megatron:
        # The transformers backend has its own FSDP, reached through DistributedConfig.fsdp. Silently
        # ignoring this flag there would leave a run that says "sharded" and replicates.
        raise ValueError('DistributedConfig.use_megatron_fsdp only applies to the megatron backend, but the active '
                         'backend is transformers. Use DistributedConfig.fsdp for the transformers path.')

    if not distributed_config.use_distributed_optimizer:
        raise ValueError('DistributedConfig.use_megatron_fsdp requires use_distributed_optimizer=True: FSDP shards '
                         'the parameters, and only the distributed optimizer keeps the matching master-weight '
                         'shards to update them from.')

    if distributed_config.context_parallel_size > 1:
        # megatron asserts the same pairing on its own CLI ('Hybrid context parallelism not supported
        # with Megatron FSDP').
        raise ValueError('DistributedConfig.use_megatron_fsdp is incompatible with context_parallel_size='
                         f'{distributed_config.context_parallel_size}. Megatron-FSDP does not support context '
                         'parallelism; use the default DDP wrapper for a CP run.')


#: (format field, param-gather field) for each low-precision format dev exposes. The amax knobs are
#: deliberately absent: their defaults are non-None, so "did the user set this?" is unanswerable and
#: a dependency check on them would fire on every run.
_QUANT_FORMATS = (('fp4_format', 'fp4_param_gather'), ('fp8_format', 'fp8_param_gather'))


def _check_quantization(model_config: 'ModelConfig', distributed_config: 'DistributedConfig',
                        is_megatron: bool) -> None:
    """Reject FP4/FP8 settings that cannot do what they say.

    Errors rather than warnings because every case below starts, reports a normal-looking loss, and
    trains nothing or trains something other than what was asked for.

    The environment preconditions (Blackwell for NVFP4, a TE new enough for the chosen recipe) are
    deliberately NOT checked here: this runs on the driver, which in Ray mode is not the process --
    nor necessarily the node -- that builds the model, so a check here would test the wrong GPU.
    mcore-bridge's ModelConfig checks them where the model is actually built.
    """
    active = [fmt for fmt, _ in _QUANT_FORMATS if getattr(model_config, fmt) is not None]
    explicit = getattr(model_config, '_explicit_fields', set())
    format_dependents = {
        'fp4_format': ('fp4_recipe', 'fp4_param_gather'),
        'fp8_format': ('fp8_recipe', 'fp8_amax_history_len', 'fp8_amax_compute_algo', 'fp8_param_gather'),
    }

    for fmt, param_gather in _QUANT_FORMATS:
        if getattr(model_config, fmt) is None:
            requested = [name for name in format_dependents[fmt] if name in explicit]
            if requested:
                raise ValueError(f'ModelConfig fields {requested} need ModelConfig.{fmt} to be set. Without it the '
                                 'model is built in its normal dtype, so these knobs would be ignored.')
            if getattr(model_config, param_gather):
                raise ValueError(f'ModelConfig.{param_gather} needs ModelConfig.{fmt} to be set. Without it the '
                                 'model is built in its normal dtype, so this knob would be ignored.')
            continue

        if not is_megatron:
            raise ValueError(f'ModelConfig.{fmt} is only implemented by the megatron backend, but the active '
                             'backend is transformers. Low-precision training here is a Megatron/Transformer-'
                             'Engine feature; the HF path has no equivalent.')

        if getattr(model_config, param_gather) and not distributed_config.use_distributed_optimizer:
            # DistributedOptimizer._copy_main_params_to_model_params is the only code that
            # re-quantizes the FP32 master shards back into the quantized parameters. Under any other
            # optimizer they keep their initial values for the whole run while the loss is computed
            # from them, so it neither errors nor learns. megatron asserts the same thing on its own
            # CLI ('--fp8-param-gather only supported with distributed optimizer, ...').
            raise ValueError(
                f'ModelConfig.{param_gather} requires DistributedConfig.use_distributed_optimizer=True: '
                'quantized parameters are updated by re-quantizing the distributed optimizer\'s master shards, '
                'and no other optimizer implements that step, so the model would never change.')

    if len(active) > 1:
        # megatron enters exactly one quantization context per transformer layer and its own
        # TransformerConfig raises on this; caught here so it fails on the driver, before a model is
        # built on every rank.
        raise ValueError(f'{" and ".join(f"ModelConfig.{fmt}" for fmt in active)} are mutually exclusive: megatron '
                         'applies a single quantization recipe per transformer layer. Pick one.')


def _check_load_quantization(model_config: 'ModelConfig', distributed_config: 'DistributedConfig',
                             tuner_config: Optional['TunerConfig'], quantize_config: Optional['QuantizeConfig'],
                             is_megatron: bool, *, training: bool) -> None:
    """Validate training-time model loading quantization before a worker loads weights."""
    if quantize_config is None or quantize_config.quant_method is None:
        return

    from swift.dev.builders.quantization import CALIBRATION_QUANT_METHODS, LOAD_TIME_QUANT_METHODS

    method = quantize_config.quant_method
    if method in CALIBRATION_QUANT_METHODS:
        if training:
            raise ValueError(
                f'quant_method={method!r} calibrates and exports weights; it is not a training load-time method. '
                'Train from an already quantized checkpoint, or use bnb/hqq/eetq/quanto/fp8 for loading.')
        return
    if method not in LOAD_TIME_QUANT_METHODS:
        raise ValueError(f'Unknown training load-time quant_method={method!r}.')
    if is_megatron:
        raise ValueError(
            f'quant_method={method!r} is a transformers load-time quantizer and is not supported by the Megatron '
            'backend. Use ModelConfig.fp4_format/fp8_format for Transformer-Engine training quantization, or '
            'convert a pre-quantized checkpoint to mcore first.')

    tuner_type = getattr(tuner_config, 'tuner_type', 'full') if tuner_config is not None else 'full'
    if training and tuner_type == 'full':
        raise ValueError(
            f'quant_method={method!r} cannot be combined with full-parameter training: load-time quantized base '
            'weights are not trainable parameters. Select a trainable adapter such as --tuner_type lora.')

    bits = quantize_config.quant_bits
    valid_bits = {
        'bnb': {4, 8},
        'hqq': {1, 2, 3, 4, 8},
        'eetq': {8},
        'quanto': {2, 4, 8, 'float8'},
        'fp8': {None, 8, 'float8'},
    }[method]
    if bits not in valid_bits:
        raise ValueError(f'quant_method={method!r} does not support quant_bits={bits!r}; expected {sorted(valid_bits, key=str)}.')

    if getattr(tuner_config, 'tuner_backend', None) == 'unsloth' and method != 'bnb':
        raise ValueError(
            f'Unsloth only exposes load_in_4bit/load_in_8bit for BNB, so quant_method={method!r} is unsupported. '
            'Use --quant_method bnb or the default tuner backend.')

    if distributed_config.fsdp and method == 'bnb' and bits == 4:
        storage = quantize_config.bnb_4bit_quant_storage
        if storage is None:
            raise ValueError(
                'FSDP QLoRA requires --bnb_4bit_quant_storage to match the model parameter dtype '
                f'(--torch_dtype {model_config.torch_dtype!r}); the bitsandbytes uint8 default cannot be sharded.')
        if model_config.torch_dtype is not None and storage != model_config.torch_dtype:
            raise ValueError(
                f'FSDP QLoRA requires bnb_4bit_quant_storage ({storage!r}) to match torch_dtype '
                f'({model_config.torch_dtype!r}) so flattened FSDP parameters have one dtype.')


def _check_mtp(model_config: 'ModelConfig', is_megatron: bool, tuner_config: Optional['TunerConfig']) -> None:
    """Reject MTP settings that cannot do what they say.

    Every failure here is one that would otherwise be silent -- an MTP run that trains nothing, or
    exports no MTP layer -- which is why these are errors rather than warnings. The only exception is
    LoRA, which *can* work if the adapter covers the MTP modules, so it warns instead.
    """
    mtp_dependents = (
        'mtp_loss_scaling_factor', 'enable_mtp_training', 'mtp_freeze', 'mtp_decoder_input_detach',
        'mtp_shared_weights')

    if model_config.mtp_num_layers is None:
        for attr in mtp_dependents:
            value = getattr(model_config, attr)
            if value:
                raise ValueError(f'ModelConfig.{attr}={value!r} needs ModelConfig.mtp_num_layers to be set. '
                                 'Without it no MTP block is built, so this knob would be ignored.')
        return

    if not is_megatron:
        raise ValueError('ModelConfig.mtp_num_layers is only implemented by the megatron backend, but the active '
                         'backend is transformers. MTP lives in mcore-bridge; the HF path has no equivalent.')

    if model_config.mtp_num_layers < 1:
        raise ValueError(f'ModelConfig.mtp_num_layers={model_config.mtp_num_layers} must be >= 1, or None to '
                         'disable MTP entirely.')

    if model_config.mtp_freeze and model_config.enable_mtp_training:
        raise ValueError('ModelConfig.mtp_freeze and ModelConfig.enable_mtp_training are contradictory: the first '
                         'drops the MTP gradient, the second asks for it. Set exactly one.')

    if model_config.mtp_loss_scaling_factor is not None and not model_config.enable_mtp_training:
        raise ValueError('ModelConfig.mtp_loss_scaling_factor only has an effect with '
                         'ModelConfig.enable_mtp_training=True; on its own the MTP loss is never computed, so the '
                         'factor scales nothing.')

    if model_config.mtp_decoder_input_detach and not model_config.enable_mtp_training:
        raise ValueError('ModelConfig.mtp_decoder_input_detach describes where the MTP gradient stops, so it needs '
                         'ModelConfig.enable_mtp_training=True to mean anything.')

    if (model_config.enable_mtp_training and tuner_config is not None
            and getattr(tuner_config, 'tuner_type', 'full') != 'full'):
        # Not an error: this is trainable if the adapter targets the MTP modules, which we cannot
        # decide from target_modules alone (it may be 'all-linear', or name them explicitly).
        # twinkle re-checks against the built model and warns if nothing ended up trainable.
        logger.warning('enable_mtp_training with tuner_type=%r: the MTP layers are base parameters, so they only '
                       'train if the adapter covers them. Otherwise the MTP loss is computed and discarded.',
                       tuner_config.tuner_type)


def _check_megatron_attn_backend(model_config: 'ModelConfig', template_config: 'TemplateConfig',
                                 is_megatron: bool) -> None:
    """padding_free needs an attention kernel that supports variable-length (THD) input.

    legacy handles this by DOWNGRADING: on an 'unfused' backend it logs a warning and sets
    args.padding_free = False (swift/megatron/model/utils.py::_check_padding_free). dev raises
    instead, deliberately not mirroring that:
      - silently rewriting the user's request is the exact failure mode this refactor has been
        removing (the warmup rounding and the min_lr override both silently ran something else);
      - 'unfused' is never a default -- reaching it means the user typed both it and padding_free,
        i.e. asked for two things that cannot hold together, which is worth reporting;
      - the downgrade also silently changes throughput/memory, so a run could look fine and be much
        slower than the config implies.
    Recorded as a break-change rather than hidden.

    legacy's other attention guard (flash + softmax_type='learnable' -> raise) is NOT mirrored: dev
    has no softmax_type or experimental_attention_variant field, so mcore keeps its default
    ('vanilla') and the condition cannot be reached. Mirroring it would add a permanently-false
    branch. If either field is ever added to dev, that guard has to come with it.
    """
    if not is_megatron or not template_config.padding_free:
        return
    if model_config.attn_impl is None:
        return
    # Resolve first: 'sdpa' also lands on the unfused kernel, so comparing the raw string would miss
    # it. Unknown/unsupported values are not this guard's business -- resolve_megatron_attn_backend
    # reports those at build time with a better message.
    from megatron.core.transformer.enums import AttnBackend

    from swift.dev.naming import resolve_megatron_attn_backend
    try:
        backend = resolve_megatron_attn_backend(model_config.attn_impl)
    except NotImplementedError:
        return
    if backend is AttnBackend.unfused:
        raise ValueError(f'padding_free=True is incompatible with attn_impl={model_config.attn_impl!r} (the '
                         'unfused attention kernel): it does not support the variable-length (THD) layout '
                         'padding_free produces. legacy silently turns padding_free off here; dev refuses instead '
                         'so the run does not quietly train a different shape. Choose attn_impl="flash"/"fused", '
                         'or set padding_free=False.')


def _check_group_by_length(dataset_config: 'DatasetConfig', template_config: 'TemplateConfig') -> None:
    if not dataset_config.group_by_length:
        return
    if template_config.padding_free:
        raise ValueError('group_by_length is incompatible with padding_free: padding_free flattens each micro '
                         'batch into a single variable-length sequence, so there is no padding left for length '
                         'grouping to remove (and clustering long samples raises peak activation memory). '
                         'Set one of them to False.')
    if dataset_config.packing:
        raise ValueError('group_by_length is incompatible with packing: packing already bin-packs samples to '
                         '~packing_length and implies padding_free, so length grouping cannot help. '
                         'Use packing alone, or set group_by_length=False.')
    # Streaming has no random access and no precomputed `lengths` column, so the sampler cannot
    # group. Previously build_dataset passed group_by_length through WITHOUT lengths, so this died
    # later as an opaque 'lengths must be provided'; failing here reports the actual cause.
    if dataset_config.streaming:
        raise ValueError('group_by_length requires a map-style dataset: streaming datasets have no `lengths` '
                         'column and no random access, so samples cannot be reordered by length. '
                         'Set streaming=False or group_by_length=False.')
    # `lengths` only exists after an EAGER encode. lazy_tokenize=None means AUTO, and auto always
    # backs off to eager when group_by_length is on (see _encode_mode), so only an EXPLICIT opt-in
    # to lazy is a conflict here.
    if dataset_config.lazy_tokenize:
        raise ValueError('group_by_length requires lazy_tokenize=False: the per-sample `lengths` column it '
                         'sorts on is only produced by eager encoding (AddLengthPreprocessor).')


def _check_lazy_tokenize(dataset_config: 'DatasetConfig') -> None:
    """Explicit lazy_tokenize=True conflicts with packing / streaming (legacy base_args.py:136-140).

    Only the EXPLICIT opt-in is a conflict: None is auto, and auto backs off to eager whenever one
    of these is on, so it can never reach here in a violating state.
    """
    if not dataset_config.lazy_tokenize:
        return
    if dataset_config.packing:
        raise ValueError('packing and lazy_tokenize are incompatible: PackingDataset reads the '
                         '`lengths` column at construction (packing.py:78), which only eager '
                         'encoding writes.')
    if dataset_config.streaming:
        raise ValueError('streaming and lazy_tokenize are incompatible.')


def _check_data_sharding(dataset_config: 'DatasetConfig') -> None:
    """data_sharding needs a shuffled order to reshuffle; it is a no-op under sequential reads.

    The group_by_length conflict is intentionally NOT fatal here: legacy downgrades data_sharding to
    False with a warning (batch_sampler.py:86-90), and existing Megatron scripts that set both must
    keep running unchanged. build_dataset performs that downgrade.
    """
    if dataset_config.data_sharding and not dataset_config.train_dataloader_shuffle:
        raise ValueError('data_sharding requires train_dataloader_shuffle=True: it only changes the SCOPE of the '
                         'per-epoch reshuffle (shuffle within a rank shard vs. globally), so with shuffling off '
                         'it does nothing.')


def _check_streaming(dataset_config: 'DatasetConfig', checkpoint_config: Optional['CheckpointConfig']) -> None:
    """Streaming/iterable datasets cannot be resumed deterministically (no epoch-aware skip)."""
    # A cached_dataset is a map-style Dataset written by save_to_disk, so it cannot participate in
    # the streaming pipeline. Legacy asserts the same in SwiftSft._prepare_dataset
    # ('Cached dataset does not support streaming.').
    if dataset_config.streaming and (dataset_config.cached_dataset or dataset_config.cached_val_dataset):
        raise ValueError('cached_dataset does not support streaming=True: the exported cache is a map-style '
                         'dataset loaded via load_from_disk. Set streaming=False, or drop cached_dataset.')
    if checkpoint_config is None:
        return
    if dataset_config.streaming and checkpoint_config.resume_from_checkpoint:
        raise NotImplementedError('Resume is not supported for streaming/iterable datasets (no deterministic '
                                  'epoch-aware skip). Use a map-style dataset, or set resume_from_checkpoint=None.')


def _check_megatron_recompute(train_config: 'TrainConfig', distributed_config: 'DistributedConfig',
                              is_megatron: bool) -> None:
    """gradient_checkpointing decides nothing on Megatron; recompute_granularity does.

    build_model forwards only recompute_granularity/method/num_layers to MegatronModel, so the
    HF-named flag is unread there. It cannot go in the _HF_ONLY table: its default is True, so every
    existing Megatron run would start failing. The two directions differ in what they deserve --
    flag-on-but-nothing-configured is the DEFAULT state and can only warn, while flag-off-yet-
    recompute-configured is a contradiction the user typed and is fatal.

    The default also disagrees with legacy, which recomputes ('selective') unless told otherwise;
    aligning that would change dev's memory/throughput baseline, so it is recorded in the design doc
    rather than changed here.
    """
    if not is_megatron:
        return
    granularity = distributed_config.recompute_granularity
    if not train_config.gradient_checkpointing and granularity:
        raise ValueError(f'gradient_checkpointing=False contradicts recompute_granularity={granularity!r}: on the '
                         'Megatron backend recompute is driven by recompute_granularity alone, so the run WOULD '
                         'recompute. Drop one of the two.')
    if train_config.gradient_checkpointing and not granularity:
        logger.warning('gradient_checkpointing=True has no effect on the Megatron backend and this run will NOT '
                       'recompute: set DistributedConfig.recompute_granularity (legacy megatron defaults to '
                       "'selective') to enable it.")


def _check_eval_iters(train_config: 'TrainConfig') -> None:
    if train_config.eval_iters == -1 or train_config.eval_iters > 0:
        return
    raise ValueError('TrainConfig.eval_iters must be -1 (evaluate the full dataset) or a positive batch count.')


def _check_megatron_microbatch_schedule(train_config: 'TrainConfig', distributed_config: 'DistributedConfig',
                                         is_megatron: bool) -> None:
    """Validate the VPP micro-batch group against the loop's actual micro-batch count."""
    group = train_config.microbatch_group_size_per_vp_stage
    if group is None:
        return
    if not is_megatron:
        raise ValueError('microbatch_group_size_per_vp_stage is only implemented by the megatron backend.')

    vpp = distributed_config.virtual_pipeline_model_parallel_size
    if vpp is None:
        raise ValueError('microbatch_group_size_per_vp_stage requires virtual_pipeline_model_parallel_size or '
                         'pipeline_model_parallel_layout; a non-interleaved pipeline does not consume this option.')
    if group <= 0:
        raise ValueError('microbatch_group_size_per_vp_stage must be > 0.')

    pp = distributed_config.pipeline_model_parallel_size
    num_microbatches = train_config.gradient_accumulation_steps
    if group < pp or group > num_microbatches:
        raise ValueError(f'microbatch_group_size_per_vp_stage={group} must be in '
                         f'[pipeline_model_parallel_size={pp}, gradient_accumulation_steps={num_microbatches}].')
    remainder = num_microbatches % group
    if 0 < remainder < pp:
        raise ValueError(f'gradient_accumulation_steps % microbatch_group_size_per_vp_stage is {remainder}, which '
                         f'must be 0 or at least pipeline_model_parallel_size={pp}.')


def _check_megatron_optimizer(train_config: 'TrainConfig', is_megatron: bool) -> None:
    """Reject an inconsistent Megatron optimizer/scheduler config before the weights are loaded."""
    if not is_megatron:
        return
    from swift.dev.optimizer import megatron_weight_decay_bounds, warmup_budget

    # Delegates so the weight-decay rule has one definition (configure_optimizer uses the same).
    start_wd, end_wd = megatron_weight_decay_bounds(train_config)
    if start_wd < 0 or end_wd < start_wd:
        raise ValueError(f'Megatron weight decay requires 0 <= start_weight_decay <= end_weight_decay; got '
                         f'{start_wd} -> {end_wd}.')
    if train_config.learning_rate <= 0:
        raise ValueError('Megatron learning_rate must be > 0.')
    if not 0 <= train_config.min_lr <= train_config.learning_rate:
        raise ValueError('Megatron min_lr must satisfy 0 <= min_lr <= learning_rate.')
    if not 0 <= train_config.lr_warmup_init <= train_config.learning_rate:
        raise ValueError('Megatron lr_warmup_init must satisfy 0 <= lr_warmup_init <= learning_rate.')

    # 'cosine_with_min_lr' is Megatron's plain cosine plus a floor, so without min_lr it would run as
    # ordinary cosine -- the name silently not doing what it says. (On the HF path transformers
    # itself raises when neither min_lr nor min_lr_rate is given.)
    if train_config.lr_scheduler_type.lower() == 'cosine_with_min_lr' and not train_config.min_lr:
        raise ValueError("lr_scheduler_type='cosine_with_min_lr' needs TrainConfig.min_lr > 0 on the Megatron "
                         'backend; with min_lr=0 it is just cosine. Set min_lr, or use '
                         "lr_scheduler_type='cosine'.")

    decay_steps = train_config.lr_decay_iters
    if decay_steps is not None and decay_steps <= 0:
        raise ValueError('Megatron lr_decay_iters must be > 0 when set.')
    if decay_steps is None and train_config.max_steps > 0:
        decay_steps = train_config.max_steps
    if decay_steps is not None:
        warmup_steps = warmup_budget(train_config, decay_steps, is_megatron=True)
        if warmup_steps >= decay_steps:
            raise ValueError(
                f'Megatron warmup ({warmup_steps} steps) must be shorter than lr decay ({decay_steps} steps).')

    style = train_config.lr_decay_style
    explicit = getattr(train_config, '_explicit_fields', set())
    wsd_style_explicit = 'lr_wsd_decay_style' in explicit
    if style == 'WSD':
        if train_config.lr_wsd_decay_iters is None or train_config.lr_wsd_decay_iters <= 0:
            raise ValueError("lr_decay_style='WSD' requires lr_wsd_decay_iters > 0.")
        if decay_steps is not None and train_config.lr_wsd_decay_iters > decay_steps:
            raise ValueError('lr_wsd_decay_iters cannot exceed the effective lr decay horizon.')
    elif train_config.lr_wsd_decay_iters is not None or wsd_style_explicit:
        raise ValueError('lr_wsd_decay_iters/lr_wsd_decay_style only apply when lr_decay_style="WSD".')

    _check_optimizer_specific_fields(train_config)


def _check_optimizer_specific_fields(train_config: 'TrainConfig') -> None:
    """Reject optimizer options that the selected Megatron optimizer would ignore."""
    import dataclasses

    explicit = getattr(train_config, '_explicit_fields', set())
    defaults = {field.name: field.default for field in dataclasses.fields(train_config)}

    def changed(name: str) -> bool:
        return name in explicit or getattr(train_config, name) != defaults[name]

    muon_fields = (
        'muon_momentum', 'muon_split_qkv', 'muon_use_nesterov', 'muon_scale_mode',
        'muon_fp32_matmul_prec', 'muon_coefficient_type', 'muon_num_ns_steps', 'muon_tp_mode',
        'muon_extra_scale_factor', 'muon_scalar_optimizer')
    if 'muon' not in train_config.optimizer:
        invalid = [name for name in muon_fields if changed(name)]
        if invalid:
            raise ValueError(f'Muon optimizer fields require optimizer="muon" or "dist_muon": {invalid}.')
    if train_config.optimizer != 'sgd' and changed('sgd_momentum'):
        raise ValueError('sgd_momentum only applies when optimizer="sgd".')
    if train_config.optimizer != 'adam':
        invalid = [name for name in ('adam_beta1', 'adam_beta2', 'adam_epsilon') if changed(name)]
        if invalid:
            raise ValueError(f'Adam optimizer fields only apply when optimizer="adam": {invalid}.')

    precision_fields = ('main_params_dtype', 'main_grads_dtype', 'exp_avg_dtype', 'exp_avg_sq_dtype')
    if not train_config.use_precision_aware_optimizer:
        invalid = [name for name in precision_fields if changed(name)]
        if invalid:
            raise ValueError(
                f'Precision-aware optimizer dtypes require use_precision_aware_optimizer=True: {invalid}.')
    if not 0 <= train_config.optimizer_offload_fraction <= 1:
        raise ValueError('optimizer_offload_fraction must be in [0, 1].')
    if not train_config.optimizer_cpu_offload and changed('optimizer_offload_fraction'):
        raise ValueError('optimizer_offload_fraction only applies when optimizer_cpu_offload=True.')


def _is_off(value, off_value) -> bool:
    if off_value is None:
        # A falsy NUMBER is a real setting, a falsy container is not. start_weight_decay=0.0 (ramp
        # up from no decay) has to count as set or the wrong-backend check skips it, while the CLI
        # normalizes an unset fsdp to [] against a None default and must still count as unset.
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return False
        return not value
    if not off_value:
        return not value
    return value == off_value


_MEGATRON_ONLY = (
    ('model_config', 'vit_attn_impl', None),
    ('model_config', 'language_model_only', False),
    ('dataset_config', 'data_sharding', False),
    # NOTE: clip_grad is NOT here -- CLI normalization folds it into max_grad_norm for both backends.
    ('train_config', 'weight_decay_incr_style', 'constant'),
    ('train_config', 'start_weight_decay', None),
    ('train_config', 'end_weight_decay', None),
    ('train_config', 'min_lr', 0.0),
    ('train_config', 'optimizer', 'adam'),
    ('train_config', 'sgd_momentum', 0.9),
    ('train_config', 'muon_momentum', 0.9),
    ('train_config', 'muon_split_qkv', True),
    ('train_config', 'muon_use_nesterov', False),
    ('train_config', 'muon_scale_mode', 'spectral'),
    ('train_config', 'muon_fp32_matmul_prec', 'medium'),
    ('train_config', 'muon_coefficient_type', 'quintic'),
    ('train_config', 'muon_num_ns_steps', 5),
    ('train_config', 'muon_tp_mode', 'blockwise'),
    ('train_config', 'muon_extra_scale_factor', 1.0),
    ('train_config', 'muon_scalar_optimizer', 'adam'),
    ('train_config', 'use_precision_aware_optimizer', False),
    ('train_config', 'main_params_dtype', 'fp32'),
    ('train_config', 'main_grads_dtype', 'fp32'),
    ('train_config', 'exp_avg_dtype', 'fp32'),
    ('train_config', 'exp_avg_sq_dtype', 'fp32'),
    ('train_config', 'optimizer_cpu_offload', False),
    ('train_config', 'optimizer_offload_fraction', 1.0),
    ('train_config', 'optimizer_cuda_graph', False),
    ('train_config', 'accumulate_allreduce_grads_in_fp32', False),
    ('train_config', 'apply_wd_to_qk_layernorm', False),
    ('train_config', 'global_batch_size', None),
    ('train_config', 'microbatch_group_size_per_vp_stage', None),
    ('train_config', 'calculate_per_token_loss', None),
    ('train_config', 'finetune', True),
    ('train_config', 'lr_decay_style', 'cosine'),
    ('train_config', 'lr_decay_iters', None),
    ('train_config', 'lr_warmup_init', 0.0),
    ('train_config', 'lr_wsd_decay_iters', None),
    ('train_config', 'lr_wsd_decay_style', 'exponential'),
    ('distributed_config', 'bridge_backend', 'mcore-bridge'),
    ('distributed_config', 'tensor_model_parallel_size', 1),
    ('distributed_config', 'pipeline_model_parallel_size', 1),
    ('distributed_config', 'context_parallel_size', 1),
    ('distributed_config', 'expert_model_parallel_size', 1),
    ('distributed_config', 'expert_tensor_parallel_size', 1),
    ('distributed_config', 'sequence_parallel', False),
    ('distributed_config', 'use_distributed_optimizer', True),
    ('distributed_config', 'use_megatron_fsdp', False),
    ('distributed_config', 'recompute_granularity', None),
    ('distributed_config', 'recompute_method', None),
    ('distributed_config', 'recompute_num_layers', None),
    ('distributed_config', 'recompute_modules', ['core_attn']),
    ('distributed_config', 'cp_comm_type', None),
    ('distributed_config', 'cp_partition_mode', 'zigzag'),
    ('distributed_config', 'data_parallel_sharding_strategy', 'optim_grads_params'),
    ('distributed_config', 'virtual_pipeline_model_parallel_size', None),
    ('distributed_config', 'pipeline_model_parallel_layout', None),
    ('distributed_config', 'decoder_first_pipeline_num_layers', None),
    ('distributed_config', 'decoder_last_pipeline_num_layers', None),
    ('distributed_config', 'account_for_embedding_in_pipeline_split', False),
    ('distributed_config', 'account_for_loss_in_pipeline_split', False),
    ('distributed_config', 'overlap_grad_reduce', False),
    ('distributed_config', 'overlap_param_gather', False),
    ('distributed_config', 'overlap_param_gather_with_optimizer_step', False),
    ('distributed_config', 'overlap_p2p_comm', True),
    ('distributed_config', 'batch_p2p_comm', None),
    ('distributed_config', 'align_grad_reduce', True),
    ('distributed_config', 'align_param_gather', True),
    ('distributed_config', 'tp_comm_overlap', False),
    ('distributed_config', 'nccl_comm_warmup', False),
)

_HF_ONLY = (
    ('model_config', 'experts_impl', None),
    ('model_config', 'new_special_tokens', []),
    ('model_config', 'device_map', None),
    ('model_config', 'max_memory', None),
    ('model_config', 'local_repo_path', None),
    ('model_config', 'model_kwargs', None),
    ('model_config', 'init_strategy', None),
    ('distributed_config', 'deepspeed', None),
    ('distributed_config', 'zero_hpz_partition_size', None),
    ('distributed_config', 'deepspeed_autotp_size', None),
    ('distributed_config', 'fsdp', None),
    ('distributed_config', 'ddp_find_unused_parameters', None),
    ('tuner_config', 'use_galore', False),
    ('train_config', 'use_liger_kernel', False),
    ('train_config', 'neftune_noise_alpha', None),
    ('train_config', 'optim', 'adamw_torch_fused'),
    ('train_config', 'optim_args', None),
    ('train_config', 'gradient_checkpointing_kwargs', None),
    ('train_config', 'router_aux_loss_coef', 0.0),
    ('train_config', 'use_logits_to_keep', None),
    ('train_config', 'predict_with_generate', False),
    ('train_config', 'eval_use_evalscope', False),
    ('train_config', 'full_determinism', False),
)

_MEGATRON_PARALLEL_SIZES = (
    'tensor_model_parallel_size',
    'pipeline_model_parallel_size',
    'context_parallel_size',
    'expert_model_parallel_size',
)


def _check_backend_specific(model_config: 'ModelConfig',
                            dataset_config: 'DatasetConfig',
                            train_config: 'TrainConfig',
                            distributed_config: 'DistributedConfig',
                            is_megatron: bool,
                            tuner_config: Optional['TunerConfig'] = None) -> None:
    """Reject knobs the active backend does not implement, so they cannot be silently ignored."""
    holders = {
        'model_config': model_config,
        'dataset_config': dataset_config,
        'train_config': train_config,
        'distributed_config': distributed_config,
        'tuner_config': tuner_config,
    }
    offending = _MEGATRON_ONLY if not is_megatron else _HF_ONLY
    wrong_backend = 'transformers' if not is_megatron else 'megatron'
    right_backend = 'megatron' if not is_megatron else 'transformers'

    for holder_name, attr, off_value in offending:
        holder = holders[holder_name]
        # tuner_config is optional (None == full-param training), in which case its tuner-only
        # knobs cannot have been set at all -- nothing to check.
        if holder is None:
            continue
        value = getattr(holder, attr)
        explicit = attr in getattr(holder, '_explicit_fields', set())
        if not explicit and _is_off(value, off_value):
            continue
        hint = (f'the {wrong_backend} backend runs with all Megatron parallel sizes == 1'
                if attr in _MEGATRON_PARALLEL_SIZES else f'the active backend is {wrong_backend}')
        raise ValueError(f'{attr}={value!r} is only implemented by the {right_backend} backend, but {hint}. '
                         f'Remove it, or switch DistributedConfig.backend.')


def _check_selective_recompute(distributed_config: 'DistributedConfig', is_megatron: bool) -> None:
    """'selective' recompute chooses WHAT to recompute; recompute_method chooses HOW MUCH of 'full'.

    Mirrors legacy megatron_args.py:799-800. Selective recomputation always targets the same
    (attention) operations, so it has no layer partitioning to configure -- `recompute_method`
    (uniform/block) only means something under 'full'. Pairing them asks Megatron to partition a mode
    that is not partitioned, which it refuses; catching it here reports the contradiction before ranks
    are spawned rather than deep in the model build.
    """
    if not is_megatron:
        return
    if distributed_config.recompute_granularity == 'selective' and distributed_config.recompute_method is not None:
        raise ValueError('DistributedConfig.recompute_method='
                         f'{distributed_config.recompute_method!r} has no effect with '
                         "recompute_granularity='selective': selective recompute always targets the attention "
                         "ops and has nothing to partition. Use recompute_granularity='full' to configure a "
                         'method, or drop recompute_method.')


def _check_pipeline_decoder_layers(distributed_config: 'DistributedConfig', is_megatron: bool) -> None:
    """The per-stage decoder layer overrides need a pipeline to distribute across.

    Mirrors legacy megatron_args.py:832-835. `decoder_first_pipeline_num_layers` /
    `decoder_last_pipeline_num_layers` move layers onto the first/last pipeline stage to balance an
    uneven split; with `pipeline_model_parallel_size == 1` there is only one stage, so both are the
    whole model and the override describes a partition that does not exist.
    """
    if not is_megatron or distributed_config.pipeline_model_parallel_size > 1:
        return
    for attr in ('decoder_first_pipeline_num_layers', 'decoder_last_pipeline_num_layers'):
        if getattr(distributed_config, attr) is not None:
            raise ValueError(f'DistributedConfig.{attr} needs pipeline_model_parallel_size > 1: with a single '
                             'pipeline stage there is no first/last stage to move layers onto. Set a pipeline '
                             f'size, or drop {attr}.')


def _check_tp_comm_overlap(distributed_config: 'DistributedConfig', is_megatron: bool) -> None:
    """Tensor-parallel comm/GEMM overlap only exists when sequence parallelism splits the activations.

    Mirrors legacy megatron_args.py:896-898. The overlap hides the tensor-parallel all-gather/
    reduce-scatter behind the GEMM, and those collectives only appear when `sequence_parallel` shards
    the activations along the sequence; without it there is nothing to overlap and Megatron asserts the
    same pairing.
    """
    if not is_megatron:
        return
    if distributed_config.tp_comm_overlap and not distributed_config.sequence_parallel:
        raise ValueError('DistributedConfig.tp_comm_overlap requires sequence_parallel=True: the overlap hides '
                         'the tensor-parallel collectives that only exist under sequence parallelism, so with it '
                         'off there is nothing to overlap.')


def _check_sequence_parallel_tp(distributed_config: 'DistributedConfig', is_megatron: bool) -> None:
    """Sequence parallelism splits activations across the tensor-parallel ranks, so it needs TP > 1.

    legacy silently sets `sequence_parallel = False` here (megatron_args.py:890-891). dev raises
    instead, for the reason the muon and attn-backend guards give: `sequence_parallel` is one the user
    typed, and quietly turning it off changes the activation memory profile so a run can look fine and
    use far more memory than the config implies. With `tensor_model_parallel_size == 1` there are no
    tensor-parallel ranks to split across, so the flag cannot do anything.
    """
    if not is_megatron:
        return
    if distributed_config.sequence_parallel and distributed_config.tensor_model_parallel_size <= 1:
        raise ValueError('DistributedConfig.sequence_parallel requires tensor_model_parallel_size > 1: it shards '
                         'activations along the sequence across the tensor-parallel ranks, and with TP=1 there are '
                         'none to shard across. legacy turned sequence_parallel off silently here; dev refuses so '
                         'the activation-memory profile of the run is not a surprise. Set a TP size, or '
                         'sequence_parallel=False.')


def _check_checkpoint_runtime(checkpoint_config: Optional['CheckpointConfig'],
                              distributed_config: 'DistributedConfig',
                              tuner_config: Optional['TunerConfig'],
                              rlhf_config: Optional['RLHFConfig'],
                              is_megatron: bool, *,
                              training: bool) -> None:
    """Reject checkpoint semantics that the selected loop/backend cannot honor."""
    if not training or checkpoint_config is None:
        return
    changed = set(_changed_fields(checkpoint_config))
    if checkpoint_config.save_strategy != 'steps':
        raise NotImplementedError(
            f'save_strategy={checkpoint_config.save_strategy!r} is not implemented by the Twinkle training loop. '
            'Use --save_strategy steps with --save_steps, or use the legacy CLI.')
    if not is_megatron:
        unsupported = {'no_save_rng', 'no_load_optim', 'no_load_rng'}.intersection(changed)
        if unsupported:
            raise NotImplementedError(
                f'Transformers checkpoints cannot selectively omit or restore {sorted(unsupported)}. '
                'Use --save_only_model/--resume_only_model, or switch to the Megatron backend.')
        tuner_type = getattr(tuner_config, 'tuner_type', 'full') if tuner_config is not None else 'full'
        if tuner_type != 'full' and 'max_shard_size' in changed:
            raise NotImplementedError(
                'Transformers PEFT adapter checkpoints do not support max_shard_size. Remove the option or use '
                'full-parameter training.')
    else:
        if not checkpoint_config.save_safetensors:
            raise NotImplementedError(
                'The dev Megatron runtime always writes HF-format safetensors checkpoints; '
                'use --save_safetensors true.')
        if not checkpoint_config.safe_serialization:
            raise NotImplementedError('Megatron HF-format checkpoints require --safe_serialization true.')
        if distributed_config.bridge_backend == 'megatron-bridge' and 'max_shard_size' in changed:
            raise NotImplementedError(
                'megatron-bridge AutoBridge does not expose max_shard_size. Use --bridge_backend mcore-bridge or '
                'remove the option.')
    if rlhf_config is not None and rlhf_config.rlhf_type in {'grpo', 'gkd', 'ppo'}:
        if 'ignore_data_skip' in changed:
            raise NotImplementedError(
                f'ignore_data_skip does not apply to {rlhf_config.rlhf_type} because its online loop has no resumable '
                'dataset iterator. Remove the option.')
        if rlhf_config.rlhf_type == 'ppo' and checkpoint_config.save_total_limit == 1:
            raise ValueError(
                'PPO requires save_total_limit >= 2 because the policy and value-model components are saved '
                'sequentially; retaining one previous complete checkpoint avoids a no-valid-checkpoint window.')


def _check_save_total_limit(checkpoint_config: Optional['CheckpointConfig'], is_megatron: bool) -> None:
    """Validate the rolling checkpoint limit and preserve legacy Megatron's stricter lower bound.

    `async_save` writes in the background, and the limit's delete-oldest step cannot tell whether an
    in-flight async save has finished, so the two are incompatible.
    """
    if checkpoint_config is None or checkpoint_config.save_total_limit is None:
        return
    if checkpoint_config.save_total_limit < 1:
        raise ValueError('CheckpointConfig.save_total_limit must be >= 1.')
    if not is_megatron:
        return
    if checkpoint_config.async_save:
        raise ValueError('CheckpointConfig.save_total_limit is incompatible with async_save=True: the rolling '
                         'delete of old checkpoints cannot tell whether a background save has finished. Disable '
                         'one of the two.')
    if checkpoint_config.save_total_limit < 2:
        raise ValueError(
            'CheckpointConfig.save_total_limit must be >= 2 on the Megatron backend, matching the legacy CLI.')


#: rlhf_type -> whether it trains against a separate reference model. CPO/ORPO fold the reference into
#: their own loss and LoRA uses the adapter-disabled base as reference, so neither takes a ref_model.
_RLHF_USES_REF_MODEL = ('dpo', 'kto', 'ppo', 'grpo')


def _check_rlhf_ref_model(model_config: 'ModelConfig', tuner_config: Optional['TunerConfig'],
                          rlhf_config: Optional['RLHFConfig']) -> None:
    """Reject a reference model passed to an algorithm that has none.

    Mirrors the trailing `elif self.ref_model is not None: raise` of legacy rlhf_args.py:297-298. The
    derivation half (defaulting ref_model to model for the algorithms that use one) lives in
    process.py::_derive_rlhf_ref_model; this is the refusal half. CPO/ORPO build the reference into
    their loss and LoRA training uses the base model with the adapter disabled, so a `--ref_model`
    there is a knob that would be silently ignored -- the class of mistake validate.py exists to catch.
    """
    if rlhf_config is None:
        return
    rlhf_type = getattr(rlhf_config, 'rlhf_type', None)
    tuner_type = getattr(tuner_config, 'tuner_type', 'full') if tuner_config is not None else 'full'
    uses_ref = rlhf_type in _RLHF_USES_REF_MODEL and (tuner_type == 'full' or bool(rlhf_config.ref_adapters))
    if rlhf_config.ref_model is None:
        if rlhf_config.ref_adapters:
            raise ValueError('ref_adapters requires a reference model; call process_configs or set ref_model.')
        return
    # grpo with beta=0 drops the KL term, so even a ref-using algorithm needs no reference then.
    if rlhf_type == 'grpo' and rlhf_config.beta == 0.0:
        uses_ref = False
    if not uses_ref:
        raise ValueError(f'RLHFConfig.ref_model={rlhf_config.ref_model!r} is not used by rlhf_type={rlhf_type!r}'
                         f' with tuner_type={tuner_type!r}: CPO/ORPO fold the reference into their loss and LoRA '
                         'uses the adapter-disabled base as the reference, so no separate ref_model is loaded. '
                         'Remove it.')


def _check_rlhf_padding_free(template_config: 'TemplateConfig', dataset_config: 'DatasetConfig',
                            rlhf_config: Optional['RLHFConfig']) -> None:
    """Only some RLHF algorithms have a padding-free/packing training path.

    Mirrors legacy rlhf_args.py::_check_padding_free. padding_free (and packing, which implies it)
    flattens a micro batch into one variable-length sequence; only GRPO/DPO/KTO/GKD implement the
    loss over that layout. For the others the flag would be accepted and then read by a code path that
    assumes padded batches, so it is refused here rather than mis-computed later.
    """
    if rlhf_config is None:
        return
    if not (template_config.padding_free or dataset_config.packing):
        return
    rlhf_type = getattr(rlhf_config, 'rlhf_type', None)
    if rlhf_type not in ('grpo', 'dpo', 'kto', 'gkd'):
        feature = 'packing' if dataset_config.packing else 'padding_free'
        raise ValueError(f'rlhf_type={rlhf_type!r} does not support {feature}: only grpo/dpo/kto/gkd implement the '
                         'variable-length training path it produces. Set the corresponding flag to False.')


def _check_rlhf_sequence_parallel(template_config: 'TemplateConfig', rlhf_config: Optional['RLHFConfig']) -> None:
    """Sequence parallelism is not wired for ANY RLHF algorithm on the dev path.

    Divergence from legacy, deliberate: legacy rlhf_args.py::_check_sequence_parallel allows grpo/dpo
    because legacy's trainers implement the sequence-parallel loss for them. dev wires NO mesh for the
    RLHF recipes at all, so allowing grpo/dpo here would silently train with SP=1 while the config says
    otherwise -- the exact failure mode validate.py exists to kill. Re-enable per algorithm when the
    RLHF SP path is wired.
    """
    if rlhf_config is None or template_config.sequence_parallel_size <= 1:
        return
    rlhf_type = getattr(rlhf_config, 'rlhf_type', None)
    raise ValueError(f'rlhf_type={rlhf_type!r} does not support sequence_parallel_size='
                     f'{template_config.sequence_parallel_size} on the dev path: no device mesh is wired for the '
                     'RLHF recipes, so the run would silently train WITHOUT sequence parallelism. '
                     'Set sequence_parallel_size=1. (legacy allows grpo/dpo here; dev diverges until RLHF SP '
                     'is wired -- see _check_hf_sequence_parallel for the SFT path.)')


#: attn_impl values whose attention kernel handles the variable-length (THD) layout that
#: padding_free produces under Ulysses SP. Mirrors legacy sft_args.py supported_impls; twinkle's
#: SP strategy enforces the same requirement at first forward (flash_attention_2/3 only).
_SP_PADDING_FREE_ATTN_IMPLS = ('flash_attn', 'flash_attention_2', 'flash_attention_3', 'flash_attention_4')


def _check_hf_sequence_parallel(model_config: 'ModelConfig', template_config: 'TemplateConfig',
                                dataset_config: 'DatasetConfig', distributed_config: 'DistributedConfig',
                                is_megatron: bool) -> None:
    """Guards for Ulysses sequence parallelism (TemplateConfig.sequence_parallel_size) on the HF backend.

    Every check raises: SP that cannot do what the config says would otherwise SILENTLY train with
    SP=1 (nothing on the HF path used to read this knob) or crash deep in the first forward.
    ``process_configs`` has already resolved packing-derived padding_free, so guard 3 sees the effective
    value. Streaming is allowed: twinkle's IterableFetcher slices by data_world_size the same way.
    """
    sp = template_config.sequence_parallel_size
    if sp <= 1:
        return

    if is_megatron:
        # TemplateConfig.sequence_parallel_size is the HF Ulysses knob; Megatron's sequence
        # parallelism is DistributedConfig.sequence_parallel (TP-SP), a different feature.
        raise ValueError(f'TemplateConfig.sequence_parallel_size={sp} only applies to the transformers backend, '
                         'but the active backend is megatron. Megatron sequence parallelism is '
                         'DistributedConfig.sequence_parallel (TP-SP over the tensor-parallel ranks) -- set that '
                         'instead, or switch DistributedConfig.backend.')

    if distributed_config.mode != 'local':
        raise NotImplementedError(f'sequence_parallel_size={sp} is only wired for mode="local" (torchrun): under '
                                  "mode='ray' the model gets a pure data-parallel mesh (_apply_ray_placement), so "
                                  'SP would silently not apply. Run with torchrun, or set sequence_parallel_size=1.')

    if distributed_config.fsdp:
        raise NotImplementedError(f'sequence_parallel_size={sp} with DistributedConfig.fsdp is not supported yet: '
                                  'the FSDP x ulysses composition in twinkle is unvalidated. Use DDP/accelerate '
                                  '(the default strategy), or set sequence_parallel_size=1.')

    # Legacy asserts this at collate time (swift/template/base.py); dev fails fast at validation.
    if template_config.padding_side != 'right':
        raise ValueError(f'sequence_parallel_size={sp} requires padding_side="right" (got '
                         f'{template_config.padding_side!r}): the SP collator injects per-row position_ids that '
                         'assume right padding. legacy asserts the same at collate time; dev refuses up front.')

    if template_config.padding_free or dataset_config.packing:
        if model_config.attn_impl not in _SP_PADDING_FREE_ATTN_IMPLS:
            raise ValueError(f'sequence_parallel_size={sp} with padding_free requires a flash attention kernel: '
                             f'twinkle\'s SP strategy rejects the variable-length layout under '
                             f'attn_impl={model_config.attn_impl!r}. Use one of '
                             f'{", ".join(repr(i) for i in _SP_PADDING_FREE_ATTN_IMPLS)}, or set padding_free=False '
                             '(packing implies padding_free, so drop packing too).')

    # twinkle.initialize(mode='local') never calls dist.init_process_group -- world size comes from
    # Platform.get_world_size() (the WORLD_SIZE env torchrun sets), the same source initialize uses
    # to build the default mesh. Requiring dist here would reject every real SP run.
    from twinkle.utils import Platform
    world = Platform.get_world_size()
    if world < 2:
        raise ValueError(f'sequence_parallel_size={sp} requires torchrun (WORLD_SIZE>=2): a single process has no '
                         'ranks to split a sequence across. Set sequence_parallel_size=1 for a single-process run.')
    if world % sp != 0:
        raise ValueError(f'world_size={world} is not divisible by sequence_parallel_size={sp}: the SP groups must '
                         'tile the ranks exactly. Adjust the rank count or sequence_parallel_size.')

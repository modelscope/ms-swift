# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

from swift.dev.utils.naming import parse_optim_args, resolve_galore_target, resolve_optim_target, resolve_scheduler
from swift.dev.utils import get_logger

if TYPE_CHECKING:
    from swift.dev2.config import DistributedConfig, PluginConfig, TrainConfig, TunerConfig
    from swift.dev.model import TrainableModel

logger = get_logger()


def megatron_weight_decay_bounds(cfg: TrainConfig) -> tuple:
    """(start_wd, end_wd) for Megatron's OptimizerParamScheduler.

    Mirrors legacy megatron_args.py: 'constant' derives both ends from weight_decay and forbids
    explicit ends; any other style requires both ends. Shared by validate_configs and here.
    """
    if cfg.weight_decay_incr_style == 'constant':
        if cfg.start_weight_decay is not None or cfg.end_weight_decay is not None:
            raise ValueError('start_weight_decay/end_weight_decay only apply to a NON-constant '
                             "weight_decay_incr_style; with the default 'constant' both ends are weight_decay "
                             f"(={cfg.weight_decay}). Set weight_decay_incr_style='linear'/'cosine' to ramp it, "
                             'or drop start/end_weight_decay.')
        return cfg.weight_decay, cfg.weight_decay
    if cfg.start_weight_decay is None or cfg.end_weight_decay is None:
        raise ValueError(f'weight_decay_incr_style={cfg.weight_decay_incr_style!r} ramps weight decay, so it '
                         f'needs BOTH start_weight_decay and end_weight_decay (got '
                         f'{cfg.start_weight_decay} -> {cfg.end_weight_decay}).')
    return cfg.start_weight_decay, cfg.end_weight_decay


def warmup_budget(cfg: TrainConfig, num_training_steps: int, *, is_megatron: bool) -> float:
    """Warmup length in OPTIMIZER steps, possibly fractional.

    The two backends resolve the ratio/absolute pair with OPPOSITE priority, each reproduced rather
    than unified: transformers takes the ABSOLUTE steps when >= 1, megatron the RATIO first.
    """
    if is_megatron:
        if cfg.warmup_ratio:
            return cfg.warmup_ratio * num_training_steps
        return float(cfg.warmup_steps)
    if cfg.warmup_steps >= 1:
        return float(cfg.warmup_steps)
    return cfg.warmup_ratio * num_training_steps


def resolve_max_grad_norm(cfg: TrainConfig) -> float:
    """Gradient-clipping threshold, folding the deprecated `clip_grad` alias into `max_grad_norm`.

    Both set: max_grad_norm wins with a warning (clip_grad is on its way out, and legacy Megatron
    scripts carrying a stray max_grad_norm would otherwise start erroring on a value legacy ignored).
    """
    if cfg.clip_grad is None:
        return cfg.max_grad_norm
    # Read the default off the dataclass, so "did the user change it" cannot drift if retuned.
    import dataclasses
    default_mgn = next(f for f in dataclasses.fields(cfg) if f.name == 'max_grad_norm').default
    if cfg.max_grad_norm != default_mgn:
        logger.warning(f'Both max_grad_norm={cfg.max_grad_norm} and clip_grad={cfg.clip_grad} are set; '
                       f'they are the same knob, so clip_grad is ignored and {cfg.max_grad_norm} is used. '
                       'clip_grad is deprecated -- drop it and keep max_grad_norm.')
        return cfg.max_grad_norm
    logger.warning(f'clip_grad={cfg.clip_grad} is deprecated; it is an alias of max_grad_norm and now applies '
                   'to both backends. Use max_grad_norm instead.')
    return cfg.clip_grad


def _galore_config(cfg: TrainConfig) -> dict:
    """The ``GaLoreConfig`` fields set_optimizer turns into low-rank projection param groups.

    QGaLore knobs ride along when galore_quantization is set; target_modules is passed only when
    set (forwarding None would override GaLoreConfig's own default).
    """
    config = {
        'rank': cfg.galore_rank,
        'update_proj_gap': cfg.galore_update_proj_gap,
        'galore_scale': cfg.galore_scale,
        'proj_type': cfg.galore_proj_type,
    }
    if cfg.galore_target_modules is not None:
        config['target_modules'] = cfg.galore_target_modules
    if cfg.galore_quantization:
        config.update({
            'quantize': True,
            'proj_quant': cfg.galore_proj_quant,
            'proj_bits': cfg.galore_proj_bits,
            'proj_group_size': cfg.galore_proj_group_size,
            'cos_threshold': cfg.galore_cos_threshold,
            'gamma_proj': cfg.galore_gamma_proj,
            'queue_size': cfg.galore_queue_size,
        })
    return config


def _muon_config(cfg: TrainConfig) -> dict:
    """The ``MuonConfig`` fields set_optimizer turns into Muon/QK-Clip param groups for MuonClip.

    adam betas/eps feed the AdamW step Muon applies to non-orthogonalised groups; qk_clip defaults
    OFF (MuonConfig's own defaults clip ~1e3 attention logits every step and collapse Q/K).
    """
    return {
        'momentum': cfg.muon_momentum,
        'nesterov': cfg.muon_use_nesterov,
        'newton_schulz_steps': cfg.muon_num_ns_steps,
        'adamw_betas': (cfg.adam_beta1, cfg.adam_beta2),
        'adamw_eps': cfg.adam_epsilon,
        'qk_clip_enabled': cfg.qk_clip_enabled,
        'qk_clip_tau': cfg.qk_clip_tau,
    }


def build_param_groups_spec(cfg: TrainConfig,
                            tuner_config: Optional[TunerConfig] = None,
                            plugin_config: Optional[PluginConfig] = None) -> Optional[list]:
    """Compose the backend-neutral optimizer param-group spec: ordered first-match-wins rules that
    twinkle's ``set_optimizer`` applies worker-side on both backends, or None when nothing is set.

    Order: LoRA+ (embedding / lora_B / ndim==1, reproducing legacy's isinstance chain) -> per-tower
    lrs (aligner before vision_tower: its prefix nests under the tower's) -> user plugins last.
    """
    from swift.dev.builders.optimizer_plugins import PARAM_GROUPS
    from swift.dev.utils.plugin import PluginRegistry

    spec: list = []

    # LoRA+ -- on only with lorap_lr_ratio (validate rejects it under a non-LoRA tuner, so reaching
    # here implies lora_A/lora_B trainables); lora_A matches no rule and keeps the base lr.
    if tuner_config is not None and tuner_config.lorap_lr_ratio is not None:
        ratio = tuner_config.lorap_lr_ratio
        spec.append({'match': {'module_type': 'Embedding'}, 'lr': tuner_config.lorap_emb_lr})
        spec.append({'match': {'name': '*lora_B*'}, 'lr_mult': ratio})
        spec.append({'match': {'ndim': 1}, 'lr_mult': ratio})

    # The aligner rule is emitted whenever any tower lr is set -- at base lr if only vit_lr is, purely
    # to claim aligner params -- so the broader vision_tower rule cannot swallow them.
    if cfg.vit_lr is not None or cfg.aligner_lr is not None:
        base_lr = cfg.learning_rate
        spec.append({'match': {'tower': 'aligner'}, 'lr': base_lr if cfg.aligner_lr is None else cfg.aligner_lr})
        if cfg.vit_lr is not None:
            spec.append({'match': {'tower': 'vision_tower'}, 'lr': cfg.vit_lr})

    # User plugins (load_configured is idempotent; a no-op when plugin_config is None).
    if cfg.param_groups_plugins:
        PluginRegistry.load_configured(plugin_config)
        for name in cfg.param_groups_plugins:
            plugin = PluginRegistry.resolve(PARAM_GROUPS, name, config=cfg)
            spec.extend(plugin(train_config=cfg, tuner_config=tuner_config))

    return spec or None


def configure_optimizer(model: TrainableModel,
                        cfg: TrainConfig,
                        *,
                        num_training_steps: int,
                        distributed_config: Optional[DistributedConfig] = None,
                        tuner_config: Optional[TunerConfig] = None,
                        plugin_config: Optional[PluginConfig] = None) -> None:
    """Set optimizer + lr_scheduler on ``model`` from a ``TrainConfig`` (backend per distributed_config)."""
    lr = cfg.learning_rate
    # Backend identity comes from the CONFIG (the model may be a Ray proxy); DistributedConfig is
    # the single source every other layer reads (is_megatron_backend). None keeps transformers.
    from swift.dev2.builders.model import is_megatron_backend
    is_megatron = distributed_config is not None and is_megatron_backend(distributed_config)
    # One backend-neutral spec feeds both branches; None leaves the default grouping untouched.
    param_groups_spec = build_param_groups_spec(cfg, tuner_config, plugin_config)

    if is_megatron:
        # Megatron's native optimizer + OptimizerParamScheduler; the step unit is OPTIMIZER steps
        # (twinkle's lr_step() advances 1/step) -- legacy spells the same curve in samples.
        start_wd, end_wd = megatron_weight_decay_bounds(cfg)
        # clip_grad MUST ride here: Megatron clips inside step() and twinkle's clip_grad_and_step
        # ignores its max_grad_norm argument entirely.
        import torch
        # min_lr must reach the OPTIMIZER too: the scheduler reads the bound off the param group
        # first, so leaving the group's 0.0 lets cosine decay past the floor all the way to 0.
        dtype_map = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp8': torch.uint8}
        model.set_optimizer(
            'MegatronOptimizer',
            optimizer=cfg.optimizer,
            lr=lr,
            min_lr=cfg.min_lr,
            weight_decay=cfg.weight_decay,
            clip_grad=resolve_max_grad_norm(cfg),
            adam_beta1=cfg.adam_beta1,
            adam_beta2=cfg.adam_beta2,
            adam_eps=cfg.adam_epsilon,
            sgd_momentum=cfg.sgd_momentum,
            muon_momentum=cfg.muon_momentum,
            muon_split_qkv=cfg.muon_split_qkv,
            muon_nesterov=cfg.muon_use_nesterov,
            muon_scale_mode=cfg.muon_scale_mode,
            muon_fp32_matmul_prec=cfg.muon_fp32_matmul_prec,
            muon_coefficient_type=cfg.muon_coefficient_type,
            muon_num_ns_steps=cfg.muon_num_ns_steps,
            muon_tp_mode=cfg.muon_tp_mode,
            muon_extra_scale_factor=cfg.muon_extra_scale_factor,
            muon_scalar_optimizer=cfg.muon_scalar_optimizer,
            use_precision_aware_optimizer=cfg.use_precision_aware_optimizer,
            main_params_dtype=dtype_map[cfg.main_params_dtype],
            main_grads_dtype=dtype_map[cfg.main_grads_dtype],
            exp_avg_dtype=dtype_map[cfg.exp_avg_dtype],
            exp_avg_sq_dtype=dtype_map[cfg.exp_avg_sq_dtype],
            optimizer_cpu_offload=cfg.optimizer_cpu_offload,
            optimizer_offload_fraction=cfg.optimizer_offload_fraction,
            optimizer_cuda_graph=cfg.optimizer_cuda_graph,
            overlap_param_gather_with_optimizer_step=bool(
                distributed_config and distributed_config.overlap_param_gather_with_optimizer_step),
            apply_wd_to_qk_layernorm=cfg.apply_wd_to_qk_layernorm,
            param_groups_spec=param_groups_spec)
        # lr_warmup_steps stays FRACTIONAL: OptimizerParamScheduler interpolates the ramp linearly on
        # it and legacy never rounds it either (0.05 over 50 = 2.5 steps; rounding shifts the peak).
        lr_decay_steps = cfg.lr_decay_iters if cfg.lr_decay_iters is not None else max(1, num_training_steps)
        warmup_steps_exact = warmup_budget(cfg, lr_decay_steps, is_megatron=True)
        lr_decay_style = cfg.lr_decay_style
        if lr_decay_style == 'cosine' and cfg.lr_scheduler != 'cosine':
            from swift.dev.utils.naming import resolve_megatron_decay_style
            lr_decay_style = resolve_megatron_decay_style(cfg.lr_scheduler)
        # wd_incr_steps is NOT passed: twinkle hardcodes it to lr_decay_steps and does not pop it
        # (passing it would raise a duplicate-keyword TypeError).
        model.set_lr_scheduler(
            'default',
            lr_decay_steps=lr_decay_steps,
            max_lr=lr,
            init_lr=cfg.lr_warmup_init,
            lr_warmup_steps=warmup_steps_exact,
            lr_decay_style=lr_decay_style,
            min_lr=cfg.min_lr,
            start_wd=start_wd,
            end_wd=end_wd,
            wd_incr_style=cfg.weight_decay_incr_style,
            wsd_decay_steps=cfg.lr_wsd_decay_iters,
            lr_wsd_decay_style=cfg.lr_wsd_decay_style)
        return

    optim_cls, extra_kwargs = resolve_optim_target(cfg.optim)
    # GaLore: upgrade the base optimizer to its variant; galore_config is inert on a non-GaLore one.
    galore_config = None
    if cfg.use_galore:
        optim_cls = resolve_galore_target(optim_cls, quantize=cfg.galore_quantization)
        # Neither GaLoreAdamW nor QGaLoreAdamW8bit has a fused CUDA kernel; `fused` is AdamW-only.
        extra_kwargs.pop('fused', None)
        galore_config = _galore_config(cfg)
    # MuonClip is more than momentum SGD only with a muon_config (param grouping + QK-Clip).
    muon_config = _muon_config(cfg) if optim_cls == 'MuonClip' else None

    opt_kwargs: dict = {'lr': lr, 'weight_decay': cfg.weight_decay}
    # betas/eps only for the Adam family (incl. GaLore variants); Adafactor/SGD/MuonClip take other
    # names -- MuonClip reads adamw_betas/adamw_eps off muon_config instead.
    if optim_cls in ('AdamW', 'Adam', 'GaLoreAdamW', 'QGaLoreAdamW8bit'):
        opt_kwargs['betas'] = (cfg.adam_beta1, cfg.adam_beta2)
        opt_kwargs['eps'] = cfg.adam_epsilon
    # Name-specific constructor args (fused / Adafactor flags) -- see naming._OPTIM_EXTRA_KWARGS.
    opt_kwargs.update(extra_kwargs)
    # optim_args last: the user's explicit escape hatch wins (matches HF's optimizer_kwargs merge).
    opt_kwargs.update(parse_optim_args(cfg.optim_args))
    if galore_config is not None:
        opt_kwargs['galore_config'] = galore_config
    if muon_config is not None:
        opt_kwargs['muon_config'] = muon_config
    if param_groups_spec is not None:
        # Applied before galore_config / muon_config reshape the groups, so a per-tower lr or LoRA+
        # rule owns its parameters first (see TransformersModel.set_optimizer).
        opt_kwargs['param_groups_spec'] = param_groups_spec
    # NOTE: `params` is deliberately NOT passed. twinkle builds the decay/no-decay groups itself
    # (get_decay_parameter_names + LoRA adapter_name filter); a flat list here would bypass that.
    model.set_optimizer(optim_cls, **opt_kwargs)

    sched_cls = resolve_scheduler(cfg.lr_scheduler)
    if sched_cls is None:
        return  # constant lr: no scheduler
    model.set_lr_scheduler(
        sched_cls,
        # Rounded UP, matching HF: TrainingArguments uses math.ceil(ratio * steps) and the schedules
        # compare against an integer boundary; round() is banker's (2.5 -> 2), off-by-one vs legacy.
        num_warmup_steps=math.ceil(warmup_budget(cfg, num_training_steps, is_megatron=False)),
        num_training_steps=num_training_steps,
        # Per-schedule extras (min_lr for cosine_with_min_lr, power for polynomial, ...).
        **(cfg.lr_scheduler_kwargs or {}),
    )

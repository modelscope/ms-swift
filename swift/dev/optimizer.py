"""Optimizer/scheduler configuration from TrainConfig.

Maps the backend-agnostic ``TrainConfig`` onto twinkle Model's
``set_optimizer`` / ``set_lr_scheduler`` (which build the instances internally).
Kept tiny and explicit: no training-plan reverse-engineering, callers pass
``num_training_steps``.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from swift.dev.naming import (parse_optim_args, parse_scheduler_kwargs, resolve_megatron_decay_style,
                              resolve_optim_target, resolve_scheduler)
from swift.dev.utils import get_logger

if TYPE_CHECKING:
    from swift.dev.config import TrainConfig
    from swift.dev.model import TrainableModel

logger = get_logger()


def megatron_weight_decay_bounds(cfg: TrainConfig) -> tuple:
    """(start_wd, end_wd) for Megatron's OptimizerParamScheduler.

    Mirrors legacy swift's MegatronArguments post-init (megatron_args.py:1096-1102): the 'constant'
    style derives both ends from weight_decay and forbids setting them, any other style requires
    both explicitly. Shared by validate_configs (so an illegal combination fails in milliseconds,
    before any weight load) and configure_optimizer, so there is one copy of the rule.
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

    The two backends resolve the ratio/absolute pair with OPPOSITE priority, and each is reproduced
    here rather than unified -- picking one rule would silently change the warmup on one backend:
      - transformers: TrainingArguments.get_warmup_steps takes warmup_steps when it is >= 1, else
        ceil(num_training_steps * warmup_ratio). So the ABSOLUTE count wins.
      - megatron: megatron_lm_utils uses lr_warmup_fraction * lr_decay_steps when the fraction is
        set, else lr_warmup_iters. So the RATIO wins.
    Returns a float so the Megatron path keeps its fractional ramp; the transformers branch rounds
    up itself (see configure_optimizer).
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

    legacy carried two names for one knob -- HF `max_grad_norm` (clipped outside the optimizer) and
    Megatron `clip_grad` (clipped inside `optimizer.step()` from `OptimizerConfig.clip_grad`). The
    clipping THRESHOLD means the same thing in both, and legacy's Megatron path never read
    `max_grad_norm` at all, so setting it there was silently dropped. dev merges them under
    `max_grad_norm` and keeps `clip_grad` working with a deprecation warning.

    Both set is not a supported combination: `max_grad_norm` wins and the conflict is warned about,
    rather than fail-fast, because `clip_grad` is on its way out and legacy Megatron scripts that
    already carry a stray `max_grad_norm` would otherwise start erroring on a value legacy ignored.
    """
    if cfg.clip_grad is None:
        return cfg.max_grad_norm
    # Read the default off the dataclass rather than hardcoding 1.0, so "did the user change it"
    # cannot drift if the default is ever retuned.
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


def _is_megatron_model(model) -> bool:
    """True for the dev MegatronModel (whose optimizer/scheduler are Megatron-native, not torch)."""
    try:
        from swift.dev.model.megatron.model import MegatronModel
    except Exception:
        return False
    return isinstance(model, MegatronModel)


def configure_optimizer(model: TrainableModel, cfg: TrainConfig, *, num_training_steps: int) -> None:
    """Set optimizer + lr_scheduler on ``model`` from a ``TrainConfig``.

    Args:
        model: a twinkle-derived Model (has set_optimizer / set_lr_scheduler).
        cfg: TrainConfig (learning_rate / optim / weight_decay / adam_* /
             lr_scheduler_type / warmup_ratio). learning_rate is read as-is: the default lives on
             TrainConfig, so no hidden fallback here can disagree with what the Config reports.
        num_training_steps: total optimizer steps (for warmup + decay schedule).
    """
    lr = cfg.learning_rate
    is_megatron = _is_megatron_model(model)
    # Fractional warmup budget. Kept fractional for Megatron and rounded for the transformers path --
    # see each branch below; the two schedulers disagree on what a fractional step count means.
    warmup_steps_exact = warmup_budget(cfg, num_training_steps, is_megatron=is_megatron)

    if is_megatron:
        # Megatron only accepts its own distributed optimizer ('Adam' routes to it) and
        # OptimizerParamScheduler ('default'); torch optim/scheduler names raise. cfg.optim is not
        # consulted here at all -- validate_configs refuses a non-default optim on this backend, so
        # an HF optimizer name can no longer be silently replaced by Adam.
        #
        # Megatron's other optimizer types are NOT reachable from dev yet, even though both legacy
        # and Megatron-LM have them (legacy: optimizer=adam/sgd/muon/dist_muon plus muon_* knobs;
        # mcore: OptimizerConfig.optimizer with 9 muon_* fields). twinkle blocks them twice:
        # set_optimizer accepts only 'MegatronOptimizer'/'default'/'Adam', and
        # _create_megatron_optimizer hardcodes optimizer='adam' inside OptimizerConfig(...) while
        # also forwarding **kwargs, so passing optimizer= would raise a duplicate-keyword TypeError.
        # Wiring sgd/muon is a separate task (it needs an upstream twinkle change plus a numerical
        # baseline); until then the only way to ask for one is cfg.optim, which fails fast above.
        #
        # Step unit: lr_decay_steps/lr_warmup_steps are OPTIMIZER steps here, because twinkle's
        # lr_step() advances the scheduler by increment=1 per optimizer step. (Legacy megatron
        # expresses the same schedule in samples -- iters * global_batch_size -- and steps by
        # global_batch_size. Same curve, different unit; do not "fix" one to look like the other.)
        #
        # clip_grad: Megatron clips inside optimizer.step() from OptimizerConfig.clip_grad, and
        # twinkle's Megatron clip_grad_and_step() ignores its max_grad_norm argument entirely, so
        # the threshold MUST be plumbed here or clipping silently runs at twinkle's own default.
        # The value comes from resolve_max_grad_norm (max_grad_norm, or the deprecated clip_grad
        # alias) -- one knob, two accepted names.
        #
        # start_wd/end_wd/wd_incr_style: OptimizerParamScheduler.step() overwrites
        # param_group['weight_decay'] on every step, so the weight_decay passed to set_optimizer
        # survives exactly zero steps -- the SCHEDULE is the authority and must be passed too, or
        # training silently uses twinkle's 0.01 default. wd_incr_steps is NOT passed: twinkle
        # hardcodes it to lr_decay_steps (the value we want) and does not pop it, so passing it
        # would raise a duplicate-keyword TypeError.
        start_wd, end_wd = megatron_weight_decay_bounds(cfg)
        # min_lr goes to the OPTIMIZER too, not just the scheduler. mcore writes
        # {'min_lr': OptimizerConfig.min_lr} into every param group (optimizer/__init__.py:399) and
        # OptimizerParamScheduler.get_lr then reads the bound OFF THE PARAM GROUP first
        # (param_group.get('min_lr', self.min_lr)). So passing min_lr only to set_lr_scheduler leaves
        # the group's 0.0 in charge and cosine decays past the floor all the way to 0 -- observed as
        # dev ending a 50-step run at lr=0 while legacy held at 1e-5 with identical flags.
        model.set_optimizer(
            'Adam',
            lr=lr,
            min_lr=cfg.min_lr,
            weight_decay=cfg.weight_decay,
            clip_grad=resolve_max_grad_norm(cfg),
            adam_beta1=cfg.adam_beta1,
            adam_beta2=cfg.adam_beta2,
            adam_eps=cfg.adam_epsilon)
        # lr_warmup_steps stays FRACTIONAL here: OptimizerParamScheduler interpolates the warmup ramp
        # linearly on this value, and legacy megatron also keeps it fractional (lr_warmup_steps =
        # lr_warmup_fraction * lr_decay_steps in SAMPLES, megatron_lm_utils.py:579-581, never rounded).
        # warmup_ratio=0.05 over 50 steps is 2.5 steps; rounding it to 2 made every warmup lr 1.25x
        # (2.5/2) too high and moved the peak a step earlier. Verified directly against
        # OptimizerParamScheduler: fractional -> dev's 50-step lr curve is bit-identical to legacy's,
        # rounded -> all 50 steps differ.
        model.set_lr_scheduler(
            'default',
            lr_decay_steps=max(1, num_training_steps),
            max_lr=lr,
            lr_warmup_steps=warmup_steps_exact,
            lr_decay_style=resolve_megatron_decay_style(cfg.lr_scheduler_type),
            min_lr=cfg.min_lr,
            start_wd=start_wd,
            end_wd=end_wd,
            wd_incr_style=cfg.weight_decay_incr_style)
        return

    optim_cls, extra_kwargs = resolve_optim_target(cfg.optim)
    opt_kwargs: dict = {'lr': lr, 'weight_decay': cfg.weight_decay}
    # betas/eps only for the Adam family; Adafactor and SGD reject them.
    if optim_cls in ('AdamW', 'Adam'):
        opt_kwargs['betas'] = (cfg.adam_beta1, cfg.adam_beta2)
        opt_kwargs['eps'] = cfg.adam_epsilon
    # Name-specific constructor args (fused / Adafactor flags) -- see naming._OPTIM_EXTRA_KWARGS.
    opt_kwargs.update(extra_kwargs)
    # optim_args last: it is the user's explicit escape hatch, so it wins over our defaults
    # (matches HF, where _parse_optim_args output is merged into optimizer_kwargs).
    opt_kwargs.update(parse_optim_args(cfg.optim_args))
    # NOTE: `params` is deliberately NOT passed. twinkle's set_optimizer builds the two
    # weight-decay param groups itself (_create_param_group), reusing transformers'
    # get_decay_parameter_names rules (bias / *norm excluded from decay) AND filtering by
    # adapter_name for LoRA. Passing a flat param list here would BYPASS that and apply
    # weight_decay to norms/biases, diverging from legacy swift.
    model.set_optimizer(optim_cls, **opt_kwargs)

    sched_cls = resolve_scheduler(cfg.lr_scheduler_type)
    if sched_cls is None:
        return  # constant lr: no scheduler
    model.set_lr_scheduler(
        sched_cls,
        # Rounded UP for the transformers path, matching HF exactly: TrainingArguments.get_warmup_steps
        # uses math.ceil(num_training_steps * warmup_ratio) (training_args.py), and HF's schedules
        # compare the step index against this as an integer boundary -- a fractional value would not
        # mean "2.5 steps of warmup" there. round() is NOT equivalent: Python rounds 2.5 to 2
        # (banker's rounding), so warmup_ratio=0.05 over 50 steps gave dev 2 warmup steps against
        # legacy's 3, shifting the whole LR curve. Measured effect: step-1/2 losses matched to 0.02%
        # while steps 3-10 drifted up to 6% before training dynamics absorbed it.
        num_warmup_steps=math.ceil(warmup_steps_exact),
        num_training_steps=num_training_steps,
        # Per-schedule extras (min_lr for cosine_with_min_lr, power for polynomial, ...). Legacy
        # swift carries them the same way, as HF's scheduler_specific_kwargs.
        **parse_scheduler_kwargs(cfg.lr_scheduler_kwargs),
    )

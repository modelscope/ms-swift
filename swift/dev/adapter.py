"""Map a TunerConfig onto the peft adapter config the requested tuner_type needs.

Every tuner here is peft-backed, so ``add_adapter_to_model`` receives a plain peft config and the
model-side path is identical for all of them. What differs is only which config class is built and
which TunerConfig fields feed it:

  - lora        -> LoraConfig. Covers QLoRA (= 4bit base model, see QuantizeConfig, + plain LoRA),
                   DoRA (use_dora) and rsLoRA (use_rslora), which are LoRA *flags*, not separate
                   tuners -- there is deliberately no 'dora'/'rslora'/'qlora' tuner_type.
  - adalora     -> AdaLoraConfig (LoraConfig subclass + rank-allocation schedule).
  - trainable_tokens -> TrainableTokensConfig, for training only a few embedding rows standalone.
                   Note LoRA can also carry trainable tokens via its own trainable_token_indices,
                   so this type is only for the "no LoRA at all" case.

LoRA+ is NOT a tuner_type: it changes the optimizer's param groups, not the module graph. It is
requested through lorap_lr_ratio/lorap_emb_lr plus the 'lorap' optimizer, and so has no config here.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from swift.dev.configs import TunerConfig
    from swift.dev.model import TrainableModel

# tuner_types that are peft-backed and reachable from dev. Anything else fails fast in apply_tuner
# rather than being silently downgraded to LoRA.
SUPPORTED_TUNER_TYPES = ('lora', 'adalora', 'trainable_tokens')


def _resolve_target_modules(cfg: TunerConfig):
    """target_regex wins over target_modules; collapse the 1-element 'all-linear' list peft wants
    as a bare string."""
    target_modules = cfg.target_regex or cfg.target_modules
    # peft accepts a str ('all-linear'/regex) or a list of module names.
    if isinstance(target_modules, list) and len(target_modules) == 1 and target_modules[0] == 'all-linear':
        target_modules = 'all-linear'
    return target_modules


def _resolve_init_weights(cfg: TunerConfig):
    """swift spells it init_weights and accepts the strings 'true'/'false'; peft wants a bool there
    and keeps the real strategy names ('gaussian'/'pissa'/'olora'/...) as-is.

    NOTE: 'loftq' additionally needs a loftq_config, which TunerConfig does not model -- peft
    raises in that case, same as the legacy swift path.
    """
    init_weights = cfg.init_weights
    if isinstance(init_weights, str) and init_weights.lower() in {'true', 'false'}:
        return init_weights.lower() == 'true'
    return init_weights


def _lora_common_kwargs(cfg: TunerConfig) -> dict:
    """The LoraConfig fields AdaLoraConfig also takes (it subclasses LoraConfig)."""
    kwargs = dict(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.lora_bias,
        target_modules=_resolve_target_modules(cfg),
        modules_to_save=(cfg.modules_to_save or None),
        use_rslora=cfg.use_rslora,
        use_dora=cfg.use_dora,
        init_lora_weights=_resolve_init_weights(cfg),
    )
    # Both are newer peft additions and both default to "unset"; only pass them when actually
    # requested so an older peft (or a config that rejects them) is not handed an unknown kwarg.
    if cfg.target_parameters:
        # requires peft>=0.17.0
        kwargs['target_parameters'] = cfg.target_parameters
    if cfg.trainable_token_indices:
        kwargs['trainable_token_indices'] = cfg.trainable_token_indices
    return kwargs


def _build_adapter_config(cfg: TunerConfig, *, num_training_steps: Optional[int] = None):
    """Build the peft config for ``cfg.tuner_type``.

    task_type is intentionally NOT set: get_peft_model then returns a base PeftModel that forwards
    straight to the wrapped model, matching every twinkle cookbook. Setting task_type='CAUSAL_LM'
    yields PeftModelForCausalLM, whose forward reads base_model.config.model_type -- fine for a HF
    model, but the Megatron path's config is mcore's ModelConfig (only hf_model_type), so it raises
    AttributeError under forward_backward. Omitting it keeps both backends on the same, safe wrapper.

    Args:
        cfg: the TunerConfig.
        num_training_steps: total optimizer steps, required by adalora only (its rank-allocation
            schedule is expressed in steps).
    """
    tuner_type = cfg.tuner_type

    if tuner_type == 'lora':
        from peft import LoraConfig
        return LoraConfig(**_lora_common_kwargs(cfg))

    if tuner_type == 'adalora':
        from peft import AdaLoraConfig
        # AdaLoRA budgets its rank allocation over the whole run, so peft rejects total_step=None
        # outright. dev knows the step count only at build time, hence the explicit argument --
        # defaulting it to something arbitrary would silently change the pruning schedule.
        if not num_training_steps:
            raise ValueError('adalora needs the total training step count for its rank-allocation '
                             'schedule; pass num_training_steps to _build_adapter_config.')
        kwargs = _lora_common_kwargs(cfg)
        # init_r is the STARTING rank AdaLoRA prunes down to target_r, so it supersedes lora_rank.
        kwargs.pop('r', None)
        # AdaLoRA reimplements the LoRA forward and has no DoRA path.
        if cfg.use_dora:
            raise ValueError('use_dora is not supported by adalora; use tuner_type="lora" instead.')
        kwargs.pop('use_dora', None)
        return AdaLoraConfig(
            target_r=cfg.adalora_target_r,
            init_r=cfg.adalora_init_r,
            tinit=cfg.adalora_tinit,
            tfinal=cfg.adalora_tfinal,
            deltaT=cfg.adalora_deltaT,
            beta1=cfg.adalora_beta1,
            beta2=cfg.adalora_beta2,
            orth_reg_weight=cfg.adalora_orth_reg_weight,
            total_step=num_training_steps,
            **kwargs,
        )

    if tuner_type == 'trainable_tokens':
        from peft import TrainableTokensConfig
        # Standalone TrainableTokens spells the indices token_indices (LoRA's own passthrough field
        # is trainable_token_indices); it needs the embedding module as its target.
        if not cfg.trainable_token_indices:
            raise ValueError('tuner_type="trainable_tokens" requires trainable_token_indices.')
        kwargs = {}
        # Default to the standard HF embedding module name only when the user did not target one.
        if cfg.target_regex or cfg.target_modules != ['all-linear']:
            kwargs['target_modules'] = _resolve_target_modules(cfg)
        return TrainableTokensConfig(token_indices=cfg.trainable_token_indices, **kwargs)

    raise NotImplementedError(f'tuner_type={tuner_type!r} is not supported by dev; '
                              f'supported: {", ".join(SUPPORTED_TUNER_TYPES)}. '
                              f'(DoRA/rsLoRA are LoRA flags -- use tuner_type="lora" with '
                              f'use_dora/use_rslora; QLoRA is LoRA + a quantized base model via '
                              f'QuantizeConfig; LoRA+ is the "lorap" optimizer, not a tuner_type.)')


def apply_tuner(model: TrainableModel,
                tuner_cfg: TunerConfig,
                *,
                adapter_name: str = 'default',
                gradient_accumulation_steps: int = 1,
                num_training_steps: Optional[int] = None) -> None:
    adapter_config = _build_adapter_config(tuner_cfg, num_training_steps=num_training_steps)
    model.add_adapter_to_model(adapter_name, adapter_config, gradient_accumulation_steps=gradient_accumulation_steps)

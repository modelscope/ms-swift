"""Map a TunerConfig onto the peft adapter config the requested tuner needs.

Every tuner here is peft-backed, so ``add_adapter_to_model`` receives a plain peft config and the
model-side path is identical for all of them. What differs is only which config class is built and
which TunerConfig fields feed it:

  - lora        -> LoraConfig. Covers QLoRA (= 4bit base model, see QuantizeConfig, + plain LoRA),
                   DoRA (use_dora) and rsLoRA (use_rslora), which are LoRA *flags*, not separate
                   tuners -- there is deliberately no 'dora'/'rslora'/'qlora' tuner.
  - adalora     -> AdaLoraConfig (LoraConfig subclass + rank-allocation schedule).
  - trainable_tokens -> TrainableTokensConfig, for training only a few embedding rows standalone.
                   Note LoRA can also carry trainable tokens via its own trainable_token_indices,
                   so this type is only for the "no LoRA at all" case.

LoRA+ is NOT a tuner: it changes the optimizer's param groups, not the module graph. It is
requested through lorap_lr_ratio/lorap_emb_lr plus the 'lorap' optimizer, and so has no config here.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from swift.dev.config import TunerConfig
    from swift.dev.model import TrainableModel

# tuners that are peft-backed and reachable from dev. Anything else fails fast in apply_tuner
# rather than being silently downgraded to LoRA.
SUPPORTED_TUNER_TYPES = ('lora', 'adalora', 'trainable_tokens')

#: The classification-head module names a HF ``*ForSequenceClassification`` model may use, added to
#: ``modules_to_save`` for a seq_cls / reranker LoRA run so the freshly-initialized head is trained AND
#: saved (otherwise the checkpoint reloads with a random head). Both are listed because the name is
#: family-specific -- Qwen/LLaMA-style use ``score``, Bert/Deberta-style use ``classifier`` -- and it
#: cannot be probed from the built model here: under Ray ``apply_tuner`` runs on the driver against a
#: model PROXY with no modules, and the peft config (modules_to_save included) is serialized to the
#: workers before ``add_adapter_to_model`` builds the adapter on the real model. peft matches
#: modules_to_save by ``name.endswith(entry)`` and silently ignores an entry that matches nothing
#: (_set_trainable, strict_module_check=False), so listing both is safe: the one the family actually
#: has gets wrapped, the other is a no-op.
_SEQ_CLS_HEAD_MODULES = ('score', 'classifier')
#: task_types that ride a separate classification head needing the modules_to_save treatment above.
#: generative_reranker is deliberately excluded -- it keeps the CausalLM vocab head (no new module),
#: and the PPO value critic rides task_type='seq_cls' so it is covered by the same rule.
_HEAD_SAVING_TASK_TYPES = ('seq_cls', 'reranker')



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


def _lora_common_kwargs(cfg: TunerConfig, task_type: Optional[str] = None) -> dict:
    """The LoraConfig fields AdaLoraConfig also takes (it subclasses LoraConfig)."""
    # A seq_cls / reranker run builds a fresh classification head on top of the base model. LoRA only
    # wraps/trains target_modules, so without the head in modules_to_save it is neither trained nor
    # saved -- the checkpoint reloads with a randomly-initialized head. Add the candidate head names
    # (peft ignores the one this model family does not have). cfg.modules_to_save is copied, not
    # mutated, so a reused TunerConfig keeps whatever the user set.
    modules_to_save = list(cfg.modules_to_save or [])
    if task_type in _HEAD_SAVING_TASK_TYPES:
        modules_to_save += [name for name in _SEQ_CLS_HEAD_MODULES if name not in modules_to_save]
    kwargs = dict(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.lora_bias,
        target_modules=_resolve_target_modules(cfg),
        modules_to_save=(modules_to_save or None),
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


def _build_adapter_config(cfg: TunerConfig,
                          *,
                          num_training_steps: Optional[int] = None,
                          task_type: Optional[str] = None):
    """Build the peft config for ``cfg.tuner``.

    peft's own ``task_type`` FIELD is intentionally NOT set: get_peft_model then returns a base
    PeftModel that forwards straight to the wrapped model, matching every twinkle cookbook. Setting it
    to 'CAUSAL_LM' yields PeftModelForCausalLM, whose forward reads base_model.config.model_type --
    fine for a HF model, but the Megatron path's config is mcore's ModelConfig (only hf_model_type),
    so it raises AttributeError under forward_backward. Omitting it keeps both backends on the same,
    safe wrapper. The ``task_type`` ARGUMENT here is dev's task_type (seq_cls/reranker/...), used
    only to decide whether the classification head must be added to modules_to_save -- unrelated to
    the peft field.

    Args:
        cfg: the TunerConfig.
        num_training_steps: total optimizer steps, required by adalora only (its rank-allocation
            schedule is expressed in steps).
        task_type: dev's task_type; seq_cls/reranker add the classification head to modules_to_save.
    """
    tuner = cfg.tuner

    if tuner == 'lora':
        from peft import LoraConfig
        return LoraConfig(**_lora_common_kwargs(cfg, task_type=task_type))

    if tuner == 'adalora':
        from peft import AdaLoraConfig
        # AdaLoRA budgets its rank allocation over the whole run, so peft rejects total_step=None
        # outright. dev knows the step count only at build time, hence the explicit argument --
        # defaulting it to something arbitrary would silently change the pruning schedule.
        if not num_training_steps:
            raise ValueError('adalora needs the total training step count for its rank-allocation '
                             'schedule; pass num_training_steps to _build_adapter_config.')
        kwargs = _lora_common_kwargs(cfg, task_type=task_type)
        # init_r is the STARTING rank AdaLoRA prunes down to target_r, so it supersedes lora_rank.
        kwargs.pop('r', None)
        # AdaLoRA reimplements the LoRA forward and has no DoRA path.
        if cfg.use_dora:
            raise ValueError('use_dora is not supported by adalora; use tuner="lora" instead.')
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

    if tuner == 'trainable_tokens':
        from peft import TrainableTokensConfig
        # Standalone TrainableTokens spells the indices token_indices (LoRA's own passthrough field
        # is trainable_token_indices); it needs the embedding module as its target.
        if not cfg.trainable_token_indices:
            raise ValueError('tuner="trainable_tokens" requires trainable_token_indices.')
        kwargs = {}
        # Default to the standard HF embedding module name only when the user did not target one.
        if cfg.target_regex or cfg.target_modules != ['all-linear']:
            kwargs['target_modules'] = _resolve_target_modules(cfg)
        return TrainableTokensConfig(token_indices=cfg.trainable_token_indices, **kwargs)

    raise NotImplementedError(f'tuner={tuner!r} is not supported by dev; '
                              f'supported: {", ".join(SUPPORTED_TUNER_TYPES)}. '
                              f'(DoRA/rsLoRA are LoRA flags -- use tuner="lora" with '
                              f'use_dora/use_rslora; QLoRA is LoRA + a quantized base model via '
                              f'QuantizeConfig; LoRA+ is the "lorap" optimizer, not a tuner.)')


def apply_tuner(model: TrainableModel,
                tuner_cfg: TunerConfig,
                *,
                adapter_name: str = 'default',
                gradient_accumulation_steps: int = 1,
                num_training_steps: Optional[int] = None,
                task_type: Optional[str] = None) -> None:
    adapter_config = _build_adapter_config(tuner_cfg, num_training_steps=num_training_steps, task_type=task_type)
    model.add_adapter_to_model(adapter_name, adapter_config, gradient_accumulation_steps=gradient_accumulation_steps)

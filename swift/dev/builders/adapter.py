# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from swift.dev.config import TunerConfig
    from swift.dev.model import TrainableModel

# tuners that are peft-backed and reachable from dev. Anything else fails fast in apply_tuner
# rather than being silently downgraded to LoRA.
SUPPORTED_TUNER_TYPES = ("lora", "trainable_tokens")

#: The classification-head module names an HF ``*ForSequenceClassification`` model may use, added to
#: ``modules_to_save`` for a seq_cls / reranker LoRA run so the fresh head is trained AND saved. Both are
#: listed (family-specific: Qwen/LLaMA use ``score``, Bert/Deberta ``classifier``) because it cannot be
#: probed here: under Ray this runs on the driver against a model PROXY, and peft matches by
#: ``name.endswith(entry)`` with strict_module_check=False, so the miss is a safe no-op.
_SEQ_CLS_HEAD_MODULES = ("score", "classifier")
#: task_types that ride a separate classification head needing the modules_to_save treatment above.
#: generative_reranker keeps the CausalLM vocab head (no new module); the PPO value critic rides seq_cls.
_HEAD_SAVING_TASK_TYPES = ("seq_cls", "reranker")


def _resolve_target_modules(cfg: TunerConfig):
    """target_regex wins over target_modules; collapse the 1-element 'all-linear' list peft wants
    as a bare string."""
    target_modules = cfg.target_regex or cfg.target_modules
    # peft accepts a str ('all-linear'/regex) or a list of module names.
    if isinstance(target_modules, list) and len(target_modules) == 1 and target_modules[0] == "all-linear":
        target_modules = "all-linear"
    return target_modules


def is_base_modifying_init(init_weights) -> bool:
    """Whether a peft ``init_lora_weights`` strategy reads or rewrites the base weight at injection.

    The pissa family (pissa / pissa_niter_[N] / olora / corda / lora_ga) subtracts its init factor from
    the base weight, and 'loftq' quantises it -- an adapter trained with one only fits the *residual*
    base it produced. Consumed by the streaming merge's fallback routing and by the checkpoint saver's
    converted-adapter flow.
    """
    return isinstance(init_weights, str) and (
        init_weights.startswith(("pissa", "corda")) or init_weights in ("olora", "lora_ga", "loftq")
    )


def _resolve_init_weights(cfg: TunerConfig):
    """swift spells it init_weights and accepts the strings 'true'/'false'; peft wants a bool there
    and keeps the real strategy names ('gaussian'/'pissa'/'olora'/...) as-is."""
    init_weights = cfg.init_weights
    if isinstance(init_weights, str) and init_weights.lower() in {"true", "false"}:
        return init_weights.lower() == "true"
    # LoRA-GA needs the lora_ga_* parameters (rejected as unsupported) and a separate weight-init pass
    # dev does not run; peft would only raise a confusing error, so reject here with the actual reason.
    if isinstance(init_weights, str) and init_weights.lower().replace("-", "_") == "lora_ga":
        raise ValueError(
            "init_weights='lora-ga' (LoRA-GA) is not supported by dev: it needs the lora_ga_* parameters "
            "(rejected as unsupported) and a separate weight-initialization pass dev does not run. Use a "
            "supported init_weights ('true'/'false'/'gaussian'/'pissa'/'pissa_niter_[N]'/'olora')."
        )
    return init_weights


def _lora_common_kwargs(cfg: TunerConfig, task_type: Optional[str] = None) -> dict:
    """The LoraConfig kwargs shared by every supported tuner here."""
    # A seq_cls / reranker run builds a fresh classification head; LoRA wraps target_modules only, so
    # without the head in modules_to_save the checkpoint reloads with a random head. Copied, not mutated.
    modules_to_save = list(cfg.modules_to_save or [])
    # The 'all-norm'/'all-embedding' shorthands ride through verbatim and are expanded worker-side
    # (twinkle _expand_modules_to_save) against the real modules; megatron rejects 'all-norm'.
    if task_type in _HEAD_SAVING_TASK_TYPES:
        modules_to_save += [name for name in _SEQ_CLS_HEAD_MODULES if name not in modules_to_save]
    kwargs = {
        "r": cfg.lora_rank,
        "lora_alpha": cfg.lora_alpha,
        "lora_dropout": cfg.lora_dropout,
        "bias": cfg.lora_bias,
        "target_modules": _resolve_target_modules(cfg),
        "modules_to_save": (modules_to_save or None),
        "use_rslora": cfg.use_rslora,
        "use_dora": cfg.use_dora,
        "init_lora_weights": _resolve_init_weights(cfg),
    }
    # Newer peft additions that default to "unset"; only pass them when requested so an older peft
    # is not handed an unknown kwarg.
    if cfg.target_parameters:
        # requires peft>=0.17.0
        kwargs["target_parameters"] = cfg.target_parameters
    if cfg.trainable_token_indices:
        kwargs["trainable_token_indices"] = cfg.trainable_token_indices
    return kwargs


def _build_adapter_config(cfg: TunerConfig, *, task_type: Optional[str] = None):
    """Build the peft config for ``cfg.tuner``: lora -> LoraConfig (QLoRA/DoRA/rsLoRA are flags),
    trainable_tokens -> TrainableTokensConfig; anything else raises.

    peft's own ``task_type`` FIELD is intentionally NOT set: get_peft_model then returns a base PeftModel
    that forwards straight to the wrapped model, matching every twinkle cookbook -- the CAUSAL_LM wrapper
    reads base_model.config.model_type, which the Megatron path's mcore ModelConfig lacks. The
    ``task_type`` ARGUMENT is dev's (seq_cls/reranker/...), used only for the head-saving rule above.
    """
    tuner = cfg.tuner

    if tuner == "lora":
        from peft import LoraConfig

        return LoraConfig(**_lora_common_kwargs(cfg, task_type=task_type))

    if tuner == "trainable_tokens":
        from peft import TrainableTokensConfig

        # Standalone TrainableTokens spells the indices token_indices (LoRA's passthrough field is
        # trainable_token_indices); with no explicit target peft infers the input-embedding module.
        if not cfg.trainable_token_indices:
            raise ValueError('tuner="trainable_tokens" requires trainable_token_indices.')
        kwargs = {}
        if cfg.target_regex or cfg.target_modules != ["all-linear"]:
            kwargs["target_modules"] = _resolve_target_modules(cfg)
        return TrainableTokensConfig(token_indices=cfg.trainable_token_indices, **kwargs)

    raise NotImplementedError(
        f"tuner={tuner!r} is not supported by dev; "
        f"supported: {', '.join(SUPPORTED_TUNER_TYPES)}. "
        f'(DoRA/rsLoRA are LoRA flags -- use tuner="lora" with '
        f"use_dora/use_rslora; QLoRA is LoRA + a quantized base model via "
        f'QuantizeConfig; LoRA+ is the "lorap" optimizer, not a tuner.)'
    )


def apply_tuner(
    model: TrainableModel,
    tuner_cfg: TunerConfig,
    *,
    adapter_name: str = "default",
    gradient_accumulation_steps: int = 1,
    task_type: Optional[str] = None,
) -> None:
    adapter_config = _build_adapter_config(tuner_cfg, task_type=task_type)
    model.add_adapter_to_model(
        adapter_name,
        adapter_config,
        gradient_accumulation_steps=gradient_accumulation_steps,
        freeze_llm=tuner_cfg.freeze_llm,
        freeze_vit=tuner_cfg.freeze_vit,
        freeze_aligner=tuner_cfg.freeze_aligner,
    )


def configure_frozen_adapter(model: Any, template: Any, adapters: List[str], *, role: str) -> Any:
    """Load one frozen adapter onto an auxiliary model and bind its processor/template group."""
    from swift.dev.processor import InputProcessor, processor_kwargs

    if len(adapters) > 1:
        raise ValueError(f"{role}_adapters currently supports exactly one adapter per auxiliary model.")
    if adapters:
        adapter_name = f"{role}_adapter"
        model.add_adapter_to_model(adapter_name, adapters[0], is_trainable=False)
        model.set_processor(InputProcessor, adapter_name=adapter_name, **processor_kwargs(template))
        model.set_template(template, adapter_name=adapter_name)
    else:
        model.set_processor(InputProcessor, **processor_kwargs(template))
        model.set_template(template)
    return model

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from swift.dev.configs import TunerConfig
    from swift.dev.model import TrainableModel


def _build_lora_config(cfg: TunerConfig):
    """Map the TunerConfig (lora fields) onto a peft LoraConfig.

    task_type is intentionally NOT set: get_peft_model then returns a base PeftModel that forwards
    straight to the wrapped model, matching every twinkle cookbook. Setting task_type='CAUSAL_LM'
    yields PeftModelForCausalLM, whose forward reads base_model.config.model_type -- fine for a HF
    model, but the Megatron path's config is mcore's ModelConfig (only hf_model_type), so it raises
    AttributeError under forward_backward. Omitting it keeps both backends on the same, safe wrapper.
    """
    from peft import LoraConfig

    target_modules = cfg.target_regex or cfg.target_modules
    # peft accepts a str ('all-linear'/regex) or a list of module names.
    if isinstance(target_modules, list) and len(target_modules) == 1 and target_modules[0] == 'all-linear':
        target_modules = 'all-linear'
    return LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.lora_bias,
        target_modules=target_modules,
        modules_to_save=(cfg.modules_to_save or None),
        use_rslora=cfg.use_rslora,
        use_dora=cfg.use_dora,
    )


def apply_tuner(model: TrainableModel,
                tuner_cfg: TunerConfig,
                *,
                adapter_name: str = 'default',
                gradient_accumulation_steps: int = 1) -> None:
    if tuner_cfg.tuner_type != 'lora':
        raise NotImplementedError(
            f"apply_tuner minimal path supports tuner_type='lora' only, got {tuner_cfg.tuner_type!r}")
    lora_config = _build_lora_config(tuner_cfg)
    model.add_adapter_to_model(adapter_name, lora_config, gradient_accumulation_steps=gradient_accumulation_steps)

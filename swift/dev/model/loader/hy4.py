# Copyright (c) ModelScope Contributors. All rights reserved.
from .base import ModelArch, ModelLoader, register_model


@register_model
class HY4FlashLoader(ModelLoader):
    model_type = "hy_v4_flash"
    model_cls = "transformers.models.hy_v4_flash.modeling_hy_v4_flash:HYV4FlashForCausalLM"
    architectures = ["HYV4FlashForCausalLM"]
    template = "hy_v4_flash"
    is_moe = True
    models = ["Tencent-Hunyuan/Hy4-Flash"]

    def process_config(self, config):
        # Match the vendor training path: its checkpoint lookup-table vocabulary differs from the config.
        text_config = getattr(config, "text_config", config)
        text_config.engram_layer_ids = None
        config.output_router_logits = False
        return config

    def process_model(self, model):
        # Disabled engram layers must not remain in PEFT's FSDP wrapping candidates.
        names = {type(module).__name__ for module in model.modules()}
        for module in model.modules():
            no_split = getattr(module, "_no_split_modules", None)
            if no_split:
                module._no_split_modules = [name for name in no_split if name in names]
        return model


@register_model
class HY4FlashVLLoader(HY4FlashLoader):
    model_type = "hy_v4_flash_vl"
    model_cls = "transformers.models.hy_v4_flash.modeling_hy_v4_flash:HYV4FlashForConditionalGeneration"
    architectures = ["HYV4FlashForConditionalGeneration"]
    template = "hy_v4_flash_vl"
    is_multimodal = True
    tags = ["vision", "video"]
    models = ["Tencent-Hunyuan/Hy4-Flash-VL"]

    @property
    def model_arch(self) -> ModelArch:
        return ModelArch(language_model=["model.language_model", "lm_head"], vision_tower="model.vision_tower")

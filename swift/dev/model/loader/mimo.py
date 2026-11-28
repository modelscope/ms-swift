# Copyright (c) ModelScope Contributors. All rights reserved.
import os

from .base import ModelArch, ModelLoader, register_model


@register_model
class MiMoV2Loader(ModelLoader):
    model_type = "mimo_v2"
    model_cls = "transformers:AutoModelForCausalLM"
    trust_remote_code = True
    architectures = ["MiMoV2ForCausalLM"]
    template = "mimo_v2"
    is_multimodal = True
    is_moe = True
    mcore_model_type = 'mimo_v2'
    requires = ["transformers>=4.57", "qwen_vl_utils>0.0.6", "decord", "torchaudio"]
    tags = ["vision", "video", "audio"]
    models = [
        "XiaomiMiMo/MiMo-V2.5",
        "XiaomiMiMo/MiMo-V2.6-Flash-RL",
        "XiaomiMiMo/MiMo-V2.6-Pro-RL",
    ]

    @property
    def model_arch(self) -> ModelArch:
        return ModelArch(
            language_model=["model", "lm_head"],
            aligner="visual.merger",
            vision_tower=["visual", "audio_encoder"],
        )

    def process_model(self, model):
        audio_dir = os.path.join(self.model_info.model_dir, "audio_tokenizer")
        if (os.path.isfile(os.path.join(audio_dir, "model.safetensors"))
                and hasattr(model, "load_audio_tokenizer") and getattr(model, "audio_tokenizer", None) is None):
            import torch

            # Dataset encoders build the language model on meta; only the audio tokenizer needs real weights.
            with torch.device("cpu"):
                model.load_audio_tokenizer(audio_dir)
            model.audio_tokenizer.eval().requires_grad_(False)
        return model

    def process_tokenizer(self, processor):
        from qwen_vl_utils import vision_process

        from ._qwen_vl_utils import patch_qwen_vl_utils

        processor.global_vars = patch_qwen_vl_utils(vision_process)
        return processor

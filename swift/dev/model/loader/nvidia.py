# Copyright (c) ModelScope Contributors. All rights reserved.
from .base import ModelArch, ModelLoader, register_model


@register_model
class NemotronHLoader(ModelLoader):
    model_type = "nemotron_h"
    model_cls = "transformers:AutoModelForCausalLM"
    architectures = ["NemotronHForCausalLM"]
    template = "nemotron_h"
    is_moe = True
    requires = ["transformers>=5.0", "mamba-ssm", "causal-conv1d>=1.2.0"]
    models = [
        ("nv-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
         "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"),
        ("nv-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
         "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"),
    ]


@register_model
class NemotronHOmniLoader(NemotronHLoader):
    model_type = "nemotron_h_omni"
    trust_remote_code = True
    architectures = ["NemotronH_Omni_Reasoning_V3"]
    template = "nemotron_h_omni"
    is_multimodal = True
    requires = ["transformers>=5.16.1", "mamba-ssm", "causal-conv1d>=1.2.0"]
    tags = ["vision", "video"]
    models = [
        ("nv-community/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
         "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"),
        ("nv-community/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4",
         "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4"),
    ]

    @property
    def model_arch(self) -> ModelArch:
        return ModelArch(language_model="language_model", aligner="vision_projector", vision_tower="vision_model")

    def process_model(self, model):
        from ._nemotron_media import patch_nemotron_omni_media

        return patch_nemotron_omni_media(model)


@register_model
class NemoSpeechLMLoader(ModelLoader):
    model_type = "nemo_speechlm"
    model_cls = "transformers:AutoModelForMultimodalLM"
    architectures = ["Nemotron3_5TranscribeForConditionalGeneration"]
    template = "nemo_speechlm"
    is_multimodal = True
    is_moe = True
    requires = ["transformers>5.21", "mamba-ssm", "causal-conv1d>=1.2.0"]
    tags = ["audio"]
    models = [("nv-community/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B",
               "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B")]

    @property
    def model_arch(self) -> ModelArch:
        return ModelArch(language_model=["model.language_model", "lm_head"],
                         vision_tower="model.audio_tower", aligner="model.multi_modal_projector")

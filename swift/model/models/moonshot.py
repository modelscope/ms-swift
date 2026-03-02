# Copyright (c) ModelScope Contributors. All rights reserved.
from transformers import PreTrainedModel
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from swift.template import TemplateType
from ..constant import MLLMModelType
from ..model_arch import ModelArch
from ..model_meta import Model, ModelGroup, ModelMeta
from ..patcher import patch_get_input_embeddings
from ..register import ModelLoader, register_model


class KimiVLLoader(ModelLoader):

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        KimiVLPreTrainedModel = get_class_from_dynamic_module('modeling_kimi_vl.KimiVLPreTrainedModel', model_dir)
        try:
            del KimiVLPreTrainedModel._supports_sdpa
        except AttributeError:
            pass
        model = super().get_model(model_dir, *args, **kwargs)
        patch_get_input_embeddings(model.vision_tower, 'patch_embed')
        return model


register_model(
    ModelMeta(
        MLLMModelType.kimi_vl,
        [
            ModelGroup([
                Model('moonshotai/Kimi-VL-A3B-Instruct', 'moonshotai/Kimi-VL-A3B-Instruct'),
                Model('moonshotai/Kimi-VL-A3B-Thinking', 'moonshotai/Kimi-VL-A3B-Thinking'),
                Model('moonshotai/Kimi-VL-A3B-Thinking-2506', 'moonshotai/Kimi-VL-A3B-Thinking-2506'),
            ])
        ],
        KimiVLLoader,
        template=TemplateType.kimi_vl,
        model_arch=ModelArch.llava_hf_legacy,
        architectures=['KimiVLForConditionalGeneration'],
        requires=['transformers<4.49'],
    ))

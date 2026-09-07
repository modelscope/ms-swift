# Copyright (c) ModelScope Contributors. All rights reserved.

from transformers import PretrainedConfig, PreTrainedModel

from swift.template import TemplateType
from ..constant import MLLMModelType
from ..model_arch import ModelArch
from ..model_meta import Model, ModelGroup, ModelMeta
from ..register import ModelLoader, register_model


def _patch_separator_token_property(model) -> None:
    """Expose `separator_token` on the top-level model class as a BC property.

    `MossVLForConditionalGeneration` ships `visual`/`language_model` BC properties but not
    `separator_token`, so generic code resolving arch paths by attribute (e.g. the lora_llm
    tuner's unfreeze loop) gets None for it. Add the missing accessor, mirroring `visual`.
    """
    if not hasattr(type(model), 'separator_token'):
        type(model).separator_token = property(lambda self: self.model.separator_token)


class MossVLLoader(ModelLoader):

    def get_model(self, model_dir: str, config: PretrainedConfig, processor, model_kwargs) -> PreTrainedModel:
        model = super().get_model(model_dir, config, processor, model_kwargs)
        _patch_separator_token_property(model)
        return model


register_model(
    ModelMeta(
        MLLMModelType.moss_vl, [
            ModelGroup([
                Model('openmoss/MOSS-VL-Instruct-0708', 'OpenMOSS-Team/MOSS-VL-Instruct-0708'),
                Model('openmoss/MOSS-VL-Base-0708', 'OpenMOSS-Team/MOSS-VL-Base-0708'),
            ]),
        ],
        loader=MossVLLoader,
        template=TemplateType.moss_vl,
        model_arch=ModelArch.moss_vl,
        architectures=['MossVLForConditionalGeneration'],
        requires=['transformers>=4.57.1,<5', 'torchcodec', 'joblib'],
        tags=['vision', 'video']))

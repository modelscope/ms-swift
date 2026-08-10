# Copyright (c) ModelScope Contributors. All rights reserved.
from swift.template import TemplateType
from swift.utils import get_logger
from ..model_arch import ModelArch
from ..constant import MLLMModelType
from ..register import ModelLoader, register_model
from ..model_meta import Model, ModelGroup, ModelMeta

logger = get_logger()


class MiMoV2Loader(ModelLoader):
    is_moe = True


register_model(
    ModelMeta(
        MLLMModelType.mimo_v2, [
            ModelGroup([
                Model('XiaomiMiMo/MiMo-V2.5', 'XiaomiMiMo/MiMo-V2.5'),
            ], TemplateType.mimo_v2),
        ],
        MiMoV2Loader,
        model_arch=ModelArch.mimo_v2,
        architectures=['MiMoV2ForCausalLM'],
        requires=['transformers>=4.57', 'qwen_vl_utils>0.0.6', 'decord'],
        tags=['vision', 'video', 'audio']))
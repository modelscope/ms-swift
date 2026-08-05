# Copyright (c) ModelScope Contributors. All rights reserved.
import logging
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

register_model(
    ModelMeta(
        MLLMModelType.kimi_k25,
        [
            ModelGroup([
                Model('moonshotai/Kimi-K2.5', 'moonshotai/Kimi-K2.5'),
                Model('moonshotai/Kimi-K2.6', 'moonshotai/Kimi-K2.6'),
                Model('moonshotai/Kimi-K2.7-Code', 'moonshotai/Kimi-K2.7-Code'),
            ])
        ],
        template=TemplateType.kimi_k25,
        model_arch=ModelArch.kimi_k25,
        architectures=['KimiK25ForConditionalGeneration'],
        requires=['transformers>=4.57.1,<5.0.0'],
    ))


class KimiK3Loader(ModelLoader):

    def get_processor(self, model_dir: str, config):
        processor = super().get_processor(model_dir, config)
        # The remote-code tokenizer (tokenization_kimi.py) warns on every
        # `encode(..., add_special_tokens=False)` call, which spams streaming
        # inference; silence that logger.
        tokenizer = self._get_tokenizer(processor)
        logging.getLogger(type(tokenizer).__module__).setLevel(logging.ERROR)
        return processor


register_model(
    ModelMeta(
        MLLMModelType.kimi_k3,
        [ModelGroup([
            Model('moonshotai/Kimi-K3', 'moonshotai/Kimi-K3'),
        ])],
        KimiK3Loader,
        template=TemplateType.kimi_k3,
        # Same module layout as Kimi-K2.5: language_model / mm_projector / vision_tower.
        model_arch=ModelArch.kimi_k25,
        architectures=['KimiK3ForConditionalGeneration'],
        requires=['transformers>=5', 'tiktoken'],
        tags=['vision'],
    ))

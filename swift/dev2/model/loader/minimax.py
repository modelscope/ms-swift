# Copyright (c) ModelScope Contributors. All rights reserved.
# minimax_m2 variants, m3_vl, remote-code minimax_vl; legacy device_map/Quanto code obsolete here.
from __future__ import annotations

from transformers import AutoProcessor
from .base import ModelArch, ModelLoader, register_model


@register_model
class MinimaxM2Loader(ModelLoader):

    model_type = 'minimax_m2'
    model_cls = 'transformers:AutoModelForCausalLM'
    architectures = ['MiniMaxM2ForCausalLM']
    template = 'minimax_m2'
    requires = ['transformers==4.57.1']
    models = [('MiniMax/MiniMax-M2', 'MiniMaxAI/MiniMax-M2')]


@register_model
class MinimaxM2_1Loader(MinimaxM2Loader):

    model_type = 'minimax_m2_1'
    architectures = []
    template = 'minimax_m2_1'
    models = [('MiniMax/MiniMax-M2.1', 'MiniMaxAI/MiniMax-M2.1')]


@register_model
class MinimaxM2_5Loader(MinimaxM2Loader):

    model_type = 'minimax_m2_5'
    architectures = []
    template = 'minimax_m2_5'
    models = [('MiniMax/MiniMax-M2.5', 'MiniMaxAI/MiniMax-M2.5')]


@register_model
class MinimaxM2_7Loader(MinimaxM2Loader):

    model_type = 'minimax_m2_7'
    architectures = []
    template = 'minimax_m2_7'
    models = [('MiniMax/MiniMax-M2.7', 'MiniMaxAI/MiniMax-M2.7')]


@register_model
class MinimaxM3VLLoader(ModelLoader):
    """MiniMax-M3 vision-language. Loads via the generic ``AutoModelForImageTextToText`` (the model
    code ships in-tree, so no ``trust_remote_code`` for the model), but its *processor* is
    remote-code and must be built with ``trust_remote_code=True`` -- the one legacy split."""

    model_type = 'minimax_m3_vl'
    model_cls = 'transformers:AutoModelForImageTextToText'
    architectures = ['MiniMaxM3SparseForConditionalGeneration']
    template = 'minimax_m3_vl'
    tags = ['vision', 'video']
    is_multimodal = True
    models = [('MiniMax/MiniMax-M3', 'MiniMaxAI/MiniMax-M3')]

    @property
    def model_arch(self) -> ModelArch:
        return ModelArch(
            language_model=['model.language_model', 'lm_head'],
            aligner='model.multi_modal_projector',
            vision_tower='model.vision_tower',
        )

    def build_processor(self, model_dir, config, **kwargs):
        kwargs['trust_remote_code'] = True
        return AutoProcessor.from_pretrained(model_dir, **kwargs)


@register_model
class MiniMaxVLLoader(ModelLoader):
    """MiniMax-VL-01: remote-code ``MiniMaxVL01ForConditionalGeneration`` (inference-oriented; legacy
    warns "does not support training").

    ``build_model`` wraps the load in ``ignore_check_imports`` because the checkpoint's
    ``modeling_*.py`` declares imports transformers' checker rejects; ``build_processor``
    re-attaches the three helpers the template calls, looked up in the loaded dynamic module.
    """

    model_type = 'minimax_vl'
    model_cls = 'transformers:AutoModelForCausalLM'
    trust_remote_code = True
    architectures = ['MiniMaxVL01ForConditionalGeneration']
    template = 'minimax_vl'
    tags = ['vision']
    is_multimodal = True
    is_moe = True
    models = [('MiniMax/MiniMax-VL-01', 'MiniMaxAI/MiniMax-VL-01')]

    @property
    def model_arch(self) -> ModelArch:
        # Legacy declared no `model_arch`; these are the module names its own device_map code
        # addressed, so the multimodal partition is recoverable from it.
        return ModelArch(
            language_model=['language_model'],
            aligner=['multi_modal_projector'],
            vision_tower=['vision_tower'],
        )

    def build_model(self, model_dir: str, config, processor, **kwargs):
        with self.ignore_check_imports():
            return super().build_model(model_dir, config, processor, **kwargs)

    def build_processor(self, model_dir, config, **kwargs):
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        with self.ignore_check_imports():
            processor = super().build_processor(model_dir, config, **kwargs)
            for name in ('MiniMaxVL01ProcessorKwargs', 'get_hw_multiple_of', 'get_num_token'):
                setattr(processor, name, get_class_from_dynamic_module(f'processing_minimax_vl_01.{name}', model_dir))
        return processor

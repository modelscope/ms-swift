# Copyright (c) ModelScope Contributors. All rights reserved.
import transformers
from contextlib import contextmanager
from packaging import version
from transformers import PreTrainedModel
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from swift.template import TemplateType
from swift.utils import get_logger
from ..constant import MLLMModelType
from ..model_arch import ModelArch
from ..model_meta import Model, ModelGroup, ModelMeta
from ..patcher import patch_output_clone
from ..register import ModelLoader, register_model

logger = get_logger()


class Molmo2Loader(ModelLoader):

    @staticmethod
    @contextmanager
    def _patch_processor_optional_attributes_compat():
        """Restrict ProcessorMixin compat to Molmo2 processor loading only."""
        if version.parse(transformers.__version__) < version.parse('5.0.0.dev'):
            yield
            return
        try:
            from transformers.processing_utils import ProcessorMixin
        except Exception:
            yield
            return

        origin_init = ProcessorMixin.__init__

        def _patched_init(self, *args, **kwargs):
            optional_attributes = getattr(self, 'optional_attributes', None) or []
            optional_values = {}
            for key in optional_attributes:
                if key in {'chat_template', 'audio_tokenizer'}:
                    continue
                if key in kwargs:
                    optional_values[key] = kwargs.pop(key)

            origin_init(self, *args, **kwargs)

            for key in optional_attributes:
                if key in {'chat_template', 'audio_tokenizer'}:
                    continue
                if key in optional_values:
                    setattr(self, key, optional_values[key])
                elif not hasattr(self, key):
                    setattr(self, key, None)

        ProcessorMixin.__init__ = _patched_init
        try:
            yield
        finally:
            ProcessorMixin.__init__ = origin_init

    @staticmethod
    def _patch_vision_pooling_attention(model: PreTrainedModel) -> None:
        inner_model = getattr(model, 'model', None)
        if inner_model is None:
            return

        vision_backbone = getattr(inner_model, 'vision_backbone', None)
        if vision_backbone is None:
            return
        pooling = getattr(vision_backbone, 'image_pooling_2d', None)
        if pooling is None or getattr(pooling, 'attn_implementation', None) != 'flash_attention_2':
            return

        pooling.attn_implementation = 'sdpa'
        adapter_config = getattr(vision_backbone, 'adapter_config', None)
        if adapter_config is not None and getattr(adapter_config, 'attn_implementation', None) == 'flash_attention_2':
            adapter_config.attn_implementation = 'sdpa'
        logger.info('Set Molmo2 vision_backbone.image_pooling_2d attention to `sdpa` to avoid '
                    'flash-attn varlen failures on padded video batches.')

    def get_processor(self, model_dir, config):
        with self._patch_processor_optional_attributes_compat():
            return super().get_processor(model_dir, config)

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        from transformers import AutoModelForImageTextToText
        model_cls = get_class_from_dynamic_module('modeling_molmo2.Molmo2ForConditionalGeneration', model_dir)
        no_split_modules = getattr(model_cls, '_no_split_modules', []) or []
        if 'MolmoSequentialBlock' not in no_split_modules:
            model_cls._no_split_modules = no_split_modules + ['MolmoSequentialBlock']
        self.auto_model_cls = self.auto_model_cls or AutoModelForImageTextToText
        model = super().get_model(model_dir, *args, **kwargs)
        patch_output_clone(model.model.transformer.wte)
        self._patch_vision_pooling_attention(model)
        return model


register_model(
    ModelMeta(
        MLLMModelType.molmo2,
        [
            ModelGroup([
                Model('LLM-Research/Molmo2-4B', 'allenai/Molmo2-4B'),
                Model('LLM-Research/Molmo2-8B', 'allenai/Molmo2-8B'),
                Model('LLM-Research/Molmo2-O-7B', 'allenai/Molmo2-O-7B'),
            ]),
        ],
        Molmo2Loader,
        template=TemplateType.molmo2,
        model_arch=ModelArch.molmo,
        architectures=['Molmo2ForConditionalGeneration'],
        tags=['vision', 'video'],
        requires=['transformers>=4.57.1', 'decord'],
    ))

# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import sys
from collections import OrderedDict
from typing import Any, Dict

from transformers import PretrainedConfig, PreTrainedModel
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from swift.template import TemplateType
from swift.utils import Processor, get_logger, git_clone_github
from ..constant import MLLMModelType
from ..model_arch import ModelArch
from ..model_meta import Model, ModelGroup, ModelMeta
from ..register import ModelLoader, register_model
from ..utils import use_submodel_func
from .qwen import QwenLoader

logger = get_logger()


class MplugOwl2Loader(ModelLoader):

    def _get_model(self, model_dir: str, vocab_size, *args, **kwargs) -> PreTrainedModel:
        local_repo_path = self.local_repo_path
        if not local_repo_path:
            local_repo_path = git_clone_github('https://github.com/X-PLUG/mPLUG-Owl')
        local_repo_path = os.path.join(local_repo_path, 'mPLUG-Owl2')
        sys.path.append(local_repo_path)
        # register
        # https://github.com/X-PLUG/mPLUG-Owl/blob/main/mPLUG-Owl2/mplug_owl2/model/modeling_mplug_owl2.py#L447
        from mplug_owl2 import MPLUGOwl2LlamaForCausalLM
        if vocab_size is not None:
            config.vocab_size = vocab_size
        model = super().get_model(model_dir, *args, **kwargs)
        logger.info('Please ignore the unimported warning.')
        return model

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        return self._get_model(model_dir, None, *args, **kwargs)

    def get_processor(self, model_dir: str, config: PretrainedConfig) -> Processor:
        from transformers.models.clip.image_processing_clip import CLIPImageProcessor
        processor = CLIPImageProcessor.from_pretrained(model_dir)
        return processor


register_model(
    ModelMeta(
        MLLMModelType.mplug_owl2, [ModelGroup([
            Model('iic/mPLUG-Owl2', 'MAGAer13/mplug-owl2-llama2-7b'),
        ])],
        MplugOwl2Loader,
        template=TemplateType.mplug_owl2,
        model_arch=ModelArch.mplug_owl2,
        requires=['transformers<4.35', 'icecream'],
        tags=['vision']), )


class MplugOwl2_1Loader(QwenLoader, MplugOwl2Loader):

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        return self._get_model(model_dir, 151851, *args, **kwargs)


register_model(
    ModelMeta(
        MLLMModelType.mplug_owl2_1, [ModelGroup([
            Model('iic/mPLUG-Owl2.1', 'Mizukiluke/mplug_owl_2_1'),
        ])],
        MplugOwl2_1Loader,
        template=TemplateType.mplug_owl2,
        model_arch=ModelArch.mplug_owl2_1,
        requires=['transformers<4.35', 'icecream'],
        tags=['vision']))


class MplugOwl3Loader(ModelLoader):

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        get_class_from_dynamic_module('configuration_hyper_qwen2.HyperQwen2Config', model_dir)
        model_cls = get_class_from_dynamic_module('modeling_mplugowl3.mPLUGOwl3Model', model_dir)
        model_cls._no_split_modules = ['SiglipEncoderLayer']
        model = super().get_model(model_dir, *args, **kwargs)
        func_list = ['generate', 'forward']
        use_submodel_func(model, 'language_model', func_list)

        all_hooks = OrderedDict()
        hooks_with_kwargs = OrderedDict()

        def append_hooks(sub_module, inc_id=0):
            for id, hook in sub_module._forward_hooks.items():
                all_hooks[inc_id] = hook
                if id in sub_module._forward_hooks_with_kwargs:
                    hooks_with_kwargs[inc_id] = sub_module._forward_hooks_with_kwargs[id]
                inc_id += 1
            return inc_id

        inc_id = append_hooks(model.language_model)
        append_hooks(model, inc_id)
        model._forward_hooks = all_hooks
        model._forward_hooks_with_kwargs = hooks_with_kwargs
        return model

    def _get_model_processor(self, model_dir, config):
        model, tokenizer = super()._get_model_processor(model_dir, config)
        if model:
            tokenizer = model.init_processor(tokenizer)
        return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.mplug_owl3, [
            ModelGroup([
                Model('iic/mPLUG-Owl3-1B-241014', 'mPLUG/mPLUG-Owl3-1B-241014'),
                Model('iic/mPLUG-Owl3-2B-241014', 'mPLUG/mPLUG-Owl3-2B-241014'),
                Model('iic/mPLUG-Owl3-7B-240728', 'mPLUG/mPLUG-Owl3-7B-240728'),
            ]),
        ],
        MplugOwl3Loader,
        template=TemplateType.mplug_owl3,
        architectures=['mPLUGOwl3Model'],
        model_arch=ModelArch.mplug_owl3,
        requires=['transformers>=4.36', 'icecream', 'decord'],
        tags=['vision', 'video']))

register_model(
    ModelMeta(
        MLLMModelType.mplug_owl3_241101, [
            ModelGroup([
                Model('iic/mPLUG-Owl3-7B-241101', 'mPLUG/mPLUG-Owl3-7B-241101'),
            ]),
        ],
        MplugOwl3Loader,
        template=TemplateType.mplug_owl3_241101,
        architectures=['mPLUGOwl3Model'],
        model_arch=ModelArch.mplug_owl3,
        requires=['transformers>=4.36', 'icecream'],
        tags=['vision', 'video']))


class DocOwl2Loader(ModelLoader):

    def _get_model_processor(self, model_dir, config):
        model, tokenizer = super()._get_model_processor(model_dir, config)
        if model:
            tokenizer = model.init_processor(tokenizer, basic_image_size=504, crop_anchors='grid_12')
        return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.doc_owl2, [
            ModelGroup([
                Model('iic/DocOwl2', 'mPLUG/DocOwl2'),
            ]),
        ],
        DocOwl2Loader,
        template=TemplateType.doc_owl2,
        architectures=['mPLUGDocOwl2'],
        model_arch=ModelArch.doc_owl2,
        requires=['transformers>=4.36', 'icecream'],
        tags=['vision']))

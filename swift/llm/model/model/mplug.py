# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys
from functools import partial
from typing import Any, Dict

from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from swift.llm import TemplateType
from swift.utils import get_logger
from ..constant import MLLMModelType
from ..model_arch import ModelArch
from ..register import Model, ModelGroup, ModelMeta, get_model_tokenizer_with_flash_attn, register_model
from ..utils import ModelInfo, git_clone_github, use_submodel_func
from .qwen import get_model_tokenizer_qwen

logger = get_logger()


def get_model_tokenizer_mplug_owl2(model_dir: str,
                                   model_info: ModelInfo,
                                   model_kwargs: Dict[str, Any],
                                   load_model: bool = True,
                                   **kwargs):
    local_repo_path = kwargs.get('local_repo_path')
    if not local_repo_path:
        local_repo_path = git_clone_github('https://github.com/X-PLUG/mPLUG-Owl')
    local_repo_path = os.path.join(local_repo_path, 'mPLUG-Owl2')
    sys.path.append(local_repo_path)

    # register
    # https://github.com/X-PLUG/mPLUG-Owl/blob/main/mPLUG-Owl2/mplug_owl2/model/modeling_mplug_owl2.py#L447
    from mplug_owl2 import MPLUGOwl2LlamaForCausalLM
    from transformers.models.clip.image_processing_clip import CLIPImageProcessor
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    vocab_size = kwargs.pop('vocab_size', None)
    if vocab_size is not None:
        model_config.vocab_size = vocab_size
    get_model_tokenizer_function = kwargs.pop('get_model_tokenizer_function', get_model_tokenizer_with_flash_attn)
    model, tokenizer = get_model_tokenizer_function(
        model_dir, model_info, model_kwargs, load_model, model_config=model_config, **kwargs)
    logger.info('Please ignore the unimported warning.')
    processor = CLIPImageProcessor.from_pretrained(model_dir)
    processor.tokenizer = tokenizer
    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.mplug_owl2, [ModelGroup([
            Model('iic/mPLUG-Owl2', 'MAGAer13/mplug-owl2-llama2-7b'),
        ])],
        TemplateType.mplug_owl2,
        get_model_tokenizer_mplug_owl2,
        model_arch=ModelArch.mplug_owl2,
        requires=['transformers<4.35', 'icecream'],
        tags=['vision']), )

register_model(
    ModelMeta(
        MLLMModelType.mplug_owl2_1, [ModelGroup([
            Model('iic/mPLUG-Owl2.1', 'Mizukiluke/mplug_owl_2_1'),
        ])],
        TemplateType.mplug_owl2,
        partial(
            get_model_tokenizer_mplug_owl2, vocab_size=151851, get_model_tokenizer_function=get_model_tokenizer_qwen),
        model_arch=ModelArch.mplug_owl2_1,
        requires=['transformers<4.35', 'icecream'],
        tags=['vision']))


def get_model_tokenizer_mplug_owl3(model_dir: str,
                                   model_info: ModelInfo,
                                   model_kwargs: Dict[str, Any],
                                   load_model: bool = True,
                                   **kwargs):
    get_class_from_dynamic_module('configuration_hyper_qwen2.HyperQwen2Config', model_dir)
    model_cls = get_class_from_dynamic_module('modeling_mplugowl3.mPLUGOwl3Model', model_dir)
    model_cls._no_split_modules = ['SiglipEncoderLayer']
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)
    processor = model.init_processor(tokenizer)
    if model is not None:
        func_list = ['generate', 'forward']
        use_submodel_func(model, 'language_model', func_list)
    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.mplug_owl3, [
            ModelGroup([
                Model('iic/mPLUG-Owl3-1B-241014', 'mPLUG/mPLUG-Owl3-1B-241014'),
                Model('iic/mPLUG-Owl3-2B-241014', 'mPLUG/mPLUG-Owl3-2B-241014'),
                Model('iic/mPLUG-Owl3-7B-240728', 'mPLUG/mPLUG-Owl3-7B-240728'),
            ]),
        ],
        TemplateType.mplug_owl3,
        get_model_tokenizer_mplug_owl3,
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
        TemplateType.mplug_owl3_241101,
        get_model_tokenizer_mplug_owl3,
        architectures=['mPLUGOwl3Model'],
        model_arch=ModelArch.mplug_owl3,
        requires=['transformers>=4.36', 'icecream'],
        tags=['vision', 'video']))


def get_model_tokenizer_doc_owl2(model_dir: str,
                                 model_info: ModelInfo,
                                 model_kwargs: Dict[str, Any],
                                 load_model: bool = True,
                                 **kwargs):
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)
    processor = model.init_processor(tokenizer, basic_image_size=504, crop_anchors='grid_12')
    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.doc_owl2, [
            ModelGroup([
                Model('iic/DocOwl2', 'mPLUG/DocOwl2'),
            ]),
        ],
        TemplateType.doc_owl2,
        get_model_tokenizer_doc_owl2,
        architectures=['mPLUGDocOwl2'],
        model_arch=ModelArch.doc_owl2,
        requires=['transformers>=4.36', 'icecream'],
        tags=['vision']))

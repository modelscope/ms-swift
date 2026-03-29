# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import sys
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedModel
from typing import Any, Dict

from swift.template import TemplateType
from swift.utils import Processor, get_logger, git_clone_github
from ..constant import LLMModelType, MLLMModelType
from ..model_arch import ModelArch
from ..model_meta import Model, ModelGroup, ModelMeta
from ..register import ModelLoader, register_model

logger = get_logger()


class YiVLLoader(ModelLoader):

    def get_config(self, model_dir: str) -> PretrainedConfig:
        local_repo_path = self.local_repo_path
        if not local_repo_path:
            local_repo_path = git_clone_github('https://github.com/01-ai/Yi')
        sys.path.append(os.path.join(local_repo_path, 'VL'))
        from llava.model import LlavaConfig
        config = LlavaConfig.from_pretrained(model_dir)
        mm_vision_tower = config.mm_vision_tower
        config.mm_vision_tower = os.path.join(model_dir, *mm_vision_tower.rsplit('/', maxsplit=2)[-2:])
        config.attention_dropout = 0.
        if not hasattr(config, 'max_sequence_length'):
            config.max_sequence_length = 2048
        return config

    def get_processor(self, model_dir: str, config: PretrainedConfig) -> Processor:
        return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)

    def get_model(self, model_dir: str, config, processor, **kwargs) -> PreTrainedModel:
        from llava.model import LlavaLlamaForCausalLM
        from llava.model.constants import key_info
        key_info['model_path'] = model_dir
        self.auto_model_cls = self.auto_model_cls or LlavaLlamaForCausalLM
        model = super().get_model(model_dir, config, processor, **kwargs)
        vision_tower = model.get_vision_tower()
        vision_tower.load_model()
        vision_tower.to(device=model.device, dtype=config.torch_dtype)

        logger.info('Please ignore the above warning.')
        logger.info('Loading the parameters of vision_tower...')
        model.resize_token_embeddings(len(processor))
        processor.image_processor = vision_tower.image_processor
        return model


register_model(
    ModelMeta(
        MLLMModelType.yi_vl,
        [
            ModelGroup([
                Model('01ai/Yi-VL-6B', '01-ai/Yi-VL-6B'),
                Model('01ai/Yi-VL-34B', '01-ai/Yi-VL-34B'),
            ], ),
        ],
        YiVLLoader,
        template=TemplateType.yi_vl,
        model_arch=ModelArch.llava_llama,
        architectures=['LlavaLlamaForCausalLM'],
        requires=['transformers>=4.34'],
        tags=['vision'],
    ))

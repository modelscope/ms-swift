# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import sys
from typing import Any, Dict

from transformers import AutoTokenizer, PretrainedConfig, PreTrainedModel

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

register_model(
    ModelMeta(
        LLMModelType.yi,
        [  # yi
            ModelGroup([
                Model('01ai/Yi-6B', '01-ai/Yi-6B'),
                Model('01ai/Yi-6B-200K', '01-ai/Yi-6B-200K'),
                Model('01ai/Yi-6B-Chat', '01-ai/Yi-6B-Chat'),
                Model('01ai/Yi-6B-Chat-4bits', '01-ai/Yi-6B-Chat-4bits'),
                Model('01ai/Yi-6B-Chat-8bits', '01-ai/Yi-6B-Chat-8bits'),
                Model('01ai/Yi-9B', '01-ai/Yi-9B'),
                Model('01ai/Yi-9B-200K', '01-ai/Yi-9B-200K'),
                Model('01ai/Yi-34B', '01-ai/Yi-34B'),
                Model('01ai/Yi-34B-200K', '01-ai/Yi-34B-200K'),
                Model('01ai/Yi-34B-Chat', '01-ai/Yi-34B-Chat'),
                Model('01ai/Yi-34B-Chat-4bits', '01-ai/Yi-34B-Chat-4bits'),
                Model('01ai/Yi-34B-Chat-8bits', '01-ai/Yi-34B-Chat-8bits'),
            ], TemplateType.chatml),
            # yi1.5
            ModelGroup([
                Model('01ai/Yi-1.5-6B', '01-ai/Yi-1.5-6B'),
                Model('01ai/Yi-1.5-6B-Chat', '01-ai/Yi-1.5-6B-Chat'),
                Model('01ai/Yi-1.5-9B', '01-ai/Yi-1.5-9B'),
                Model('01ai/Yi-1.5-9B-Chat', '01-ai/Yi-1.5-9B-Chat'),
                Model('01ai/Yi-1.5-9B-Chat-16K', '01-ai/Yi-1.5-9B-Chat-16K'),
                Model('01ai/Yi-1.5-34B', '01-ai/Yi-1.5-34B'),
                Model('01ai/Yi-1.5-34B-Chat', '01-ai/Yi-1.5-34B-Chat'),
                Model('01ai/Yi-1.5-34B-Chat-16K', '01-ai/Yi-1.5-34B-Chat-16K'),
            ], TemplateType.chatml),
            # yi1.5-quant
            ModelGroup([
                Model('AI-ModelScope/Yi-1.5-6B-Chat-GPTQ', 'modelscope/Yi-1.5-6B-Chat-GPTQ'),
                Model('AI-ModelScope/Yi-1.5-6B-Chat-AWQ', 'modelscope/Yi-1.5-6B-Chat-AWQ'),
                Model('AI-ModelScope/Yi-1.5-9B-Chat-GPTQ', 'modelscope/Yi-1.5-9B-Chat-GPTQ'),
                Model('AI-ModelScope/Yi-1.5-9B-Chat-AWQ', 'modelscope/Yi-1.5-9B-Chat-AWQ'),
                Model('AI-ModelScope/Yi-1.5-34B-Chat-GPTQ', 'modelscope/Yi-1.5-34B-Chat-GPTQ'),
                Model('AI-ModelScope/Yi-1.5-34B-Chat-AWQ', 'modelscope/Yi-1.5-34B-Chat-AWQ'),
            ], TemplateType.chatml),
            ModelGroup([
                Model('01ai/Yi-Coder-1.5B', '01-ai/Yi-Coder-1.5B'),
                Model('01ai/Yi-Coder-9B', '01-ai/Yi-Coder-9B'),
                Model('01ai/Yi-Coder-1.5B-Chat', '01-ai/Yi-Coder-1.5B-Chat'),
                Model('01ai/Yi-Coder-9B-Chat', '01-ai/Yi-Coder-9B-Chat'),
            ],
                       TemplateType.yi_coder,
                       tags=['coding']),
            ModelGroup([
                Model('SUSTC/SUS-Chat-34B', 'SUSTech/SUS-Chat-34B'),
            ], TemplateType.sus),
        ],
        architectures=['LlamaForCausalLM'],
        model_arch=ModelArch.llama,
    ))

# Copyright (c) Alibaba, Inc. and its affiliates.
from transformers import AutoConfig
from transformers import AutoModel
from swift.utils import get_logger
from ..constant import LLMModelType
from ..register import Model, ModelGroup, ModelMeta, get_model_tokenizer_from_local, register_model

logger = get_logger()


def get_model_tokenizer_reward_model(model_dir, *args, **kwargs):
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    if 'AutoModel' in (getattr(model_config, 'auto_map', None) or {}):
        kwargs['automodel_class'] = AutoModel
    return get_model_tokenizer_from_local(model_dir, *args, **kwargs)


register_model(
    ModelMeta(
        LLMModelType.reward_model, [
            ModelGroup([
                Model('Qwen/Qwen2.5-Math-RM-72B', 'Qwen/Qwen2.5-Math-RM-72B'),
                Model('Qwen/Qwen2-Math-RM-72B', 'Qwen/Qwen2-Math-RM-72B'),
            ]),
            ModelGroup([
                Model('Shanghai_AI_Laboratory/internlm2-1_8b-reward', 'internlm/internlm2-1_8b-reward'),
                Model('Shanghai_AI_Laboratory/internlm2-7b-reward', 'internlm/internlm2-7b-reward'),
                Model('Shanghai_AI_Laboratory/internlm2-20b-reward', 'internlm/internlm2-20b-reward'),
            ]),
        ],
        None,
        get_model_tokenizer_reward_model,
        tags=['reward_model']))

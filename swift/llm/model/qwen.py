from typing import Any, Dict

import torch
from transformers import AutoConfig, BitsAndBytesConfig

from swift.llm import TemplateType
from swift.utils import get_logger
from .constant import ModelType
from .model import Model, ModelGroup, TemplateGroup, get_model_tokenizer_from_repo, register_model

logger = get_logger()
dtype_mapping = {torch.float16: 'fp16', torch.bfloat16: 'bf16', torch.float32: 'fp32'}



def get_model_tokenizer_qwen(model_dir: str,
                             torch_dtype: torch.dtype,
                             model_kwargs: Dict[str, Any],
                             load_model: bool = True,
                             model_config=None,
                             **kwargs):
    if model_config is None:
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    if torch_dtype is not None:
        k_true = dtype_mapping[torch_dtype]
        for k in dtype_mapping.values():
            v = False
            if k == k_true:
                v = True
            setattr(model_config, k, v)

    if model_kwargs.get('quantization_config') is None or not isinstance(model_kwargs['quantization_config'],
                                                                         BitsAndBytesConfig):
        # not (quantization + bnb)
        torch_dtype = None
    use_flash_attn = kwargs.pop('use_flash_attn', None)
    if use_flash_attn is None:
        use_flash_attn = 'auto'
    model_config.use_flash_attn = use_flash_attn
    model, tokenizer = get_model_tokenizer_from_repo(
        model_dir, torch_dtype, model_kwargs, load_model, model_config=model_config, **kwargs)
    try:
        # fix mp+ddp bug
        model.transformer.registered_causal_mask = model.transformer.registered_causal_mask.cuda()
        logger.info('registered_causal_mask to cuda')
    except AttributeError:
        pass
    return model, tokenizer



register_model(
    ModelType.qwen,
    'QWenLMHeadModel',
    [
        ModelGroup([
            Model('qwen/Qwen-1_8B', 'Qwen/Qwen-1_8B'),
            Model('qwen/Qwen-7B', 'Qwen/Qwen-7B'),
            Model('qwen/Qwen-14B', 'Qwen/Qwen-14B'),
            Model('qwen/Qwen-72B', 'Qwen/Qwen-72B'),
            Model('qwen/Qwen-1_8B-Chat', 'Qwen/Qwen-1_8B-Chat'),
            Model('qwen/Qwen-7B-Chat', 'Qwen/Qwen-7B-Chat'),
            Model('qwen/Qwen-14B-Chat', 'Qwen/Qwen-14B-Chat'),
            Model('qwen/Qwen-72B-Chat', 'Qwen/Qwen-72B-Chat'),
        ], TemplateGroup(TemplateType.qwen)),
        ModelGroup([
            Model('TongyiFinance/Tongyi-Finance-14B'),
            Model('TongyiFinance/Tongyi-Finance-14B-Chat', 'jxy/Tongyi-Finance-14B-Chat'),
        ], TemplateGroup(TemplateType.qwen), tags=['financial']),
        ModelGroup([
            Model('codefuse-ai/CodeFuse-QWen-14B', 'codefuse-ai/CodeFuse-QWen-14B'),
        ], TemplateGroup(TemplateType.codefuse), tags=['coding'])
    ],
    get_model_tokenizer_qwen,
    support_flash_attn=True,
    support_vllm=True,
    support_lmdeploy=True)

register_model(
    ModelType.modelscope_agent,
    'QWenLMHeadModel',
    [ModelGroup([
        Model('iic/ModelScope-Agent-7B'),
        Model('iic/ModelScope-Agent-14B'),
    ], TemplateGroup(TemplateType.modelscope_agent))],
    support_flash_attn=True,
    support_vllm=False)


register_model(
    ModelType.qwen_audio,
    'QWenLMHeadModel',
    [ModelGroup([
        Model('qwen/Qwen-Audio', 'Qwen/Qwen-Audio'),
        Model('qwen/Qwen-Audio-Chat', 'Qwen/Qwen-Audio-Chat'),
    ], TemplateGroup(TemplateType.qwen_audio, TemplateType.qwen_audio_generation))],
    support_flash_attn=True)
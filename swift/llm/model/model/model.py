# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect
import os
import sys
from contextlib import contextmanager
from functools import partial, update_wrapper, wraps
from types import MethodType
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.utils.checkpoint
import transformers
from accelerate.utils import find_device
from modelscope import (AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig,
                        snapshot_download)
from modelscope.hub.utils.utils import get_cache_dir
from packaging import version
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.auto.tokenization_auto import get_tokenizer_config

from swift import get_logger
from swift.llm.template.template import TemplateType, get_env_args
from swift.llm.utils import to_device
from swift.utils import get_dist_setting, safe_ddp_context, subprocess_run, use_torchacc
from ..patcher import patch_fixed_device, patch_output_clone, patch_output_to_input_device, patch_rope_scaling

logger = get_logger()


@register_model(
    ModelType.cogvlm2_video_13b_chat,
    'ZhipuAI/cogvlm2-video-llama3-chat',
    template=TemplateType.cogvlm2_video,
    support_gradient_checkpointing=False,
    requires=['decord', 'pytorchvideo', 'transformers>=4.42'],
    placeholder_tokens=['<|reserved_special_token_0|>'],
    tags=['multi-modal', 'vision', 'video'],
    hf_model_id='THUDM/cogvlm2-video-llama3-chat')
@register_model(
    ModelType.cogvlm2_en_19b_chat,
    'ZhipuAI/cogvlm2-llama3-chat-19B',
    template=TemplateType.cogvlm,
    support_gradient_checkpointing=False,
    support_lmdeploy=True,
    requires=['transformers<4.42'],
    placeholder_tokens=['<|reserved_special_token_0|>'],
    tags=['multi-modal', 'vision'],
    hf_model_id='THUDM/cogvlm2-llama3-chat-19B')
@register_model(
    ModelType.cogvlm2_19b_chat,
    'ZhipuAI/cogvlm2-llama3-chinese-chat-19B',
    template=TemplateType.cogvlm,
    support_gradient_checkpointing=False,
    support_lmdeploy=True,
    requires=['transformers<4.42'],
    placeholder_tokens=['<|reserved_special_token_0|>'],
    tags=['multi-modal', 'vision'],
    hf_model_id='THUDM/cogvlm2-llama3-chinese-chat-19B')
def get_model_tokenizer_cogvlm2(*args, **kwargs):
    model, tokenizer = get_model_tokenizer_from_repo(*args, **kwargs)
    if model is not None:
        # fix device map 4
        for layer in model.model.vision.transformer.layers:
            patch_output_to_input_device(layer.mlp)
            patch_output_to_input_device(layer.post_attention_layernorm)

        device = next(model.model.vision.linear_proj.parameters()).device
        model.model.vision.boi.data = model.model.vision.boi.to(device)
        model.model.vision.eoi.data = model.model.vision.eoi.to(device)
        tokenizer.build_conversation_input_ids = MethodType(model.build_conversation_input_ids, tokenizer)
    return model, tokenizer


@register_model(
    ModelType.llava_llama3_8b_v1_1,
    'AI-ModelScope/llava-llama-3-8b-v1_1-transformers',
    template=TemplateType.llava_llama_instruct,
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.36'],
    tags=['multi-modal', 'vision'],
    hf_model_id='xtuner/llava-llama-3-8b-v1_1-transformers')
def get_model_tokenizer_llava_llama(model_dir: str,
                                    torch_dtype: torch.dtype,
                                    model_kwargs: Dict[str, Any],
                                    load_model: bool = True,
                                    **kwargs):
    from transformers import LlavaForConditionalGeneration, LlavaConfig, AutoProcessor

    model_config = LlavaConfig.from_pretrained(model_dir)
    processor = AutoProcessor.from_pretrained(model_dir)
    model, tokenizer = get_model_tokenizer_with_flash_attn(
        model_dir,
        torch_dtype,
        model_kwargs,
        load_model,
        model_config=model_config,
        automodel_class=LlavaForConditionalGeneration,
        **kwargs)
    tokenizer.processor = processor
    return model, tokenizer


def patch_baichuan2_lm_head_forward(self, hidden_states: Tensor) -> Tensor:
    # patch: baichuan2 lm_head (fp32 bug)
    if self.training:
        norm_weight = F.normalize(self.weight).to(self.weight.dtype)
    elif self.first_flag:
        self.first_flag = False
        self.weight.data = F.normalize(self.weight).to(self.weight.dtype)
        norm_weight = self.weight
    else:
        norm_weight = self.weight
    return F.linear(hidden_states, norm_weight)


@register_model(
    ModelType.grok_1,
    'colossalai/grok-1-pytorch',
    template=TemplateType.default_generation,
    support_vllm=False,
    support_flash_attn=False,
    hf_model_id='hpcai-tech/grok-1')
def get_model_tokenizer_grok(model_dir: str,
                             torch_dtype: Optional[torch.dtype],
                             model_kwargs: Dict[str, Any],
                             load_model: bool = True,
                             model_config=None,
                             tokenizer=None,
                             automodel_class=AutoModelForCausalLM,
                             **kwargs):
    if model_config is None:
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    if torch_dtype is not None:
        model_config.torch_dtype = torch_dtype
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            'AI-ModelScope/grok-1-tokenizer', revision='master', trust_remote_code=True)
    eos_token = kwargs.get('eos_token')
    if eos_token is not None:
        tokenizer.eos_token = eos_token
    model = None
    if load_model:
        model = automodel_class.from_pretrained(
            model_dir, config=model_config, torch_dtype=torch_dtype, trust_remote_code=True, **model_kwargs)
    return model, tokenizer


@register_model(
    ModelType.mamba_130m,
    'AI-ModelScope/mamba-130m-hf',
    template=TemplateType.default_generation,
    requires=['transformers>=4.39.0'],
    support_vllm=False,
    hf_model_id='state-spaces/mamba-130m-hf')
@register_model(
    ModelType.mamba_370m,
    'AI-ModelScope/mamba-370m-hf',
    template=TemplateType.default_generation,
    requires=['transformers>=4.39.0'],
    support_vllm=False,
    hf_model_id='state-spaces/mamba-370m-hf')
@register_model(
    ModelType.mamba_390m,
    'AI-ModelScope/mamba-390m-hf',
    template=TemplateType.default_generation,
    requires=['transformers>=4.39.0'],
    support_vllm=False,
    hf_model_id='state-spaces/mamba-390m-hf')
@register_model(
    ModelType.mamba_790m,
    'AI-ModelScope/mamba-790m-hf',
    template=TemplateType.default_generation,
    requires=['transformers>=4.39.0'],
    support_vllm=False,
    hf_model_id='state-spaces/mamba-790m-hf')
@register_model(
    ModelType.mamba_1_4b,
    'AI-ModelScope/mamba-1.4b-hf',
    template=TemplateType.default_generation,
    requires=['transformers>=4.39.0'],
    support_vllm=False,
    hf_model_id='state-spaces/mamba-1.4b-hf')
@register_model(
    ModelType.mamba_2_8b,
    'AI-ModelScope/mamba-2.8b-hf',
    template=TemplateType.default_generation,
    requires=['transformers>=4.39.0'],
    support_vllm=False,
    hf_model_id='state-spaces/mamba-2.8b-hf')
def get_model_tokenizer_mamba(model_dir: str,
                              torch_dtype: Optional[torch.dtype],
                              model_kwargs: Dict[str, Any],
                              load_model: bool = True,
                              **kwargs):
    logger.info('[IMPORTANT] Remember installing causal-conv1d>=1.2.0 and mamba-ssm, or you training and inference will'
                'be really slow!')
    return get_model_tokenizer_from_repo(model_dir, torch_dtype, model_kwargs, load_model, **kwargs)


@register_model(
    ModelType.cogvlm_17b_chat,
    'ZhipuAI/cogvlm-chat',
    template=TemplateType.cogvlm,
    support_gradient_checkpointing=False,
    requires=['transformers<4.42'],
    tags=['multi-modal', 'vision'],
    hf_model_id='THUDM/cogvlm-chat-hf')
@register_model(
    ModelType.cogagent_18b_chat,
    'ZhipuAI/cogagent-chat',
    template=TemplateType.cogagent_chat,
    support_gradient_checkpointing=False,
    requires=['timm'],
    tags=['multi-modal', 'vision'],
    hf_model_id='THUDM/cogagent-chat-hf')
@register_model(
    ModelType.cogagent_18b_instruct,
    'ZhipuAI/cogagent-vqa',
    template=TemplateType.cogagent_instruct,
    support_gradient_checkpointing=False,
    requires=['timm'],
    tags=['multi-modal', 'vision'],
    hf_model_id='THUDM/cogagent-vqa-hf')
def get_model_tokenizer_cogagent(model_dir: str,
                                 torch_dtype: torch.dtype,
                                 model_kwargs: Dict[str, Any],
                                 load_model: bool = True,
                                 **kwargs):
    tokenizer = AutoTokenizer.from_pretrained('AI-ModelScope/vicuna-7b-v1.5', revision='master', trust_remote_code=True)
    if load_model:
        logger.warning('CogAgent with FusedLayerNorm will cause an training loss of NAN, '
                       'to avoid this, please uninstall apex.')
    model, tokenizer = get_model_tokenizer_from_repo(
        model_dir, torch_dtype, model_kwargs, load_model, tokenizer=tokenizer, **kwargs)
    tokenizer.build_conversation_input_ids = MethodType(model.build_conversation_input_ids, tokenizer)
    logger.info('Please ignore the un-imported warning.')
    return model, tokenizer


@register_model(
    ModelType.internlm_20b_chat,
    'Shanghai_AI_Laboratory/internlm-chat-20b',
    template=TemplateType.internlm,
    support_vllm=True,
    support_lmdeploy=True,
    hf_model_id='internlm/internlm-chat-20b')
@register_model(
    ModelType.internlm_7b_chat_8k,
    'Shanghai_AI_Laboratory/internlm-chat-7b-8k',
    template=TemplateType.internlm,
    support_vllm=True,
    support_lmdeploy=True)
@register_model(
    ModelType.internlm_7b_chat,
    'Shanghai_AI_Laboratory/internlm-chat-7b',
    template=TemplateType.internlm,
    support_vllm=True,
    support_lmdeploy=True,
    hf_model_id='internlm/internlm-chat-7b')
def get_model_tokenizer_internlm_chat(model_dir: str,
                                      torch_dtype: torch.dtype,
                                      model_kwargs: Dict[str, Any],
                                      load_model: bool = True,
                                      **kwargs):
    model, tokenizer = get_model_tokenizer_from_repo(model_dir, torch_dtype, model_kwargs, load_model, **kwargs)
    if getattr(tokenizer.__class__.eos_token_id, 'fset', None) is None:
        del tokenizer.__class__.eos_token_id
    tokenizer.eos_token = '<eoa>'
    return model, tokenizer


@register_model(
    ModelType.baichuan_13b,
    'baichuan-inc/Baichuan-13B-Base',
    template=TemplateType.default_generation,
    requires=['transformers<4.34'],
    support_vllm=True,
    support_lmdeploy=True,
    hf_model_id='baichuan-inc/Baichuan-13B-Base')
def get_model_tokenizer_baichuan_13b(model_dir: str,
                                     torch_dtype: torch.dtype,
                                     model_kwargs: Dict[str, Any],
                                     load_model: bool = True,
                                     **kwargs):
    model, tokenizer = get_model_tokenizer_from_repo(model_dir, torch_dtype, model_kwargs, load_model, **kwargs)
    # baichuan-13b does not implement the `get_input_embeddings` function
    # fix gradient_checkpointing bug
    try:
        model.get_input_embeddings()
    except NotImplementedError:
        model.__class__.get_input_embeddings = lambda self: self.model.embed_tokens
    return model, tokenizer


@register_model(
    ModelType.paligemma_3b_pt_224,
    'AI-ModelScope/paligemma-3b-pt-224',
    template=TemplateType.paligemma,
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.41'],
    placeholder_tokens=['<image>'],
    tags=['multi-modal', 'vision'],
    hf_model_id='google/paligemma-3b-pt-224')
@register_model(
    ModelType.paligemma_3b_pt_448,
    'AI-ModelScope/paligemma-3b-pt-448',
    template=TemplateType.paligemma,
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.41'],
    placeholder_tokens=['<image>'],
    tags=['multi-modal', 'vision'],
    hf_model_id='google/paligemma-3b-pt-448')
@register_model(
    ModelType.paligemma_3b_pt_896,
    'AI-ModelScope/paligemma-3b-pt-896',
    template=TemplateType.paligemma,
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.41'],
    placeholder_tokens=['<image>'],
    tags=['multi-modal', 'vision'],
    hf_model_id='google/paligemma-3b-pt-896')
@register_model(
    ModelType.paligemma_3b_mix_224,
    'AI-ModelScope/paligemma-3b-mix-224',
    template=TemplateType.paligemma,
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.41'],
    placeholder_tokens=['<image>'],
    tags=['multi-modal', 'vision'],
    hf_model_id='google/paligemma-3b-mix-224')
@register_model(
    ModelType.paligemma_3b_mix_448,
    'AI-ModelScope/paligemma-3b-mix-448',
    template=TemplateType.paligemma,
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.41'],
    placeholder_tokens=['<image>'],
    tags=['multi-modal', 'vision'],
    hf_model_id='google/paligemma-3b-mix-448')
def get_model_tokenizer_paligemma_vision(model_dir: str,
                                         torch_dtype: torch.dtype,
                                         model_kwargs: Dict[str, Any],
                                         load_model: bool = True,
                                         **kwargs):
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model, tokenizer = get_model_tokenizer_from_repo(
        model_dir, torch_dtype, model_kwargs, load_model, automodel_class=PaliGemmaForConditionalGeneration, **kwargs)
    tokenizer.processor = processor
    return model, tokenizer


@register_model(
    ModelType.phi3_vision_128k_instruct,
    'LLM-Research/Phi-3-vision-128k-instruct',
    template=TemplateType.phi3_vl,
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.36'],
    tags=['multi-modal', 'vision'],
    hf_model_id='microsoft/Phi-3-vision-128k-instruct')
@register_model(
    ModelType.phi3_5_vision_instruct,
    'LLM-Research/Phi-3.5-vision-instruct',
    template=TemplateType.phi3_vl,
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.36'],
    tags=['multi-modal', 'vision'],
    function_kwargs={'num_crops': 4},
    hf_model_id='microsoft/Phi-3.5-vision-instruct')
def get_model_tokenizer_phi3_vision(model_dir: str,
                                    torch_dtype: torch.dtype,
                                    model_kwargs: Dict[str, Any],
                                    load_model: bool = True,
                                    **kwargs):
    processor_kwargs = {}
    if 'num_crops' in kwargs:
        processor_kwargs['num_crops'] = get_env_args('num_crops', int, kwargs['num_crops'])
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True, **processor_kwargs)
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, torch_dtype, model_kwargs, load_model, **kwargs)
    tokenizer.processor = processor

    if load_model:
        patch_output_clone(model.model.vision_embed_tokens.wte)

    return model, tokenizer


@register_model(
    ModelType.baichuan2_13b_chat,
    'baichuan-inc/Baichuan2-13B-Chat',
    template=TemplateType.baichuan,
    support_vllm=True,
    support_lmdeploy=True,
    hf_model_id='baichuan-inc/Baichuan2-13B-Chat')
@register_model(
    ModelType.baichuan2_13b,
    'baichuan-inc/Baichuan2-13B-Base',
    template=TemplateType.default_generation,
    support_vllm=True,
    support_lmdeploy=True,
    hf_model_id='baichuan-inc/Baichuan2-13B-Base')
def get_model_tokenizer_baichuan2_13b(model_dir: str,
                                      torch_dtype: torch.dtype,
                                      model_kwargs: Dict[str, Any],
                                      load_model: bool = True,
                                      **kwargs):
    # patch: baichuan2_13b configuration_baichuan.py bug
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    gradient_checkpointing = model_config.gradient_checkpointing
    if isinstance(gradient_checkpointing, (tuple, list)):
        model_config.gradient_checkpointing = gradient_checkpointing[0]
    return get_model_tokenizer_baichuan2(
        model_dir, torch_dtype, model_kwargs, load_model, model_config=model_config, **kwargs)


@register_model(
    ModelType.baichuan2_7b_chat,
    'baichuan-inc/Baichuan2-7B-Chat',
    template=TemplateType.baichuan,
    support_vllm=True,
    support_lmdeploy=True,
    hf_model_id='baichuan-inc/Baichuan2-7B-Chat')
@register_model(
    ModelType.baichuan2_7b,
    'baichuan-inc/Baichuan2-7B-Base',
    template=TemplateType.default_generation,
    support_vllm=True,
    support_lmdeploy=True,
    hf_model_id='baichuan-inc/Baichuan2-7B-Base')
def get_model_tokenizer_baichuan2(model_dir: str,
                                  torch_dtype: torch.dtype,
                                  model_kwargs: Dict[str, Any],
                                  load_model: bool = True,
                                  model_config=None,
                                  **kwargs):
    if model_config is None:
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    if not hasattr(model_config, 'z_loss_weight'):
        model_config.z_loss_weight = 0
    model, tokenizer = get_model_tokenizer_from_repo(
        model_dir, torch_dtype, model_kwargs, load_model, model_config=model_config, **kwargs)
    model_ori = model
    if model is not None:
        if not hasattr(model, 'lm_head'):  # fix awq
            model = model.model
        new_forward = MethodType(patch_baichuan2_lm_head_forward, model.lm_head)
        if hasattr(model, '_old_forward'):  # device_map
            model.lm_head._old_forward = new_forward
        else:
            model.lm_head.forward = new_forward
    return model_ori, tokenizer


@register_model(
    ModelType.baichuan2_13b_chat_int4,
    'baichuan-inc/Baichuan2-13B-Chat-4bits',
    template=TemplateType.baichuan,
    function_kwargs={'get_baichuan2_function': get_model_tokenizer_baichuan2_13b},
    torch_dtype=torch.bfloat16,
    requires=['bitsandbytes<0.41.2', 'accelerate<0.26'],
    hf_model_id='baichuan-inc/Baichuan2-13B-Chat-4bits')
@register_model(
    ModelType.baichuan2_7b_chat_int4,
    'baichuan-inc/Baichuan2-7B-Chat-4bits',
    template=TemplateType.baichuan,
    torch_dtype=torch.bfloat16,
    requires=['bitsandbytes<0.41.2', 'accelerate<0.26'],
    hf_model_id='baichuan-inc/Baichuan2-7B-Chat-4bits')
def get_model_tokenizer_baichuan2_int4(model_dir: str,
                                       torch_dtype: torch.dtype,
                                       model_kwargs: Dict[str, Any],
                                       load_model: bool = True,
                                       **kwargs):
    logger.info('use `model_config.quantization_config`, ignore bnb arguments')
    model_kwargs.pop('quantization_config', None)

    # fix device_map bug
    import accelerate
    _old_infer_auto_device_map = accelerate.infer_auto_device_map
    device_map = model_kwargs.get('device_map', None)
    if device_map != 'auto':
        accelerate.infer_auto_device_map = lambda *args, **kwargs: device_map
    get_baichuan2_function = kwargs.pop('get_baichuan2_function', get_model_tokenizer_baichuan2)
    model, tokenizer = get_baichuan2_function(model_dir, torch_dtype, model_kwargs, load_model, **kwargs)
    if device_map != 'auto':
        accelerate.infer_auto_device_map = _old_infer_auto_device_map
    if model is not None:
        model.config.quantization_config = BitsAndBytesConfig(**model.config.quantization_config)
        model.train()
        model._is_quantized_training_enabled = True
        model.is_loaded_in_4bit = True
    return model, tokenizer


@register_model(
    ModelType.codefuse_codegeex2_6b_chat,
    'codefuse-ai/CodeFuse-CodeGeeX2-6B',
    template=TemplateType.codefuse,
    requires=['transformers<4.34'],
    support_vllm=True,
    tags=['coding'],
    hf_model_id='codefuse-ai/CodeFuse-CodeGeeX2-6B')
@register_model(
    ModelType.chatglm3_6b_32k,
    'ZhipuAI/chatglm3-6b-32k',
    template=TemplateType.chatglm3,
    support_vllm=True,
    requires=['transformers<4.42'],
    hf_model_id='THUDM/chatglm3-6b-32k')
@register_model(
    ModelType.chatglm3_6b_128k,
    'ZhipuAI/chatglm3-6b-128k',
    template=TemplateType.chatglm3,
    support_vllm=True,
    requires=['transformers<4.42'],
    hf_model_id='THUDM/chatglm3-6b-128k')
@register_model(
    ModelType.chatglm3_6b,
    'ZhipuAI/chatglm3-6b',
    template=TemplateType.chatglm3,
    support_vllm=True,
    requires=['transformers<4.42'],
    hf_model_id='THUDM/chatglm3-6b')
@register_model(
    ModelType.chatglm3_6b_base,
    'ZhipuAI/chatglm3-6b-base',
    template=TemplateType.chatglm_generation,
    support_vllm=True,
    requires=['transformers<4.42'],
    hf_model_id='THUDM/chatglm3-6b-base')
@register_model(
    ModelType.chatglm2_6b_32k,
    'ZhipuAI/chatglm2-6b-32k',
    template=TemplateType.chatglm2,
    support_vllm=True,
    requires=['transformers<4.42'],
    hf_model_id='THUDM/chatglm2-6b-32k')
@register_model(
    ModelType.chatglm2_6b,
    'ZhipuAI/chatglm2-6b',
    template=TemplateType.chatglm2,
    support_vllm=True,
    requires=['transformers<4.42'],
    hf_model_id='THUDM/chatglm2-6b')
@register_model(
    ModelType.codegeex2_6b,
    'ZhipuAI/codegeex2-6b',
    template=TemplateType.chatglm_generation,
    requires=['transformers<4.34'],
    support_vllm=True,
    tags=['coding'],
    hf_model_id='THUDM/codegeex2-6b')
def get_model_tokenizer_chatglm(model_dir: str,
                                torch_dtype: torch.dtype,
                                model_kwargs: Dict[str, Any],
                                load_model: bool = True,
                                **kwargs):

    def remove_property(tokenizer_cls: Type[PreTrainedTokenizerBase], tokenizer_config: Dict[str, Any]) -> None:
        for k, v in tokenizer_cls.__dict__.items():
            if k.endswith('_token') and isinstance(v, property) and k in tokenizer_config:
                setattr(tokenizer_cls, k, tokenizer_config[k])

    if model_kwargs.get('quantization_config') is not None:
        model_kwargs['quantization_config'].llm_int8_skip_modules = ['output_layer']
    # fix transformers>=4.34 bug
    if version.parse(transformers.__version__) >= version.parse('4.34'):
        tokenizer_config = get_tokenizer_config(model_dir)
        class_ref = tokenizer_config['auto_map']['AutoTokenizer'][0]
        # noinspection PyTypeChecker
        tokenizer_cls: Type[PreTrainedTokenizerBase] = get_class_from_dynamic_module(class_ref, model_dir)
        tokenizer_cls._auto_class = 'AutoTokenizer'
        remove_property(tokenizer_cls, tokenizer_config)
        kwargs['tokenizer'] = tokenizer_cls.from_pretrained(model_dir, trust_remote_code=True)
    model, tokenizer = get_model_tokenizer_from_repo(model_dir, torch_dtype, model_kwargs, load_model, **kwargs)
    if model is not None:
        from torch.nn import CrossEntropyLoss
        __old_forward = CrossEntropyLoss.forward

        def cross_entropy_forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            target = target.to(device=inputs.device)
            return __old_forward(self, inputs, target)

        CrossEntropyLoss.forward = cross_entropy_forward

    return model, tokenizer


@register_model(
    ModelType.codegeex4_9b_chat,
    'ZhipuAI/codegeex4-all-9b',
    template=TemplateType.codegeex4,
    support_vllm=True,
    support_flash_attn=True,
    support_lmdeploy=True,
    tags=['coding'],
    requires=['transformers<4.42'],
    hf_model_id='THUDM/codegeex4-all-9b')
@register_model(
    ModelType.glm4_9b,
    'ZhipuAI/glm-4-9b',
    template=TemplateType.chatglm_generation,
    support_vllm=True,
    support_flash_attn=True,
    support_lmdeploy=True,
    requires=['transformers>=4.42'],
    hf_model_id='THUDM/glm-4-9b')
@register_model(
    ModelType.glm4_9b_chat,
    'ZhipuAI/glm-4-9b-chat',
    template=TemplateType.chatglm4,
    support_flash_attn=True,
    support_vllm=True,
    support_lmdeploy=True,
    requires=['transformers>=4.42'],
    hf_model_id='THUDM/glm-4-9b-chat')
@register_model(
    ModelType.glm4_9b_chat_1m,
    'ZhipuAI/glm-4-9b-chat-1m',
    template=TemplateType.chatglm4,
    support_flash_attn=True,
    support_vllm=True,
    support_lmdeploy=True,
    requires=['transformers>=4.42'],
    hf_model_id='THUDM/glm-4-9b-chat-1m')
def get_model_tokenizer_glm4(model_dir: str,
                             torch_dtype: torch.dtype,
                             model_kwargs: Dict[str, Any],
                             load_model: bool = True,
                             **kwargs):
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    attn_type = AttentionType(kwargs.pop('use_flash_attn', None), kwargs.pop('attn_type', None))
    model_config._attn_implementation = attn_type.to_impl(attn_version=2)
    return get_model_tokenizer_chatglm(
        model_dir, torch_dtype, model_kwargs, load_model, model_config=model_config, **kwargs)


@register_model(
    ModelType.longwriter_glm4_9b,
    'ZhipuAI/LongWriter-glm4-9b',
    template=TemplateType.chatglm4,
    support_flash_attn=True,
    support_vllm=True,
    support_lmdeploy=True,
    requires=['transformers>=4.42'],
    hf_model_id='THUDM/LongWriter-glm4-9b')
def get_model_tokenizer_longwriter_glm4(*args, **kwargs):
    model, tokenizer = get_model_tokenizer_glm4(*args, **kwargs)
    for k in tokenizer.special_tokens.keys():
        tokenizer.add_tokens(k)
    return model, tokenizer


@register_model(
    ModelType.glm4v_9b_chat,
    'ZhipuAI/glm-4v-9b',
    template=TemplateType.glm4v,
    eos_token='<|endoftext|>',
    requires=['transformers>=4.42'],
    tags=['multi-modal', 'vision'],
    hf_model_id='THUDM/glm-4v-9b')
def get_model_tokenizer_glm4v(model_dir: str,
                              torch_dtype: torch.dtype,
                              model_kwargs: Dict[str, Any],
                              load_model: bool = True,
                              **kwargs):
    model, tokenizer = get_model_tokenizer_glm4(model_dir, torch_dtype, model_kwargs, load_model, **kwargs)
    # fix merge-lora
    tokenizer.init_kwargs['image_size'] = 1120
    if load_model:
        # fix device_map 4
        n_gpu = torch.cuda.device_count()
        local_world_size = get_dist_setting()[3]
        if n_gpu // local_world_size >= 4:
            for layer in model.transformer.vision.transformer.layers:
                patch_output_to_input_device(layer.mlp)
                patch_output_to_input_device(layer.post_attention_layernorm)
            device = next(model.transformer.vision.linear_proj.parameters()).device
            model.transformer.vision.boi.data = model.transformer.vision.boi.to(device)
            model.transformer.vision.eoi.data = model.transformer.vision.eoi.to(device)
    return model, tokenizer


@register_model(
    ModelType.mplug_owl3_7b_chat,
    'iic/mPLUG-Owl3-7B-240728',
    template=TemplateType.mplug_owl3,
    requires=['transformers>=4.36', 'icecream'],  # decord
    support_flash_attn=True,
    tags=['multi-modal', 'vision'],
    hf_model_id='mPLUG/mPLUG-Owl3-7B-240728')
def get_model_tokenizer_mplug_owl3(model_dir: str,
                                   torch_dtype: torch.dtype,
                                   model_kwargs: Dict[str, Any],
                                   load_model: bool = True,
                                   **kwargs):
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, torch_dtype, model_kwargs, load_model, **kwargs)
    processor = model.init_processor(tokenizer)
    tokenizer.processor = processor
    func_list = ['generate', 'forward']
    _use_submodel_func(model, 'language_model', func_list)
    return model, tokenizer


def get_model_tokenizer_yi1_5(model_dir, *args, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)
    return get_model_tokenizer_with_flash_attn(model_dir, *args, tokenizer=tokenizer, **kwargs)


@register_model(
    ModelType.florence_2_base,
    'AI-ModelScope/Florence-2-base',
    template=TemplateType.florence,
    support_flash_attn=True,
    hf_model_id='microsoft/Florence-2-base',
    tags=['multi-modal', 'vision'])
@register_model(
    ModelType.florence_2_base_ft,
    'AI-ModelScope/Florence-2-base-ft',
    template=TemplateType.florence,
    support_flash_attn=True,
    hf_model_id='microsoft/Florence-2-base-ft',
    tags=['multi-modal', 'vision'])
@register_model(
    ModelType.florence_2_large,
    'AI-ModelScope/Florence-2-large',
    template=TemplateType.florence,
    support_flash_attn=True,
    hf_model_id='microsoft/Florence-2-large',
    tags=['multi-modal', 'vision'])
@register_model(
    ModelType.florence_2_large_ft,
    'AI-ModelScope/Florence-2-large-ft',
    template=TemplateType.florence,
    support_flash_attn=True,
    hf_model_id='microsoft/Florence-2-large-ft',
    tags=['multi-modal', 'vision'])
def get_model_tokenizer_florence(model_dir: str,
                                 torch_dtype: torch.dtype,
                                 model_kwargs: Dict[str, Any],
                                 load_model: bool = True,
                                 model_config=None,
                                 **kwargs):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    with ignore_check_imports():
        model, tokenizer = get_model_tokenizer_with_flash_attn(
            model_dir, torch_dtype, model_kwargs, load_model, tokenizer=processor.tokenizer, **kwargs)

    tokenizer.processor = processor
    # model.vision_tower.enable_checkpoint = True
    _use_submodel_func(model, 'language_model', ['generate', 'forward'])
    return model, tokenizer


@register_model(
    ModelType.phi3_small_8k_instruct,
    'LLM-Research/Phi-3-small-8k-instruct',
    template=TemplateType.phi3,
    requires=['transformers>=4.36'],
    support_flash_attn=True,
    support_gradient_checkpointing=False,
    support_vllm=True,
    hf_model_id='microsoft/Phi-3-small-8k-instruct')
@register_model(
    ModelType.phi3_small_128k_instruct,
    'LLM-Research/Phi-3-small-128k-instruct',
    template=TemplateType.phi3,
    requires=['transformers>=4.36'],
    support_flash_attn=True,
    support_gradient_checkpointing=False,
    support_vllm=True,
    hf_model_id='microsoft/Phi-3-small-128k-instruct')
def get_model_tokenizer_phi3_small(model_dir: str,
                                   torch_dtype: torch.dtype,
                                   model_kwargs: Dict[str, Any],
                                   load_model: bool = True,
                                   model_config=None,
                                   **kwargs):
    if model_config is None:
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    attn_type = AttentionType(kwargs.pop('use_flash_attn', None), kwargs.pop('attn_type', 'sdpa'))
    attn_type.update_config(model_config)
    model, tokenizer = get_model_tokenizer_from_repo(
        model_dir, torch_dtype, model_kwargs, load_model, model_config=model_config, **kwargs)

    def rotary_emb(self, query_states, key_states, **kwargs):
        q_type = query_states.dtype
        k_type = key_states.dtype
        query_states, key_states = self.rotory_emb_origin(query_states, key_states, **kwargs)
        query_states = query_states.to(q_type)
        key_states = key_states.to(k_type)
        return query_states, key_states

    for i in range(32):
        re = model.model.layers[i].self_attn.rotary_emb
        re.rotory_emb_origin = re.forward
        re.forward = MethodType(rotary_emb, re)
    return model, tokenizer


def get_model_tokenizer_qwen2_chat(model_dir: str,
                                   torch_dtype: torch.dtype,
                                   model_kwargs: Dict[str, Any],
                                   load_model: bool = True,
                                   **kwargs):
    kwargs['eos_token'] = '<|im_end|>'
    return get_model_tokenizer_with_flash_attn(model_dir, torch_dtype, model_kwargs, load_model, **kwargs)


@register_model(
    ModelType.qwen2_audio_7b_instruct,
    'qwen/Qwen2-Audio-7B-Instruct',
    template=TemplateType.qwen2_audio,
    support_flash_attn=True,
    requires=['librosa', 'transformers>=4.45.0.dev0'],
    tags=['multi-modal', 'audio'],
    hf_model_id='Qwen/Qwen2-Audio-7B-Instruct')
@register_model(
    ModelType.qwen2_audio_7b,
    'qwen/Qwen2-Audio-7B',
    template=TemplateType.qwen2_audio_generation,
    support_flash_attn=True,
    requires=['librosa', 'transformers>=4.45.0.dev0'],
    eos_token='<|endoftext|>',
    tags=['multi-modal', 'audio'],
    hf_model_id='Qwen/Qwen2-Audio-7B')
def get_model_tokenizer_qwen2_audio(model_dir: str,
                                    torch_dtype: torch.dtype,
                                    model_kwargs: Dict[str, Any],
                                    load_model: bool = True,
                                    **kwargs):
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_dir)
    kwargs['automodel_class'] = Qwen2AudioForConditionalGeneration
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, torch_dtype, model_kwargs, load_model, **kwargs)
    tokenizer.processor = processor
    return model, tokenizer


@register_model(
    ModelType.qwen2_vl_7b_instruct,
    'qwen/Qwen2-VL-7B-Instruct',
    template=TemplateType.qwen2_vl,
    support_flash_attn=True,
    placeholder_tokens=['<|image_pad|>', '<|video_pad|>'],
    # pip install qwen_vl_utils
    requires=['transformers>=4.45.0.dev0', 'qwen_vl_utils'],  # 'pyav'
    tags=['multi-modal', 'vision'],
    hf_model_id='Qwen/Qwen2-VL-7B-Instruct')
@register_model(
    ModelType.qwen2_vl_7b_instruct_gptq_int4,
    'qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4',
    template=TemplateType.qwen2_vl,
    support_flash_attn=True,
    placeholder_tokens=['<|image_pad|>', '<|video_pad|>'],
    requires=['transformers>=4.45.0.dev0', 'qwen_vl_utils', 'auto_gptq>=0.5'],
    tags=['multi-modal', 'vision'],
    torch_dtype=torch.float16,
    hf_model_id='Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4')
@register_model(
    ModelType.qwen2_vl_7b_instruct_gptq_int8,
    'qwen/Qwen2-VL-7B-Instruct-GPTQ-Int8',
    template=TemplateType.qwen2_vl,
    support_flash_attn=True,
    placeholder_tokens=['<|image_pad|>', '<|video_pad|>'],
    requires=['transformers>=4.45.0.dev0', 'qwen_vl_utils', 'auto_gptq>=0.5'],
    tags=['multi-modal', 'vision'],
    torch_dtype=torch.float16,
    hf_model_id='Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int8')
@register_model(
    ModelType.qwen2_vl_7b_instruct_awq,
    'qwen/Qwen2-VL-7B-Instruct-AWQ',
    template=TemplateType.qwen2_vl,
    support_flash_attn=True,
    placeholder_tokens=['<|image_pad|>', '<|video_pad|>'],
    requires=['transformers>=4.45.0.dev0', 'qwen_vl_utils', 'autoawq'],
    tags=['multi-modal', 'vision'],
    torch_dtype=torch.float16,
    hf_model_id='Qwen/Qwen2-VL-7B-Instruct-AWQ')
@register_model(
    ModelType.qwen2_vl_2b_instruct,
    'qwen/Qwen2-VL-2B-Instruct',
    template=TemplateType.qwen2_vl,
    support_flash_attn=True,
    placeholder_tokens=['<|image_pad|>', '<|video_pad|>'],
    requires=['transformers>=4.45.0.dev0', 'qwen_vl_utils'],  # 'pyav'
    tags=['multi-modal', 'vision'],
    hf_model_id='Qwen/Qwen2-VL-2B-Instruct')
@register_model(
    ModelType.qwen2_vl_2b_instruct_gptq_int4,
    'qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4',
    template=TemplateType.qwen2_vl,
    support_flash_attn=True,
    placeholder_tokens=['<|image_pad|>', '<|video_pad|>'],
    requires=['transformers>=4.45.0.dev0', 'qwen_vl_utils', 'auto_gptq>=0.5'],
    tags=['multi-modal', 'vision'],
    torch_dtype=torch.float16,
    hf_model_id='Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4')
@register_model(
    ModelType.qwen2_vl_2b_instruct_gptq_int8,
    'qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8',
    template=TemplateType.qwen2_vl,
    support_flash_attn=True,
    placeholder_tokens=['<|image_pad|>', '<|video_pad|>'],
    requires=['transformers>=4.45.0.dev0', 'qwen_vl_utils', 'auto_gptq>=0.5'],
    tags=['multi-modal', 'vision'],
    torch_dtype=torch.float16,
    hf_model_id='Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8')
@register_model(
    ModelType.qwen2_vl_2b_instruct_awq,
    'qwen/Qwen2-VL-2B-Instruct-AWQ',
    template=TemplateType.qwen2_vl,
    support_flash_attn=True,
    placeholder_tokens=['<|image_pad|>', '<|video_pad|>'],
    requires=['transformers>=4.45.0.dev0', 'qwen_vl_utils', 'autoawq'],
    tags=['multi-modal', 'vision'],
    torch_dtype=torch.float16,
    hf_model_id='Qwen/Qwen2-VL-2B-Instruct-AWQ')
def get_model_tokenizer_qwen2_vl(model_dir: str,
                                 torch_dtype: torch.dtype,
                                 model_kwargs: Dict[str, Any],
                                 load_model: bool = True,
                                 **kwargs):
    try:
        from torchvision.io import video
        if not hasattr(video, '_patching'):
            # not read audio
            video._patching = True
            _old_read_from_stream = video._read_from_stream

            def _read_from_stream(container: 'av.container.Container', start_offset: float, end_offset: float,
                                  pts_unit: str, stream: 'av.stream.Stream', *args, **kwargs) -> List['av.frame.Frame']:
                if stream.type == 'video':
                    return _old_read_from_stream(container, start_offset, end_offset, pts_unit, stream, *args, **kwargs)
                return []

            video._read_from_stream = _read_from_stream
    except Exception:
        pass

    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_dir)
    kwargs['automodel_class'] = Qwen2VLForConditionalGeneration
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, torch_dtype, model_kwargs, load_model, **kwargs)
    tokenizer.processor = processor
    if model is not None:
        patch_output_clone(model.model.embed_tokens)
    return model, tokenizer


def get_model_tokenizer_qwen2_intx(model_dir: str,
                                   torch_dtype: torch.dtype,
                                   model_kwargs: Dict[str, Any],
                                   load_model: bool = True,
                                   **kwargs):
    kwargs['get_qwen_function'] = get_model_tokenizer_qwen2_chat
    return get_model_tokenizer_qwen_intx(model_dir, torch_dtype, model_kwargs, load_model, **kwargs)


def get_model_tokenizer_internlm2(model_dir: str,
                                  torch_dtype: torch.dtype,
                                  model_kwargs: Dict[str, Any],
                                  load_model: bool = True,
                                  model_config=None,
                                  **kwargs):
    if model_config is None:
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)

    attn_type = AttentionType(kwargs.pop('use_flash_attn', None), kwargs.pop('attn_type', None))
    model_config.attn_implementation = attn_type.to_impl(attn_version=2)

    eos_token = kwargs.pop('eos_token', None)
    model, tokenizer = get_model_tokenizer_from_repo(
        model_dir, torch_dtype, model_kwargs, load_model, model_config=model_config, **kwargs)
    if eos_token is not None:
        if getattr(tokenizer.__class__.eos_token_id, 'fset', None) is None:
            del tokenizer.__class__.eos_token_id
        tokenizer.eos_token = eos_token

    return model, tokenizer


@register_model(
    ModelType.deepseek_coder_v2,
    'deepseek-ai/DeepSeek-Coder-V2-Base',
    template=TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    tags=['coding', 'moe'],
    requires=['transformers>=4.39.3'],
    hf_model_id='deepseek-ai/DeepSeek-Coder-V2-Base')
@register_model(
    ModelType.deepseek_coder_v2_lite,
    'deepseek-ai/DeepSeek-Coder-V2-Lite-Base',
    template=TemplateType.default_generation,
    tags=['coding', 'moe'],
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.39.3'],
    hf_model_id='deepseek-ai/DeepSeek-Coder-V2-Lite-Base')
@register_model(
    ModelType.deepseek_coder_v2_instruct,
    'deepseek-ai/DeepSeek-Coder-V2-Instruct',
    template=TemplateType.deepseek2,
    support_flash_attn=True,
    support_vllm=True,
    tags=['coding', 'moe'],
    requires=['transformers>=4.39.3'],
    hf_model_id='deepseek-ai/DeepSeek-Coder-V2-Instruct')
@register_model(
    ModelType.deepseek_coder_v2_lite_instruct,
    'deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct',
    template=TemplateType.deepseek2,
    support_flash_attn=True,
    support_vllm=True,
    tags=['coding', 'moe'],
    requires=['transformers>=4.39.3'],
    hf_model_id='deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct')
@register_model(
    ModelType.deepseek_v2_lite,
    'deepseek-ai/DeepSeek-V2-Lite',
    template=TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    tags=['moe'],
    requires=['transformers>=4.39.3'],
    hf_model_id='deepseek-ai/DeepSeek-V2-Lite')
@register_model(
    ModelType.deepseek_v2_lite_chat,
    'deepseek-ai/DeepSeek-V2-Lite-Chat',
    template=TemplateType.deepseek2,
    support_flash_attn=True,
    support_vllm=True,
    tags=['moe'],
    requires=['transformers>=4.39.3'],
    hf_model_id='deepseek-ai/DeepSeek-V2-Lite-Chat')
@register_model(
    ModelType.deepseek_v2,
    'deepseek-ai/DeepSeek-V2',
    template=TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    tags=['moe'],
    requires=['transformers>=4.39.3'],
    hf_model_id='deepseek-ai/DeepSeek-V2')
@register_model(
    ModelType.deepseek_v2_chat,
    'deepseek-ai/DeepSeek-V2-Chat',
    template=TemplateType.deepseek2,
    support_flash_attn=True,
    support_vllm=True,
    tags=['moe'],
    requires=['transformers>=4.39.3'],
    hf_model_id='deepseek-ai/DeepSeek-V2-Chat')
@register_model(
    ModelType.deepseek_v2_5,
    'deepseek-ai/DeepSeek-V2.5',
    template=TemplateType.deepseek2_5,
    support_flash_attn=True,
    support_vllm=True,
    tags=['moe'],
    requires=['transformers>=4.39.3'],
    hf_model_id='deepseek-ai/DeepSeek-V2.5')
def get_model_tokenizer_deepseek2(model_dir: str,
                                  torch_dtype: torch.dtype,
                                  model_kwargs: Dict[str, Any],
                                  load_model: bool = True,
                                  **kwargs):
    model, tokenizer = get_model_tokenizer_deepseek_moe(model_dir, torch_dtype, model_kwargs, load_model, **kwargs)
    if model is not None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
    return model, tokenizer


@register_model(
    ModelType.internvl_chat_v1_5,
    'AI-ModelScope/InternVL-Chat-V1-5',
    template=TemplateType.internvl,
    requires=['transformers>=4.35', 'timm'],
    support_flash_attn=True,
    support_lmdeploy=True,
    support_vllm=True,
    placeholder_tokens=['<IMG_CONTEXT>'],
    tags=['multi-modal', 'vision'],
    hf_model_id='OpenGVLab/InternVL-Chat-V1-5')
@register_model(
    ModelType.internvl_chat_v1_5_int8,
    'AI-ModelScope/InternVL-Chat-V1-5-int8',
    template=TemplateType.internvl,
    requires=['transformers>=4.35', 'timm'],
    support_flash_attn=True,
    placeholder_tokens=['<IMG_CONTEXT>'],
    tags=['multi-modal', 'vision'],
    hf_model_id='OpenGVLab/InternVL-Chat-V1-5-int8')
@register_model(
    ModelType.mini_internvl_chat_2b_v1_5,
    'OpenGVLab/Mini-InternVL-Chat-2B-V1-5',
    template=TemplateType.internvl,
    requires=['transformers>=4.35', 'timm'],
    support_flash_attn=True,
    support_lmdeploy=True,
    support_vllm=True,
    placeholder_tokens=['<IMG_CONTEXT>'],
    tags=['multi-modal', 'vision'],
    hf_model_id='OpenGVLab/Mini-InternVL-Chat-2B-V1-5')
@register_model(
    ModelType.mini_internvl_chat_4b_v1_5,
    'OpenGVLab/Mini-InternVL-Chat-4B-V1-5',
    template=TemplateType.internvl_phi3,
    requires=['transformers>=4.35,<4.42', 'timm'],
    support_flash_attn=True,
    support_vllm=True,
    eos_token='<|end|>',
    placeholder_tokens=['<IMG_CONTEXT>'],
    tags=['multi-modal', 'vision'],
    hf_model_id='OpenGVLab/Mini-InternVL-Chat-4B-V1-5')
@register_model(
    ModelType.internvl2_1b,
    'OpenGVLab/InternVL2-1B',
    template=TemplateType.internvl2,
    requires=['transformers>=4.36', 'timm'],
    ignore_file_pattern=[r'.+\.zip$'],
    support_flash_attn=True,
    support_lmdeploy=True,
    support_vllm=True,
    placeholder_tokens=['<IMG_CONTEXT>'],
    tags=['multi-modal', 'vision', 'video'],
    hf_model_id='OpenGVLab/InternVL2-1B')
@register_model(
    ModelType.internvl2_2b,
    'OpenGVLab/InternVL2-2B',
    template=TemplateType.internvl2,
    requires=['transformers>=4.36', 'timm'],
    ignore_file_pattern=[r'.+\.zip$'],
    support_flash_attn=True,
    support_lmdeploy=True,
    support_vllm=True,
    placeholder_tokens=['<IMG_CONTEXT>'],
    tags=['multi-modal', 'vision', 'video'],
    hf_model_id='OpenGVLab/InternVL2-2B')
@register_model(
    ModelType.internvl2_4b,
    'OpenGVLab/InternVL2-4B',
    template=TemplateType.internvl2_phi3,
    requires=['transformers>=4.36,<4.42', 'timm'],
    ignore_file_pattern=[r'.+\.zip$'],
    support_flash_attn=True,
    support_lmdeploy=True,
    support_vllm=True,
    eos_token='<|end|>',
    placeholder_tokens=['<IMG_CONTEXT>'],
    tags=['multi-modal', 'vision', 'video'],
    hf_model_id='OpenGVLab/InternVL2-4B')
@register_model(
    ModelType.internvl2_8b,
    'OpenGVLab/InternVL2-8B',
    template=TemplateType.internvl2,
    requires=['transformers>=4.36', 'timm'],
    ignore_file_pattern=[r'.+\.zip$'],
    support_flash_attn=True,
    support_lmdeploy=True,
    support_vllm=True,
    placeholder_tokens=['<IMG_CONTEXT>'],
    tags=['multi-modal', 'vision', 'video'],
    hf_model_id='OpenGVLab/InternVL2-8B')
@register_model(
    ModelType.internvl2_26b,
    'OpenGVLab/InternVL2-26B',
    template=TemplateType.internvl2,
    requires=['transformers>=4.36', 'timm'],
    ignore_file_pattern=[r'.+\.zip$'],
    support_flash_attn=True,
    support_vllm=True,
    support_lmdeploy=True,
    placeholder_tokens=['<IMG_CONTEXT>'],
    tags=['multi-modal', 'vision', 'video'],
    hf_model_id='OpenGVLab/InternVL2-26B')
@register_model(
    ModelType.internvl2_40b,
    'OpenGVLab/InternVL2-40B',
    template=TemplateType.internvl2,
    requires=['transformers>=4.36', 'timm'],
    ignore_file_pattern=[r'.+\.zip$'],
    support_flash_attn=True,
    support_vllm=True,
    support_lmdeploy=True,
    placeholder_tokens=['<IMG_CONTEXT>'],
    tags=['multi-modal', 'vision', 'video'],
    hf_model_id='OpenGVLab/InternVL2-40B')
@register_model(
    ModelType.internvl2_llama3_76b,
    'OpenGVLab/InternVL2-Llama3-76B',
    template=TemplateType.internvl2,
    requires=['transformers>=4.36', 'timm'],
    ignore_file_pattern=[r'.+\.zip$'],
    support_flash_attn=True,
    support_vllm=True,
    support_lmdeploy=True,
    placeholder_tokens=['<IMG_CONTEXT>'],
    tags=['multi-modal', 'vision', 'video'],
    hf_model_id='OpenGVLab/InternVL2-Llama3-76B')
@register_model(
    ModelType.internvl2_2b_awq,
    'OpenGVLab/InternVL2-2B-AWQ',
    template=TemplateType.internvl2,
    requires=['transformers>=4.36', 'timm'],
    ignore_file_pattern=[r'.+\.zip$'],
    support_flash_attn=True,
    support_lmdeploy=True,
    support_vllm=False,
    torch_dtype=torch.float16,
    placeholder_tokens=['<IMG_CONTEXT>'],
    tags=['multi-modal', 'vision', 'video'],
    hf_model_id='OpenGVLab/InternVL2-2B-AWQ')
@register_model(
    ModelType.internvl2_8b_awq,
    'OpenGVLab/InternVL2-8B-AWQ',
    template=TemplateType.internvl2,
    requires=['transformers>=4.36', 'timm'],
    ignore_file_pattern=[r'.+\.zip$'],
    support_flash_attn=True,
    support_lmdeploy=True,
    support_vllm=False,
    torch_dtype=torch.float16,
    placeholder_tokens=['<IMG_CONTEXT>'],
    tags=['multi-modal', 'vision', 'video'],
    hf_model_id='OpenGVLab/InternVL2-8B-AWQ')
@register_model(
    ModelType.internvl2_26b_awq,
    'OpenGVLab/InternVL2-26B-AWQ',
    template=TemplateType.internvl2,
    requires=['transformers>=4.36', 'timm'],
    ignore_file_pattern=[r'.+\.zip$'],
    support_flash_attn=True,
    support_lmdeploy=True,
    support_vllm=False,
    torch_dtype=torch.float16,
    placeholder_tokens=['<IMG_CONTEXT>'],
    tags=['multi-modal', 'vision', 'video'],
    hf_model_id='OpenGVLab/InternVL2-26B-AWQ')
@register_model(
    ModelType.internvl2_40b_awq,
    'OpenGVLab/InternVL2-40B-AWQ',
    template=TemplateType.internvl2,
    requires=['transformers>=4.36', 'timm'],
    ignore_file_pattern=[r'.+\.zip$'],
    support_flash_attn=True,
    support_lmdeploy=True,
    support_vllm=False,
    torch_dtype=torch.float16,
    placeholder_tokens=['<IMG_CONTEXT>'],
    tags=['multi-modal', 'vision', 'video'],
    hf_model_id='OpenGVLab/InternVL2-40B-AWQ')
@register_model(
    ModelType.internvl2_llama3_76b_awq,
    'OpenGVLab/InternVL2-Llama3-76B-AWQ',
    template=TemplateType.internvl2,
    requires=['transformers>=4.36', 'timm'],
    ignore_file_pattern=[r'.+\.zip$'],
    support_flash_attn=True,
    support_lmdeploy=True,
    support_vllm=False,
    torch_dtype=torch.float16,
    placeholder_tokens=['<IMG_CONTEXT>'],
    tags=['multi-modal', 'vision', 'video'],
    hf_model_id='OpenGVLab/InternVL2-Llama3-76B-AWQ')
def get_model_tokenizer_internvl(model_dir: str,
                                 torch_dtype: torch.dtype,
                                 model_kwargs: Dict[str, Any],
                                 load_model: bool = True,
                                 **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)
    if kwargs.get('eos_token') is None and tokenizer.eos_token != '<|im_end|>':
        try:
            del tokenizer.__class__.eos_token_id
        except AttributeError:
            pass
        tokenizer.eos_token = '<|im_end|>'

    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    attn_type = AttentionType(kwargs.pop('use_flash_attn', None), kwargs.pop('attn_type', None))
    model_config.llm_config.attn_implementation = attn_type.to_impl(attn_version=2)
    model_quant_config = getattr(model_config, 'quantization_config', None)

    use_bnb = False
    if model_quant_config is not None:
        use_bnb = model_quant_config.get('quant_method', None) == 'bitsandbytes'
    quantization_config = model_kwargs.get('quantization_config', None)
    if isinstance(quantization_config, BitsAndBytesConfig):
        use_bnb = True

    model, tokenizer = get_model_tokenizer_from_local(
        model_dir, torch_dtype, model_kwargs, load_model, tokenizer=tokenizer, model_config=model_config, **kwargs)

    if use_bnb and kwargs.get('is_training'):
        # patch: bnb backward shape mismatch bug
        if model is not None and model.language_model is not None:
            model.language_model.output.state.force_no_igemmlt = True

    if model is not None:
        func_list = ['generate', 'get_input_embeddings', 'gradient_checkpointing_enable', 'forward']
        _use_submodel_func(model, 'language_model', func_list)
        patch_output_clone(model.language_model.get_input_embeddings())

    return model, tokenizer


@register_model(
    ModelType.internlm_xcomposer2_5_7b_chat,
    'Shanghai_AI_Laboratory/internlm-xcomposer2d5-7b',
    template=TemplateType.internlm_xcomposer2_5,
    eos_token='<|im_end|>',
    support_flash_attn=True,
    support_lmdeploy=True,
    # requires=['decord'],
    tags=['multi-modal', 'vision'],
    function_kwargs={'version': 'v2.5'},
    hf_model_id='internlm/internlm-xcomposer2d5-7b')
@register_model(
    ModelType.internlm_xcomposer2_7b_chat,
    'Shanghai_AI_Laboratory/internlm-xcomposer2-7b',
    template=TemplateType.internlm_xcomposer2,
    support_flash_attn=True,
    support_lmdeploy=True,
    eos_token='[UNUSED_TOKEN_145]',
    tags=['multi-modal', 'vision'],
    hf_model_id='internlm/internlm-xcomposer2-7b')
@register_model(
    ModelType.internlm_xcomposer2_4khd_7b_chat,
    'Shanghai_AI_Laboratory/internlm-xcomposer2-4khd-7b',
    template=TemplateType.internlm_xcomposer2_4khd,
    support_flash_attn=True,
    support_lmdeploy=True,
    eos_token='<|im_end|>',
    function_kwargs={'version': 'v2-4khd'},
    tags=['multi-modal', 'vision'],
    hf_model_id='internlm/internlm-xcomposer2-4khd-7b')
def get_model_tokenizer_internlm_xcomposer2(model_dir: str,
                                            torch_dtype: torch.dtype,
                                            model_kwargs: Dict[str, Any],
                                            load_model: bool = True,
                                            **kwargs):
    version = kwargs.pop('version', 'v2')
    model_config = None
    attn_type = AttentionType(kwargs.get('use_flash_attn', None), kwargs.get('attn_type', None))
    if version == 'v2-4khd':
        from transformers import CLIPVisionModel

        def load_model(self):
            self.vision_tower_name = snapshot_download('AI-ModelScope/clip-vit-large-patch14-336')
            self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
            self.vision_tower.requires_grad_(False)
            self.is_loaded = True

        CLIPVisionTower = get_class_from_dynamic_module('build_mlp.CLIPVisionTower', model_dir)
        CLIPVisionTower.load_model = load_model
    elif version == 'v2':
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        model_config._flash_attn_2_enabled = attn_type.to_bool()

    model, tokenizer = get_model_tokenizer_internlm2(
        model_dir, torch_dtype, model_kwargs, load_model, model_config=model_config, **kwargs)
    if model is not None:
        if version == 'v2' and attn_type.to_bool():
            # fix AttributeError: no attribute 'attention_dropout'
            model.model.layers[0].attention.__class__.attention_dropout = 0.

        if version == 'v2.5':
            patch_output_to_input_device(model.vision_proj)
            patch_output_to_input_device(model.vit)
        tokenizer.vis_processor = model.vis_processor

    return model, tokenizer


def git_clone_github(github_url: str,
                     local_repo_name: Optional[str] = None,
                     branch: Optional[str] = None,
                     commit_hash: Optional[str] = None) -> str:
    git_cache_dir = os.path.join(get_cache_dir(), '_github')
    os.makedirs(git_cache_dir, exist_ok=True)
    if local_repo_name is None:
        github_url = github_url.rstrip('/')
        local_repo_name = github_url.rsplit('/', 1)[1]
    local_repo_path = os.path.join(git_cache_dir, local_repo_name)
    with safe_ddp_context():
        if not os.path.exists(local_repo_path):
            if not github_url.endswith('.git'):
                github_url = f'{github_url}.git'
            command = ['git', '-C', git_cache_dir, 'clone', github_url, local_repo_name]
            command_str = f"git -C '{git_cache_dir}' clone '{github_url}' {local_repo_name}"
            if branch is not None:
                command += ['--branch', branch]
                command_str += f' --branch {branch}'
            logger.info(f'Run the command: `{command_str}`')
            subprocess_run(command)

            if commit_hash is not None:
                git_cache_path = os.path.join(git_cache_dir, local_repo_name)
                command = ['git', '-C', git_cache_path, 'reset', '--hard', commit_hash]
                command_str = f"git -C '{git_cache_path}' reset '--hard' {commit_hash}"
                logger.info(f'Run the command: `{command_str}`')
                subprocess_run(command)

        logger.info(f'local_repo_path: {local_repo_path}')
    return local_repo_path


def _use_submodel_func(model, submodel_name: str, func_list: List[str]) -> None:
    submodel = getattr(model, submodel_name)

    def _get_new_func(func_name: str):
        _old_func = getattr(submodel.__class__, func_name)

        @wraps(_old_func)
        def _new_func(self, *args, **kwargs):
            res = _old_func(submodel, *args, **kwargs)
            if func_name == 'forward':
                device = find_device(args)
                if device is None:
                    device = find_device(kwargs)
                res.logits = to_device(res.logits, device)
                res.loss = to_device(res.loss, device)
            return res

        return _new_func

    for key in func_list:
        setattr(model, key, MethodType(_get_new_func(key), model))
        if key == 'generate' and model.device != submodel.device:
            submodel.__class__.device = model.device
        if key == 'forward' and 'generate' in func_list:
            setattr(submodel, key, MethodType(_get_new_func(key), submodel))  # fix device_map


@register_model(
    ModelType.deepseek_vl_7b_chat,
    'deepseek-ai/deepseek-vl-7b-chat',
    template=TemplateType.deepseek_vl,
    support_flash_attn=True,
    support_lmdeploy=True,
    tags=['multi-modal', 'vision'],
    placeholder_tokens=['<image_placeholder>'],
    hf_model_id='deepseek-ai/deepseek-vl-7b-chat')
@register_model(
    ModelType.deepseek_vl_1_3b_chat,
    'deepseek-ai/deepseek-vl-1.3b-chat',
    template=TemplateType.deepseek_vl,
    support_flash_attn=True,
    support_lmdeploy=True,
    tags=['multi-modal', 'vision'],
    placeholder_tokens=['<image_placeholder>'],
    hf_model_id='deepseek-ai/deepseek-vl-1.3b-chat')
def get_model_tokenizer_deepseek_vl(model_dir: str,
                                    torch_dtype: torch.dtype,
                                    model_kwargs: Dict[str, Any],
                                    load_model: bool = True,
                                    **kwargs):
    # compat with python==3.10
    if sys.version_info.minor >= 10:
        import collections
        import collections.abc
        for type_name in collections.abc.__all__:
            setattr(collections, type_name, getattr(collections.abc, type_name))
    if 'local_repo_path' in kwargs:
        local_repo_path = kwargs['local_repo_path']
    else:
        local_repo_path = git_clone_github('https://github.com/deepseek-ai/DeepSeek-VL')
    sys.path.append(os.path.join(local_repo_path))
    from deepseek_vl.models import VLChatProcessor
    processor = VLChatProcessor.from_pretrained(model_dir)
    tokenizer = processor.tokenizer
    # flash_attn
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    attn_type = AttentionType(kwargs.pop('use_flash_attn', None), kwargs.pop('attn_type', None))
    attn_type.update_config(model_config)
    model, tokenizer = get_model_tokenizer_from_repo(
        model_dir, torch_dtype, model_kwargs, load_model, model_config=model_config, tokenizer=tokenizer, **kwargs)
    tokenizer.processor = processor
    if load_model:
        patch_output_clone(model.language_model.model.embed_tokens)
        patch_output_to_input_device(model.language_model.model.embed_tokens)
        func_list = ['generate', 'get_input_embeddings', 'gradient_checkpointing_enable', 'forward']
        _use_submodel_func(model, 'language_model', func_list)
        model.generation_config = model.language_model.generation_config
    return model, tokenizer


def get_model_tokenizer_llama2(model_dir: str,
                               torch_dtype: torch.dtype,
                               model_kwargs: Dict[str, Any],
                               load_model: bool = True,
                               **kwargs):
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model_config.pretraining_tp = 1
    return get_model_tokenizer_with_flash_attn(
        model_dir, torch_dtype, model_kwargs, load_model, model_config=model_config, **kwargs)


@register_model(
    ModelType.polylm_13b,
    'damo/nlp_polylm_13b_text_generation',
    template=TemplateType.default_generation,
    hf_model_id='DAMO-NLP-MT/polylm-13b')
def get_model_tokenizer_polylm(model_dir: str,
                               torch_dtype: torch.dtype,
                               model_kwargs: Dict[str, Any],
                               load_model: bool = True,
                               **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False, legacy=True)
    return get_model_tokenizer_from_repo(
        model_dir, torch_dtype, model_kwargs, load_model, tokenizer=tokenizer, **kwargs)


@register_model(
    ModelType.tongyi_finance_14b_chat_int4,
    'TongyiFinance/Tongyi-Finance-14B-Chat-Int4',
    template=TemplateType.qwen,
    requires=['auto_gptq>=0.5'],
    torch_dtype=torch.float16,
    support_flash_attn=True,
    support_vllm=True,
    tags=['financial'],
    hf_model_id='jxy/Tongyi-Finance-14B-Chat-Int4')
@register_model(
    ModelType.qwen_vl_chat_int4,
    'qwen/Qwen-VL-Chat-Int4',
    template=TemplateType.qwen_vl,
    requires=['auto_gptq>=0.5'],
    torch_dtype=torch.float16,
    function_kwargs={'get_qwen_function': get_model_tokenizer_qwen_vl},
    support_flash_attn=True,
    tags=['multi-modal', 'vision'],
    hf_model_id='Qwen/Qwen-VL-Chat-Int4')
def get_model_tokenizer_qwen_intx(model_dir: str,
                                  torch_dtype: torch.dtype,
                                  model_kwargs: Dict[str, Any],
                                  load_model: bool = True,
                                  **kwargs):
    get_qwen_function = kwargs.pop('get_qwen_function', get_model_tokenizer_qwen_chat)
    model, tokenizer = get_qwen_function(model_dir, torch_dtype, model_kwargs, load_model, **kwargs)
    return model, tokenizer


register_model(
    ModelType.skywork_13b,
    'skywork/Skywork-13B-base',
    template=TemplateType.default_generation,
    get_function=get_model_tokenizer_from_repo,
    hf_model_id='Skywork/Skywork-13B-base')


@register_model(ModelType.skywork_13b_chat, 'skywork/Skywork-13B-chat', template=TemplateType.skywork)
def get_skywork_model_tokenizer(model_dir: str,
                                torch_dtype: torch.dtype,
                                model_kwargs: Dict[str, Any],
                                load_model: bool = True,
                                **kwargs):
    model, tokenizer = get_model_tokenizer_from_repo(model_dir, torch_dtype, model_kwargs, load_model, **kwargs)
    tokenizer.add_tokens('[USER]')
    tokenizer.add_tokens('[BOT]')
    tokenizer.add_tokens('[SEP]')
    return model, tokenizer


@register_model(
    ModelType.codefuse_codellama_34b_chat,
    'codefuse-ai/CodeFuse-CodeLlama-34B',
    template=TemplateType.codefuse_codellama,
    support_flash_attn=True,
    support_vllm=True,
    support_lmdeploy=True,
    tags=['coding'],
    hf_model_id='codefuse-ai/CodeFuse-CodeLlama-34B')
def get_model_tokenizer_codellama(model_dir: str,
                                  torch_dtype: torch.dtype,
                                  model_kwargs: Dict[str, Any],
                                  load_model: bool = True,
                                  **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False, legacy=False)
    return get_model_tokenizer_with_flash_attn(
        model_dir, torch_dtype, model_kwargs, load_model, tokenizer=tokenizer, **kwargs)


@register_model(
    ModelType.phi2_3b,
    'AI-ModelScope/phi-2',
    template=TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    support_gradient_checkpointing=False,
    tags=['coding'],
    hf_model_id='microsoft/phi-2')
@register_model(
    ModelType.telechat_12b,
    'TeleAI/TeleChat-12B',
    template=TemplateType.telechat,
    support_flash_attn=True,
    hf_model_id='Tele-AI/TeleChat-12B')
@register_model(
    ModelType.telechat_12b_v2,
    'TeleAI/TeleChat-12B-v2',
    template=TemplateType.telechat_v2,
    eos_token=2,
    support_flash_attn=True,
    hf_model_id='Tele-AI/TeleChat-12B-v2')
@register_model(
    ModelType.telechat_12b_v2_gptq_int4,
    'swift/TeleChat-12B-V2-GPTQ-Int4',
    template=TemplateType.telechat_v2,
    eos_token=2,
    requires=['auto_gptq>=0.5'],
    support_flash_attn=True)
def get_model_tokenizer_phi(model_dir: str,
                            torch_dtype: torch.dtype,
                            model_kwargs: Dict[str, Any],
                            load_model: bool = True,
                            **kwargs):
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    attn_type = AttentionType(kwargs.pop('use_flash_attn', None), kwargs.pop('attn_type', None))
    model_config.flash_attn = attn_type.to_bool()
    return get_model_tokenizer_from_repo(
        model_dir, torch_dtype, model_kwargs, load_model, model_config=model_config, **kwargs)


@register_model(
    ModelType.telechat_7b,
    'TeleAI/TeleChat-7B',
    template=TemplateType.telechat,
    support_flash_attn=True,
    hf_model_id='Tele-AI/telechat-7B')
def get_model_tokenizer_telechat(model_dir: str,
                                 torch_dtype: torch.dtype,
                                 model_kwargs: Dict[str, Any],
                                 load_model: bool = True,
                                 **kwargs):
    if torch_dtype == torch.bfloat16:
        logger.info('telechat-7b does not support the bf16 dtype; the dtype is converted to fp16.')
        torch_dtype = torch.float16
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    attn_type = AttentionType(kwargs.pop('use_flash_attn', None), kwargs.pop('attn_type', None))
    model_config.flash_attn = attn_type.to_bool()
    return get_model_tokenizer_from_repo(
        model_dir, torch_dtype, model_kwargs, load_model, model_config=model_config, **kwargs)


@register_model(
    ModelType.deepseek_moe_16b_chat,
    'deepseek-ai/deepseek-moe-16b-chat',
    template=TemplateType.deepseek,
    support_flash_attn=True,
    support_vllm=True,
    tags=['moe'],
    hf_model_id='deepseek-ai/deepseek-moe-16b-chat')
@register_model(
    ModelType.deepseek_moe_16b,
    'deepseek-ai/deepseek-moe-16b-base',
    template=TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    tags=['moe'],
    hf_model_id='deepseek-ai/deepseek-moe-16b-base')
@register_model(
    ModelType.minicpm_moe_8x2b,
    'OpenBMB/MiniCPM-MoE-8x2B',
    template=TemplateType.minicpm,
    requires=['transformers>=4.36.0'],
    support_flash_attn=True,
    support_vllm=True,
    tags=['moe'],
    hf_model_id='openbmb/MiniCPM-MoE-8x2B')
def get_model_tokenizer_deepseek_moe(model_dir: str,
                                     torch_dtype: torch.dtype,
                                     model_kwargs: Dict[str, Any],
                                     load_model: bool = True,
                                     **kwargs):
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, torch_dtype, model_kwargs, load_model, **kwargs)
    if model is not None:
        # fix dtype bug
        mlp_cls = model.model.layers[1].mlp.__class__

        def _dtype_hook(module, input, output):
            return output.to(input[0].dtype)

        for module in model.modules():
            if isinstance(module, mlp_cls):
                module.register_forward_hook(_dtype_hook)
    return model, tokenizer


@register_model(
    ModelType.yuan2_2b_instruct,
    'YuanLLM/Yuan2.0-2B-hf',
    template=TemplateType.yuan,
    support_flash_attn=True,
    hf_model_id='IEITYuan/Yuan2-2B-hf')
@register_model(
    ModelType.yuan2_51b_instruct,
    'YuanLLM/Yuan2.0-51B-hf',
    template=TemplateType.yuan,
    support_flash_attn=True,
    hf_model_id='IEITYuan/Yuan2-51B-hf')
@register_model(
    ModelType.yuan2_102b_instruct,
    'YuanLLM/Yuan2.0-102B-hf',
    template=TemplateType.yuan,
    support_flash_attn=True,
    hf_model_id='IEITYuan/Yuan2-102B-hf')
@register_model(
    ModelType.yuan2_2b_janus_instruct,
    'YuanLLM/Yuan2-2B-Janus-hf',
    template=TemplateType.yuan,
    support_flash_attn=True,
    hf_model_id='IEITYuan/Yuan2-2B-Janus-hf')
@register_model(
    ModelType.yuan2_m32,
    'YuanLLM/Yuan2-M32-hf',
    template=TemplateType.yuan,
    tags=['moe'],
    support_flash_attn=True,
    hf_model_id='IEITYuan/Yuan2-M32-hf')
def get_model_tokenizer_yuan(model_dir: str,
                             torch_dtype: torch.dtype,
                             model_kwargs: Dict[str, Any],
                             load_model: bool = True,
                             **kwargs):
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    attn_type = AttentionType(kwargs.pop('use_flash_attn', None), kwargs.pop('attn_type', None))
    model_config.use_flash_attention = attn_type.to_bool()
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, add_eos_token=False, add_bos_token=False, eos_token='<eod>', legacy=True)
    addi_tokens = [
        '<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>', '<commit_before>',
        '<commit_msg>', '<commit_after>', '<jupyter_start>', '<jupyter_text>', '<jupyter_code>', '<jupyter_output>',
        '<empty_output>'
    ]
    tokenizer.add_tokens(addi_tokens, special_tokens=True)
    model, tokenizer = get_model_tokenizer_from_repo(
        model_dir, torch_dtype, model_kwargs, load_model, model_config=model_config, tokenizer=tokenizer, **kwargs)
    return model, tokenizer


@register_model(
    ModelType.orion_14b,
    'OrionStarAI/Orion-14B-Base',
    template=TemplateType.default_generation,
    support_flash_attn=True,
    hf_model_id='OrionStarAI/Orion-14B-Base')
@register_model(
    ModelType.orion_14b_chat,
    'OrionStarAI/Orion-14B-Chat',
    template=TemplateType.orion,
    support_flash_attn=True,
    ignore_file_pattern=[r'.+\.gguf$'],
    hf_model_id='OrionStarAI/Orion-14B-Chat')
def get_model_tokenizer_orion(model_dir: str,
                              torch_dtype: torch.dtype,
                              model_kwargs: Dict[str, Any],
                              load_model: bool = True,
                              **kwargs):
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    attn_type = AttentionType(kwargs.pop('use_flash_attn', None), kwargs.pop('attn_type', None))
    model_config._flash_attn_2_enabled = attn_type.to_bool()
    return get_model_tokenizer_from_repo(
        model_dir, torch_dtype, model_kwargs, load_model, model_config=model_config, **kwargs)


@register_model(
    ModelType.yi_vl_34b_chat,
    '01ai/Yi-VL-34B',
    template=TemplateType.yi_vl,
    support_flash_attn=True,
    requires=['transformers>=4.34'],
    tags=['multi-modal', 'vision'],
    hf_model_id='01-ai/Yi-VL-34B')
@register_model(
    ModelType.yi_vl_6b_chat,
    '01ai/Yi-VL-6B',
    template=TemplateType.yi_vl,
    support_flash_attn=True,
    requires=['transformers>=4.34'],
    tags=['multi-modal', 'vision'],
    hf_model_id='01-ai/Yi-VL-6B')
def get_model_tokenizer_yi_vl(model_dir: str,
                              torch_dtype: torch.dtype,
                              model_kwargs: Dict[str, Any],
                              load_model: bool = True,
                              **kwargs):
    if 'local_repo_path' in kwargs:
        local_repo_path = kwargs['local_repo_path']
    else:
        local_repo_path = git_clone_github('https://github.com/01-ai/Yi')
    sys.path.append(os.path.join(local_repo_path, 'VL'))
    from llava.model import LlavaLlamaForCausalLM, LlavaConfig
    from llava.model.constants import key_info

    model_config = LlavaConfig.from_pretrained(model_dir)
    mm_vision_tower = model_config.mm_vision_tower
    model_config.mm_vision_tower = os.path.join(model_dir, *mm_vision_tower.rsplit('/', maxsplit=2)[-2:])
    model_config.attention_dropout = 0.
    key_info['model_path'] = model_dir
    model, tokenizer = get_model_tokenizer_with_flash_attn(
        model_dir,
        torch_dtype,
        model_kwargs,
        load_model,
        model_config=model_config,
        automodel_class=LlavaLlamaForCausalLM,
        **kwargs)
    if model is not None:
        logger.info('Please ignore the above warning.')
        logger.info('Loading the parameters of vision_tower...')
        model.resize_token_embeddings(len(tokenizer))
        vision_tower = model.get_vision_tower()
        vision_tower.load_model()
        vision_tower.to(device=model.device, dtype=torch_dtype)
        if not hasattr(model.config, 'max_sequence_length'):
            model.config.max_sequence_length = 2048
        tokenizer.image_processor = vision_tower.image_processor
    return model, tokenizer


def _patch_minicpm_v_device_map(model) -> None:
    if not hasattr(model, 'hf_device_map') or len(model.hf_device_map.values()) == 1:
        return

    device = list(model.hf_device_map.values())[0]
    if hasattr(model, 'get_vision_embedding') and not hasattr(model, '_old_get_vision_embedding'):
        # minicpm-v-v2-chat; avoid double patching
        _old_get_vision_embedding = model.__class__.get_vision_embedding

        def _get_vision_embedding(self, pixel_values):
            if len(pixel_values) == 0:
                return _old_get_vision_embedding(self, pixel_values)
            output = _old_get_vision_embedding(self, pixel_values)
            return output.to(device=device)

        model.__class__._old_get_vision_embedding = _old_get_vision_embedding
        model.__class__.get_vision_embedding = _get_vision_embedding

    if hasattr(model, 'resampler'):  # minicpm-v-v2_5-chat
        patch_fixed_device(model.resampler, device)


@register_model(
    ModelType.minicpm_v_3b_chat,
    'OpenBMB/MiniCPM-V',
    template=TemplateType.minicpm_v,
    support_flash_attn=True,
    requires=['timm', 'transformers<4.42'],
    tags=['multi-modal', 'vision'],
    hf_model_id='openbmb/MiniCPM-V')
@register_model(
    ModelType.minicpm_v_v2_chat,
    'OpenBMB/MiniCPM-V-2',
    template=TemplateType.minicpm_v,
    support_flash_attn=True,
    requires=['timm', 'transformers<4.42'],
    tags=['multi-modal', 'vision'],
    hf_model_id='openbmb/MiniCPM-V-2')
def get_model_tokenizer_minicpm_v(model_dir: str,
                                  torch_dtype: torch.dtype,
                                  model_kwargs: Dict[str, Any],
                                  load_model: bool = True,
                                  **kwargs):
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, torch_dtype, model_kwargs, load_model, **kwargs)
    if load_model:
        model.resampler.to(torch_dtype)  # fix float32
        _patch_minicpm_v_device_map(model)
        func_list = ['generate', 'get_input_embeddings', 'forward']
        _use_submodel_func(model, 'llm', func_list)
        if hasattr(model, 'get_slice_image_placeholder'):
            tokenizer.get_slice_image_placeholder = MethodType(model.get_slice_image_placeholder, tokenizer)
            tokenizer.transform = MethodType(model.transform, tokenizer)
    return model, tokenizer


@contextmanager
def ignore_check_imports():
    import transformers.dynamic_module_utils as td

    @wraps(td.check_imports)
    def _check_imports(filename) -> List[str]:
        return td.get_relative_imports(filename)

    td._old_check_imports = td.check_imports
    td.check_imports = _check_imports
    yield
    td.check_imports = td._old_check_imports


@register_model(
    ModelType.minicpm_v_v2_6_chat,
    'OpenBMB/MiniCPM-V-2_6',
    template=TemplateType.minicpm_v_v2_6,
    support_flash_attn=True,
    support_vllm=True,
    requires=['timm', 'transformers>=4.36'],  # 'decord'
    placeholder_tokens=['<unk>'],
    function_kwargs={'version': 'v2.6'},
    tags=['multi-modal', 'vision', 'video'],
    hf_model_id='openbmb/MiniCPM-V-2_6')
@register_model(
    ModelType.minicpm_v_v2_5_chat,
    'OpenBMB/MiniCPM-Llama3-V-2_5',
    template=TemplateType.minicpm_v_v2_5,
    support_flash_attn=True,
    support_vllm=True,
    requires=['timm', 'transformers>=4.36'],
    placeholder_tokens=['<unk>'],
    tags=['multi-modal', 'vision'],
    hf_model_id='openbmb/MiniCPM-Llama3-V-2_5')
def get_model_tokenizer_minicpm_v_2_x(model_dir: str,
                                      torch_dtype: torch.dtype,
                                      model_kwargs: Dict[str, Any],
                                      load_model: bool = True,
                                      **kwargs):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    version = kwargs.get('version', 'v2.5')
    if load_model and version == 'v2.6':
        with ignore_check_imports():
            model_cls = get_class_from_dynamic_module('modeling_navit_siglip.SiglipVisionTransformer', model_dir)
            model_cls._no_split_modules = []
    model, tokenizer = get_model_tokenizer_minicpm_v(model_dir, torch_dtype, model_kwargs, load_model, **kwargs)
    tokenizer.processor = processor
    if load_model:
        embedding = model.get_input_embeddings()
        patch_output_clone(embedding)

    return model, tokenizer


def _patch_llava(model):
    if hasattr(model, '__old_generate'):
        return
    generate = model.generate
    model.__old_generate = generate

    @wraps(generate)
    def _new_generate(inputs=None, *args, **kwargs):
        input_ids = kwargs.pop('input_ids', None)
        if inputs is None and input_ids is not None:
            inputs = input_ids
        return generate(inputs, *args, **kwargs)

    model.generate = _new_generate


def get_model_tokenizer_llava_hf(model_dir: str, *args, **kwargs):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_dir)
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, *args, **kwargs)
    tokenizer.processor = processor
    return model, tokenizer


@register_model(
    ModelType.llava1_5_13b_instruct,
    'swift/llava-1.5-13b-hf',
    template=TemplateType.llava1_5,
    eos_token='</s>',
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.36'],
    tags=['multi-modal', 'vision'],
    hf_model_id='llava-hf/llava-1.5-13b-hf')
@register_model(
    ModelType.llava1_5_7b_instruct,
    'swift/llava-1.5-7b-hf',
    template=TemplateType.llava1_5,
    eos_token='</s>',
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.36'],
    tags=['multi-modal', 'vision'],
    hf_model_id='llava-hf/llava-1.5-7b-hf')
def get_model_tokenizer_llava_1_5(*args, **kwargs):
    from transformers import LlavaForConditionalGeneration
    kwargs['automodel_class'] = LlavaForConditionalGeneration
    return get_model_tokenizer_llava_hf(*args, **kwargs)


@register_model(
    ModelType.llava_onevision_qwen2_0_5b_ov,
    'AI-ModelScope/llava-onevision-qwen2-0.5b-ov-hf',
    template=TemplateType.llava_onevision_qwen,
    support_flash_attn=True,
    requires=['transformers>=4.45.0.dev0'],
    tags=['multi-modal', 'vision', 'video'],
    ignore_file_pattern=['onnx'],
    placeholder_tokens=['<image>'],
    hf_model_id='llava-hf/llava-onevision-qwen2-0.5b-ov-hf')
@register_model(
    ModelType.llava_onevision_qwen2_7b_ov,
    'AI-ModelScope/llava-onevision-qwen2-7b-ov-hf',
    template=TemplateType.llava_onevision_qwen,
    support_flash_attn=True,
    requires=['transformers>=4.45.0.dev0'],
    tags=['multi-modal', 'vision', 'video'],
    placeholder_tokens=['<image>'],
    hf_model_id='llava-hf/llava-onevision-qwen2-7b-ov-hf')
@register_model(
    ModelType.llava_onevision_qwen2_72b_ov,
    'AI-ModelScope/llava-onevision-qwen2-72b-ov-hf',
    template=TemplateType.llava_onevision_qwen,
    support_flash_attn=True,
    requires=['transformers>=4.45.0.dev0'],
    tags=['multi-modal', 'vision', 'video'],
    placeholder_tokens=['<image>'],
    hf_model_id='llava-hf/llava-onevision-qwen2-72b-ov-hf')
def get_model_tokenizer_llava_onevision(*args, **kwargs):
    from transformers import LlavaOnevisionForConditionalGeneration
    kwargs['automodel_class'] = LlavaOnevisionForConditionalGeneration
    return get_model_tokenizer_llava_hf(*args, **kwargs)


@register_model(
    ModelType.llava_next_72b_hf,
    'AI-ModelScope/llava-next-72b-hf',
    template=TemplateType.llava_qwen_hf,
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.39'],
    tags=['multi-modal', 'vision'],
    hf_model_id='llava-hf/llava-next-72b-hf')
@register_model(
    ModelType.llava_next_110b_hf,
    'AI-ModelScope/llava-next-110b-hf',
    template=TemplateType.llava_qwen_hf,
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.39'],
    tags=['multi-modal', 'vision'],
    hf_model_id='llava-hf/llava-next-110b-hf')
@register_model(
    ModelType.llama3_llava_next_8b_hf,
    'swift/llama3-llava-next-8b-hf',
    template=TemplateType.llama3_llava_next_hf,
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.39'],
    tags=['multi-modal', 'vision'],
    hf_model_id='llava-hf/llama3-llava-next-8b-hf')
@register_model(
    ModelType.llava1_6_vicuna_7b_instruct,
    'swift/llava-v1.6-vicuna-7b-hf',
    template=TemplateType.llava_vicuna,
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.39'],
    tags=['multi-modal', 'vision'],
    hf_model_id='llava-hf/llava-v1.6-vicuna-7b-hf')
@register_model(
    ModelType.llava1_6_vicuna_13b_instruct,
    'swift/llava-v1.6-vicuna-13b-hf',
    template=TemplateType.llava_vicuna,
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.39'],
    tags=['multi-modal', 'vision'],
    hf_model_id='llava-hf/llava-v1.6-vicuna-13b-hf')
@register_model(
    ModelType.llava1_6_mistral_7b_instruct,
    'swift/llava-v1.6-mistral-7b-hf',
    template=TemplateType.llava_mistral,
    support_flash_attn=True,
    support_vllm=True,
    requires=['transformers>=4.39'],
    tags=['multi-modal', 'vision'],
    hf_model_id='llava-hf/llava-v1.6-mistral-7b-hf')
@register_model(
    ModelType.llava1_6_llama3_1_8b_instruct,
    'DaozeZhang/llava-llama3.1-8b',
    template=TemplateType.llava_next_llama3,
    support_flash_attn=True,
    support_vllm=False,
    requires=['transformers>=4.41'],
    tags=['multi-modal', 'vision'])
def get_model_tokenizer_llava_next(*args, **kwargs):
    from transformers import LlavaNextForConditionalGeneration
    kwargs['automodel_class'] = LlavaNextForConditionalGeneration
    return get_model_tokenizer_llava_hf(*args, **kwargs)


@register_model(
    ModelType.llava1_6_yi_34b_instruct,
    'swift/llava-v1.6-34b-hf',
    template=TemplateType.llava_yi,
    support_flash_attn=True,
    support_vllm=True,
    eos_token='<|im_end|>',
    requires=['transformers>=4.39'],
    tags=['multi-modal', 'vision'],
    hf_model_id='llava-hf/llava-v1.6-34b-hf')
def get_model_tokenizer_llava_next_yi(*args, **kwargs):
    model, tokenizer = get_model_tokenizer_llava_next(*args, **kwargs)
    if model is not None:
        model.config.image_token_index = 64003
    return model, tokenizer


@register_model(
    ModelType.llava_next_video_7b_dpo_instruct,
    'swift/LLaVA-NeXT-Video-7B-DPO-hf',
    template=TemplateType.llava_next_video,
    support_flash_attn=True,
    requires=['transformers>=4.42', 'av'],
    tags=['multi-modal', 'video'],
    hf_model_id='llava-hf/LLaVA-NeXT-Video-7B-DPO-hf')
@register_model(
    ModelType.llava_next_video_7b_32k_instruct,
    'swift/LLaVA-NeXT-Video-7B-32K-hf',
    template=TemplateType.llava_next_video,
    support_flash_attn=True,
    requires=['transformers>=4.42', 'av'],
    tags=['multi-modal', 'video'],
    hf_model_id='llava-hf/LLaVA-NeXT-Video-7B-32K-hf')
@register_model(
    ModelType.llava_next_video_7b_instruct,
    'swift/LLaVA-NeXT-Video-7B-hf',
    template=TemplateType.llava_next_video,
    support_flash_attn=True,
    requires=['transformers>=4.42', 'av'],
    tags=['multi-modal', 'video'],
    hf_model_id='llava-hf/LLaVA-NeXT-Video-7B-hf')
def get_model_tokenizer_llava_next_video(*args, **kwargs):
    from transformers import LlavaNextVideoForConditionalGeneration
    kwargs['automodel_class'] = LlavaNextVideoForConditionalGeneration
    return get_model_tokenizer_llava_hf(*args, **kwargs)


@register_model(
    ModelType.llava_next_video_34b_instruct,
    'swift/LLaVA-NeXT-Video-34B-hf',
    template=TemplateType.llava_next_video_yi,
    support_flash_attn=True,
    requires=['transformers>=4.42', 'av'],
    tags=['multi-modal', 'video'],
    hf_model_id='llava-hf/LLaVA-NeXT-Video-34B-hf')
def get_model_tokenizer_llava_next_video_yi(*args, **kwargs):
    model, tokenizer = get_model_tokenizer_llava_next_video(*args, **kwargs)
    if model is not None:
        model.config.video_token_index = 64003
        model.config.image_token_index = 64004
    return model, tokenizer


@register_model(
    ModelType.llama3_llava_next_8b,
    'AI-Modelscope/llama3-llava-next-8b',
    template=TemplateType.llama3_llava_next,
    support_flash_attn=True,
    tags=['multi-modal', 'vision'],
    function_kwargs={'llm_model_type': 'next_llama'},
    hf_model_id='lmms-lab/llama3-llava-next-8b')
@register_model(
    ModelType.llava_next_72b,
    'AI-Modelscope/llava-next-72b',
    template=TemplateType.llava_qwen,
    support_flash_attn=True,
    tags=['multi-modal', 'vision'],
    function_kwargs={'llm_model_type': 'next_qwen'},
    hf_model_id='lmms-lab/llava-next-72b')
@register_model(
    ModelType.llava_next_110b,
    'AI-Modelscope/llava-next-110b',
    template=TemplateType.llava_qwen,
    support_flash_attn=True,
    tags=['multi-modal', 'vision'],
    function_kwargs={'llm_model_type': 'next_qwen'},
    hf_model_id='lmms-lab/llava-next-110b')
def get_model_tokenizer_llava(model_dir: str,
                              torch_dtype: torch.dtype,
                              model_kwargs: Dict[str, Any],
                              load_model: bool = True,
                              **kwargs):
    llm_model_type = kwargs.pop('llm_model_type')
    if 'local_repo_path' in kwargs:
        local_repo_path = kwargs['local_repo_path']
    elif 'next' in llm_model_type:
        repo_path = 'https://github.com/LLaVA-VL/LLaVA-NeXT'
        local_repo_path = git_clone_github(repo_path)
    else:
        repo_path = 'https://github.com/haotian-liu/LLaVA'
        local_repo_path = git_clone_github(repo_path)
    sys.path.append(os.path.join(local_repo_path))

    if llm_model_type == 'mistral':
        from llava.model import LlavaMistralForCausalLM, LlavaMistralConfig
        model_config = LlavaMistralConfig.from_pretrained(model_dir)
        automodel_class = LlavaMistralForCausalLM
    elif 'llama' in llm_model_type:  # llama
        from llava.model import LlavaLlamaForCausalLM, LlavaConfig
        if not hasattr(LlavaLlamaForCausalLM, '__old_forward'):  # Avoid double patching
            forward = LlavaLlamaForCausalLM.forward
            LlavaLlamaForCausalLM.__old_forward = forward

            @wraps(forward)
            def _new_forward(*args, **kwargs):
                kwargs.pop('cache_position', None)
                return forward(*args, **kwargs)

            LlavaLlamaForCausalLM.forward = _new_forward
        model_config = LlavaConfig.from_pretrained(model_dir)
        automodel_class = LlavaLlamaForCausalLM
    else:  # qwen
        from llava.model import LlavaQwenForCausalLM
        automodel_class = LlavaQwenForCausalLM
        model_config = AutoConfig.from_pretrained(model_dir)

    model_config.mm_vision_tower = snapshot_download('AI-ModelScope/clip-vit-large-patch14-336')
    model, tokenizer = get_model_tokenizer_with_flash_attn(
        model_dir,
        torch_dtype,
        model_kwargs,
        load_model,
        model_config=model_config,
        automodel_class=automodel_class,
        **kwargs)

    if model is not None:
        model.resize_token_embeddings(len(tokenizer))
        vision_tower = model.get_vision_tower()
        device_map = str(model_kwargs.get('device_map', str(model.device)))
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        if not hasattr(model.config, 'max_sequence_length'):
            model.config.max_sequence_length = 2048
        _patch_llava(model)
        tokenizer.image_processor = vision_tower.image_processor
    return model, tokenizer


@register_model(
    ModelType.idefics3_8b_llama3,
    'AI-ModelScope/Idefics3-8B-Llama3',
    template=TemplateType.idefics3,
    support_flash_attn=True,
    placeholder_tokens=['<image>'],
    requires=['transformers>=4.45.0.dev0'],
    tags=['multi-modal', 'vision'],
    hf_model_id='HuggingFaceM4/Idefics3-8B-Llama3')
def get_model_tokenizer_idefics(model_dir: str, *args, **kwargs):
    from transformers import AutoProcessor, AutoModelForVision2Seq
    processor = AutoProcessor.from_pretrained(model_dir)
    kwargs['automodel_class'] = AutoModelForVision2Seq
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, *args, **kwargs)
    tokenizer.processor = processor
    return model, tokenizer


@register_model(
    ModelType.mplug_owl2_chat,
    'iic/mPLUG-Owl2',
    template=TemplateType.mplug_owl2,
    requires=['transformers<4.35', 'icecream'],
    eos_token='</s>',
    function_kwargs={'get_model_tokenizer_function': get_model_tokenizer_with_flash_attn},
    support_flash_attn=True,
    tags=['multi-modal', 'vision'],
    hf_model_id='MAGAer13/mplug-owl2-llama2-7b')
@register_model(
    ModelType.mplug_owl2_1_chat,
    'iic/mPLUG-Owl2.1',
    template=TemplateType.mplug_owl2,
    requires=['transformers<4.35', 'icecream'],
    eos_token='<|endoftext|>',
    function_kwargs={
        'vocab_size': 151851,
        'get_model_tokenizer_function': get_model_tokenizer_qwen
    },
    support_flash_attn=True,
    tags=['multi-modal', 'vision'],
    hf_model_id='Mizukiluke/mplug_owl_2_1')
def get_model_tokenizer_mplug_owl2(model_dir: str,
                                   torch_dtype: torch.dtype,
                                   model_kwargs: Dict[str, Any],
                                   load_model: bool = True,
                                   **kwargs):
    if 'local_repo_path' in kwargs:
        local_repo_path = kwargs['local_repo_path']
    else:
        local_repo_path = git_clone_github('https://github.com/X-PLUG/mPLUG-Owl')
    local_repo_path = os.path.join(local_repo_path, 'mPLUG-Owl2')
    sys.path.append(os.path.join(local_repo_path))

    # register
    # https://github.com/X-PLUG/mPLUG-Owl/blob/main/mPLUG-Owl2/mplug_owl2/model/modeling_mplug_owl2.py#L447
    from transformers.models.clip.image_processing_clip import CLIPImageProcessor
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    vocab_size = kwargs.pop('vocab_size', None)
    if vocab_size is not None:
        model_config.vocab_size = vocab_size
    get_model_tokenizer_function = kwargs.pop('get_model_tokenizer_function')
    model, tokenizer = get_model_tokenizer_function(
        model_dir, torch_dtype, model_kwargs, load_model, model_config=model_config, **kwargs)
    logger.info('Please ignore the unimported warning.')
    processor = CLIPImageProcessor.from_pretrained(model_dir)
    tokenizer.processor = processor
    return model, tokenizer

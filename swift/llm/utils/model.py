# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect
import os
from functools import partial, update_wrapper
from types import MethodType
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Type

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from modelscope import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                        BitsAndBytesConfig, GenerationConfig, GPTQConfig,
                        snapshot_download)
from packaging import version
from torch import Tensor
from torch import dtype as Dtype
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.auto.auto_factory import _BaseAutoModelClass
from transformers.models.auto.tokenization_auto import get_tokenizer_config
from transformers.utils.versions import require_version

from swift import get_logger
from swift.utils import is_dist, is_local_master
from .template import TemplateType

logger = get_logger()

# Model Home: 'https://modelscope.cn/models/{model_id_or_path}/summary'
MODEL_MAPPING: Dict[str, Dict[str, Any]] = {}


class ModelType:
    # qwen
    qwen_1_8b = 'qwen-1_8b'
    qwen_1_8b_chat = 'qwen-1_8b-chat'
    qwen_1_8b_chat_int4 = 'qwen-1_8b-chat-int4'
    qwen_1_8b_chat_int8 = 'qwen-1_8b-chat-int8'
    qwen_7b = 'qwen-7b'
    qwen_7b_chat = 'qwen-7b-chat'
    qwen_7b_chat_int4 = 'qwen-7b-chat-int4'
    qwen_7b_chat_int8 = 'qwen-7b-chat-int8'
    qwen_14b = 'qwen-14b'
    qwen_14b_chat = 'qwen-14b-chat'
    qwen_14b_chat_int4 = 'qwen-14b-chat-int4'
    qwen_14b_chat_int8 = 'qwen-14b-chat-int8'
    qwen_72b = 'qwen-72b'
    qwen_72b_chat = 'qwen-72b-chat'
    qwen_72b_chat_int4 = 'qwen-72b-chat-int4'
    qwen_72b_chat_int8 = 'qwen-72b-chat-int8'
    # qwen-vl
    qwen_vl = 'qwen-vl'
    qwen_vl_chat = 'qwen-vl-chat'
    qwen_vl_chat_int4 = 'qwen-vl-chat-int4'
    # qwen-audio
    qwen_audio = 'qwen-audio'
    qwen_audio_chat = 'qwen-audio-chat'
    # chatglm
    chatglm2_6b = 'chatglm2-6b'
    chatglm2_6b_32k = 'chatglm2-6b-32k'
    chatglm3_6b_base = 'chatglm3-6b-base'
    chatglm3_6b = 'chatglm3-6b'
    chatglm3_6b_32k = 'chatglm3-6b-32k'
    # llama2
    llama2_7b = 'llama2-7b'
    llama2_7b_chat = 'llama2-7b-chat'
    llama2_13b = 'llama2-13b'
    llama2_13b_chat = 'llama2-13b-chat'
    llama2_70b = 'llama2-70b'
    llama2_70b_chat = 'llama2-70b-chat'
    # yi
    yi_6b = 'yi-6b'
    yi_6b_200k = 'yi-6b-200k'
    yi_6b_chat = 'yi-6b-chat'
    yi_34b = 'yi-34b'
    yi_34b_200k = 'yi-34b-200k'
    yi_34b_chat = 'yi-34b-chat'
    # deepseek
    deepseek_7b = 'deepseek-7b'
    deepseek_7b_chat = 'deepseek-7b-chat'
    deepseek_67b = 'deepseek-67b'
    deepseek_67b_chat = 'deepseek-67b-chat'
    # openbuddy
    openbuddy_llama2_13b_chat = 'openbuddy-llama2-13b-chat'
    openbuddy_llama2_65b_chat = 'openbuddy-llama-65b-chat'
    openbuddy_llama2_70b_chat = 'openbuddy-llama2-70b-chat'
    openbuddy_mistral_7b_chat = 'openbuddy-mistral-7b-chat'
    openbuddy_zephyr_7b_chat = 'openbuddy-zephyr-7b-chat'
    openbuddy_deepseek_67b_chat = 'openbuddy-deepseek-67b-chat'
    # mistral
    mistral_7b = 'mistral-7b'
    mistral_7b_chat = 'mistral-7b-chat'
    mistral_7b_chat_v2 = 'mistral-7b-chat-v2'
    mixtral_7b_moe = 'mixtral-7b-moe'
    mixtral_7b_moe_chat = 'mixtral-7b-moe-chat'
    # baichuan
    baichuan_7b = 'baichuan-7b'
    baichuan_13b = 'baichuan-13b'
    baichuan_13b_chat = 'baichuan-13b-chat'
    baichuan2_7b = 'baichuan2-7b'
    baichuan2_7b_chat = 'baichuan2-7b-chat'
    baichuan2_7b_chat_int4 = 'baichuan2-7b-chat-int4'
    baichuan2_13b = 'baichuan2-13b'
    baichuan2_13b_chat = 'baichuan2-13b-chat'
    baichuan2_13b_chat_int4 = 'baichuan2-13b-chat-int4'
    # internlm
    internlm_7b = 'internlm-7b'
    internlm_7b_chat = 'internlm-7b-chat'
    internlm_7b_chat_8k = 'internlm-7b-chat-8k'
    internlm_20b = 'internlm-20b'
    internlm_20b_chat = 'internlm-20b-chat'
    # xverse
    xverse_7b = 'xverse-7b'
    xverse_7b_chat = 'xverse-7b-chat'
    xverse_13b = 'xverse-13b'
    xverse_13b_chat = 'xverse-13b-chat'
    xverse_65b = 'xverse-65b'
    # vivo
    bluelm_7b = 'bluelm-7b'
    bluelm_7b_32k = 'bluelm-7b-32k'
    bluelm_7b_chat = 'bluelm-7b-chat'
    bluelm_7b_chat_32k = 'bluelm-7b-chat-32k'
    # ziya
    ziya2_13b = 'ziya2-13b'
    ziya2_13b_chat = 'ziya2-13b-chat'
    # skywork
    skywork_13b = 'skywork-13b'
    skywork_13b_chat = 'skywork-13b-chat'
    # zephyr
    zephyr_7b_beta_chat = 'zephyr-7b-beta-chat'
    # sus
    sus_34b_chat = 'sus-34b-chat'
    # other
    polylm_13b = 'polylm-13b'
    seqgpt_560m = 'seqgpt-560m'

    # domain-specific
    # financial
    tongyi_finance_14b = 'tongyi-finance-14b'
    tongyi_finance_14b_chat = 'tongyi-finance-14b-chat'
    tongyi_finance_14b_chat_int4 = 'tongyi-finance-14b-chat-int4'
    # coding
    # codefuse
    codefuse_codellama_34b_chat = 'codefuse-codellama-34b-chat'
    # deepseek-coder
    deepseek_coder_1_3b = 'deepseek-coder-1_3b'
    deepseek_coder_1_3b_chat = 'deepseek-coder-1_3b-chat'
    deepseek_coder_6_7b = 'deepseek-coder-6_7b'
    deepseek_coder_6_7b_chat = 'deepseek-coder-6_7b-chat'
    deepseek_coder_33b = 'deepseek-coder-33b'
    deepseek_coder_33b_chat = 'deepseek-coder-33b-chat'
    # phi
    phi2_3b = 'phi2-3b'

    cogagent_chat = 'cogagent-chat'
    cogagent_vqa = 'cogagent-vqa'

    @classmethod
    def get_model_name_list(cls) -> List[str]:
        res = []
        for k in cls.__dict__.keys():
            if k.startswith('__') or k == 'get_model_name_list':
                continue
            res.append(cls.__dict__[k])
        return res


class LoRATM(NamedTuple):
    # default lora target modules. qkv
    baichuan = ['W_pack']
    chatglm = ['query_key_value']
    llama2 = ['q_proj', 'k_proj', 'v_proj']
    qwen = ['c_attn']
    polylm = ['c_attn']
    bloom = ['query_key_value']
    cogagent = [
        'vision_expert_query_key_value', 'vision_expert_dense',
        'language_expert_query_key_value', 'language_expert_dense', 'query',
        'key_value', 'dense'
    ]
    phi = ['Wqkv']


GetModelTokenizerFunction = Callable[..., Tuple[Optional[PreTrainedModel],
                                                PreTrainedTokenizerBase]]


def register_model(
    model_type: str,
    model_id_or_path: Optional[str],
    lora_target_modules: Optional[List[str]] = None,
    template: str = TemplateType.default,
    get_function: Optional[GetModelTokenizerFunction] = None,
    *,
    requires: Optional[List[str]] = None,
    torch_dtype: Optional[Dtype] = None,
    automodel_class: Type[_BaseAutoModelClass] = AutoModelForCausalLM,
    revision: str = 'master',
    ignore_file_pattern: Optional[List[str]] = None,
    function_kwargs: Optional[Dict[str, Any]] = None,
    exists_ok: bool = False,
    eos_token: Optional[str] = None,
    **kwargs
) -> Optional[Callable[[GetModelTokenizerFunction],
                       GetModelTokenizerFunction]]:
    if not exists_ok and model_type in MODEL_MAPPING:
        raise ValueError(
            f'The `{model_type}` has already been registered in the MODEL_MAPPING.'
        )
    if requires is None:
        requires = []
    if function_kwargs is None:
        function_kwargs = {}
    model_info = {
        'model_id_or_path': model_id_or_path,
        'lora_target_modules': lora_target_modules,
        'template': template,
        'requires': requires,
        'torch_dtype': torch_dtype,
        'automodel_class': automodel_class,
        'ignore_file_pattern': ignore_file_pattern,
        'revision': revision,
        'eos_token': eos_token,
        **kwargs
    }

    if get_function is not None:
        if len(function_kwargs) > 0:
            get_function = partial(get_function, **function_kwargs)
        model_info['get_function'] = get_function
        MODEL_MAPPING[model_type] = model_info
        return

    def _register_model(
            get_function: GetModelTokenizerFunction
    ) -> GetModelTokenizerFunction:
        _old_get_function = get_function
        if len(function_kwargs) > 0:
            get_function = partial(get_function, **function_kwargs)
        model_info['get_function'] = get_function
        MODEL_MAPPING[model_type] = model_info
        return _old_get_function

    return _register_model


@register_model(
    ModelType.internlm_20b,
    'Shanghai_AI_Laboratory/internlm-20b',
    LoRATM.llama2,
    TemplateType.default_generation_bos,
    support_vllm=True)
@register_model(
    ModelType.internlm_7b,
    'Shanghai_AI_Laboratory/internlm-7b',
    LoRATM.llama2,
    TemplateType.default_generation_bos,
    support_vllm=True)
@register_model(ModelType.bluelm_7b_chat_32k, 'vivo-ai/BlueLM-7B-Chat-32K',
                LoRATM.llama2, TemplateType.bluelm)
@register_model(ModelType.bluelm_7b_chat, 'vivo-ai/BlueLM-7B-Chat',
                LoRATM.llama2, TemplateType.bluelm)
@register_model(ModelType.bluelm_7b_32k, 'vivo-ai/BlueLM-7B-Base-32K',
                LoRATM.llama2, TemplateType.default_generation_bos)
@register_model(ModelType.bluelm_7b, 'vivo-ai/BlueLM-7B-Base', LoRATM.llama2,
                TemplateType.default_generation_bos)
@register_model(
    ModelType.seqgpt_560m,
    'damo/nlp_seqgpt-560m',
    LoRATM.bloom,
    TemplateType.default_generation,
    support_vllm=True)
@register_model(ModelType.xverse_13b_chat, 'xverse/XVERSE-13B-Chat',
                LoRATM.llama2, TemplateType.xverse)
@register_model(ModelType.xverse_13b, 'xverse/XVERSE-13B', LoRATM.llama2,
                TemplateType.default_generation)
@register_model(ModelType.xverse_65b, 'xverse/XVERSE-65B', LoRATM.llama2,
                TemplateType.default_generation)
@register_model(ModelType.xverse_7b_chat, 'xverse/XVERSE-7B-Chat',
                LoRATM.llama2, TemplateType.xverse)
@register_model(ModelType.xverse_7b, 'xverse/XVERSE-7B', LoRATM.llama2,
                TemplateType.default_generation)
@register_model(
    ModelType.baichuan_13b_chat,
    'baichuan-inc/Baichuan-13B-Chat',
    LoRATM.baichuan,
    TemplateType.baichuan,
    requires=['transformers<4.34'],
    support_vllm=True)
@register_model(
    ModelType.baichuan_7b,
    'baichuan-inc/baichuan-7B',
    LoRATM.baichuan,
    TemplateType.default_generation,
    requires=['transformers<4.34'],
    support_vllm=True)
def get_model_tokenizer_from_repo(model_dir: str,
                                  torch_dtype: Dtype,
                                  model_kwargs: Dict[str, Any],
                                  load_model: bool = True,
                                  model_config=None,
                                  tokenizer=None,
                                  automodel_class=AutoModelForCausalLM,
                                  **kwargs):
    """load from an independent repository"""
    if model_config is None:
        model_config = AutoConfig.from_pretrained(
            model_dir, trust_remote_code=True)
    model_config.torch_dtype = torch_dtype
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=True)
    eos_token = kwargs.get('eos_token')
    if eos_token is not None:
        tokenizer.eos_token = eos_token
    model = None
    if load_model:
        model = automodel_class.from_pretrained(
            model_dir,
            config=model_config,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            **model_kwargs)
    return model, tokenizer


@register_model(
    ModelType.cogagent_chat,
    'ZhipuAI/cogagent-chat',
    LoRATM.cogagent,
    TemplateType.cogagent,
    requires=['transformers>=4.36'],
    support_vllm=False)
@register_model(
    ModelType.cogagent_vqa,
    'ZhipuAI/cogagent-vqa',
    LoRATM.cogagent,
    TemplateType.cogagent,
    requires=['transformers>=4.36'],
    support_vllm=False)
def get_model_tokenizer_from_repo_cogagent(
        model_dir: str,
        torch_dtype: Dtype,
        model_kwargs: Dict[str, Any],
        load_model: bool = True,
        model_config=None,
        tokenizer=None,
        automodel_class=AutoModelForCausalLM,
        **kwargs):
    """load from an independent repository"""
    if model_config is None:
        model_config = AutoConfig.from_pretrained(
            model_dir, trust_remote_code=True)
    model_config.torch_dtype = torch_dtype
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            'AI-ModelScope/vicuna-7b-v1.5',
            trust_remote_code=True,
            padding_side='left')
    eos_token = kwargs.get('eos_token')
    if eos_token is not None:
        tokenizer.eos_token = eos_token
    model = None
    if load_model:
        model = automodel_class.from_pretrained(
            model_dir,
            config=model_config,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            **model_kwargs)
        logger.info(
            'CogAgent with FusedLayerNorm will cause an training loss of Nan, '
            'to avoid this, please uninstall apex.')
    return model, tokenizer


@register_model(
    ModelType.internlm_20b_chat,
    'Shanghai_AI_Laboratory/internlm-chat-20b',
    LoRATM.llama2,
    TemplateType.internlm,
    support_vllm=True)
@register_model(
    ModelType.internlm_7b_chat_8k,
    'Shanghai_AI_Laboratory/internlm-chat-7b-8k',
    LoRATM.llama2,
    TemplateType.internlm,
    support_vllm=True)
@register_model(
    ModelType.internlm_7b_chat,
    'Shanghai_AI_Laboratory/internlm-chat-7b-v1_1',
    LoRATM.llama2,
    TemplateType.internlm,
    support_vllm=True)
def get_model_tokenizer_internlm_chat(model_dir: str,
                                      torch_dtype: Dtype,
                                      model_kwargs: Dict[str, Any],
                                      load_model: bool = True,
                                      **kwargs):
    model, tokenizer = get_model_tokenizer_from_repo(model_dir, torch_dtype,
                                                     model_kwargs, load_model,
                                                     **kwargs)
    del tokenizer.__class__.eos_token_id
    tokenizer.eos_token = '<eoa>'
    return model, tokenizer


@register_model(
    ModelType.baichuan_13b,
    'baichuan-inc/Baichuan-13B-Base',
    LoRATM.baichuan,
    TemplateType.default_generation,
    requires=['transformers<4.34'],
    support_vllm=True)
def get_model_tokenizer_baichuan_13b(model_dir: str,
                                     torch_dtype: Dtype,
                                     model_kwargs: Dict[str, Any],
                                     load_model: bool = True,
                                     **kwargs):
    model, tokenizer = get_model_tokenizer_from_repo(model_dir, torch_dtype,
                                                     model_kwargs, load_model,
                                                     **kwargs)
    # baichuan-13b does not implement the `get_input_embeddings` function
    # fix gradient_checkpointing bug
    try:
        model.get_input_embeddings()
    except NotImplementedError:
        model.__class__.get_input_embeddings = lambda self: self.model.embed_tokens
    return model, tokenizer


@register_model(
    ModelType.baichuan2_13b_chat,
    'baichuan-inc/Baichuan2-13B-Chat',
    LoRATM.baichuan,
    TemplateType.baichuan,
    support_vllm=True)
@register_model(
    ModelType.baichuan2_13b,
    'baichuan-inc/Baichuan2-13B-Base',
    LoRATM.baichuan,
    TemplateType.default_generation,
    support_vllm=True)
def get_model_tokenizer_baichuan2_13b(model_dir: str,
                                      torch_dtype: Dtype,
                                      model_kwargs: Dict[str, Any],
                                      load_model: bool = True,
                                      **kwargs):
    # patch: baichuan2_13b configuration_baichuan.py bug
    model_config = AutoConfig.from_pretrained(
        model_dir, trust_remote_code=True)
    gradient_checkpointing = model_config.gradient_checkpointing
    if isinstance(gradient_checkpointing, (tuple, list)):
        model_config.gradient_checkpointing = gradient_checkpointing[0]
    return get_model_tokenizer_baichuan2(model_dir, torch_dtype, model_kwargs,
                                         load_model, model_config, **kwargs)


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
    ModelType.baichuan2_7b_chat,
    'baichuan-inc/Baichuan2-7B-Chat',
    LoRATM.baichuan,
    TemplateType.baichuan,
    support_vllm=True)
@register_model(
    ModelType.baichuan2_7b,
    'baichuan-inc/Baichuan2-7B-Base',
    LoRATM.baichuan,
    TemplateType.default_generation,
    support_vllm=True)
def get_model_tokenizer_baichuan2(model_dir: str,
                                  torch_dtype: Dtype,
                                  model_kwargs: Dict[str, Any],
                                  load_model: bool = True,
                                  model_config=None,
                                  **kwargs):
    model, tokenizer = get_model_tokenizer_from_repo(model_dir, torch_dtype,
                                                     model_kwargs, load_model,
                                                     model_config, **kwargs)
    if model is not None:
        new_forward = MethodType(patch_baichuan2_lm_head_forward,
                                 model.lm_head)
        if hasattr(model, '_old_forward'):  # device_map
            model.lm_head._old_forward = new_forward
        else:
            model.lm_head.forward = new_forward
    return model, tokenizer


@register_model(
    ModelType.baichuan2_13b_chat_int4,
    'baichuan-inc/Baichuan2-13B-Chat-4bits',
    LoRATM.baichuan,
    TemplateType.baichuan,
    function_kwargs={
        'get_baichuan2_function': get_model_tokenizer_baichuan2_13b
    },
    torch_dtype=torch.bfloat16)
@register_model(
    ModelType.baichuan2_7b_chat_int4,
    'baichuan-inc/Baichuan2-7B-Chat-4bits',
    LoRATM.baichuan,
    TemplateType.baichuan,
    torch_dtype=torch.bfloat16)
def get_model_tokenizer_baichuan2_int4(model_dir: str,
                                       torch_dtype: Dtype,
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
    get_baichuan2_function = kwargs.pop('get_baichuan2_function',
                                        get_model_tokenizer_baichuan2)
    model, tokenizer = get_baichuan2_function(model_dir, torch_dtype,
                                              model_kwargs, load_model,
                                              **kwargs)
    if device_map != 'auto':
        accelerate.infer_auto_device_map = _old_infer_auto_device_map
    if model is not None:
        model.train()
        model._is_quantized_training_enabled = True
        model.is_loaded_in_4bit = True
    return model, tokenizer


def remove_property(tokenizer_cls: Type[PreTrainedTokenizerBase],
                    tokenizer_config: Dict[str, Any]) -> None:
    for k, v in tokenizer_cls.__dict__.items():
        if k.endswith('_token') and isinstance(
                v, property) and k in tokenizer_config:
            setattr(tokenizer_cls, k, tokenizer_config[k])


@register_model(
    ModelType.chatglm3_6b_32k,
    'ZhipuAI/chatglm3-6b-32k',
    LoRATM.chatglm,
    TemplateType.chatglm3,
    support_vllm=True)
@register_model(
    ModelType.chatglm3_6b,
    'ZhipuAI/chatglm3-6b',
    LoRATM.chatglm,
    TemplateType.chatglm3,
    support_vllm=True)
@register_model(
    ModelType.chatglm3_6b_base,
    'ZhipuAI/chatglm3-6b-base',
    LoRATM.chatglm,
    TemplateType.chatglm_generation,
    support_vllm=True)
@register_model(
    ModelType.chatglm2_6b_32k,
    'ZhipuAI/chatglm2-6b-32k',
    LoRATM.chatglm,
    TemplateType.chatglm2,
    support_vllm=True)
@register_model(
    ModelType.chatglm2_6b,
    'ZhipuAI/chatglm2-6b',
    LoRATM.chatglm,
    TemplateType.chatglm2,
    support_vllm=True)
def get_model_tokenizer_chatglm(model_dir: str,
                                torch_dtype: Dtype,
                                model_kwargs: Dict[str, Any],
                                load_model: bool = True,
                                **kwargs):
    if model_kwargs.get('quantization_config') is not None:
        model_kwargs['quantization_config'].llm_int8_skip_modules = [
            'output_layer'
        ]
    # fix transformers>=4.34 bug
    if version.parse(transformers.__version__) >= version.parse('4.34'):
        tokenizer_config = get_tokenizer_config(model_dir)
        class_ref = tokenizer_config['auto_map']['AutoTokenizer'][0]
        tokenizer_cls = get_class_from_dynamic_module(class_ref, model_dir)
        tokenizer_cls._auto_class = 'AutoTokenizer'
        remove_property(tokenizer_cls, tokenizer_config)
        kwargs['tokenizer'] = tokenizer_cls.from_pretrained(
            model_dir, trust_remote_code=True)
    model, tokenizer = get_model_tokenizer_from_repo(model_dir, torch_dtype,
                                                     model_kwargs, load_model,
                                                     **kwargs)
    if model is not None:
        from torch.nn import CrossEntropyLoss
        __old_forward = CrossEntropyLoss.forward

        def cross_entropy_forward(self, inputs: Tensor,
                                  target: Tensor) -> Tensor:
            target = target.to(device=inputs.device)
            return __old_forward(self, inputs, target)

        CrossEntropyLoss.forward = cross_entropy_forward
    return model, tokenizer


@register_model(
    ModelType.deepseek_coder_1_3b,
    'deepseek-ai/deepseek-coder-1.3b-base',
    LoRATM.llama2,
    TemplateType.default_generation_bos,
    support_flash_attn=True,
    support_vllm=True)
@register_model(
    ModelType.deepseek_coder_6_7b,
    'deepseek-ai/deepseek-coder-6.7b-base',
    LoRATM.llama2,
    TemplateType.default_generation_bos,
    support_flash_attn=True,
    support_vllm=True)
@register_model(
    ModelType.deepseek_coder_33b,
    'deepseek-ai/deepseek-coder-33b-base',
    LoRATM.llama2,
    TemplateType.default_generation_bos,
    support_flash_attn=True,
    support_vllm=True)
@register_model(
    ModelType.deepseek_coder_1_3b_chat,
    'deepseek-ai/deepseek-coder-1.3b-instruct',
    LoRATM.llama2,
    TemplateType.deepseek_coder,
    eos_token='<|EOT|>',
    support_flash_attn=True,
    support_vllm=True)
@register_model(
    ModelType.deepseek_coder_6_7b_chat,
    'deepseek-ai/deepseek-coder-6.7b-instruct',
    LoRATM.llama2,
    TemplateType.deepseek_coder,
    eos_token='<|EOT|>',
    support_flash_attn=True,
    support_vllm=True)
@register_model(
    ModelType.deepseek_coder_33b_chat,
    'deepseek-ai/deepseek-coder-33b-instruct',
    LoRATM.llama2,
    TemplateType.deepseek_coder,
    eos_token='<|EOT|>',
    support_flash_attn=True,
    support_vllm=True)
@register_model(
    ModelType.openbuddy_deepseek_67b_chat,
    'OpenBuddy/openbuddy-deepseek-67b-v15.2',
    LoRATM.llama2,
    TemplateType.openbuddy,
    support_flash_attn=True,
    support_vllm=True)
@register_model(
    ModelType.deepseek_67b_chat,
    'deepseek-ai/deepseek-llm-67b-chat',
    LoRATM.llama2,
    TemplateType.deepseek,
    support_flash_attn=True,
    support_vllm=True)
@register_model(
    ModelType.deepseek_67b,
    'deepseek-ai/deepseek-llm-67b-base',
    LoRATM.llama2,
    TemplateType.default_generation_bos,
    support_flash_attn=True,
    support_vllm=True)
@register_model(
    ModelType.deepseek_7b_chat,
    'deepseek-ai/deepseek-llm-7b-chat',
    LoRATM.llama2,
    TemplateType.deepseek,
    support_flash_attn=True,
    support_vllm=True)
@register_model(
    ModelType.deepseek_7b,
    'deepseek-ai/deepseek-llm-7b-base',
    LoRATM.llama2,
    TemplateType.default_generation_bos,
    support_flash_attn=True,
    support_vllm=True)
@register_model(
    ModelType.sus_34b_chat,
    'SUSTC/SUS-Chat-34B',
    LoRATM.llama2,
    TemplateType.sus,
    support_flash_attn=True,
    support_vllm=True)
@register_model(
    ModelType.openbuddy_zephyr_7b_chat,
    'OpenBuddy/openbuddy-zephyr-7b-v14.1',
    LoRATM.llama2,
    TemplateType.openbuddy,
    requires=['transformers>=4.34'],
    support_flash_attn=True,
    support_vllm=True)
@register_model(
    ModelType.zephyr_7b_beta_chat,
    'modelscope/zephyr-7b-beta',
    LoRATM.llama2,
    TemplateType.zephyr,
    requires=['transformers>=4.34'],
    support_flash_attn=True,
    support_vllm=True)
@register_model(
    ModelType.yi_6b_chat,
    '01ai/Yi-6B-Chat',
    LoRATM.llama2,
    TemplateType.yi,
    eos_token='<|im_end|>',
    support_flash_attn=True,
    support_vllm=True)
@register_model(
    ModelType.yi_34b_chat,
    '01ai/Yi-34B-Chat',
    LoRATM.llama2,
    TemplateType.yi,
    eos_token='<|im_end|>',
    support_flash_attn=True,
    support_vllm=True)
@register_model(
    ModelType.yi_34b_200k,
    '01ai/Yi-34B-200K',
    LoRATM.llama2,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True)
@register_model(
    ModelType.yi_34b,
    '01ai/Yi-34B',
    LoRATM.llama2,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True)
@register_model(
    ModelType.yi_6b_200k,
    '01ai/Yi-6B-200K',
    LoRATM.llama2,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True)
@register_model(
    ModelType.yi_6b,
    '01ai/Yi-6B',
    LoRATM.llama2,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True)
@register_model(
    ModelType.ziya2_13b_chat,
    'Fengshenbang/Ziya2-13B-Chat',
    LoRATM.llama2,
    TemplateType.ziya,
    support_flash_attn=True,
    support_vllm=True)
@register_model(
    ModelType.ziya2_13b,
    'Fengshenbang/Ziya2-13B-Base',
    LoRATM.llama2,
    TemplateType.default_generation_bos,
    support_flash_attn=True,
    support_vllm=True)
@register_model(
    ModelType.openbuddy_mistral_7b_chat,
    'OpenBuddy/openbuddy-mistral-7b-v13.1',
    LoRATM.llama2,
    TemplateType.openbuddy,
    requires=['transformers>=4.34'],
    support_flash_attn=True,
    support_vllm=True)
@register_model(
    ModelType.openbuddy_llama2_70b_chat,
    'OpenBuddy/openbuddy-llama2-70b-v10.1-bf16',
    LoRATM.llama2,
    TemplateType.openbuddy,
    support_flash_attn=True,
    support_vllm=True)
@register_model(
    ModelType.openbuddy_llama2_65b_chat,
    'OpenBuddy/openbuddy-llama-65b-v8-bf16',
    LoRATM.llama2,
    TemplateType.openbuddy,
    support_flash_attn=True,
    support_vllm=True)
@register_model(
    ModelType.openbuddy_llama2_13b_chat,
    'OpenBuddy/openbuddy-llama2-13b-v8.1-fp16',
    LoRATM.llama2,
    TemplateType.openbuddy,
    support_flash_attn=True,
    support_vllm=True)
@register_model(
    ModelType.mistral_7b_chat,
    'AI-ModelScope/Mistral-7B-Instruct-v0.1',
    LoRATM.llama2,
    TemplateType.llama,
    requires=['transformers>=4.34'],
    support_flash_attn=True,
    support_vllm=True)
@register_model(
    ModelType.mistral_7b_chat_v2,
    'AI-ModelScope/Mistral-7B-Instruct-v0.2',
    LoRATM.llama2,
    TemplateType.llama,
    requires=['transformers>=4.34'],
    support_flash_attn=True,
    support_vllm=True)
@register_model(
    ModelType.mistral_7b,
    'AI-ModelScope/Mistral-7B-v0.1',
    LoRATM.llama2,
    TemplateType.default_generation_bos,
    requires=['transformers>=4.34'],
    support_flash_attn=True,
    support_vllm=True)
@register_model(
    ModelType.mixtral_7b_moe,
    'AI-ModelScope/Mixtral-8x7B-v0.1',
    LoRATM.llama2,
    TemplateType.default_generation_bos,
    requires=['transformers>=4.36'],
    support_flash_attn=True,
    support_vllm=True,
    support_gradient_checkpointing=False)
@register_model(
    ModelType.mixtral_7b_moe_chat,
    'AI-ModelScope/Mixtral-8x7B-Instruct-v0.1',
    LoRATM.llama2,
    TemplateType.llama,
    requires=['transformers>=4.36'],
    support_flash_attn=True,
    support_vllm=True,
    support_gradient_checkpointing=False)
def get_model_tokenizer_with_flash_attn(model_dir: str,
                                        torch_dtype: Dtype,
                                        model_kwargs: Dict[str, Any],
                                        load_model: bool = True,
                                        model_config=None,
                                        **kwargs):
    if model_config is None:
        model_config = AutoConfig.from_pretrained(
            model_dir, trust_remote_code=True)
    use_flash_attn = kwargs.pop('use_flash_attn', False)
    if version.parse(transformers.__version__) >= version.parse('4.36'):
        if use_flash_attn:
            model_config._attn_implementation = 'flash_attention_2'
    else:
        model_config._flash_attn_2_enabled = use_flash_attn
    return get_model_tokenizer_from_repo(model_dir, torch_dtype, model_kwargs,
                                         load_model, model_config, **kwargs)


@register_model(
    ModelType.llama2_7b,
    'modelscope/Llama-2-7b-ms',
    LoRATM.llama2,
    TemplateType.default_generation_bos,
    ignore_file_pattern=[r'.+\.bin$'],
    support_flash_attn=True,
    support_vllm=True)
@register_model(
    ModelType.llama2_13b,
    'modelscope/Llama-2-13b-ms',
    LoRATM.llama2,
    TemplateType.default_generation_bos,
    ignore_file_pattern=[r'.+\.bin$'],
    support_flash_attn=True,
    support_vllm=True)
@register_model(
    ModelType.llama2_70b,
    'modelscope/Llama-2-70b-ms',
    LoRATM.llama2,
    TemplateType.default_generation_bos,
    ignore_file_pattern=[r'.+\.bin$'],
    support_flash_attn=True,
    support_vllm=True)
@register_model(
    ModelType.llama2_7b_chat,
    'modelscope/Llama-2-7b-chat-ms',
    LoRATM.llama2,
    TemplateType.llama,
    ignore_file_pattern=[r'.+\.bin$'],
    support_flash_attn=True,
    support_vllm=True)
@register_model(
    ModelType.llama2_13b_chat,
    'modelscope/Llama-2-13b-chat-ms',
    LoRATM.llama2,
    TemplateType.llama,
    ignore_file_pattern=[r'.+\.bin$'],
    support_flash_attn=True,
    support_vllm=True)
@register_model(
    ModelType.llama2_70b_chat,
    'modelscope/Llama-2-70b-chat-ms',
    LoRATM.llama2,
    TemplateType.llama,
    ignore_file_pattern=[r'.+\.bin$'],
    support_flash_attn=True,
    support_vllm=True)
def get_model_tokenizer_llama2(model_dir: str,
                               torch_dtype: Dtype,
                               model_kwargs: Dict[str, Any],
                               load_model: bool = True,
                               **kwargs):
    model_config = AutoConfig.from_pretrained(
        model_dir, trust_remote_code=True)
    model_config.pretraining_tp = 1
    return get_model_tokenizer_with_flash_attn(model_dir, torch_dtype,
                                               model_kwargs, load_model,
                                               model_config, **kwargs)


@register_model(ModelType.polylm_13b, 'damo/nlp_polylm_13b_text_generation',
                LoRATM.polylm, TemplateType.default_generation)
def get_model_tokenizer_polylm(model_dir: str,
                               torch_dtype: Dtype,
                               model_kwargs: Dict[str, Any],
                               load_model: bool = True,
                               **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True, use_fast=False, legacy=True)
    return get_model_tokenizer_from_repo(
        model_dir,
        torch_dtype,
        model_kwargs,
        load_model,
        tokenizer=tokenizer,
        **kwargs)


dtype_mapping = {
    torch.float16: 'fp16',
    torch.bfloat16: 'bf16',
    torch.float32: 'fp32'
}


def get_model_tokenizer_qwen(model_dir: str,
                             torch_dtype: Dtype,
                             model_kwargs: Dict[str, Any],
                             load_model: bool = True,
                             **kwargs):
    model_config = AutoConfig.from_pretrained(
        model_dir, trust_remote_code=True)
    if torch_dtype is not None:
        k_true = dtype_mapping[torch_dtype]
        for k in dtype_mapping.values():
            v = False
            if k == k_true:
                v = True
            setattr(model_config, k, v)

    use_flash_attn = kwargs.pop('use_flash_attn', None)
    if use_flash_attn is None:
        use_flash_attn = 'auto'
    model_config.use_flash_attn = use_flash_attn
    model, tokenizer = get_model_tokenizer_from_repo(model_dir, torch_dtype,
                                                     model_kwargs, load_model,
                                                     model_config, **kwargs)
    try:
        # fix mp+ddp bug
        model.transformer.registered_causal_mask = model.transformer.registered_causal_mask.cuda(
        )
        logger.info('registered_causal_mask to cuda')
    except AttributeError:
        pass
    return model, tokenizer


@register_model(
    ModelType.qwen_1_8b,
    'qwen/Qwen-1_8B',
    LoRATM.qwen,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True)
@register_model(
    ModelType.qwen_72b,
    'qwen/Qwen-72B',
    LoRATM.qwen,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True)
@register_model(
    ModelType.tongyi_finance_14b,
    'TongyiFinance/Tongyi-Finance-14B',
    LoRATM.qwen,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True)
@register_model(
    ModelType.qwen_14b,
    'qwen/Qwen-14B',
    LoRATM.qwen,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True)
@register_model(
    ModelType.qwen_7b,
    'qwen/Qwen-7B',
    LoRATM.qwen,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True)
def get_model_tokenizer_qwen_base(*args, **kwargs):
    model, tokenizer = get_model_tokenizer_qwen(*args, **kwargs)
    tokenizer.eos_token_id = tokenizer.eod_id
    return model, tokenizer


@register_model(
    ModelType.qwen_1_8b_chat,
    'qwen/Qwen-1_8B-Chat',
    LoRATM.qwen,
    TemplateType.chatml,
    support_flash_attn=True,
    support_vllm=True)
@register_model(
    ModelType.qwen_72b_chat,
    'qwen/Qwen-72B-Chat',
    LoRATM.qwen,
    TemplateType.chatml,
    support_flash_attn=True,
    support_vllm=True)
@register_model(
    ModelType.tongyi_finance_14b_chat,
    'TongyiFinance/Tongyi-Finance-14B-Chat',
    LoRATM.qwen,
    TemplateType.chatml,
    support_flash_attn=True,
    support_vllm=True)
@register_model(
    ModelType.qwen_14b_chat,
    'qwen/Qwen-14B-Chat',
    LoRATM.qwen,
    TemplateType.chatml,
    support_flash_attn=True,
    support_vllm=True)
@register_model(
    ModelType.qwen_7b_chat,
    'qwen/Qwen-7B-Chat',
    LoRATM.qwen,
    TemplateType.chatml,
    support_flash_attn=True,
    support_vllm=True)
def get_model_tokenizer_qwen_chat(*args, **kwargs):
    model, tokenizer = get_model_tokenizer_qwen(*args, **kwargs)
    tokenizer.eos_token_id = tokenizer.im_end_id
    return model, tokenizer


def fix_qwen_inplace_bug(model) -> None:
    first_drop = model.transformer.drop
    if first_drop.p == 0.:
        # fix in-place operation bug
        if not hasattr(first_drop, '__old_forward'):  # Avoid double patching
            if hasattr(first_drop, '_old_forward'):  # device_map
                __old_forward = first_drop._old_forward
                first_drop._old_forward = lambda *args, **kwargs: __old_forward(
                    *args, **kwargs).clone()
            else:
                __old_forward = first_drop.forward
                first_drop.forward = lambda *args, **kwargs: __old_forward(
                    *args, **kwargs).clone()
            first_drop.__old_forward = __old_forward


def _qwen_vl_audio_decode(self,
                          *args,
                          skip_special_tokens=False,
                          **kwargs) -> str:
    if skip_special_tokens:
        token_ids = kwargs['token_ids']
        while len(token_ids) > 0 and token_ids[-1] in {151645, 151643}:
            token_ids.pop()
        return self._old_decode(*args, skip_special_tokens=False, **kwargs)
    else:
        return self._old_decode(*args, skip_special_tokens=False, **kwargs)


@register_model(
    ModelType.qwen_vl_chat,
    'qwen/Qwen-VL-Chat',
    LoRATM.qwen,
    TemplateType.chatml,
    support_flash_attn=True)
@register_model(
    ModelType.qwen_vl,
    'qwen/Qwen-VL',
    LoRATM.qwen,
    TemplateType.default_generation,
    function_kwargs={'get_qwen_function': get_model_tokenizer_qwen_base},
    support_flash_attn=True)
def get_model_tokenizer_qwen_vl(model_dir: str,
                                torch_dtype: Dtype,
                                model_kwargs: Dict[str, Any],
                                load_model: bool = True,
                                **kwargs):
    if (model_kwargs.get('quantization_config') is not None and isinstance(
            model_kwargs['quantization_config'], BitsAndBytesConfig)):
        # https://github.com/pytorch/pytorch/issues/58969
        model_kwargs['quantization_config'].llm_int8_skip_modules = [
            'lm_head', 'attn_pool.attn'
        ]
    get_qwen_function = kwargs.pop('get_qwen_function',
                                   get_model_tokenizer_qwen_chat)
    tokenizer_config = get_tokenizer_config(model_dir)
    class_ref = tokenizer_config['auto_map']['AutoTokenizer'][0]
    tokenizer_cls = get_class_from_dynamic_module(class_ref, model_dir)
    tokenizer_cls._auto_class = 'AutoTokenizer'
    tokenizer_cls.IMAGE_ST = ()  # fix no attr `self.IMAGE_ST` bug
    if not hasattr(tokenizer_cls, '_old_decode'):  # avoid double patching
        tokenizer_cls._old_decode = tokenizer_cls._decode
        tokenizer_cls._decode = _qwen_vl_audio_decode
    kwargs['tokenizer'] = tokenizer_cls.from_pretrained(
        model_dir, trust_remote_code=True)
    model, tokenizer = get_qwen_function(model_dir, torch_dtype, model_kwargs,
                                         load_model, **kwargs)
    if model is not None:
        fix_qwen_inplace_bug(model)

    return model, tokenizer


@register_model(
    ModelType.qwen_audio_chat,
    'qwen/Qwen-Audio-Chat',
    LoRATM.qwen,
    TemplateType.chatml,
    support_flash_attn=True,
    function_kwargs={'get_qwen_function': get_model_tokenizer_qwen_chat})
@register_model(
    ModelType.qwen_audio,
    'qwen/Qwen-Audio',
    LoRATM.qwen,
    TemplateType.default_generation,
    support_flash_attn=True,
    function_kwargs={'get_qwen_function': get_model_tokenizer_qwen_base})
def get_model_tokenizer_qwen_audio(model_dir: str,
                                   torch_dtype: Dtype,
                                   model_kwargs: Dict[str, Any],
                                   load_model: bool = True,
                                   **kwargs):
    get_qwen_function = kwargs.pop('get_qwen_function')
    tokenizer_config = get_tokenizer_config(model_dir)
    class_ref = tokenizer_config['auto_map']['AutoTokenizer'][0]
    tokenizer_cls = get_class_from_dynamic_module(class_ref, model_dir)
    tokenizer_cls._auto_class = 'AutoTokenizer'
    tokenizer_cls.AUDIO_ST = ()  # fix no attr `self.AUDIO_ST` bug
    if not hasattr(tokenizer_cls, '_old_decode'):  # avoid double patching
        tokenizer_cls._old_decode = tokenizer_cls._decode
        tokenizer_cls._decode = _qwen_vl_audio_decode
    kwargs['tokenizer'] = tokenizer_cls.from_pretrained(
        model_dir, trust_remote_code=True)
    model, tokenizer = get_qwen_function(model_dir, torch_dtype, model_kwargs,
                                         load_model, **kwargs)
    if model is not None:
        fix_qwen_inplace_bug(model)

    return model, tokenizer


@register_model(
    ModelType.qwen_1_8b_chat_int8,
    'qwen/Qwen-1_8B-Chat-Int8',
    LoRATM.qwen,
    TemplateType.chatml,
    requires=['auto_gptq>=0.5'],
    torch_dtype=torch.float16,
    function_kwargs={'bits': 8},
    support_flash_attn=True)
@register_model(
    ModelType.qwen_1_8b_chat_int4,
    'qwen/Qwen-1_8B-Chat-Int4',
    LoRATM.qwen,
    TemplateType.chatml,
    requires=['auto_gptq>=0.5'],
    torch_dtype=torch.float16,
    function_kwargs={'bits': 4},
    support_flash_attn=True)
@register_model(
    ModelType.qwen_72b_chat_int8,
    'qwen/Qwen-72B-Chat-Int8',
    LoRATM.qwen,
    TemplateType.chatml,
    requires=['auto_gptq>=0.5'],
    torch_dtype=torch.float16,
    function_kwargs={'bits': 8},
    support_flash_attn=True)
@register_model(
    ModelType.qwen_72b_chat_int4,
    'qwen/Qwen-72B-Chat-Int4',
    LoRATM.qwen,
    TemplateType.chatml,
    requires=['auto_gptq>=0.5'],
    torch_dtype=torch.float16,
    function_kwargs={'bits': 4},
    support_flash_attn=True)
@register_model(
    ModelType.tongyi_finance_14b_chat_int4,
    'TongyiFinance/Tongyi-Finance-14B-Chat-Int4',
    LoRATM.qwen,
    TemplateType.chatml,
    requires=['auto_gptq>=0.5'],
    torch_dtype=torch.float16,
    function_kwargs={'bits': 4},
    support_flash_attn=True)
@register_model(
    ModelType.qwen_vl_chat_int4,
    'qwen/Qwen-VL-Chat-Int4',
    LoRATM.qwen,
    TemplateType.chatml,
    requires=['auto_gptq>=0.5'],
    torch_dtype=torch.float16,
    function_kwargs={
        'get_qwen_function': get_model_tokenizer_qwen_vl,
        'bits': 4
    },
    support_flash_attn=True)
@register_model(
    ModelType.qwen_14b_chat_int8,
    'qwen/Qwen-14B-Chat-Int8',
    LoRATM.qwen,
    TemplateType.chatml,
    requires=['auto_gptq>=0.5'],
    torch_dtype=torch.float16,
    function_kwargs={'bits': 8},
    support_flash_attn=True)
@register_model(
    ModelType.qwen_7b_chat_int8,
    'qwen/Qwen-7B-Chat-Int8',
    LoRATM.qwen,
    TemplateType.chatml,
    requires=['auto_gptq>=0.5'],
    torch_dtype=torch.float16,
    function_kwargs={'bits': 8},
    support_flash_attn=True)
@register_model(
    ModelType.qwen_14b_chat_int4,
    'qwen/Qwen-14B-Chat-Int4',
    LoRATM.qwen,
    TemplateType.chatml,
    requires=['auto_gptq>=0.5'],
    torch_dtype=torch.float16,
    function_kwargs={'bits': 4},
    support_flash_attn=True)
@register_model(
    ModelType.qwen_7b_chat_int4,
    'qwen/Qwen-7B-Chat-Int4',
    LoRATM.qwen,
    TemplateType.chatml,
    requires=['auto_gptq>=0.5'],
    torch_dtype=torch.float16,
    function_kwargs={'bits': 4},
    support_flash_attn=True)
def get_model_tokenizer_qwen_intx(model_dir: str,
                                  torch_dtype: Dtype,
                                  model_kwargs: Dict[str, Any],
                                  load_model: bool = True,
                                  **kwargs):

    logger.info('use gptq, ignore bnb arguments')
    bits = kwargs.pop('bits')
    if version.parse(transformers.__version__) >= version.parse('4.35'):
        model_kwargs['quantization_config'] = GPTQConfig(
            bits=bits, use_exllama=False)
    else:
        model_kwargs['quantization_config'] = GPTQConfig(
            bits=bits, disable_exllama=True)

    # fix quantlinear bug
    from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import QuantLinear
    __old_forward = QuantLinear.forward

    def _new_forward(self, x):
        if not self.training or not self.autogptq_cuda_available:
            return self.__old_forward(x)
        # fix sft no grad
        self.autogptq_cuda_available = False
        res = self.__old_forward(x)
        self.autogptq_cuda_available = True
        return res

    if not hasattr(QuantLinear, '__old_forward'):  # avoid double patching
        QuantLinear.__old_forward = __old_forward
        QuantLinear.forward = _new_forward
    get_qwen_function = kwargs.pop('get_qwen_function',
                                   get_model_tokenizer_qwen_chat)
    model, tokenizer = get_qwen_function(model_dir, torch_dtype, model_kwargs,
                                         load_model, **kwargs)
    return model, tokenizer


register_model(ModelType.skywork_13b, 'skywork/Skywork-13B-base',
               LoRATM.llama2, TemplateType.default_generation_bos,
               get_model_tokenizer_from_repo)


@register_model(ModelType.skywork_13b_chat, 'skywork/Skywork-13B-chat',
                LoRATM.llama2, TemplateType.skywork)
def get_skywork_model_tokenizer(model_dir: str,
                                torch_dtype: Dtype,
                                model_kwargs: Dict[str, Any],
                                load_model: bool = True,
                                **kwargs):
    model, tokenizer = get_model_tokenizer_from_repo(model_dir, torch_dtype,
                                                     model_kwargs, load_model,
                                                     **kwargs)
    tokenizer.add_tokens('[USER]')
    tokenizer.add_tokens('[BOT]')
    tokenizer.add_tokens('[SEP]')
    return model, tokenizer


@register_model(
    ModelType.codefuse_codellama_34b_chat,
    'codefuse-ai/CodeFuse-CodeLlama-34B',
    LoRATM.llama2,
    TemplateType.codefuse_codellama,
    support_flash_attn=True,
    support_vllm=True)
def get_model_tokenizer_codellama(model_dir: str,
                                  torch_dtype: Dtype,
                                  model_kwargs: Dict[str, Any],
                                  load_model: bool = True,
                                  **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True, use_fast=False, legacy=False)
    return get_model_tokenizer_with_flash_attn(
        model_dir,
        torch_dtype,
        model_kwargs,
        load_model,
        tokenizer=tokenizer,
        **kwargs)


@register_model(
    ModelType.phi2_3b,
    'AI-ModelScope/phi-2',
    LoRATM.phi,
    TemplateType.default_generation,
    support_flash_attn=True,
    support_vllm=True,
    support_gradient_checkpointing=False)
def get_model_tokenizer_phi(model_dir: str,
                            torch_dtype: Dtype,
                            model_kwargs: Dict[str, Any],
                            load_model: bool = True,
                            **kwargs):
    model_config = AutoConfig.from_pretrained(
        model_dir, trust_remote_code=True)
    use_flash_attn = kwargs.get('use_flash_attn', False)
    model_config.flash_attn = use_flash_attn
    return get_model_tokenizer_from_repo(model_dir, torch_dtype, model_kwargs,
                                         load_model, model_config, **kwargs)


def fix_transformers_upgrade(module: PreTrainedModel) -> None:
    # from 4.35, transformers changes its arguments of _set_gradient_checkpointing
    if version.parse(transformers.__version__) >= version.parse('4.35'):
        if isinstance(module, PreTrainedModel) and hasattr(module, '_set_gradient_checkpointing') \
                and 'value' in inspect.signature(module._set_gradient_checkpointing).parameters.keys():
            module._set_gradient_checkpointing = MethodType(
                PreTrainedModel._set_gradient_checkpointing, module)


def fix_gradient_checkpointing_warning() -> None:
    if version.parse(torch.__version__) < version.parse('2'):
        return
    _old_checkpoint = torch.utils.checkpoint.checkpoint
    if not hasattr(torch.utils.checkpoint,
                   '_old_checkpoint'):  # avoid double patching

        torch.utils.checkpoint._old_checkpoint = _old_checkpoint
        torch.utils.checkpoint.checkpoint = update_wrapper(
            lambda *args, use_reentrant=False, **kwargs: _old_checkpoint(
                *args, use_reentrant=use_reentrant, **kwargs),
            _old_checkpoint)
    try:
        import transformers.modeling_utils
        if hasattr(transformers.modeling_utils, 'checkpoint'):
            transformers.modeling_utils.checkpoint = (
                lambda *args, use_reentrant=False, **kwargs: _old_checkpoint(
                    *args, use_reentrant=use_reentrant, **kwargs))
    except ImportError:
        pass


def get_model_tokenizer(
        model_type: str,
        torch_dtype: Optional[Dtype] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        load_model: bool = True,
        **kwargs) -> Tuple[Optional[PreTrainedModel], PreTrainedTokenizerBase]:
    """
    torch_dtype: If you use None, it will retrieve the torch_dtype from the config.json file.
        However, if torch.float32 is retrieved, torch.float16 will be used.
    """
    model_info = MODEL_MAPPING[model_type]
    requires = model_info['requires']
    for require in requires:
        require_version(require)

    model_id_or_path = model_info['model_id_or_path']
    get_function = model_info['get_function']
    ignore_file_pattern = model_info['ignore_file_pattern']
    if model_kwargs is None:
        model_kwargs = {}
    if 'device_map' not in model_kwargs:
        model_kwargs['device_map'] = 'auto'

    model_dir = kwargs.pop('model_dir', None)
    if model_dir is None:
        if is_dist() and not is_local_master():
            dist.barrier()
        model_dir = model_id_or_path
        if model_id_or_path is not None and not os.path.exists(
                model_id_or_path):
            revision = model_info['revision']
            model_dir = snapshot_download(
                model_id_or_path,
                revision,
                ignore_file_pattern=ignore_file_pattern)
        if is_dist() and is_local_master():
            dist.barrier()
    model_dir = os.path.expanduser(model_dir)
    assert os.path.isdir(model_dir)
    if model_info.get('torch_dtype') is not None:
        model_torch_dtype = model_info['torch_dtype']
        if torch_dtype is None:
            torch_dtype = model_torch_dtype
            logger.info(f'Setting torch_dtype: {torch_dtype}')
        else:
            assert torch_dtype == model_torch_dtype, f'please use `{model_torch_dtype}`'
    else:
        if torch_dtype is None:
            model_config = AutoConfig.from_pretrained(
                model_dir, trust_remote_code=True)
            torch_dtype = getattr(model_config, 'torch_dtype', None)
            if torch_dtype == torch.float32:
                torch_dtype = torch.float16
            logger.info(f'Setting torch_dtype: {torch_dtype}')
    kwargs['automodel_class'] = model_info['automodel_class']
    kwargs['eos_token'] = model_info['eos_token']
    model, tokenizer = get_function(model_dir, torch_dtype, model_kwargs,
                                    load_model, **kwargs)
    if model is not None:
        model.model_type = model_type
        fix_transformers_upgrade(model)
        fix_gradient_checkpointing_warning()
    tokenizer.model_type = model_type
    assert tokenizer.eos_token is not None
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model is not None and model_dir is not None:
        generation_config_path = os.path.join(model_dir,
                                              'generation_config.json')
        generation_config = getattr(model, 'generation_config', None)
        if os.path.isfile(
                generation_config_path) and generation_config is None:
            model.generation_config = GenerationConfig.from_pretrained(
                model_dir)
    return model, tokenizer


def get_additional_saved_files(model_type: str) -> List[str]:
    if 'qwen-vl' in model_type:
        return ['SimSun.ttf']
    elif 'qwen-audio' in model_type:
        return ['mel_filters.npz']
    return []


def get_default_template_type(model_type: str) -> Optional[str]:
    return MODEL_MAPPING[model_type].get('template')


def get_default_lora_target_modules(model_type: str) -> Optional[List[str]]:
    return MODEL_MAPPING[model_type].get('lora_target_modules')

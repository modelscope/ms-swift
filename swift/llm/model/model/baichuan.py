# Copyright (c) Alibaba, Inc. and its affiliates.
from types import MethodType
from typing import Any, Dict

import torch.nn.functional as F
from modelscope import AutoConfig
from torch import Tensor
from transformers import BitsAndBytesConfig, PretrainedConfig

from ..model_arch import ModelArch
from swift.llm import TemplateType
from swift.utils import get_logger
from ..constant import LLMModelType
from ..register import Model, ModelGroup, ModelMeta, get_model_tokenizer_with_flash_attn, register_model
from ..utils import ModelInfo

logger = get_logger()


def get_model_tokenizer_baichuan(model_dir: str,
                                 model_config: PretrainedConfig,
                                 model_kwargs: Dict[str, Any],
                                 load_model: bool = True,
                                 **kwargs):
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, model_config, model_kwargs, load_model, **kwargs)
    # baichuan-13b does not implement the `get_input_embeddings` function
    # fix gradient_checkpointing bug
    try:
        model.get_input_embeddings()
    except NotImplementedError:
        model.__class__.get_input_embeddings = lambda self: self.model.embed_tokens
    return model, tokenizer


register_model(
    ModelMeta(
        LLMModelType.baichuan,
        [
            ModelGroup([
                Model('baichuan-inc/baichuan-7B', 'baichuan-inc/Baichuan-7B'),
                Model('baichuan-inc/Baichuan-13B-Base', 'baichuan-inc/Baichuan-13B-Base'),
                Model('baichuan-inc/Baichuan-13B-Chat', 'baichuan-inc/Baichuan-13B-Chat'),
            ],
                       requires=['transformers<4.33.3']),
        ],
        TemplateType.baichuan,
        get_model_tokenizer_baichuan,
        architectures=['BaiChuanForCausalLM'],
        model_arch=ModelArch.baichuan,
    ))


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


def get_model_tokenizer_baichuan2(model_dir: str,
                                  model_info: ModelInfo,
                                  model_kwargs: Dict[str, Any],
                                  load_model: bool = True,
                                  model_config=None,
                                  **kwargs):
    if model_config is None:
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    if not hasattr(model_config, 'z_loss_weight'):
        model_config.z_loss_weight = 0
    # patch: baichuan2_13b configuration_baichuan.py bug
    if hasattr(model_config, 'gradient_checkpointing'):
        gradient_checkpointing = model_config.gradient_checkpointing
        if isinstance(gradient_checkpointing, (tuple, list)):
            model_config.gradient_checkpointing = gradient_checkpointing[0]
    model, tokenizer = get_model_tokenizer_with_flash_attn(
        model_dir, model_info, model_kwargs, load_model, model_config=model_config, **kwargs)
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


register_model(
    ModelMeta(
        LLMModelType.baichuan2,
        [
            ModelGroup([
                Model('baichuan-inc/Baichuan2-7B-Base', 'baichuan-inc/Baichuan2-7B-Base'),
                Model('baichuan-inc/Baichuan2-7B-Chat', 'baichuan-inc/Baichuan2-7B-Chat'),
                Model('baichuan-inc/Baichuan2-13B-Base', 'baichuan-inc/Baichuan2-13B-Base'),
                Model('baichuan-inc/Baichuan2-13B-Chat', 'baichuan-inc/Baichuan2-13B-Chat'),
            ]),
        ],
        TemplateType.baichuan,
        get_model_tokenizer_baichuan2,
        architectures=['BaichuanForCausalLM'],
        model_arch=ModelArch.baichuan,
    ))


def get_model_tokenizer_baichuan2_int4(model_dir: str,
                                       model_info: ModelInfo,
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
    model, tokenizer = get_baichuan2_function(model_dir, model_info, model_kwargs, load_model, **kwargs)
    if device_map != 'auto':
        accelerate.infer_auto_device_map = _old_infer_auto_device_map
    if model is not None:
        model.config.quantization_config = BitsAndBytesConfig(**model.config.quantization_config)
        model.train()
        model._is_quantized_training_enabled = True
        model.is_loaded_in_4bit = True
    return model, tokenizer


register_model(
    ModelMeta(
        LLMModelType.baichuan2_int4,
        [
            ModelGroup([
                Model('baichuan-inc/Baichuan2-7B-Chat-4bits', 'baichuan-inc/Baichuan2-7B-Chat-4bits'),
                Model('baichuan-inc/Baichuan2-13B-Chat-4bits', 'baichuan-inc/Baichuan2-13B-Chat-4bits'),
            ],
                       requires=['bitsandbytes<0.41.2', 'accelerate<0.26']),
        ],
        TemplateType.baichuan,
        get_model_tokenizer_baichuan2_int4,
        architectures=['BaichuanForCausalLM'],
        model_arch=ModelArch.baichuan,
    ))

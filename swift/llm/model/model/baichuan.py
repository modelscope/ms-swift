# Copyright (c) Alibaba, Inc. and its affiliates.
from types import MethodType
from typing import Any, Dict

import torch.nn.functional as F
from torch import Tensor
from transformers import AutoConfig

from swift.llm import TemplateType
from swift.utils import get_logger
from ..constant import LLMModelType
from ..model_arch import ModelArch
from ..register import Model, ModelGroup, ModelMeta, get_model_tokenizer_with_flash_attn, register_model
from ..utils import ModelInfo

logger = get_logger()


def get_model_tokenizer_baichuan(model_dir: str,
                                 model_info: ModelInfo,
                                 model_kwargs: Dict[str, Any],
                                 load_model: bool = True,
                                 **kwargs):
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)
    # baichuan-13b does not implement the `get_input_embeddings` function
    # fix gradient_checkpointing bug
    try:
        if model is not None:
            model.get_input_embeddings()
    except NotImplementedError:
        model.__class__.get_input_embeddings = lambda self: self.model.embed_tokens
    return model, tokenizer


register_model(
    ModelMeta(
        LLMModelType.baichuan, [
            ModelGroup([
                Model('baichuan-inc/Baichuan-13B-Chat', 'baichuan-inc/Baichuan-13B-Chat'),
                Model('baichuan-inc/Baichuan-13B-Base', 'baichuan-inc/Baichuan-13B-Base'),
                Model('baichuan-inc/baichuan-7B', 'baichuan-inc/Baichuan-7B'),
            ]),
        ],
        TemplateType.baichuan,
        get_model_tokenizer_baichuan,
        architectures=['BaichuanForCausalLM', 'BaiChuanForCausalLM'],
        model_arch=ModelArch.baichuan,
        requires=['transformers<4.34']))


def get_model_tokenizer_baichuan_m1(model_dir: str,
                                    model_info: ModelInfo,
                                    model_kwargs: Dict[str, Any],
                                    load_model: bool = True,
                                    **kwargs):
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    rotary_embedding = get_class_from_dynamic_module('modeling_baichuan.RotaryEmbedding', model_dir)
    _old_forward = rotary_embedding.forward

    def _new_forward(self, q, k, seqlen_offset=None, cu_seqlens=None, max_seqlen=None):
        q = q.to(k.dtype)
        res = _old_forward(self, q, k, seqlen_offset, cu_seqlens, max_seqlen)
        return res

    rotary_embedding.forward = _new_forward

    model, tokenizer = get_model_tokenizer_baichuan(model_dir, model_info, model_kwargs, load_model, **kwargs)
    return model, tokenizer


register_model(
    ModelMeta(
        LLMModelType.baichuan_m1, [
            ModelGroup([
                Model('baichuan-inc/Baichuan-M1-14B-Instruct', 'baichuan-inc/Baichuan-M1-14B-Instruct'),
            ]),
        ],
        TemplateType.baichuan_m1,
        get_model_tokenizer_baichuan_m1,
        architectures=['BaichuanM1ForCausalLM'],
        model_arch=ModelArch.baichuan,
        requires=['transformers>=4.48']))


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
                Model('baichuan-inc/Baichuan2-7B-Chat', 'baichuan-inc/Baichuan2-7B-Chat'),
                Model('baichuan-inc/Baichuan2-7B-Base', 'baichuan-inc/Baichuan2-7B-Base'),
                Model('baichuan-inc/Baichuan2-13B-Chat', 'baichuan-inc/Baichuan2-13B-Chat'),
                Model('baichuan-inc/Baichuan2-13B-Base', 'baichuan-inc/Baichuan2-13B-Base'),
            ]),
            ModelGroup([
                Model('baichuan-inc/Baichuan2-7B-Chat-4bits', 'baichuan-inc/Baichuan2-7B-Chat-4bits'),
                Model('baichuan-inc/Baichuan2-13B-Chat-4bits', 'baichuan-inc/Baichuan2-13B-Chat-4bits'),
            ],
                       requires=['bitsandbytes<0.41.2', 'accelerate<0.26'])
        ],
        TemplateType.baichuan,
        get_model_tokenizer_baichuan2,
        architectures=['BaichuanForCausalLM', 'BaiChuanForCausalLM'],
        model_arch=ModelArch.baichuan,
    ))

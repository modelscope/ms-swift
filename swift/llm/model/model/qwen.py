# Copyright (c) Alibaba, Inc. and its affiliates.
import importlib.metadata
import os
from types import MethodType
from typing import Any, Dict, Optional, Tuple, Type, Union

import torch
from packaging import version
from PIL import Image
from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizerBase
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.auto.tokenization_auto import get_tokenizer_config
from transformers.utils.versions import require_version

from swift.llm import TemplateType, to_device
from swift.utils import get_device_count, get_dist_setting, get_env_args, get_logger, is_deepspeed_enabled
from ..constant import LLMModelType, MLLMModelType, RerankerModelType, RMModelType
from ..model_arch import ModelArch
from ..patcher import patch_fixed_device, patch_get_input_embeddings, patch_output_clone
from ..register import (Model, ModelGroup, ModelMeta, get_model_tokenizer_multimodal, get_model_tokenizer_reward_model,
                        get_model_tokenizer_with_flash_attn, register_model)
from ..utils import AttnImpl, ModelInfo, use_submodel_func

logger = get_logger()
dtype_mapping = {torch.float16: 'fp16', torch.bfloat16: 'bf16', torch.float32: 'fp32'}


def get_model_tokenizer_qwen(model_dir: str,
                             model_info: ModelInfo,
                             model_kwargs: Dict[str, Any],
                             load_model: bool = True,
                             model_config=None,
                             **kwargs):
    if model_config is None:
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    if model_info.torch_dtype is not None:
        k_true = dtype_mapping[model_info.torch_dtype]
        for k in dtype_mapping.values():
            setattr(model_config, k, k == k_true)

    quantization_config = model_kwargs.get('quantization_config')
    if not isinstance(quantization_config, BitsAndBytesConfig):
        # not bnb quant
        model_config.torch_dtype = None
    use_flash_attn = AttnImpl.to_use_flash_attn(kwargs.pop('attn_impl', None), 'auto')
    model_config.use_flash_attn = use_flash_attn
    kwargs['model_config'] = model_config
    tokenizer = kwargs.get('tokenizer')
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.eod_id
    kwargs['tokenizer'] = tokenizer
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)
    try:
        # fix mp+ddp bug
        model.transformer.registered_causal_mask = model.transformer.registered_causal_mask.cuda()
        logger.info('registered_causal_mask to cuda')
    except AttributeError:
        pass
    return model, tokenizer


register_model(
    ModelMeta(
        LLMModelType.qwen,
        [
            # qwen
            ModelGroup([
                # chat
                Model('Qwen/Qwen-1_8B-Chat', 'Qwen/Qwen-1_8B-Chat'),
                Model('Qwen/Qwen-7B-Chat', 'Qwen/Qwen-7B-Chat'),
                Model('Qwen/Qwen-14B-Chat', 'Qwen/Qwen-14B-Chat'),
                Model('Qwen/Qwen-72B-Chat', 'Qwen/Qwen-72B-Chat'),
                # base
                Model('Qwen/Qwen-1_8B', 'Qwen/Qwen-1_8B'),
                Model('Qwen/Qwen-7B', 'Qwen/Qwen-7B'),
                Model('Qwen/Qwen-14B', 'Qwen/Qwen-14B'),
                Model('Qwen/Qwen-72B', 'Qwen/Qwen-72B'),
                # gptq-int4
                Model('Qwen/Qwen-1_8B-Chat-Int4', 'Qwen/Qwen-1_8B-Chat-Int4'),
                Model('Qwen/Qwen-7B-Chat-Int4', 'Qwen/Qwen-7B-Chat-Int4'),
                Model('Qwen/Qwen-14B-Chat-Int4', 'Qwen/Qwen-14B-Chat-Int4'),
                Model('Qwen/Qwen-72B-Chat-Int4', 'Qwen/Qwen-72B-Chat-Int4'),
                # gptq-int8
                Model('Qwen/Qwen-1_8B-Chat-Int8', 'Qwen/Qwen-1_8B-Chat-Int8'),
                Model('Qwen/Qwen-7B-Chat-Int8', 'Qwen/Qwen-7B-Chat-Int8'),
                Model('Qwen/Qwen-14B-Chat-Int8', 'Qwen/Qwen-14B-Chat-Int8'),
                Model('Qwen/Qwen-72B-Chat-Int8', 'Qwen/Qwen-72B-Chat-Int8'),
            ]),
            # tongyi-finance
            ModelGroup([
                Model('TongyiFinance/Tongyi-Finance-14B-Chat', 'jxy/Tongyi-Finance-14B-Chat'),
                Model('TongyiFinance/Tongyi-Finance-14B'),
                Model('TongyiFinance/Tongyi-Finance-14B-Chat-Int4', 'jxy/Tongyi-Finance-14B-Chat-Int4'),
            ],
                       tags=['financial']),
        ],
        TemplateType.qwen,
        get_model_tokenizer_qwen,
        architectures=['QWenLMHeadModel'],
        model_arch=ModelArch.qwen))

register_model(
    ModelMeta(
        LLMModelType.modelscope_agent,
        [ModelGroup([
            Model('iic/ModelScope-Agent-7B'),
            Model('iic/ModelScope-Agent-14B'),
        ])],
        TemplateType.modelscope_agent,
        get_model_tokenizer_qwen,
        architectures=['QWenLMHeadModel'],
        model_arch=ModelArch.qwen))


def _qwen_vl_audio_decode(self, *args, skip_special_tokens=False, **kwargs) -> str:
    if skip_special_tokens:
        token_ids = kwargs['token_ids']
        while len(token_ids) > 0 and token_ids[-1] in {151645, 151643}:
            token_ids.pop()
        return self._old_decode(*args, skip_special_tokens=False, **kwargs)
    else:
        return self._old_decode(*args, skip_special_tokens=False, **kwargs)


def fix_qwen_inplace_bug(model) -> None:
    # qwen-vl, qwen-audio
    first_drop = model.transformer.drop
    if first_drop.p == 0.:
        # fix in-place operation bug
        patch_output_clone(first_drop)


def get_model_tokenizer_qwen_audio(model_dir: str,
                                   model_info: ModelInfo,
                                   model_kwargs: Dict[str, Any],
                                   load_model: bool = True,
                                   **kwargs):
    tokenizer_config = get_tokenizer_config(model_dir)
    class_ref = tokenizer_config['auto_map']['AutoTokenizer'][0]
    tokenizer_cls: Type[PreTrainedTokenizerBase] = get_class_from_dynamic_module(class_ref, model_dir)
    tokenizer_cls._auto_class = 'AutoTokenizer'
    tokenizer_cls.AUDIO_ST = ()  # fix no attr `self.AUDIO_ST` bug
    if not hasattr(tokenizer_cls, '_old_decode'):
        tokenizer_cls._old_decode = tokenizer_cls._decode
        tokenizer_cls._decode = _qwen_vl_audio_decode
    kwargs['tokenizer'] = tokenizer_cls.from_pretrained(model_dir, trust_remote_code=True)
    model, tokenizer = get_model_tokenizer_qwen(model_dir, model_info, model_kwargs, load_model, **kwargs)
    if model is not None:
        fix_qwen_inplace_bug(model)

    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.qwen_audio, [
            ModelGroup([
                Model('Qwen/Qwen-Audio-Chat', 'Qwen/Qwen-Audio-Chat'),
                Model('Qwen/Qwen-Audio', 'Qwen/Qwen-Audio'),
            ])
        ],
        TemplateType.qwen_audio,
        get_model_tokenizer_qwen_audio,
        model_arch=ModelArch.qwen_audio,
        architectures=['QWenLMHeadModel'],
        additional_saved_files=['mel_filters.npz'],
        tags=['audio']))


def _qwen_vl_visual_block_forward(
    self,
    q_x: torch.Tensor,
    k_x: Optional[torch.Tensor] = None,
    v_x: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
):
    k_x = self.ln_1_kv(k_x) if hasattr(self, 'ln_1_kv') and k_x is not None else None
    v_x = self.ln_1_kv(v_x) if hasattr(self, 'ln_1_kv') and v_x is not None else None

    x = q_x + self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask)
    z = self.mlp(self.ln_2(x))
    x = x.to(z.device) + z  # FIX
    return x


def get_model_tokenizer_qwen_vl(model_dir: str,
                                model_info: ModelInfo,
                                model_kwargs: Dict[str, Any],
                                load_model: bool = True,
                                **kwargs):
    if (model_kwargs.get('quantization_config') is not None
            and isinstance(model_kwargs['quantization_config'], BitsAndBytesConfig)):
        # https://github.com/pytorch/pytorch/issues/58969
        model_kwargs['quantization_config'].llm_int8_skip_modules = ['lm_head', 'attn_pool.attn']
        _TransformerBlock = get_class_from_dynamic_module('visual.TransformerBlock', model_dir)

        def _get_cast_dtype(self) -> torch.dtype:
            return self.resblocks[0].ln_1.weight.dtype

        _TransformerBlock.__old_get_cast_dtype = _TransformerBlock.get_cast_dtype
        _TransformerBlock.get_cast_dtype = _get_cast_dtype

    tokenizer_config = get_tokenizer_config(model_dir)
    class_ref = tokenizer_config['auto_map']['AutoTokenizer'][0]
    tokenizer_cls: Type[PreTrainedTokenizerBase] = get_class_from_dynamic_module(class_ref, model_dir)
    tokenizer_cls._auto_class = 'AutoTokenizer'
    tokenizer_cls.IMAGE_ST = ()  # fix no attr `self.IMAGE_ST` bug
    if not hasattr(tokenizer_cls, '_old_decode'):
        tokenizer_cls._old_decode = tokenizer_cls._decode
        tokenizer_cls._decode = _qwen_vl_audio_decode
    # fix device_map is 4
    n_gpu = get_device_count()
    local_world_size = get_dist_setting()[3]
    if n_gpu // local_world_size >= 4:
        visual_block_cls = get_class_from_dynamic_module('visual.VisualAttentionBlock', model_dir)
        visual_block_cls.__old_forward = visual_block_cls.forward
        visual_block_cls.forward = _qwen_vl_visual_block_forward

    kwargs['tokenizer'] = tokenizer_cls.from_pretrained(model_dir, trust_remote_code=True)
    model, tokenizer = get_model_tokenizer_qwen(model_dir, model_info, model_kwargs, load_model, **kwargs)
    if model is not None:
        device_type = next(model.parameters()).device.type
        fix_qwen_inplace_bug(model)
        # fix device_map is 4
        if n_gpu // local_world_size >= 4:
            model.transformer.visual.proj.data = model.transformer.visual.proj.to(
                model.transformer.visual.ln_post.bias.device)
        # fix images cuda:1 bug
        patch_fixed_device(model.transformer.visual, f'{device_type}:0')
    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.qwen_vl, [
            ModelGroup([
                Model('Qwen/Qwen-VL-Chat', 'Qwen/Qwen-VL-Chat'),
                Model('Qwen/Qwen-VL', 'Qwen/Qwen-VL'),
                Model('Qwen/Qwen-VL-Chat-Int4', 'Qwen/Qwen-VL-Chat-Int4'),
            ])
        ],
        TemplateType.qwen_vl,
        get_model_tokenizer_qwen_vl,
        model_arch=ModelArch.qwen_vl,
        architectures=['QWenLMHeadModel'],
        additional_saved_files=['SimSun.ttf'],
        tags=['vision']))

register_model(
    ModelMeta(
        LLMModelType.qwen2,
        [
            # qwen1.5
            ModelGroup([
                # chat
                Model('Qwen/Qwen1.5-0.5B-Chat', 'Qwen/Qwen1.5-0.5B-Chat'),
                Model('Qwen/Qwen1.5-1.8B-Chat', 'Qwen/Qwen1.5-1.8B-Chat'),
                Model('Qwen/Qwen1.5-4B-Chat', 'Qwen/Qwen1.5-4B-Chat'),
                Model('Qwen/Qwen1.5-7B-Chat', 'Qwen/Qwen1.5-7B-Chat'),
                Model('Qwen/Qwen1.5-14B-Chat', 'Qwen/Qwen1.5-14B-Chat'),
                Model('Qwen/Qwen1.5-32B-Chat', 'Qwen/Qwen1.5-32B-Chat'),
                Model('Qwen/Qwen1.5-72B-Chat', 'Qwen/Qwen1.5-72B-Chat'),
                Model('Qwen/Qwen1.5-110B-Chat', 'Qwen/Qwen1.5-110B-Chat'),
                # base
                Model('Qwen/Qwen1.5-0.5B', 'Qwen/Qwen1.5-0.5B'),
                Model('Qwen/Qwen1.5-1.8B', 'Qwen/Qwen1.5-1.8B'),
                Model('Qwen/Qwen1.5-4B', 'Qwen/Qwen1.5-4B'),
                Model('Qwen/Qwen1.5-7B', 'Qwen/Qwen1.5-7B'),
                Model('Qwen/Qwen1.5-14B', 'Qwen/Qwen1.5-14B'),
                Model('Qwen/Qwen1.5-32B', 'Qwen/Qwen1.5-32B'),
                Model('Qwen/Qwen1.5-72B', 'Qwen/Qwen1.5-72B'),
                Model('Qwen/Qwen1.5-110B', 'Qwen/Qwen1.5-110B'),
                # gptq-int4
                Model('Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4', 'Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4'),
                Model('Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int4', 'Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int4'),
                Model('Qwen/Qwen1.5-4B-Chat-GPTQ-Int4', 'Qwen/Qwen1.5-4B-Chat-GPTQ-Int4'),
                Model('Qwen/Qwen1.5-7B-Chat-GPTQ-Int4', 'Qwen/Qwen1.5-7B-Chat-GPTQ-Int4'),
                Model('Qwen/Qwen1.5-14B-Chat-GPTQ-Int4', 'Qwen/Qwen1.5-14B-Chat-GPTQ-Int4'),
                Model('Qwen/Qwen1.5-32B-Chat-GPTQ-Int4', 'Qwen/Qwen1.5-32B-Chat-GPTQ-Int4'),
                Model('Qwen/Qwen1.5-72B-Chat-GPTQ-Int4', 'Qwen/Qwen1.5-72B-Chat-GPTQ-Int4'),
                Model('Qwen/Qwen1.5-110B-Chat-GPTQ-Int4', 'Qwen/Qwen1.5-110B-Chat-GPTQ-Int4'),
                # gptq-int8
                Model('Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int8', 'Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int8'),
                Model('Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int8', 'Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int8'),
                Model('Qwen/Qwen1.5-4B-Chat-GPTQ-Int8', 'Qwen/Qwen1.5-4B-Chat-GPTQ-Int8'),
                Model('Qwen/Qwen1.5-7B-Chat-GPTQ-Int8', 'Qwen/Qwen1.5-7B-Chat-GPTQ-Int8'),
                Model('Qwen/Qwen1.5-14B-Chat-GPTQ-Int8', 'Qwen/Qwen1.5-14B-Chat-GPTQ-Int8'),
                Model('Qwen/Qwen1.5-72B-Chat-GPTQ-Int8', 'Qwen/Qwen1.5-72B-Chat-GPTQ-Int8'),
                # awq-int4
                Model('Qwen/Qwen1.5-0.5B-Chat-AWQ', 'Qwen/Qwen1.5-0.5B-Chat-AWQ'),
                Model('Qwen/Qwen1.5-1.8B-Chat-AWQ', 'Qwen/Qwen1.5-1.8B-Chat-AWQ'),
                Model('Qwen/Qwen1.5-4B-Chat-AWQ', 'Qwen/Qwen1.5-4B-Chat-AWQ'),
                Model('Qwen/Qwen1.5-7B-Chat-AWQ', 'Qwen/Qwen1.5-7B-Chat-AWQ'),
                Model('Qwen/Qwen1.5-14B-Chat-AWQ', 'Qwen/Qwen1.5-14B-Chat-AWQ'),
                Model('Qwen/Qwen1.5-32B-Chat-AWQ', 'Qwen/Qwen1.5-32B-Chat-AWQ'),
                Model('Qwen/Qwen1.5-72B-Chat-AWQ', 'Qwen/Qwen1.5-72B-Chat-AWQ'),
                Model('Qwen/Qwen1.5-110B-Chat-AWQ', 'Qwen/Qwen1.5-110B-Chat-AWQ'),
            ]),
            # code-qwen1.5
            ModelGroup([
                Model('Qwen/CodeQwen1.5-7B', 'Qwen/CodeQwen1.5-7B'),
                Model('Qwen/CodeQwen1.5-7B-Chat', 'Qwen/CodeQwen1.5-7B-Chat'),
                Model('Qwen/CodeQwen1.5-7B-Chat-AWQ', 'Qwen/CodeQwen1.5-7B-Chat-AWQ'),
            ],
                       tags=['coding']),
            # qwen2
            ModelGroup([
                # instruct
                Model('Qwen/Qwen2-0.5B-Instruct', 'Qwen/Qwen2-0.5B-Instruct'),
                Model('Qwen/Qwen2-1.5B-Instruct', 'Qwen/Qwen2-1.5B-Instruct'),
                Model('Qwen/Qwen2-7B-Instruct', 'Qwen/Qwen2-7B-Instruct'),
                Model('Qwen/Qwen2-72B-Instruct', 'Qwen/Qwen2-72B-Instruct'),
                # base
                Model('Qwen/Qwen2-0.5B', 'Qwen/Qwen2-0.5B'),
                Model('Qwen/Qwen2-1.5B', 'Qwen/Qwen2-1.5B'),
                Model('Qwen/Qwen2-7B', 'Qwen/Qwen2-7B'),
                Model('Qwen/Qwen2-72B', 'Qwen/Qwen2-72B'),
                # gptq-int4
                Model('Qwen/Qwen2-0.5B-Instruct-GPTQ-Int4', 'Qwen/Qwen2-0.5B-Instruct-GPTQ-Int4'),
                Model('Qwen/Qwen2-1.5B-Instruct-GPTQ-Int4', 'Qwen/Qwen2-1.5B-Instruct-GPTQ-Int4'),
                Model('Qwen/Qwen2-7B-Instruct-GPTQ-Int4', 'Qwen/Qwen2-7B-Instruct-GPTQ-Int4'),
                Model('Qwen/Qwen2-72B-Instruct-GPTQ-Int4', 'Qwen/Qwen2-72B-Instruct-GPTQ-Int4'),
                # gptq-int8
                Model('Qwen/Qwen2-0.5B-Instruct-GPTQ-Int8', 'Qwen/Qwen2-0.5B-Instruct-GPTQ-Int8'),
                Model('Qwen/Qwen2-1.5B-Instruct-GPTQ-Int8', 'Qwen/Qwen2-1.5B-Instruct-GPTQ-Int8'),
                Model('Qwen/Qwen2-7B-Instruct-GPTQ-Int8', 'Qwen/Qwen2-7B-Instruct-GPTQ-Int8'),
                Model('Qwen/Qwen2-72B-Instruct-GPTQ-Int8', 'Qwen/Qwen2-72B-Instruct-GPTQ-Int8'),
                # awq-int4
                Model('Qwen/Qwen2-0.5B-Instruct-AWQ', 'Qwen/Qwen2-0.5B-Instruct-AWQ'),
                Model('Qwen/Qwen2-1.5B-Instruct-AWQ', 'Qwen/Qwen2-1.5B-Instruct-AWQ'),
                Model('Qwen/Qwen2-7B-Instruct-AWQ', 'Qwen/Qwen2-7B-Instruct-AWQ'),
                Model('Qwen/Qwen2-72B-Instruct-AWQ', 'Qwen/Qwen2-72B-Instruct-AWQ'),
            ]),
            # qwen2-math
            ModelGroup(
                [
                    # instruct
                    Model('Qwen/Qwen2-Math-1.5B-Instruct', 'Qwen/Qwen2-Math-1.5B-Instruct'),
                    Model('Qwen/Qwen2-Math-7B-Instruct', 'Qwen/Qwen2-Math-7B-Instruct'),
                    Model('Qwen/Qwen2-Math-72B-Instruct', 'Qwen/Qwen2-Math-72B-Instruct'),
                    # base
                    Model('Qwen/Qwen2-Math-1.5B', 'Qwen/Qwen2-Math-1.5B'),
                    Model('Qwen/Qwen2-Math-7B', 'Qwen/Qwen2-Math-7B'),
                    Model('Qwen/Qwen2-Math-72B', 'Qwen/Qwen2-Math-72B'),
                ],
                tags=['math']),
            # qwen2.5-1m
            ModelGroup([
                Model('Qwen/Qwen2.5-7B-Instruct-1M', 'Qwen/Qwen2.5-7B-Instruct-1M'),
                Model('Qwen/Qwen2.5-14B-Instruct-1M', 'Qwen/Qwen2.5-14B-Instruct-1M'),
            ]),
            # other
            ModelGroup([Model('PowerInfer/SmallThinker-3B-Preview', 'PowerInfer/SmallThinker-3B-Preview')]),
        ],
        TemplateType.qwen,
        get_model_tokenizer_with_flash_attn,
        architectures=['Qwen2ForCausalLM'],
        requires=['transformers>=4.37'],
        model_arch=ModelArch.llama))

register_model(
    ModelMeta(
        LLMModelType.qwen2_5,
        [
            # qwen2.5
            ModelGroup([
                # instruct
                Model('Qwen/Qwen2.5-0.5B-Instruct', 'Qwen/Qwen2.5-0.5B-Instruct'),
                Model('Qwen/Qwen2.5-1.5B-Instruct', 'Qwen/Qwen2.5-1.5B-Instruct'),
                Model('Qwen/Qwen2.5-3B-Instruct', 'Qwen/Qwen2.5-3B-Instruct'),
                Model('Qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen2.5-7B-Instruct'),
                Model('Qwen/Qwen2.5-14B-Instruct', 'Qwen/Qwen2.5-14B-Instruct'),
                Model('Qwen/Qwen2.5-32B-Instruct', 'Qwen/Qwen2.5-32B-Instruct'),
                Model('Qwen/Qwen2.5-72B-Instruct', 'Qwen/Qwen2.5-72B-Instruct'),
                # base
                Model('Qwen/Qwen2.5-0.5B', 'Qwen/Qwen2.5-0.5B'),
                Model('Qwen/Qwen2.5-1.5B', 'Qwen/Qwen2.5-1.5B'),
                Model('Qwen/Qwen2.5-3B', 'Qwen/Qwen2.5-3B'),
                Model('Qwen/Qwen2.5-7B', 'Qwen/Qwen2.5-7B'),
                Model('Qwen/Qwen2.5-14B', 'Qwen/Qwen2.5-14B'),
                Model('Qwen/Qwen2.5-32B', 'Qwen/Qwen2.5-32B'),
                Model('Qwen/Qwen2.5-72B', 'Qwen/Qwen2.5-72B'),
                # gptq-int4
                Model('Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4', 'Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4'),
                Model('Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4', 'Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4'),
                Model('Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4', 'Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4'),
                Model('Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4', 'Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4'),
                Model('Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4', 'Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4'),
                Model('Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4', 'Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4'),
                Model('Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4', 'Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4'),
                # gptq-int8
                Model('Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8', 'Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8'),
                Model('Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8', 'Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8'),
                Model('Qwen/Qwen2.5-3B-Instruct-GPTQ-Int8', 'Qwen/Qwen2.5-3B-Instruct-GPTQ-Int8'),
                Model('Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8', 'Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8'),
                Model('Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8', 'Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8'),
                Model('Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8', 'Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8'),
                Model('Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8', 'Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8'),
                # awq-int4
                Model('Qwen/Qwen2.5-0.5B-Instruct-AWQ', 'Qwen/Qwen2.5-0.5B-Instruct-AWQ'),
                Model('Qwen/Qwen2.5-1.5B-Instruct-AWQ', 'Qwen/Qwen2.5-1.5B-Instruct-AWQ'),
                Model('Qwen/Qwen2.5-3B-Instruct-AWQ', 'Qwen/Qwen2.5-3B-Instruct-AWQ'),
                Model('Qwen/Qwen2.5-7B-Instruct-AWQ', 'Qwen/Qwen2.5-7B-Instruct-AWQ'),
                Model('Qwen/Qwen2.5-14B-Instruct-AWQ', 'Qwen/Qwen2.5-14B-Instruct-AWQ'),
                Model('Qwen/Qwen2.5-32B-Instruct-AWQ', 'Qwen/Qwen2.5-32B-Instruct-AWQ'),
                Model('Qwen/Qwen2.5-72B-Instruct-AWQ', 'Qwen/Qwen2.5-72B-Instruct-AWQ'),
            ]),
            # qwen2.5-coder
            ModelGroup(
                [
                    # instruct
                    Model('Qwen/Qwen2.5-Coder-0.5B-Instruct', 'Qwen/Qwen2.5-Coder-0.5B-Instruct'),
                    Model('Qwen/Qwen2.5-Coder-1.5B-Instruct', 'Qwen/Qwen2.5-Coder-1.5B-Instruct'),
                    Model('Qwen/Qwen2.5-Coder-3B-Instruct', 'Qwen/Qwen2.5-Coder-3B-Instruct'),
                    Model('Qwen/Qwen2.5-Coder-7B-Instruct', 'Qwen/Qwen2.5-Coder-7B-Instruct'),
                    Model('Qwen/Qwen2.5-Coder-14B-Instruct', 'Qwen/Qwen2.5-Coder-14B-Instruct'),
                    Model('Qwen/Qwen2.5-Coder-32B-Instruct', 'Qwen/Qwen2.5-Coder-32B-Instruct'),
                    # base
                    Model('Qwen/Qwen2.5-Coder-0.5B', 'Qwen/Qwen2.5-Coder-0.5B'),
                    Model('Qwen/Qwen2.5-Coder-1.5B', 'Qwen/Qwen2.5-Coder-1.5B'),
                    Model('Qwen/Qwen2.5-Coder-3B', 'Qwen/Qwen2.5-Coder-3B'),
                    Model('Qwen/Qwen2.5-Coder-7B', 'Qwen/Qwen2.5-Coder-7B'),
                    Model('Qwen/Qwen2.5-Coder-14B', 'Qwen/Qwen2.5-Coder-14B'),
                    Model('Qwen/Qwen2.5-Coder-32B', 'Qwen/Qwen2.5-Coder-32B'),
                    # AWQ
                    Model('Qwen/Qwen2.5-Coder-0.5B-Instruct-AWQ', 'Qwen/Qwen2.5-Coder-0.5B-Instruct-AWQ'),
                    Model('Qwen/Qwen2.5-Coder-1.5B-Instruct-AWQ', 'Qwen/Qwen2.5-Coder-1.5B-Instruct-AWQ'),
                    Model('Qwen/Qwen2.5-Coder-3B-Instruct-AWQ', 'Qwen/Qwen2.5-Coder-3B-Instruct-AWQ'),
                    Model('Qwen/Qwen2.5-Coder-7B-Instruct-AWQ', 'Qwen/Qwen2.5-Coder-7B-Instruct-AWQ'),
                    Model('Qwen/Qwen2.5-Coder-14B-Instruct-AWQ', 'Qwen/Qwen2.5-Coder-14B-Instruct-AWQ'),
                    Model('Qwen/Qwen2.5-Coder-32B-Instruct-AWQ', 'Qwen/Qwen2.5-Coder-32B-Instruct-AWQ'),
                    # GPTQ
                    Model('Qwen/Qwen2.5-Coder-0.5B-Instruct-GPTQ-Int4', 'Qwen/Qwen2.5-Coder-0.5B-Instruct-GPTQ-Int4'),
                    Model('Qwen/Qwen2.5-Coder-0.5B-Instruct-GPTQ-Int8', 'Qwen/Qwen2.5-Coder-0.5B-Instruct-GPTQ-Int8'),
                    Model('Qwen/Qwen2.5-Coder-1.5B-Instruct-GPTQ-Int4', 'Qwen/Qwen2.5-Coder-1.5B-Instruct-GPTQ-Int4'),
                    Model('Qwen/Qwen2.5-Coder-1.5B-Instruct-GPTQ-Int8', 'Qwen/Qwen2.5-Coder-1.5B-Instruct-GPTQ-Int8'),
                    Model('Qwen/Qwen2.5-Coder-3B-Instruct-GPTQ-Int4', 'Qwen/Qwen2.5-Coder-3B-Instruct-GPTQ-Int4'),
                    Model('Qwen/Qwen2.5-Coder-3B-Instruct-GPTQ-Int8', 'Qwen/Qwen2.5-Coder-3B-Instruct-GPTQ-Int8'),
                    Model('Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-Int4', 'Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-Int4'),
                    Model('Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-Int8', 'Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-Int8'),
                    Model('Qwen/Qwen2.5-Coder-14B-Instruct-GPTQ-Int4', 'Qwen/Qwen2.5-Coder-14B-Instruct-GPTQ-Int4'),
                    Model('Qwen/Qwen2.5-Coder-14B-Instruct-GPTQ-Int8', 'Qwen/Qwen2.5-Coder-14B-Instruct-GPTQ-Int8'),
                    Model('Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ-Int4', 'Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ-Int4'),
                    Model('Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ-Int8', 'Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ-Int8'),
                ],
                tags=['coding']),
            ModelGroup([
                Model('moonshotai/Kimi-Dev-72B', 'moonshotai/Kimi-Dev-72B'),
            ]),
        ],
        TemplateType.qwen2_5,
        get_model_tokenizer_with_flash_attn,
        architectures=['Qwen2ForCausalLM'],
        requires=['transformers>=4.37'],
        model_arch=ModelArch.llama))

register_model(
    ModelMeta(
        LLMModelType.qwen2_5_math,
        [
            # qwen2.5-math
            ModelGroup(
                [
                    # instruct
                    Model('Qwen/Qwen2.5-Math-1.5B-Instruct', 'Qwen/Qwen2.5-Math-1.5B-Instruct'),
                    Model('Qwen/Qwen2.5-Math-7B-Instruct', 'Qwen/Qwen2.5-Math-7B-Instruct'),
                    Model('Qwen/Qwen2.5-Math-72B-Instruct', 'Qwen/Qwen2.5-Math-72B-Instruct'),
                    # base
                    Model('Qwen/Qwen2.5-Math-1.5B', 'Qwen/Qwen2.5-Math-1.5B'),
                    Model('Qwen/Qwen2.5-Math-7B', 'Qwen/Qwen2.5-Math-7B'),
                    Model('Qwen/Qwen2.5-Math-72B', 'Qwen/Qwen2.5-Math-72B'),
                ],
                tags=['math']),
        ],
        TemplateType.qwen2_5_math,
        get_model_tokenizer_with_flash_attn,
        architectures=['Qwen2ForCausalLM'],
        requires=['transformers>=4.37'],
        model_arch=ModelArch.llama))

register_model(
    ModelMeta(
        LLMModelType.qwen2_moe,
        [
            # qwen1.5-moe
            ModelGroup([
                Model('Qwen/Qwen1.5-MoE-A2.7B-Chat', 'Qwen/Qwen1.5-MoE-A2.7B-Chat'),
                Model('Qwen/Qwen1.5-MoE-A2.7B', 'Qwen/Qwen1.5-MoE-A2.7B'),
                Model('Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4', 'Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4'),
            ]),
            ModelGroup([
                Model('Qwen/Qwen2-57B-A14B-Instruct', 'Qwen/Qwen2-57B-A14B-Instruct'),
                Model('Qwen/Qwen2-57B-A14B', 'Qwen/Qwen2-57B-A14B'),
                Model('Qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4', 'Qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4'),
            ])
        ],
        TemplateType.qwen,
        get_model_tokenizer_with_flash_attn,
        architectures=['Qwen2MoeForCausalLM'],
        requires=['transformers>=4.40'],
    ))

register_model(
    ModelMeta(
        LLMModelType.qwen3,
        [
            ModelGroup([
                Model('Qwen/Qwen3-0.6B-Base', 'Qwen/Qwen3-0.6B-Base'),
                Model('Qwen/Qwen3-1.7B-Base', 'Qwen/Qwen3-1.7B-Base'),
                Model('Qwen/Qwen3-4B-Base', 'Qwen/Qwen3-4B-Base'),
                Model('Qwen/Qwen3-8B-Base', 'Qwen/Qwen3-8B-Base'),
                Model('Qwen/Qwen3-14B-Base', 'Qwen/Qwen3-14B-Base'),
                # instruct
                Model('Qwen/Qwen3-0.6B', 'Qwen/Qwen3-0.6B'),
                Model('Qwen/Qwen3-1.7B', 'Qwen/Qwen3-1.7B'),
                Model('Qwen/Qwen3-4B', 'Qwen/Qwen3-4B'),
                Model('Qwen/Qwen3-8B', 'Qwen/Qwen3-8B'),
                Model('Qwen/Qwen3-14B', 'Qwen/Qwen3-14B'),
                Model('Qwen/Qwen3-32B', 'Qwen/Qwen3-32B'),
                # fp8
                Model('Qwen/Qwen3-0.6B-FP8', 'Qwen/Qwen3-0.6B-FP8'),
                Model('Qwen/Qwen3-1.7B-FP8', 'Qwen/Qwen3-1.7B-FP8'),
                Model('Qwen/Qwen3-4B-FP8', 'Qwen/Qwen3-4B-FP8'),
                Model('Qwen/Qwen3-8B-FP8', 'Qwen/Qwen3-8B-FP8'),
                Model('Qwen/Qwen3-14B-FP8', 'Qwen/Qwen3-14B-FP8'),
                Model('Qwen/Qwen3-32B-FP8', 'Qwen/Qwen3-32B-FP8'),
                # awq
                Model('Qwen/Qwen3-4B-AWQ', 'Qwen/Qwen3-4B-AWQ'),
                Model('Qwen/Qwen3-8B-AWQ', 'Qwen/Qwen3-8B-AWQ'),
                Model('Qwen/Qwen3-14B-AWQ', 'Qwen/Qwen3-14B-AWQ'),
                Model('Qwen/Qwen3-32B-AWQ', 'Qwen/Qwen3-32B-AWQ'),
                # swift
                Model('swift/Qwen3-32B-AWQ'),
            ]),
        ],
        TemplateType.qwen3,
        get_model_tokenizer_with_flash_attn,
        architectures=['Qwen3ForCausalLM'],
        requires=['transformers>=4.51'],
        model_arch=ModelArch.llama))

register_model(
    ModelMeta(
        LLMModelType.qwen3_moe,
        [
            ModelGroup([
                Model('Qwen/Qwen3-30B-A3B-Base', 'Qwen/Qwen3-30B-A3B-Base'),
                # instruct
                Model('Qwen/Qwen3-30B-A3B', 'Qwen/Qwen3-30B-A3B'),
                Model('Qwen/Qwen3-235B-A22B', 'Qwen/Qwen3-235B-A22B'),
                # fp8
                Model('Qwen/Qwen3-30B-A3B-FP8', 'Qwen/Qwen3-30B-A3B-FP8'),
                Model('Qwen/Qwen3-235B-A22B-FP8', 'Qwen/Qwen3-235B-A22B-FP8'),
                # awq
                Model('swift/Qwen3-30B-A3B-AWQ', 'cognitivecomputations/Qwen3-30B-A3B-AWQ'),
                Model('swift/Qwen3-235B-A22B-AWQ', 'cognitivecomputations/Qwen3-235B-A22B-AWQ'),
            ]),
            ModelGroup([
                Model('iic/Tongyi-DeepResearch-30B-A3B', 'Alibaba-NLP/Tongyi-DeepResearch-30B-A3B'),
            ])
        ],
        TemplateType.qwen3,
        get_model_tokenizer_with_flash_attn,
        architectures=['Qwen3MoeForCausalLM'],
        requires=['transformers>=4.51'],
    ))

register_model(
    ModelMeta(
        LLMModelType.qwen3_guard,
        [
            ModelGroup([
                Model('Qwen/Qwen3Guard-Gen-0.6B', 'Qwen/Qwen3Guard-Gen-0.6B'),
                Model('Qwen/Qwen3Guard-Gen-4B', 'Qwen/Qwen3Guard-Gen-4B'),
                Model('Qwen/Qwen3Guard-Gen-8B', 'Qwen/Qwen3Guard-Gen-8B'),
            ])
        ],
        TemplateType.qwen3_guard,
        get_model_tokenizer_with_flash_attn,
        architectures=['Qwen3ForCausalLM'],
        requires=['transformers>=4.51'],
    ))

register_model(
    ModelMeta(
        LLMModelType.qwen3_thinking,
        [
            ModelGroup([
                Model('Qwen/Qwen3-4B-Thinking-2507', 'Qwen/Qwen3-4B-Thinking-2507'),
                Model('Qwen/Qwen3-4B-Thinking-2507-FP8', 'Qwen/Qwen3-4B-Thinking-2507-FP8'),
            ]),
        ],
        TemplateType.qwen3_thinking,
        get_model_tokenizer_with_flash_attn,
        architectures=['Qwen3ForCausalLM'],
        requires=['transformers>=4.51'],
    ))

register_model(
    ModelMeta(
        LLMModelType.qwen3_nothinking,
        [
            ModelGroup([
                Model('Qwen/Qwen3-30B-A3B-Instruct-2507', 'Qwen/Qwen3-30B-A3B-Instruct-2507'),
                Model('Qwen/Qwen3-30B-A3B-Instruct-2507-FP8', 'Qwen/Qwen3-30B-A3B-Instruct-2507-FP8'),
                Model('Qwen/Qwen3-235B-A22B-Instruct-2507', 'Qwen/Qwen3-235B-A22B-Instruct-2507'),
                Model('Qwen/Qwen3-235B-A22B-Instruct-2507-FP8', 'Qwen/Qwen3-235B-A22B-Instruct-2507-FP8'),
                # awq
                Model('swift/Qwen3-235B-A22B-Instruct-2507-AWQ'),
            ]),
            ModelGroup([
                Model('Qwen/Qwen3-4B-Instruct-2507', 'Qwen/Qwen3-4B-Instruct-2507'),
                Model('Qwen/Qwen3-4B-Instruct-2507-FP8', 'Qwen/Qwen3-4B-Instruct-2507-FP8'),
            ])
        ],
        TemplateType.qwen3_nothinking,
        get_model_tokenizer_with_flash_attn,
        architectures=['Qwen3MoeForCausalLM', 'Qwen3ForCausalLM'],
        requires=['transformers>=4.51'],
    ))

register_model(
    ModelMeta(
        LLMModelType.qwen3_coder,
        [
            ModelGroup([
                Model('Qwen/Qwen3-Coder-30B-A3B-Instruct', 'Qwen/Qwen3-Coder-30B-A3B-Instruct'),
                Model('Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8', 'Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8'),
                Model('Qwen/Qwen3-Coder-480B-A35B-Instruct', 'Qwen/Qwen3-Coder-480B-A35B-Instruct'),
                Model('Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8', 'Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8'),
                Model('swift/Qwen3-Coder-480B-A35B-Instruct-AWQ'),
            ],
                       tags=['coding']),
        ],
        TemplateType.qwen3_coder,
        get_model_tokenizer_with_flash_attn,
        architectures=['Qwen3MoeForCausalLM'],
        requires=['transformers>=4.51'],
    ))

register_model(
    ModelMeta(
        LLMModelType.qwen3_moe_thinking,
        [
            ModelGroup([
                Model('Qwen/Qwen3-30B-A3B-Thinking-2507', 'Qwen/Qwen3-30B-A3B-Thinking-2507'),
                Model('Qwen/Qwen3-30B-A3B-Thinking-2507-FP8', 'Qwen/Qwen3-30B-A3B-Thinking-2507-FP8'),
                Model('Qwen/Qwen3-235B-A22B-Thinking-2507', 'Qwen/Qwen3-235B-A22B-Thinking-2507'),
                Model('Qwen/Qwen3-235B-A22B-Thinking-2507-FP8', 'Qwen/Qwen3-235B-A22B-Thinking-2507-FP8'),
                # awq
                Model('swift/Qwen3-235B-A22B-Thinking-2507-AWQ'),
            ]),
        ],
        TemplateType.qwen3_thinking,
        get_model_tokenizer_with_flash_attn,
        architectures=['Qwen3MoeForCausalLM'],
        requires=['transformers>=4.51'],
    ))

register_model(
    ModelMeta(
        LLMModelType.qwen3_next,
        [ModelGroup([
            Model('Qwen/Qwen3-Next-80B-A3B-Instruct'),
            Model('Qwen/Qwen3-Next-80B-A3B-Instruct-FP8'),
        ])],
        TemplateType.qwen3_nothinking,
        get_model_tokenizer_with_flash_attn,
        architectures=['Qwen3NextForCausalLM'],
        requires=['transformers>=4.57'],
    ))

register_model(
    ModelMeta(
        LLMModelType.qwen3_next_thinking,
        [ModelGroup([
            Model('Qwen/Qwen3-Next-80B-A3B-Thinking'),
            Model('Qwen/Qwen3-Next-80B-A3B-Thinking-FP8'),
        ])],
        TemplateType.qwen3_thinking,
        get_model_tokenizer_with_flash_attn,
        architectures=['Qwen3NextForCausalLM'],
        requires=['transformers>=4.57'],
    ))


def patch_qwen_vl_utils(vision_process):
    if hasattr(vision_process, '_patch'):
        return
    if os.getenv('VIDEO_MAX_PIXELS') and not os.getenv('VIDEO_TOTAL_PIXELS'):
        # https://github.com/QwenLM/Qwen2.5-VL/issues/1120
        os.environ['VIDEO_TOTAL_PIXELS'] = str(int(128000 * 28 * 28 * 0.9))
    res = {}
    for key in [
            'image_factor',  # image_patch_size * SPATIAL_MERGE_SIZE
            'min_pixels',  # IMAGE_MIN_TOKEN_NUM * image_factor ** 2
            'max_pixels',
            'video_min_pixels',
            'video_max_pixels',
            'video_total_pixels',
            #
            'max_ratio',
            'frame_factor',
            'fps',
            'fps_min_frames',
            'fps_max_frames',
            # qwen3_vl
            'image_max_token_num',
            'image_min_token_num',
            'spatial_merge_size',
            'video_max_token_num',
            'video_min_token_num',
    ]:
        type_func = float if key == 'fps' else int
        default_value = getattr(vision_process, key.upper(), None)
        if default_value is None:
            # Skip keys not supported by the specific vision_process implementation
            continue
        val = get_env_args(key, type_func, default_value)
        setattr(vision_process, key.upper(), val)
        res[key] = val
    # Patch decord video reader if available
    _read_video_decord = getattr(vision_process, '_read_video_decord', None)
    if _read_video_decord is not None:

        def _new_read_video_decord(ele: dict):
            from swift.llm import load_file
            ele['video'] = load_file(ele['video'])
            return _read_video_decord(ele)

        backends = getattr(vision_process, 'VIDEO_READER_BACKENDS', None)
        if isinstance(backends, dict):
            backends['decord'] = _new_read_video_decord
        elif backends is None:  # keye_vl
            vision_process._read_video_decord = _new_read_video_decord
    vision_process._patch = True
    return res


def compat_qwen_vl_utils(image_patch_size: int):
    spatial_merge_size = int(os.getenv('SPATIAL_MERGE_SIZE', '2'))
    image_factor = image_patch_size * spatial_merge_size
    env_vars_to_process = {
        'MAX_PIXELS': 'IMAGE_MAX_TOKEN_NUM',
        'MIN_PIXELS': 'IMAGE_MIN_TOKEN_NUM',
        'VIDEO_MAX_PIXELS': 'VIDEO_MAX_TOKEN_NUM',
        'VIDEO_MIN_PIXELS': 'VIDEO_MIN_TOKEN_NUM',
    }
    for source_var, target_var in env_vars_to_process.items():
        value = os.getenv(source_var)
        if value and not os.getenv(target_var):
            os.environ[target_var] = str(int(value) // image_factor**2)


def get_model_tokenizer_qwen2_vl(*args, **kwargs):
    from transformers import Qwen2VLForConditionalGeneration
    kwargs['automodel_class'] = kwargs['automodel_class'] or Qwen2VLForConditionalGeneration
    model, tokenizer = get_model_tokenizer_multimodal(*args, **kwargs)
    if model is not None:
        base_model = model.model if 'AWQ' in model.__class__.__name__ else model
        patch_get_input_embeddings(base_model.visual, 'patch_embed')

    from qwen_vl_utils import vision_process
    import qwen_vl_utils
    check_qwen_vl_utils = kwargs.get('_check_qwen_vl_utils', True)
    if check_qwen_vl_utils:
        try:
            qwen_vl_utils_version = importlib.metadata.version('qwen_vl_utils')
        except importlib.metadata.PackageNotFoundError:
            raise importlib.metadata.PackageNotFoundError(
                "The 'qwen_vl_utils' distribution was not found and is required by this application.")
        if version.parse(qwen_vl_utils_version) >= version.parse('0.0.14'):
            compat_qwen_vl_utils(image_patch_size=14)
        else:
            require_version('qwen_vl_utils<0.0.12')
    global_vars = patch_qwen_vl_utils(vision_process)
    tokenizer.global_vars = global_vars  # In order to have different hashes for the template.
    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.qwen2_vl,
        [
            ModelGroup(
                [
                    # chat
                    Model('Qwen/Qwen2-VL-2B-Instruct', 'Qwen/Qwen2-VL-2B-Instruct'),
                    Model('Qwen/Qwen2-VL-7B-Instruct', 'Qwen/Qwen2-VL-7B-Instruct'),
                    Model('Qwen/Qwen2-VL-72B-Instruct', 'Qwen/Qwen2-VL-72B-Instruct'),
                    # base
                    Model('Qwen/Qwen2-VL-2B', 'Qwen/Qwen2-VL-2B'),
                    Model('Qwen/Qwen2-VL-7B', 'Qwen/Qwen2-VL-7B'),
                    Model('Qwen/Qwen2-VL-72B', 'Qwen/Qwen2-VL-72B'),
                    # gptq-int4
                    Model('Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4', 'Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4'),
                    Model('Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4', 'Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4'),
                    Model('Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4', 'Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4'),
                    # gptq-int8
                    Model('Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8', 'Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8'),
                    Model('Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int8', 'Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int8'),
                    Model('Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int8', 'Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int8'),
                    # awq-int4
                    Model('Qwen/Qwen2-VL-2B-Instruct-AWQ', 'Qwen/Qwen2-VL-2B-Instruct-AWQ'),
                    Model('Qwen/Qwen2-VL-7B-Instruct-AWQ', 'Qwen/Qwen2-VL-7B-Instruct-AWQ'),
                    Model('Qwen/Qwen2-VL-72B-Instruct-AWQ', 'Qwen/Qwen2-VL-72B-Instruct-AWQ'),
                ], ),
            ModelGroup([
                Model('bytedance-research/UI-TARS-2B-SFT', 'bytedance-research/UI-TARS-2B-SFT'),
                Model('bytedance-research/UI-TARS-7B-SFT', 'bytedance-research/UI-TARS-7B-SFT'),
                Model('bytedance-research/UI-TARS-7B-DPO', 'bytedance-research/UI-TARS-7B-DPO'),
                Model('bytedance-research/UI-TARS-72B-SFT', 'bytedance-research/UI-TARS-72B-SFT'),
                Model('bytedance-research/UI-TARS-72B-DPO', 'bytedance-research/UI-TARS-72B-DPO'),
            ]),
            ModelGroup([
                Model('allenai/olmOCR-7B-0225-preview', 'allenai/olmOCR-7B-0225-preview'),
            ]),
        ],
        TemplateType.qwen2_vl,
        get_model_tokenizer_qwen2_vl,
        model_arch=ModelArch.qwen2_vl,
        architectures=['Qwen2VLForConditionalGeneration'],
        requires=['transformers>=4.45', 'qwen_vl_utils>=0.0.6', 'decord'],
        tags=['vision', 'video']))

register_model(
    ModelMeta(
        MLLMModelType.qvq, [
            ModelGroup([
                Model('Qwen/QVQ-72B-Preview', 'Qwen/QVQ-72B-Preview'),
            ]),
        ],
        TemplateType.qvq,
        get_model_tokenizer_qwen2_vl,
        model_arch=ModelArch.qwen2_vl,
        architectures=['Qwen2VLForConditionalGeneration'],
        requires=['transformers>=4.45', 'qwen_vl_utils>=0.0.6', 'decord'],
        tags=['vision', 'video']))


def get_model_tokenizer_qwen2_5_vl(*args, **kwargs):
    from transformers import Qwen2_5_VLForConditionalGeneration
    kwargs['automodel_class'] = kwargs['automodel_class'] or Qwen2_5_VLForConditionalGeneration
    return get_model_tokenizer_qwen2_vl(*args, **kwargs)


register_model(
    ModelMeta(
        MLLMModelType.qwen2_5_vl, [
            ModelGroup([
                Model('Qwen/Qwen2.5-VL-3B-Instruct', 'Qwen/Qwen2.5-VL-3B-Instruct'),
                Model('Qwen/Qwen2.5-VL-7B-Instruct', 'Qwen/Qwen2.5-VL-7B-Instruct'),
                Model('Qwen/Qwen2.5-VL-32B-Instruct', 'Qwen/Qwen2.5-VL-32B-Instruct'),
                Model('Qwen/Qwen2.5-VL-72B-Instruct', 'Qwen/Qwen2.5-VL-72B-Instruct'),
            ]),
            ModelGroup([
                Model('Qwen/Qwen2.5-VL-3B-Instruct-AWQ', 'Qwen/Qwen2.5-VL-3B-Instruct-AWQ'),
                Model('Qwen/Qwen2.5-VL-7B-Instruct-AWQ', 'Qwen/Qwen2.5-VL-7B-Instruct-AWQ'),
                Model('Qwen/Qwen2.5-VL-32B-Instruct-AWQ', 'Qwen/Qwen2.5-VL-32B-Instruct-AWQ'),
                Model('Qwen/Qwen2.5-VL-72B-Instruct-AWQ', 'Qwen/Qwen2.5-VL-72B-Instruct-AWQ'),
            ]),
        ],
        TemplateType.qwen2_5_vl,
        get_model_tokenizer_qwen2_5_vl,
        model_arch=ModelArch.qwen2_vl,
        architectures=['Qwen2_5_VLForConditionalGeneration'],
        requires=['transformers>=4.49', 'qwen_vl_utils>=0.0.6', 'decord'],
        tags=['vision', 'video']))


def patch_Qwen3VLMoeTextExperts_dtype():
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextExperts
    if hasattr(Qwen3VLMoeTextExperts, '_patch'):
        return
    Qwen3VLMoeTextExperts._patch = True
    origin_forward = Qwen3VLMoeTextExperts.forward

    def forward(self, hidden_states, *args, **kwargs):
        res = origin_forward(self, hidden_states, *args, **kwargs)
        return res.to(hidden_states.dtype)

    Qwen3VLMoeTextExperts.forward = forward


def _forward_qwen3_vl_or_qwen3_omni(
    self,
    processor,
    input_ids,
    inputs_embeds,
    pixel_values,
    pixel_values_videos,
    image_grid_thw,
    video_grid_thw,
):
    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    dtype = self.visual.dtype
    if pixel_values is None and pixel_values_videos is None:  # plain-text
        images = [Image.new('RGB', (32, 32), (0, 0, 0))]
        media_inputs = processor.image_processor(images=images, return_tensors='pt')
        media_inputs = to_device(media_inputs, input_ids.device)
        pixel_values = media_inputs['pixel_values'].type(dtype)
        image_embeds, deepstack_visual_embeds = self.visual(pixel_values, grid_thw=media_inputs['image_grid_thw'])
        inputs_embeds = inputs_embeds + image_embeds.mean().to(device=inputs_embeds.device) * 0.
        visual_pos_masks = None
    else:
        if pixel_values is None:
            pixel_values_mixed = pixel_values_videos
            grid_thw = video_grid_thw
        elif pixel_values_videos is None:
            pixel_values_mixed = pixel_values
            grid_thw = image_grid_thw
        else:
            pixel_values_mixed = torch.concat([pixel_values, pixel_values_videos], dim=0)
            grid_thw = torch.concat([image_grid_thw, video_grid_thw], dim=0)
        pixel_values_mixed = pixel_values_mixed.type(dtype)
        mixed_embeds, deepstack_visual_embeds = self.visual(pixel_values_mixed, grid_thw=grid_thw)
        if pixel_values is None:
            image_embeds = None
            video_embeds = mixed_embeds
        elif pixel_values_videos is None:
            image_embeds = mixed_embeds
            video_embeds = None
        else:
            merge_length = processor.image_processor.merge_size**2
            image_tokens = (image_grid_thw.prod(dim=-1) // merge_length).sum()
            image_embeds = mixed_embeds[:image_tokens]
            video_embeds = mixed_embeds[image_tokens:]

        image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        if image_embeds is not None:
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask = image_mask.to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if video_embeds is not None:
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            video_mask = video_mask.to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
        image_mask, video_mask = image_mask[..., 0], video_mask[..., 0]
        visual_pos_masks = image_mask | video_mask
        if image_embeds is not None and video_embeds is not None:
            deepstack_image_embeds = [tensor[:image_tokens] for tensor in deepstack_visual_embeds]
            deepstack_video_embeds = [tensor[image_tokens:] for tensor in deepstack_visual_embeds]
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
    return inputs_embeds, visual_pos_masks, deepstack_visual_embeds


def _patch_deepstack_process(model):

    def _deepstack_process(self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor,
                           visual_embeds: torch.Tensor):
        from swift.trainers.sequence_parallel import sequence_parallel
        world_size = sequence_parallel.world_size
        if world_size and world_size > 1 and visual_pos_masks is not None:
            visual_pos_masks, visual_embeds = sequence_parallel.pad_and_split_mm_tokens(visual_pos_masks, visual_embeds)
        if visual_pos_masks is None:
            return hidden_states + visual_embeds.mean() * 0
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states

    model._deepstack_process = MethodType(_deepstack_process, model)


def _compat_qwen3_vl_mixed_data(model, processor, is_moe: bool = False):
    if not is_deepspeed_enabled() or hasattr(model, 'origin_forward'):
        return
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (Qwen3VLModelOutputWithPast, TransformersKwargs, Unpack,
                                                                check_model_inputs, Cache, is_torchdynamo_compiling)
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeModelOutputWithPast
    output_cls = Qwen3VLMoeModelOutputWithPast if is_moe else Qwen3VLModelOutputWithPast

    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, output_cls]:
        if not self.training and not is_deepspeed_enabled():
            return self.origin_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                cache_position=cache_position,
                **kwargs,
            )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError('You must specify exactly one of input_ids or inputs_embeds')

        inputs_embeds, visual_pos_masks, deepstack_visual_embeds = _forward_qwen3_vl_or_qwen3_omni(
            self, processor, input_ids, inputs_embeds, pixel_values, pixel_values_videos, image_grid_thw,
            video_grid_thw)
        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask['full_attention'])
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                # Only apply conversion for floating point tensors (inverted masks)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1) or
                (inputs_embeds is not None and inputs_embeds.shape[1] != 1))
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0) or
                (past_key_values is None or past_key_values.get_seq_length() == 0))
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = ((cache_position[0]
                          + self.rope_deltas).to(inputs_embeds.device) if cache_position is not None else 0)
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        return output_cls(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )

    model.origin_forward = model.forward
    model.forward = MethodType(forward, model)
    _patch_deepstack_process(model.language_model)


def get_model_tokenizer_qwen3_vl(model_dir, *args, **kwargs):
    from transformers import Qwen3VLForConditionalGeneration
    require_version('qwen_vl_utils>=0.0.14')
    compat_qwen_vl_utils(image_patch_size=16)
    kwargs['automodel_class'] = kwargs['automodel_class'] or Qwen3VLForConditionalGeneration
    kwargs['_check_qwen_vl_utils'] = False
    model, processor = get_model_tokenizer_qwen2_vl(model_dir, *args, **kwargs)
    if model is not None:
        _compat_qwen3_vl_mixed_data(model.model, processor)
    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.qwen3_vl, [
            ModelGroup([
                Model('Qwen/Qwen3-VL-2B-Instruct', 'Qwen/Qwen3-VL-2B-Instruct'),
                Model('Qwen/Qwen3-VL-2B-Thinking', 'Qwen/Qwen3-VL-2B-Thinking'),
                Model('Qwen/Qwen3-VL-2B-Instruct-FP8', 'Qwen/Qwen3-VL-2B-Instruct-FP8'),
                Model('Qwen/Qwen3-VL-2B-Thinking-FP8', 'Qwen/Qwen3-VL-2B-Thinking-FP8'),
                Model('Qwen/Qwen3-VL-4B-Instruct', 'Qwen/Qwen3-VL-4B-Instruct'),
                Model('Qwen/Qwen3-VL-4B-Thinking', 'Qwen/Qwen3-VL-4B-Thinking'),
                Model('Qwen/Qwen3-VL-4B-Instruct-FP8', 'Qwen/Qwen3-VL-4B-Instruct-FP8'),
                Model('Qwen/Qwen3-VL-4B-Thinking-FP8', 'Qwen/Qwen3-VL-4B-Thinking-FP8'),
                Model('Qwen/Qwen3-VL-8B-Instruct', 'Qwen/Qwen3-VL-8B-Instruct'),
                Model('Qwen/Qwen3-VL-8B-Thinking', 'Qwen/Qwen3-VL-8B-Thinking'),
                Model('Qwen/Qwen3-VL-8B-Instruct-FP8', 'Qwen/Qwen3-VL-8B-Instruct-FP8'),
                Model('Qwen/Qwen3-VL-8B-Thinking-FP8', 'Qwen/Qwen3-VL-8B-Thinking-FP8'),
                Model('Qwen/Qwen3-VL-32B-Instruct', 'Qwen/Qwen3-VL-32B-Instruct'),
                Model('Qwen/Qwen3-VL-32B-Thinking', 'Qwen/Qwen3-VL-32B-Thinking'),
                Model('Qwen/Qwen3-VL-32B-Instruct-FP8', 'Qwen/Qwen3-VL-32B-Instruct-FP8'),
                Model('Qwen/Qwen3-VL-32B-Thinking-FP8', 'Qwen/Qwen3-VL-32B-Thinking-FP8'),
            ]),
        ],
        TemplateType.qwen3_vl,
        get_model_tokenizer_qwen3_vl,
        model_arch=ModelArch.qwen3_vl,
        architectures=['Qwen3VLForConditionalGeneration'],
        requires=['transformers>=4.57', 'qwen_vl_utils>=0.0.14', 'decord'],
        tags=['vision', 'video']))


def get_model_tokenizer_qwen3_moe_vl(model_dir, *args, **kwargs):
    from transformers import Qwen3VLMoeForConditionalGeneration
    require_version('qwen_vl_utils>=0.0.14')
    compat_qwen_vl_utils(image_patch_size=16)
    kwargs['automodel_class'] = kwargs['automodel_class'] or Qwen3VLMoeForConditionalGeneration
    kwargs['_check_qwen_vl_utils'] = False
    model, processor = get_model_tokenizer_qwen2_vl(model_dir, *args, **kwargs)
    patch_Qwen3VLMoeTextExperts_dtype()
    if model is not None:
        _compat_qwen3_vl_mixed_data(model.model, processor, True)
    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.qwen3_moe_vl, [
            ModelGroup([
                Model('Qwen/Qwen3-VL-30B-A3B-Instruct', 'Qwen/Qwen3-VL-30B-A3B-Instruct'),
                Model('Qwen/Qwen3-VL-30B-A3B-Thinking', 'Qwen/Qwen3-VL-30B-A3B-Thinking'),
                Model('Qwen/Qwen3-VL-30B-A3B-Instruct-FP8', 'Qwen/Qwen3-VL-30B-A3B-Instruct-FP8'),
                Model('Qwen/Qwen3-VL-30B-A3B-Thinking-FP8', 'Qwen/Qwen3-VL-30B-A3B-Thinking-FP8'),
                Model('Qwen/Qwen3-VL-235B-A22B-Instruct', 'Qwen/Qwen3-VL-235B-A22B-Instruct'),
                Model('Qwen/Qwen3-VL-235B-A22B-Thinking', 'Qwen/Qwen3-VL-235B-A22B-Thinking'),
                Model('Qwen/Qwen3-VL-235B-A22B-Instruct-FP8', 'Qwen/Qwen3-VL-235B-A22B-Instruct-FP8'),
                Model('Qwen/Qwen3-VL-235B-A22B-Thinking-FP8', 'Qwen/Qwen3-VL-235B-A22B-Thinking-FP8'),
            ]),
        ],
        TemplateType.qwen3_vl,
        get_model_tokenizer_qwen3_moe_vl,
        model_arch=ModelArch.qwen3_vl,
        architectures=['Qwen3VLMoeForConditionalGeneration'],
        requires=['transformers>=4.57', 'qwen_vl_utils>=0.0.14', 'decord'],
        tags=['vision', 'video']))

register_model(
    ModelMeta(
        MLLMModelType.mimo_vl, [
            ModelGroup([
                Model('XiaomiMiMo/MiMo-VL-7B-SFT', 'XiaomiMiMo/MiMo-VL-7B-SFT'),
                Model('XiaomiMiMo/MiMo-VL-7B-RL', 'XiaomiMiMo/MiMo-VL-7B-RL'),
            ])
        ],
        TemplateType.mimo_vl,
        get_model_tokenizer_qwen2_5_vl,
        model_arch=ModelArch.qwen2_vl,
        architectures=['Qwen2_5_VLForConditionalGeneration'],
        requires=['transformers>=4.49', 'qwen_vl_utils>=0.0.6', 'decord'],
        tags=['vision', 'video']))


def get_model_tokenizer_qwen2_5_omni(model_dir, *args, **kwargs):
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, Qwen2_5OmniConfig
    from qwen_omni_utils import vision_process
    kwargs['automodel_class'] = kwargs['automodel_class'] or Qwen2_5OmniForConditionalGeneration
    processor = Qwen2_5OmniProcessor.from_pretrained(model_dir, trust_remote_code=True)
    kwargs['tokenizer'] = processor.tokenizer
    kwargs['model_config'] = Qwen2_5OmniConfig.from_pretrained(model_dir, trust_remote_code=True)
    global_vars = patch_qwen_vl_utils(vision_process)
    processor.global_vars = global_vars
    enable_audio_output = get_env_args('ENABLE_AUDIO_OUTPUT', bool, None)
    if enable_audio_output is not None:
        kwargs['model_config'].enable_audio_output = enable_audio_output
    model, _ = get_model_tokenizer_with_flash_attn(model_dir, *args, **kwargs)
    if model:
        base_model = model.model if 'AWQ' in model.__class__.__name__ else model
        use_submodel_func(base_model, 'thinker')
        base_model.config.keys_to_ignore_at_inference += ['hidden_states', 'attention_mask']
        base_model.config.talker_config.pad_token_id = None
        patch_get_input_embeddings(base_model.thinker.visual, 'patch_embed')
    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.qwen2_5_omni,
        [
            ModelGroup([
                Model('Qwen/Qwen2.5-Omni-3B', 'Qwen/Qwen2.5-Omni-3B'),
                Model('Qwen/Qwen2.5-Omni-7B', 'Qwen/Qwen2.5-Omni-7B'),
            ]),
        ],
        TemplateType.qwen2_5_omni,
        get_model_tokenizer_qwen2_5_omni,
        model_arch=ModelArch.qwen2_5_omni,
        architectures=['Qwen2_5OmniModel', 'Qwen2_5OmniForConditionalGeneration'],
        requires=['transformers>=4.50', 'soundfile', 'qwen_omni_utils', 'decord'],
        tags=['vision', 'video', 'audio'],
        additional_saved_files=['spk_dict.pt'],
        ignore_patterns=[],
    ))


def _compat_qwen3_omni_mixed_data(model, processor):
    if not is_deepspeed_enabled() or hasattr(model, 'origin_forward'):
        return
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (Qwen3OmniMoeThinkerCausalLMOutputWithPast,
                                                                            can_return_tuple, load_balancing_loss_func)

    @can_return_tuple
    def forward(
        self,
        input_ids=None,
        input_features=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
        feature_attention_mask=None,
        audio_feature_lengths=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        rope_deltas=None,
        labels=None,
        use_cache=None,
        output_router_logits: Optional[bool] = None,
        use_audio_in_video=None,
        cache_position=None,
        video_second_per_grid=None,
        **kwargs,
    ) -> Union[tuple, Qwen3OmniMoeThinkerCausalLMOutputWithPast]:
        if not self.training and not is_deepspeed_enabled():
            return self.origin_forward(
                input_ids=input_ids,
                input_features=input_features,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                feature_attention_mask=feature_attention_mask,
                audio_feature_lengths=audio_feature_lengths,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                rope_deltas=rope_deltas,
                labels=labels,
                use_cache=use_cache,
                output_router_logits=output_router_logits,
                use_audio_in_video=use_audio_in_video,
                cache_position=cache_position,
                video_second_per_grid=video_second_per_grid,
                **kwargs,
            )
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.text_config.output_router_logits)

        inputs_embeds, visual_pos_masks, visual_embeds_multiscale = _forward_qwen3_vl_or_qwen3_omni(
            self, processor, input_ids, inputs_embeds, pixel_values, pixel_values_videos, image_grid_thw,
            video_grid_thw)

        if input_features is None:
            input_features = input_ids.new_zeros([1, 128, 128], dtype=self.audio_tower.dtype)
            feature_attention_mask = input_ids.new_ones([1, 128], dtype=torch.bool)
            audio_embeds = self.get_audio_features(input_features, feature_attention_mask)
            inputs_embeds = inputs_embeds + audio_embeds.mean() * 0.
        else:
            audio_embeds = self.get_audio_features(input_features, feature_attention_mask)
            audio_mask = (input_ids == self.config.audio_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            audio_embeds = audio_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_embeds)

        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        else:
            audio_feature_lengths = None

        if attention_mask is not None and position_ids is None:
            if (cache_position is None or (cache_position is not None and cache_position[0] == 0)
                    or self.rope_deltas is None):
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                    use_audio_in_video,
                    audio_feature_lengths,
                    video_second_per_grid,
                )
                rope_deltas = rope_deltas - delta0
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length = input_ids.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            cache_position=cache_position,
            deepstack_visual_embeds=visual_embeds_multiscale,
            visual_pos_masks=visual_pos_masks,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.get_text_config().vocab_size)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.config.router_aux_loss_coef * aux_loss.to(
                    loss.device)  # make sure to reside in the same device

        return Qwen3OmniMoeThinkerCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            aux_loss=aux_loss,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )

    model.origin_forward = model.forward
    model.forward = MethodType(forward, model)
    _patch_deepstack_process(model.model)


def get_model_tokenizer_qwen3_omni(model_dir, *args, **kwargs):
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor, Qwen3OmniMoeConfig
    from qwen_omni_utils import vision_process
    kwargs['automodel_class'] = kwargs['automodel_class'] or Qwen3OmniMoeForConditionalGeneration
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_dir, trust_remote_code=True)
    kwargs['tokenizer'] = processor.tokenizer
    kwargs['model_config'] = Qwen3OmniMoeConfig.from_pretrained(model_dir, trust_remote_code=True)
    kwargs['model_config'].thinker_config.audio_token_id = processor.tokenizer.encode('<|audio_pad|>')[0]
    global_vars = patch_qwen_vl_utils(vision_process)
    processor.global_vars = global_vars
    enable_audio_output = get_env_args('ENABLE_AUDIO_OUTPUT', bool, None)
    if enable_audio_output is not None:
        kwargs['model_config'].enable_audio_output = enable_audio_output
    model, _ = get_model_tokenizer_with_flash_attn(model_dir, *args, **kwargs)
    if model:
        _compat_qwen3_omni_mixed_data(model.thinker, processor)
        base_model = model.model if 'AWQ' in model.__class__.__name__ else model
        use_submodel_func(base_model, 'thinker')
        base_model.config.keys_to_ignore_at_inference += ['hidden_states', 'attention_mask']
        base_model.config.talker_config.pad_token_id = None
        patch_get_input_embeddings(base_model.thinker.visual, 'patch_embed')
        patch_get_input_embeddings(base_model.thinker.audio_tower, 'conv_out')
    return model, processor


register_model(
    ModelMeta(
        MLLMModelType.qwen3_omni,
        [
            ModelGroup([
                Model('Qwen/Qwen3-Omni-30B-A3B-Instruct', 'Qwen/Qwen3-Omni-30B-A3B-Instruct'),
                Model('Qwen/Qwen3-Omni-30B-A3B-Thinking', 'Qwen/Qwen3-Omni-30B-A3B-Thinking'),
                Model('Qwen/Qwen3-Omni-30B-A3B-Captioner', 'Qwen/Qwen3-Omni-30B-A3B-Captioner'),
            ])
        ],
        TemplateType.qwen3_omni,
        get_model_tokenizer_qwen3_omni,
        model_arch=ModelArch.qwen3_omni,
        architectures=['Qwen3OmniMoeForConditionalGeneration'],
        requires=['transformers>=4.57.dev0', 'soundfile', 'decord', 'qwen_omni_utils'],
        tags=['vision', 'video', 'audio'],
    ))


def get_model_tokenizer_midashenglm(*args, **kwargs):
    model, tokenizer = get_model_tokenizer_multimodal(*args, **kwargs)
    if model is not None:
        model.audio_encoder.float()
        patch_output_clone(model.decoder.model.embed_tokens)
    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.midashenglm,
        [ModelGroup([
            Model('mispeech/midashenglm-7b', 'mispeech/midashenglm-7b'),
        ])],
        TemplateType.midashenglm,
        get_model_tokenizer_midashenglm,
        model_arch=ModelArch.midashenglm,
        architectures=['MiDashengLMModel'],
        requires=['transformers>=4.52', 'soundfile'],
        tags=['audio'],
    ))


def get_model_tokenizer_qwen2_audio(*args, **kwargs):
    from transformers import Qwen2AudioForConditionalGeneration
    kwargs['automodel_class'] = kwargs['automodel_class'] or Qwen2AudioForConditionalGeneration
    return get_model_tokenizer_multimodal(*args, **kwargs)


register_model(
    ModelMeta(
        MLLMModelType.qwen2_audio,
        [
            ModelGroup([
                Model('Qwen/Qwen2-Audio-7B-Instruct', 'Qwen/Qwen2-Audio-7B-Instruct'),
                Model('Qwen/Qwen2-Audio-7B', 'Qwen/Qwen2-Audio-7B'),
            ]),
        ],
        TemplateType.qwen2_audio,
        get_model_tokenizer_qwen2_audio,
        model_arch=ModelArch.qwen2_audio,
        architectures=['Qwen2AudioForConditionalGeneration'],
        requires=['transformers>=4.45,<4.49', 'librosa'],
        tags=['audio'],
    ))

register_model(
    ModelMeta(
        LLMModelType.marco_o1, [ModelGroup([Model('AIDC-AI/Marco-o1', 'AIDC-AI/Marco-o1')])],
        TemplateType.marco_o1,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.llama,
        architectures=['Qwen2ForCausalLM'],
        requires=['transformers>=4.37']))

register_model(
    ModelMeta(
        LLMModelType.qwq_preview, [ModelGroup([Model('Qwen/QwQ-32B-Preview', 'Qwen/QwQ-32B-Preview')])],
        TemplateType.qwq_preview,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.llama,
        architectures=['Qwen2ForCausalLM'],
        requires=['transformers>=4.37']))

register_model(
    ModelMeta(
        LLMModelType.qwq,
        [ModelGroup([
            Model('Qwen/QwQ-32B', 'Qwen/QwQ-32B'),
            Model('Qwen/QwQ-32B-AWQ', 'Qwen/QwQ-32B-AWQ'),
        ])],
        TemplateType.qwq,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.llama,
        architectures=['Qwen2ForCausalLM'],
        requires=['transformers>=4.37']))


def get_model_tokenizer_ovis(*args, **kwargs):
    kwargs['attn_impl_keys'] = ['llm_attn_implementation']
    model, tokenizer = get_model_tokenizer_with_flash_attn(*args, **kwargs)
    if model is not None:
        model.visual_tokenizer.to(model.dtype)
        model.vte.to(model.dtype)

        model.generation_config.cache_implementation = None
        func_list = ['generate', 'forward', 'get_input_embeddings']
        use_submodel_func(model, 'llm', func_list)
        embedding = model.get_input_embeddings()
        patch_output_clone(embedding)
        if hasattr(model.visual_tokenizer, 'backbone'):
            backbone = model.visual_tokenizer.backbone
            if hasattr(backbone, 'vision_model'):
                patch_get_input_embeddings(model.visual_tokenizer, 'backbone.vision_model.embeddings')
            elif hasattr(backbone, 'preprocessor'):
                patch_get_input_embeddings(model.visual_tokenizer, 'backbone.preprocessor.patchifier')
    try:
        # fix device_map
        from transformers.cache_utils import HybridCache

        def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, *args,
                   **kwargs) -> Tuple[torch.Tensor]:
            self.key_cache[layer_idx] = self.key_cache[layer_idx].to(key_states.device)
            self.value_cache[layer_idx] = self.value_cache[layer_idx].to(value_states.device)
            return self._update_origin(key_states, value_states, layer_idx, *args, **kwargs)

        if not hasattr(HybridCache, '_update_origin'):
            HybridCache._update_origin = HybridCache.update
            HybridCache.update = update
    except ImportError:
        pass
    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.ovis1_6,
        [
            ModelGroup([
                Model('AIDC-AI/Ovis1.6-Gemma2-9B', 'AIDC-AI/Ovis1.6-Gemma2-9B'),
                Model('AIDC-AI/Ovis1.6-Gemma2-9B-GPTQ-Int4', 'AIDC-AI/Ovis1.6-Gemma2-9B-GPTQ-Int4'),
                Model('AIDC-AI/Ovis1.6-Gemma2-27B', 'AIDC-AI/Ovis1.6-Gemma2-27B'),
            ]),
        ],
        TemplateType.ovis1_6,
        get_model_tokenizer_ovis,
        model_arch=ModelArch.ovis,
        architectures=['Ovis'],
        tags=['vision'],
        requires=['transformers>=4.42'],
    ))

register_model(
    ModelMeta(
        MLLMModelType.ovis1_6_llama3,
        [
            ModelGroup([
                Model('AIDC-AI/Ovis1.6-Llama3.2-3B', 'AIDC-AI/Ovis1.6-Llama3.2-3B'),
            ]),
        ],
        TemplateType.ovis1_6_llama3,
        get_model_tokenizer_ovis,
        model_arch=ModelArch.ovis,
        architectures=['Ovis'],
        tags=['vision'],
    ))

register_model(
    ModelMeta(
        MLLMModelType.ovis2,
        [
            ModelGroup([
                Model('AIDC-AI/Ovis2-1B', 'AIDC-AI/Ovis2-1B'),
                Model('AIDC-AI/Ovis2-2B', 'AIDC-AI/Ovis2-2B'),
                Model('AIDC-AI/Ovis2-4B', 'AIDC-AI/Ovis2-4B'),
                Model('AIDC-AI/Ovis2-8B', 'AIDC-AI/Ovis2-8B'),
                Model('AIDC-AI/Ovis2-16B', 'AIDC-AI/Ovis2-16B'),
                Model('AIDC-AI/Ovis2-34B', 'AIDC-AI/Ovis2-34B'),
            ]),
        ],
        TemplateType.ovis2,
        get_model_tokenizer_ovis,
        model_arch=ModelArch.ovis,
        architectures=['Ovis'],
        tags=['vision'],
        requires=['transformers>=4.46.2', 'moviepy<2'],
    ))


def get_model_tokenizer_ovis2_5(*args, **kwargs):
    model, tokenizer = get_model_tokenizer_with_flash_attn(*args, **kwargs)
    if model is not None:
        model.visual_tokenizer.to(model.dtype)
        model.vte.to(model.dtype)

        func_list = ['generate', 'forward', 'get_input_embeddings']
        use_submodel_func(model, 'llm', func_list)
        embedding = model.get_input_embeddings()
        patch_output_clone(embedding)
        patch_get_input_embeddings(model.visual_tokenizer.vit, 'vision_model.embeddings.patch_embedding')

    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.ovis2_5,
        [
            ModelGroup([
                Model('AIDC-AI/Ovis2.5-2B', 'AIDC-AI/Ovis2.5-2B'),
                Model('AIDC-AI/Ovis2.5-9B', 'AIDC-AI/Ovis2.5-9B'),
            ]),
        ],
        TemplateType.ovis2_5,
        get_model_tokenizer_ovis2_5,
        model_arch=ModelArch.ovis2_5,
        architectures=['Ovis'],
        tags=['vision'],
        requires=['transformers>=4.46.2', 'moviepy<2'],
    ))

register_model(
    ModelMeta(
        RMModelType.qwen2_reward,
        [
            ModelGroup([
                Model('Qwen/Qwen2-Math-RM-72B', 'Qwen/Qwen2-Math-RM-72B'),
            ]),
        ],
        TemplateType.qwen,
        get_model_tokenizer_reward_model,
        architectures=['Qwen2ForRewardModel'],
        requires=['transformers>=4.37'],
    ))

register_model(
    ModelMeta(
        RMModelType.qwen2_5_prm,
        [
            ModelGroup([
                Model('Qwen/Qwen2.5-Math-PRM-7B', 'Qwen/Qwen2.5-Math-PRM-7B'),
                Model('Qwen/Qwen2.5-Math-7B-PRM800K', 'Qwen/Qwen2.5-Math-7B-PRM800K'),
                Model('Qwen/Qwen2.5-Math-PRM-72B', 'Qwen/Qwen2.5-Math-PRM-72B'),
            ]),
        ],
        TemplateType.qwen2_5_math_prm,
        get_model_tokenizer_reward_model,
        task_type='prm',
        architectures=['Qwen2ForProcessRewardModel'],
        requires=['transformers>=4.37'],
    ))

register_model(
    ModelMeta(
        RMModelType.qwen2_5_math_reward,
        [
            ModelGroup([
                Model('Qwen/Qwen2.5-Math-RM-72B', 'Qwen/Qwen2.5-Math-RM-72B'),
            ]),
        ],
        TemplateType.qwen2_5_math,
        get_model_tokenizer_reward_model,
        architectures=['Qwen2ForRewardModel'],
        requires=['transformers>=4.37'],
    ))

register_model(
    ModelMeta(
        LLMModelType.qwen3_emb, [
            ModelGroup([
                Model('Qwen/Qwen3-Embedding-0.6B', 'Qwen/Qwen3-Embedding-0.6B'),
                Model('Qwen/Qwen3-Embedding-4B', 'Qwen/Qwen3-Embedding-4B'),
                Model('Qwen/Qwen3-Embedding-8B', 'Qwen/Qwen3-Embedding-8B'),
            ]),
        ],
        TemplateType.qwen3_emb,
        get_model_tokenizer_with_flash_attn,
        additional_saved_files=['config_sentence_transformers.json', '1_Pooling', 'modules.json'],
        architectures=['Qwen3ForCausalLM']))

register_model(
    ModelMeta(
        RerankerModelType.qwen3_reranker, [
            ModelGroup([
                Model('Qwen/Qwen3-Reranker-0.6B', 'Qwen/Qwen3-Reranker-0.6B'),
                Model('Qwen/Qwen3-Reranker-4B', 'Qwen/Qwen3-Reranker-4B'),
                Model('Qwen/Qwen3-Reranker-8B', 'Qwen/Qwen3-Reranker-8B'),
            ]),
        ],
        TemplateType.qwen3_reranker,
        get_model_tokenizer_with_flash_attn,
        architectures=['Qwen3ForCausalLM'],
        task_type='reranker'))

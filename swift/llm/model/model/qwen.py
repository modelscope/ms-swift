# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
from transformers import AutoConfig, BitsAndBytesConfig, PreTrainedTokenizerBase
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.auto.tokenization_auto import get_tokenizer_config

from swift.llm import TemplateType
from swift.utils import get_dist_setting, get_logger
from ..constant import LLMModelType, MLLMModelType
from ..model_arch import ModelArch
from ..patcher import patch_fixed_device, patch_output_clone, patch_output_to_input_device
from ..register import (Model, ModelGroup, ModelMeta, get_model_tokenizer_multimodal,
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
    if model_config.torch_dtype is not None:
        k_true = dtype_mapping[model_config.torch_dtype]
        for k in dtype_mapping.values():
            setattr(model_config, k, k == k_true)

    quantization_config = model_kwargs.get('quantization_config')
    if not isinstance(quantization_config, BitsAndBytesConfig):
        # not bnb quant
        model_config.torch_dtype = None
    use_flash_attn = AttnImpl.to_use_flash_attn(kwargs.pop('attn_impl', None), 'auto')
    model_config.use_flash_attn = use_flash_attn
    kwargs['model_config'] = model_config
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)
    try:
        # fix mp+ddp bug
        model.transformer.registered_causal_mask = model.transformer.registered_causal_mask.cuda()
        logger.info('registered_causal_mask to cuda')
    except AttributeError:
        pass
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.eod_id
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
    n_gpu = torch.cuda.device_count()
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
                tags=['coding'])
        ],
        TemplateType.qwen2_5,
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
        model_arch=ModelArch.llama))


def get_model_tokenizer_qwen2_vl(model_dir: str,
                                 model_info: ModelInfo,
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

    from transformers import Qwen2VLForConditionalGeneration
    kwargs['automodel_class'] = kwargs['automodel_class'] or Qwen2VLForConditionalGeneration
    model, tokenizer = get_model_tokenizer_multimodal(model_dir, model_info, model_kwargs, load_model, **kwargs)
    if model is not None and hasattr(model.model, 'embed_tokens'):
        patch_output_clone(model.model.embed_tokens)
        patch_output_to_input_device(model.model.embed_tokens)
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
        ],
        TemplateType.qwen2_vl,
        get_model_tokenizer_qwen2_vl,
        model_arch=ModelArch.qwen2_vl,
        architectures=['Qwen2VLForConditionalGeneration'],
        requires=['transformers>=4.45', 'qwen_vl_utils', 'pyav'],
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
        requires=['transformers>=4.45', 'qwen_vl_utils', 'pyav'],
        tags=['vision', 'video']))


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
        requires=['transformers>=4.45', 'librosa'],
        tags=['audio'],
    ))

register_model(
    ModelMeta(
        LLMModelType.marco_o1, [ModelGroup([Model('AIDC-AI/Marco-o1', 'AIDC-AI/Marco-o1')])],
        LLMModelType.marco_o1,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.llama,
        architectures=['Qwen2ForCausalLM'],
        requires=['transformers>=4.37']))

register_model(
    ModelMeta(
        LLMModelType.qwq, [ModelGroup([Model('Qwen/QwQ-32B-Preview', 'Qwen/QwQ-32B-Preview')])],
        LLMModelType.qwq,
        get_model_tokenizer_with_flash_attn,
        model_arch=ModelArch.llama,
        architectures=['Qwen2ForCausalLM'],
        requires=['transformers>=4.37']))


def get_model_tokenizer_ovis(*args, **kwargs):
    model, tokenizer = get_model_tokenizer_with_flash_attn(*args, **kwargs)
    if model is not None:
        model.generation_config.cache_implementation = None
        func_list = ['generate', 'forward', 'get_input_embeddings']
        use_submodel_func(model, 'llm', func_list)
        embedding = model.get_input_embeddings()
        patch_output_clone(embedding)
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
            ]),
        ],
        TemplateType.ovis1_6,
        get_model_tokenizer_ovis,
        model_arch=ModelArch.ovis1_6,
        architectures=['Ovis'],
        tags=['vision'],
        requires=['transformers>=4.42'],
    ))

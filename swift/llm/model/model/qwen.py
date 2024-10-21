from typing import Any, Dict, List, Optional, Type

import torch
from transformers import AutoConfig, BitsAndBytesConfig, PreTrainedTokenizerBase
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.auto.tokenization_auto import get_tokenizer_config

from swift.llm import TemplateType
from swift.utils import get_dist_setting, get_logger
from ..constant import LLMModelType, MLLMModelType
from ..patcher import patch_fixed_device, patch_output_clone, patch_output_to_input_device
from ..register import (Model, ModelGroup, TemplateGroup, get_model_tokenizer_from_local, get_model_tokenizer_multimodal,
                       get_model_tokenizer_with_flash_attn, register_model)
from ..utils import AttnImpl

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
            setattr(model_config, k, k == k_true)

    quantization_config = model_kwargs.get('quantization_config')
    if not isinstance(quantization_config, BitsAndBytesConfig):
        # not bnb quant
        torch_dtype = None
    use_flash_attn = AttnImpl.to_use_flash_attn(kwargs.pop('attn_impl', None), 'auto')
    model_config.use_flash_attn = use_flash_attn
    model, tokenizer = get_model_tokenizer_from_local(
        model_dir, torch_dtype, model_kwargs, load_model, model_config=model_config, **kwargs)
    try:
        # fix mp+ddp bug
        model.transformer.registered_causal_mask = model.transformer.registered_causal_mask.cuda()
        logger.info('registered_causal_mask to cuda')
    except AttributeError:
        pass
    return model, tokenizer


register_model(
    LLMModelType.qwen,
    'QWenLMHeadModel',
    [
        # qwen
        ModelGroup([
            # base
            Model('qwen/Qwen-1_8B', 'Qwen/Qwen-1_8B'),
            Model('qwen/Qwen-7B', 'Qwen/Qwen-7B'),
            Model('qwen/Qwen-14B', 'Qwen/Qwen-14B'),
            Model('qwen/Qwen-72B', 'Qwen/Qwen-72B'),
            # chat
            Model('qwen/Qwen-1_8B-Chat', 'Qwen/Qwen-1_8B-Chat'),
            Model('qwen/Qwen-7B-Chat', 'Qwen/Qwen-7B-Chat'),
            Model('qwen/Qwen-14B-Chat', 'Qwen/Qwen-14B-Chat'),
            Model('qwen/Qwen-72B-Chat', 'Qwen/Qwen-72B-Chat'),
            # gptq-int4
            Model('qwen/Qwen-1_8B-Chat-Int4', 'Qwen/Qwen-1_8B-Chat-Int4'),
            Model('qwen/Qwen-7B-Chat-Int4', 'Qwen/Qwen-7B-Chat-Int4'),
            Model('qwen/Qwen-14B-Chat-Int4', 'Qwen/Qwen-14B-Chat-Int4'),
            Model('qwen/Qwen-72B-Chat-Int4', 'Qwen/Qwen-72B-Chat-Int4'),
            # gptq-int8
            Model('qwen/Qwen-1_8B-Chat-Int8', 'Qwen/Qwen-1_8B-Chat-Int8'),
            Model('qwen/Qwen-7B-Chat-Int8', 'Qwen/Qwen-7B-Chat-Int8'),
            Model('qwen/Qwen-14B-Chat-Int8', 'Qwen/Qwen-14B-Chat-Int8'),
            Model('qwen/Qwen-72B-Chat-Int8', 'Qwen/Qwen-72B-Chat-Int8'),
        ]),
        # tongyi-finance
        ModelGroup([
            Model('TongyiFinance/Tongyi-Finance-14B'),
            Model('TongyiFinance/Tongyi-Finance-14B-Chat', 'jxy/Tongyi-Finance-14B-Chat'),
            Model('TongyiFinance/Tongyi-Finance-14B-Chat-Int4', 'jxy/Tongyi-Finance-14B-Chat-Int4'),
        ],
                   tags=['financial']),
    ],
    TemplateGroup(TemplateType.qwen),
    get_model_tokenizer_qwen,
    support_flash_attn=True,
    support_vllm=True,
    support_lmdeploy=True)

register_model(
    LLMModelType.codefuse_qwen,
    'QWenLMHeadModel',
    [ModelGroup([
        Model('codefuse-ai/CodeFuse-QWen-14B', 'codefuse-ai/CodeFuse-QWen-14B'),
    ], tags=['coding'])],
    TemplateGroup(TemplateType.codefuse),
    get_model_tokenizer_qwen,
    support_flash_attn=True,
    support_vllm=True,
    support_lmdeploy=True)

register_model(
    LLMModelType.modelscope_agent,
    'QWenLMHeadModel', [ModelGroup([
        Model('iic/ModelScope-Agent-7B'),
        Model('iic/ModelScope-Agent-14B'),
    ])],
    TemplateGroup(TemplateType.modelscope_agent),
    get_model_tokenizer_qwen,
    support_flash_attn=True,
    support_vllm=False)


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
                                   torch_dtype: torch.dtype,
                                   model_kwargs: Dict[str, Any],
                                   load_model: bool = True,
                                   **kwargs):
    tokenizer_config = get_tokenizer_config(model_dir)
    class_ref = tokenizer_config['auto_map']['AutoTokenizer'][0]
    tokenizer_cls: Type[PreTrainedTokenizerBase] = get_class_from_dynamic_module(class_ref, model_dir)
    tokenizer_cls._auto_class = 'AutoTokenizer'
    tokenizer_cls.AUDIO_ST = ()  # fix no attr `self.AUDIO_ST` bug
    tokenizer_cls._old_decode = tokenizer_cls._decode
    tokenizer_cls._decode = _qwen_vl_audio_decode
    kwargs['tokenizer'] = tokenizer_cls.from_pretrained(model_dir, trust_remote_code=True)
    model, tokenizer = get_model_tokenizer_qwen(model_dir, torch_dtype, model_kwargs, load_model, **kwargs)
    if model is not None:
        fix_qwen_inplace_bug(model)

    return model, tokenizer


register_model(
    MLLMModelType.qwen_audio,
    'QWenLMHeadModel', [
        ModelGroup([
            Model('qwen/Qwen-Audio', 'Qwen/Qwen-Audio'),
            Model('qwen/Qwen-Audio-Chat', 'Qwen/Qwen-Audio-Chat'),
        ],
                   tags=['audio'])
    ],
    TemplateGroup(TemplateType.qwen_audio, TemplateType.qwen_audio_generation),
    get_model_tokenizer_qwen_audio,
    support_flash_attn=True,
    additional_saved_files=['mel_filters.npz'])


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
                                torch_dtype: torch.dtype,
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
    model, tokenizer = get_model_tokenizer_qwen(model_dir, torch_dtype, model_kwargs, load_model, **kwargs)
    if model is not None:
        fix_qwen_inplace_bug(model)
        # fix device_map is 4
        if n_gpu // local_world_size >= 4:
            model.transformer.visual.proj.data = model.transformer.visual.proj.to(
                model.transformer.visual.ln_post.bias.device)
        # fix images cuda:1 bug
        patch_fixed_device(model.transformer.visual, 0)
    return model, tokenizer


register_model(
    MLLMModelType.qwen_vl,
    'QWenLMHeadModel', [
        ModelGroup([
            Model('qwen/Qwen-VL', 'Qwen/Qwen-VL'),
            Model('qwen/Qwen-VL-Chat', 'Qwen/Qwen-VL-Chat'),
            Model('qwen/Qwen-VL-Chat-Int4', 'Qwen/Qwen-VL-Chat-Int4'),
        ],
                   tags=['vision'])
    ],
    TemplateGroup(TemplateType.qwen_vl, TemplateType.qwen_vl_generation),
    get_model_tokenizer_qwen_vl,
    support_flash_attn=True,
    support_vllm=True,
    support_lmdeploy=True,
    additional_saved_files=['SimSun.ttf'])

register_model(
    LLMModelType.qwen2,
    'Qwen2ForCausalLM',
    [
        # qwen1.5
        ModelGroup([
            # base
            Model('qwen/Qwen1.5-0.5B', 'Qwen/Qwen1.5-0.5B'),
            Model('qwen/Qwen1.5-1.8B', 'Qwen/Qwen1.5-1.8B'),
            Model('qwen/Qwen1.5-4B', 'Qwen/Qwen1.5-4B'),
            Model('qwen/Qwen1.5-7B', 'Qwen/Qwen1.5-7B'),
            Model('qwen/Qwen1.5-14B', 'Qwen/Qwen1.5-14B'),
            Model('qwen/Qwen1.5-32B', 'Qwen/Qwen1.5-32B'),
            Model('qwen/Qwen1.5-72B', 'Qwen/Qwen1.5-72B'),
            Model('qwen/Qwen1.5-110B', 'Qwen/Qwen1.5-110B'),
            # chat
            Model('qwen/Qwen1.5-0.5B-Chat', 'Qwen/Qwen1.5-0.5B-Chat'),
            Model('qwen/Qwen1.5-1.8B-Chat', 'Qwen/Qwen1.5-1.8B-Chat'),
            Model('qwen/Qwen1.5-4B-Chat', 'Qwen/Qwen1.5-4B-Chat'),
            Model('qwen/Qwen1.5-7B-Chat', 'Qwen/Qwen1.5-7B-Chat'),
            Model('qwen/Qwen1.5-14B-Chat', 'Qwen/Qwen1.5-14B-Chat'),
            Model('qwen/Qwen1.5-32B-Chat', 'Qwen/Qwen1.5-32B-Chat'),
            Model('qwen/Qwen1.5-72B-Chat', 'Qwen/Qwen1.5-72B-Chat'),
            Model('qwen/Qwen1.5-110B-Chat', 'Qwen/Qwen1.5-110B-Chat'),
            # gptq-int4
            Model('qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4', 'Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4'),
            Model('qwen/Qwen1.5-1.8B-Chat-GPTQ-Int4', 'Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int4'),
            Model('qwen/Qwen1.5-4B-Chat-GPTQ-Int4', 'Qwen/Qwen1.5-4B-Chat-GPTQ-Int4'),
            Model('qwen/Qwen1.5-7B-Chat-GPTQ-Int4', 'Qwen/Qwen1.5-7B-Chat-GPTQ-Int4'),
            Model('qwen/Qwen1.5-14B-Chat-GPTQ-Int4', 'Qwen/Qwen1.5-14B-Chat-GPTQ-Int4'),
            Model('qwen/Qwen1.5-32B-Chat-GPTQ-Int4', 'Qwen/Qwen1.5-32B-Chat-GPTQ-Int4'),
            Model('qwen/Qwen1.5-72B-Chat-GPTQ-Int4', 'Qwen/Qwen1.5-72B-Chat-GPTQ-Int4'),
            Model('qwen/Qwen1.5-110B-Chat-GPTQ-Int4', 'Qwen/Qwen1.5-110B-Chat-GPTQ-Int4'),
            # gptq-int8
            Model('qwen/Qwen1.5-0.5B-Chat-GPTQ-Int8', 'Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int8'),
            Model('qwen/Qwen1.5-1.8B-Chat-GPTQ-Int8', 'Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int8'),
            Model('qwen/Qwen1.5-4B-Chat-GPTQ-Int8', 'Qwen/Qwen1.5-4B-Chat-GPTQ-Int8'),
            Model('qwen/Qwen1.5-7B-Chat-GPTQ-Int8', 'Qwen/Qwen1.5-7B-Chat-GPTQ-Int8'),
            Model('qwen/Qwen1.5-14B-Chat-GPTQ-Int8', 'Qwen/Qwen1.5-14B-Chat-GPTQ-Int8'),
            Model('qwen/Qwen1.5-72B-Chat-GPTQ-Int8', 'Qwen/Qwen1.5-72B-Chat-GPTQ-Int8'),
            # awq-int4
            Model('qwen/Qwen1.5-0.5B-Chat-AWQ', 'Qwen/Qwen1.5-0.5B-Chat-AWQ'),
            Model('qwen/Qwen1.5-1.8B-Chat-AWQ', 'Qwen/Qwen1.5-1.8B-Chat-AWQ'),
            Model('qwen/Qwen1.5-4B-Chat-AWQ', 'Qwen/Qwen1.5-4B-Chat-AWQ'),
            Model('qwen/Qwen1.5-7B-Chat-AWQ', 'Qwen/Qwen1.5-7B-Chat-AWQ'),
            Model('qwen/Qwen1.5-14B-Chat-AWQ', 'Qwen/Qwen1.5-14B-Chat-AWQ'),
            Model('qwen/Qwen1.5-32B-Chat-AWQ', 'Qwen/Qwen1.5-32B-Chat-AWQ'),
            Model('qwen/Qwen1.5-72B-Chat-AWQ', 'Qwen/Qwen1.5-72B-Chat-AWQ'),
            Model('qwen/Qwen1.5-110B-Chat-AWQ', 'Qwen/Qwen1.5-110B-Chat-AWQ'),
        ]),
        # code-qwen1.5
        ModelGroup([
            Model('qwen/CodeQwen1.5-7B', 'Qwen/CodeQwen1.5-7B'),
            Model('qwen/CodeQwen1.5-7B-Chat', 'Qwen/CodeQwen1.5-7B-Chat'),
            Model('qwen/CodeQwen1.5-7B-Chat-AWQ', 'Qwen/CodeQwen1.5-7B-Chat-AWQ'),
        ],
                   tags=['coding']),
        # qwen2
        ModelGroup([
            # base
            Model('qwen/Qwen2-0.5B', 'Qwen/Qwen2-0.5B'),
            Model('qwen/Qwen2-1.5B', 'Qwen/Qwen2-1.5B'),
            Model('qwen/Qwen2-7B', 'Qwen/Qwen2-7B'),
            Model('qwen/Qwen2-72B', 'Qwen/Qwen2-72B'),
            # instruct
            Model('qwen/Qwen2-0.5B-Instruct', 'Qwen/Qwen2-0.5B-Instruct'),
            Model('qwen/Qwen2-1.5B-Instruct', 'Qwen/Qwen2-1.5B-Instruct'),
            Model('qwen/Qwen2-7B-Instruct', 'Qwen/Qwen2-7B-Instruct'),
            Model('qwen/Qwen2-72B-Instruct', 'Qwen/Qwen2-72B-Instruct'),
            # gptq-int4
            Model('qwen/Qwen2-0.5B-Instruct-GPTQ-Int4', 'Qwen/Qwen2-0.5B-Instruct-GPTQ-Int4'),
            Model('qwen/Qwen2-1.5B-Instruct-GPTQ-Int4', 'Qwen/Qwen2-1.5B-Instruct-GPTQ-Int4'),
            Model('qwen/Qwen2-7B-Instruct-GPTQ-Int4', 'Qwen/Qwen2-7B-Instruct-GPTQ-Int4'),
            Model('qwen/Qwen2-72B-Instruct-GPTQ-Int4', 'Qwen/Qwen2-72B-Instruct-GPTQ-Int4'),
            # gptq-int8
            Model('qwen/Qwen2-0.5B-Instruct-GPTQ-Int8', 'Qwen/Qwen2-0.5B-Instruct-GPTQ-Int8'),
            Model('qwen/Qwen2-1.5B-Instruct-GPTQ-Int8', 'Qwen/Qwen2-1.5B-Instruct-GPTQ-Int8'),
            Model('qwen/Qwen2-7B-Instruct-GPTQ-Int8', 'Qwen/Qwen2-7B-Instruct-GPTQ-Int8'),
            Model('qwen/Qwen2-72B-Instruct-GPTQ-Int8', 'Qwen/Qwen2-72B-Instruct-GPTQ-Int8'),
            # awq-int4
            Model('qwen/Qwen2-0.5B-Instruct-AWQ', 'Qwen/Qwen2-0.5B-Instruct-AWQ'),
            Model('qwen/Qwen2-1.5B-Instruct-AWQ', 'Qwen/Qwen2-1.5B-Instruct-AWQ'),
            Model('qwen/Qwen2-7B-Instruct-AWQ', 'Qwen/Qwen2-7B-Instruct-AWQ'),
            Model('qwen/Qwen2-72B-Instruct-AWQ', 'Qwen/Qwen2-72B-Instruct-AWQ'),
        ]),
        # qwen2-math
        ModelGroup(
            [
                # base
                Model('qwen/Qwen2-Math-1.5B', 'Qwen/Qwen2-Math-1.5B'),
                Model('qwen/Qwen2-Math-7B', 'Qwen/Qwen2-Math-7B'),
                Model('qwen/Qwen2-Math-72B', 'Qwen/Qwen2-Math-72B'),
                # instruct
                Model('qwen/Qwen2-Math-1.5B-Instruct', 'Qwen/Qwen2-Math-1.5B-Instruct'),
                Model('qwen/Qwen2-Math-7B-Instruct', 'Qwen/Qwen2-Math-7B-Instruct'),
                Model('qwen/Qwen2-Math-72B-Instruct', 'Qwen/Qwen2-Math-72B-Instruct'),
            ],
            tags=['math']),
    ],
    TemplateGroup(TemplateType.qwen),
    get_model_tokenizer_with_flash_attn,
    requires=['transformers>=4.37'],
    support_flash_attn=True,
    support_vllm=True,
    support_lmdeploy=True,
    support_megatron=True)

register_model(
    LLMModelType.qwen2_moe,
    'Qwen2MoeForCausalLM',
    [
        # qwen1.5-moe
        ModelGroup([
            Model('qwen/Qwen1.5-MoE-A2.7B', 'Qwen/Qwen1.5-MoE-A2.7B'),
            Model('qwen/Qwen1.5-MoE-A2.7B-Chat', 'Qwen/Qwen1.5-MoE-A2.7B-Chat'),
            Model('qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4', 'Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4'),
        ]),
        ModelGroup([
            Model('qwen/Qwen2-57B-A14B', 'Qwen/Qwen2-57B-A14B'),
            Model('Qwen/Qwen2-57B-A14B-Instruct', 'Qwen/Qwen2-57B-A14B-Instruct'),
            Model('qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4', 'Qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4'),
        ])
    ],
    TemplateGroup(TemplateType.qwen),
    get_model_tokenizer_with_flash_attn,
    requires=['transformers>=4.40'],
    is_moe=True,
    support_flash_attn=True,
    support_vllm=True)

register_model(
    LLMModelType.qwen2_5,
    'Qwen2ForCausalLM',
    [
        # qwen2.5
        ModelGroup([
            # base
            Model('qwen/Qwen2.5-0.5B', 'Qwen/Qwen2.5-0.5B'),
            Model('qwen/Qwen2.5-1.5B', 'Qwen/Qwen2.5-1.5B'),
            Model('qwen/Qwen2.5-3B', 'Qwen/Qwen2.5-3B'),
            Model('qwen/Qwen2.5-7B', 'Qwen/Qwen2.5-7B'),
            Model('qwen/Qwen2.5-14B', 'Qwen/Qwen2.5-14B'),
            Model('qwen/Qwen2.5-32B', 'Qwen/Qwen2.5-32B'),
            Model('qwen/Qwen2.5-72B', 'Qwen/Qwen2.5-72B'),
            # instruct
            Model('qwen/Qwen2.5-0.5B-Instruct', 'Qwen/Qwen2.5-0.5B-Instruct'),
            Model('qwen/Qwen2.5-1.5B-Instruct', 'Qwen/Qwen2.5-1.5B-Instruct'),
            Model('qwen/Qwen2.5-3B-Instruct', 'Qwen/Qwen2.5-3B-Instruct'),
            Model('qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen2.5-7B-Instruct'),
            Model('qwen/Qwen2.5-14B-Instruct', 'Qwen/Qwen2.5-14B-Instruct'),
            Model('qwen/Qwen2.5-32B-Instruct', 'Qwen/Qwen2.5-32B-Instruct'),
            Model('qwen/Qwen2.5-72B-Instruct', 'Qwen/Qwen2.5-72B-Instruct'),
            # gptq-int4
            Model('qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4', 'Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4'),
            Model('qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4', 'Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4'),
            Model('qwen/Qwen2.5-3B-Instruct-GPTQ-Int4', 'Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4'),
            Model('qwen/Qwen2.5-7B-Instruct-GPTQ-Int4', 'Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4'),
            Model('qwen/Qwen2.5-14B-Instruct-GPTQ-Int4', 'Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4'),
            Model('qwen/Qwen2.5-32B-Instruct-GPTQ-Int4', 'Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4'),
            Model('qwen/Qwen2.5-72B-Instruct-GPTQ-Int4', 'Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4'),
            # gptq-int8
            Model('qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8', 'Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8'),
            Model('qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8', 'Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8'),
            Model('qwen/Qwen2.5-3B-Instruct-GPTQ-Int8', 'Qwen/Qwen2.5-3B-Instruct-GPTQ-Int8'),
            Model('qwen/Qwen2.5-7B-Instruct-GPTQ-Int8', 'Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8'),
            Model('qwen/Qwen2.5-14B-Instruct-GPTQ-Int8', 'Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8'),
            Model('qwen/Qwen2.5-32B-Instruct-GPTQ-Int8', 'Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8'),
            Model('qwen/Qwen2.5-72B-Instruct-GPTQ-Int8', 'Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8'),
            # awq-int4
            Model('qwen/Qwen2.5-0.5B-Instruct-AWQ', 'Qwen/Qwen2.5-0.5B-Instruct-AWQ'),
            Model('qwen/Qwen2.5-1.5B-Instruct-AWQ', 'Qwen/Qwen2.5-1.5B-Instruct-AWQ'),
            Model('qwen/Qwen2.5-3B-Instruct-AWQ', 'Qwen/Qwen2.5-3B-Instruct-AWQ'),
            Model('qwen/Qwen2.5-7B-Instruct-AWQ', 'Qwen/Qwen2.5-7B-Instruct-AWQ'),
            Model('qwen/Qwen2.5-14B-Instruct-AWQ', 'Qwen/Qwen2.5-14B-Instruct-AWQ'),
            Model('qwen/Qwen2.5-32B-Instruct-AWQ', 'Qwen/Qwen2.5-32B-Instruct-AWQ'),
            Model('qwen/Qwen2.5-72B-Instruct-AWQ', 'Qwen/Qwen2.5-72B-Instruct-AWQ'),
        ]),
        # qwen2.5-math
        ModelGroup(
            [
                # base
                Model('qwen/Qwen2.5-Math-1.5B', 'Qwen/Qwen2.5-Math-1.5B'),
                Model('qwen/Qwen2.5-Math-7B', 'Qwen/Qwen2.5-Math-7B'),
                Model('qwen/Qwen2.5-Math-72B', 'Qwen/Qwen2.5-Math-72B'),
                # instruct
                Model('qwen/Qwen2.5-Math-1.5B-Instruct', 'Qwen/Qwen2.5-Math-1.5B-Instruct'),
                Model('qwen/Qwen2.5-Math-7B-Instruct', 'Qwen/Qwen2.5-Math-7B-Instruct'),
                Model('qwen/Qwen2.5-Math-72B-Instruct', 'Qwen/Qwen2.5-Math-72B-Instruct'),
            ],
            tags=['math']),
        # qwen2.5-coder
        ModelGroup(
            [
                # base
                Model('qwen/Qwen2.5-Coder-1.5B', 'Qwen/Qwen2.5-Coder-1.5B'),
                Model('qwen/Qwen2.5-Coder-7B', 'Qwen/Qwen2.5-Coder-7B'),
                # instruct
                Model('qwen/Qwen2.5-Coder-1.5B-Instruct', 'Qwen/Qwen2.5-Coder-1.5B-Instruct'),
                Model('qwen/Qwen2.5-Coder-7B-Instruct', 'Qwen/Qwen2.5-Coder-7B-Instruct'),
            ],
            tags=['coding'])
    ],
    TemplateGroup(TemplateType.qwen2_5),
    get_model_tokenizer_with_flash_attn,
    requires=['transformers>=4.37'],
    support_flash_attn=True,
    support_vllm=True,
    support_lmdeploy=True)


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

    from transformers import Qwen2VLForConditionalGeneration
    kwargs['automodel_class'] = Qwen2VLForConditionalGeneration
    model, tokenizer = get_model_tokenizer_multimodal(model_dir, torch_dtype, model_kwargs, load_model, **kwargs)
    if model is not None:
        patch_output_clone(model.model.embed_tokens)
        patch_output_to_input_device(model.model.embed_tokens)
    return model, tokenizer


register_model(
    MLLMModelType.qwen2_vl,
    'Qwen2VLForConditionalGeneration',
    [
        ModelGroup(
            [
                # base
                Model('qwen/Qwen2-VL-2B', 'Qwen/Qwen2-VL-2B'),
                Model('qwen/Qwen2-VL-7B', 'Qwen/Qwen2-VL-7B'),
                Model('qwen/Qwen2-VL-72B', 'Qwen/Qwen2-VL-72B'),
                # chat
                Model('qwen/Qwen2-VL-2B-Instruct', 'Qwen/Qwen2-VL-2B-Instruct'),
                Model('qwen/Qwen2-VL-7B-Instruct', 'Qwen/Qwen2-VL-7B-Instruct'),
                Model('qwen/Qwen2-VL-72B-Instruct', 'Qwen/Qwen2-VL-72B-Instruct'),
                # gptq-int4
                Model('qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4', 'Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4'),
                Model('qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4', 'Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4'),
                Model('qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4', 'Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4'),
                # gptq-int8
                Model('qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8', 'Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8'),
                Model('qwen/Qwen2-VL-7B-Instruct-GPTQ-Int8', 'Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int8'),
                Model('qwen/Qwen2-VL-72B-Instruct-GPTQ-Int8', 'Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int8'),
                # awq-int4
                Model('qwen/Qwen2-VL-2B-Instruct-AWQ', 'Qwen/Qwen2-VL-2B-Instruct-AWQ'),
                Model('qwen/Qwen2-VL-7B-Instruct-AWQ', 'Qwen/Qwen2-VL-7B-Instruct-AWQ'),
                Model('qwen/Qwen2-VL-72B-Instruct-AWQ', 'Qwen/Qwen2-VL-72B-Instruct-AWQ'),
            ],
            tags=['vision', 'video']),
    ],
    TemplateGroup(TemplateType.qwen2_vl, TemplateType.qwen2_vl_generation),
    get_model_tokenizer_qwen2_vl,
    requires=['transformers>=4.45.dev.0'],  # pip install qwen_vl_utils
    support_flash_attn=True,
    support_vllm=True)


def get_model_tokenizer_qwen2_audio(*args, **kwargs):
    from transformers import Qwen2AudioForConditionalGeneration
    kwargs['automodel_class'] = Qwen2AudioForConditionalGeneration
    return get_model_tokenizer_multimodal(*args, **kwargs)


register_model(
    MLLMModelType.qwen2_audio,
    'Qwen2AudioForConditionalGeneration', [
        ModelGroup([
            Model('qwen/Qwen2-Audio-7B', 'Qwen/Qwen2-Audio-7B'),
            Model('qwen/Qwen2-Audio-7B-Instruct', 'Qwen/Qwen2-Audio-7B-Instruct'),
            Model('qwen/Qwen2-Audio-7B', 'Qwen/Qwen2-Audio-7B'),
        ],
                   tags=['audio']),
    ],
    TemplateGroup(TemplateType.qwen2_audio, TemplateType.qwen2_audio_generation),
    get_model_tokenizer_qwen2_audio,
    requires=['transformers>=4.45.dev.0'],
    support_flash_attn=True)

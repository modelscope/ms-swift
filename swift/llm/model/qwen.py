from typing import Any, Dict, Type

import torch
from transformers import AutoConfig, BitsAndBytesConfig, PreTrainedTokenizerBase
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.auto.tokenization_auto import get_tokenizer_config

from swift.llm import TemplateType
from swift.utils import get_logger
from .constant import LLMModelType, MLLMModelType
from .model import Model, ModelGroup, TemplateGroup, get_model_tokenizer_from_repo, register_model
from .patcher import patch_output_clone
from .utils import AttnImpl

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
    use_flash_attn = AttnImpl.to_use_flash_attn(kwargs.pop('attn_impl', None), 'auto')
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
    LLMModelType.qwen,
    'QWenLMHeadModel',
    [
        # qwen
        ModelGroup(
            [
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
            ],
            TemplateGroup(TemplateType.qwen)),
        # tongyi-finance
        ModelGroup([
            Model('TongyiFinance/Tongyi-Finance-14B'),
            Model('TongyiFinance/Tongyi-Finance-14B-Chat', 'jxy/Tongyi-Finance-14B-Chat'),
            Model('TongyiFinance/Tongyi-Finance-14B-Chat-Int4', 'jxy/Tongyi-Finance-14B-Chat-Int4'),
        ],
                   TemplateGroup(TemplateType.qwen),
                   tags=['financial']),
        # codefuse-qwen
        ModelGroup([
            Model('codefuse-ai/CodeFuse-QWen-14B', 'codefuse-ai/CodeFuse-QWen-14B'),
        ],
                   TemplateGroup(TemplateType.codefuse),
                   tags=['coding'])
    ],
    get_model_tokenizer_qwen,
    support_flash_attn=True,
    support_vllm=True,
    support_lmdeploy=True)

register_model(
    LLMModelType.modelscope_agent,
    'QWenLMHeadModel', [
        ModelGroup([
            Model('iic/ModelScope-Agent-7B'),
            Model('iic/ModelScope-Agent-14B'),
        ], TemplateGroup(TemplateType.modelscope_agent))
    ],
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
                   TemplateGroup(TemplateType.qwen_audio, TemplateType.qwen_audio_generation),
                   tags=['audio'])
    ],
    get_model_tokenizer_qwen_audio,
    support_flash_attn=True)


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

    get_qwen_function = kwargs.pop('get_qwen_function', get_model_tokenizer_qwen_chat)
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
    model, tokenizer = get_qwen_function(model_dir, torch_dtype, model_kwargs, load_model, **kwargs)
    if model is not None:
        fix_qwen_inplace_bug(model)
        # fix device_map is 4
        if n_gpu // local_world_size >= 4:
            model.transformer.visual.proj.data = model.transformer.visual.proj.to(
                model.transformer.visual.ln_post.bias.device)
        # fix images cuda:1 bug
        model.transformer.visual.register_forward_hook(get_device_hook(0))
    return model, tokenizer


register_model(
    MLLMModelType.qwen_vl,
    'QWenLMHeadModel', [
        ModelGroup([
            Model('qwen/Qwen-VL', 'Qwen/Qwen-VL'),
            Model('qwen/Qwen-VL-Chat', 'Qwen/Qwen-VL-Chat'),
            Model('qwen/Qwen-VL-Chat-Int4', 'Qwen/Qwen-VL-Chat-Int4'),
        ],
                   TemplateGroup(TemplateType.qwen_vl, TemplateType.qwen_vl_generation),
                   tags=['vision'])
    ],
    get_model_tokenizer_qwen_vl,
    support_flash_attn=True,
    support_vllm=True,
    support_lmdeploy=True)

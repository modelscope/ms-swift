from typing import Any, Dict, Type

import torch
import transformers
from modelscope import snapshot_download
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from swift.llm import TemplateType
from .constant import LLMModelType, MLLMModelType
from .patcher import patch_output_to_input_device
from .register import Model, ModelGroup, TemplateGroup, get_model_tokenizer_from_local, register_model


def get_model_tokenizer_internlm_chat(model_dir: str,
                                      torch_dtype: torch.dtype,
                                      model_kwargs: Dict[str, Any],
                                      load_model: bool = True,
                                      **kwargs):
    model, tokenizer = get_model_tokenizer_from_local(model_dir, torch_dtype, model_kwargs, load_model, **kwargs)
    if getattr(tokenizer.__class__.eos_token_id, 'fset', None) is None:
        del tokenizer.__class__.eos_token_id
    tokenizer.eos_token = '<eoa>'
    return model, tokenizer


register_model(
    LLMModelType.internlm,
    'InternLMForCausalLM',
    ModelGroup([
        Model('Shanghai_AI_Laboratory/internlm-7b', 'internlm/internlm-7b'),
        Model('Shanghai_AI_Laboratory/internlm-chat-7b', 'internlm/internlm-chat-7b'),
        Model('Shanghai_AI_Laboratory/internlm-chat-7b-8k'),
        Model('Shanghai_AI_Laboratory/internlm-20b', 'internlm/internlm-20b'),
        Model('Shanghai_AI_Laboratory/internlm-chat-20b', 'internlm/internlm-chat-20b'),
    ]),
    TemplateGroup(TemplateType.internlm),
    get_model_tokenizer_internlm_chat,
    support_vllm=True,
    support_lmdeploy=True,
)


def get_model_tokenizer_internlm2(model_dir: str,
                                  torch_dtype: torch.dtype,
                                  model_kwargs: Dict[str, Any],
                                  load_model: bool = True,
                                  model_config=None,
                                  **kwargs):
    if model_config is None:
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    use_flash_attn = kwargs.pop('use_flash_attn', False)
    if use_flash_attn:
        model_config.attn_implementation = 'flash_attention_2'

    eos_token = kwargs.pop('eos_token', None)
    model, tokenizer = get_model_tokenizer_from_local(
        model_dir, torch_dtype, model_kwargs, load_model, model_config=model_config, **kwargs)
    if eos_token is not None:
        if getattr(tokenizer.__class__.eos_token_id, 'fset', None) is None:
            del tokenizer.__class__.eos_token_id
        tokenizer.eos_token = eos_token

    return model, tokenizer


register_model(
    LLMModelType.internlm2,
    'InternLM2ForCausalLM',
    [
        ModelGroup([
            Model('Shanghai_AI_Laboratory/internlm2-1_8b', 'internlm/internlm2-1_8b'),
            Model('Shanghai_AI_Laboratory/internlm2-chat-1_8b', 'internlm/internlm2-chat-1_8b'),
            Model('Shanghai_AI_Laboratory/internlm2-chat-1_8b-sft', 'internlm/internlm2-chat-1_8b-sft'),
            Model('Shanghai_AI_Laboratory/internlm2-base-7b', 'internlm/internlm2-base-7b'),
            Model('Shanghai_AI_Laboratory/internlm2-7b', 'internlm/internlm2-7b'),
            Model('Shanghai_AI_Laboratory/internlm2-chat-7b', 'internlm/internlm2-chat-7b'),
            Model('Shanghai_AI_Laboratory/internlm2-chat-7b-sft', 'internlm/internlm2-chat-7b-sft'),
            Model('Shanghai_AI_Laboratory/internlm2-base-20b', 'internlm/internlm2-base-20b'),
            Model('Shanghai_AI_Laboratory/internlm2-20b', 'internlm/internlm2-20b'),
            Model('Shanghai_AI_Laboratory/internlm2-chat-20b', 'internlm/internlm2-chat-20b'),
            Model('Shanghai_AI_Laboratory/internlm2-chat-20b-sft', 'internlm/internlm2-chat-20b-sft'),
        ]),
        ModelGroup([
            Model('Shanghai_AI_Laboratory/internlm2-math-base-7b', 'internlm/internlm2-math-base-7b'),
            Model('Shanghai_AI_Laboratory/internlm2-math-7b', 'internlm/internlm2-math-7b'),
            Model('Shanghai_AI_Laboratory/internlm2-math-base-20b', 'internlm/internlm2-math-base-20b'),
            Model('Shanghai_AI_Laboratory/internlm2-math-20b', 'internlm/internlm2-math-20b'),
        ],
                   tags=['math']),
        ModelGroup([
            Model('Shanghai_AI_Laboratory/internlm2_5-1_8b', 'internlm/internlm2_5-1_8b'),
            Model('Shanghai_AI_Laboratory/internlm2_5-1_8b-chat', 'internlm/internlm2_5-1_8b-chat'),
            Model('Shanghai_AI_Laboratory/internlm2_5-7b', 'internlm/internlm2_5-7b'),
            Model('Shanghai_AI_Laboratory/internlm2_5-7b-chat', 'internlm/internlm2_5-7b-chat'),
            Model('Shanghai_AI_Laboratory/internlm2_5-7b-chat-1m', 'internlm/internlm2_5-7b-chat-1m'),
            Model('Shanghai_AI_Laboratory/internlm2_5-20b', 'internlm/internlm2_5-20b'),
            Model('Shanghai_AI_Laboratory/internlm2_5-20b-chat', 'internlm/internlm2_5-20b-chat'),
        ],
                   tags=['math'])
    ],
    TemplateGroup(TemplateType.internlm2),
    get_model_tokenizer_internlm2,
    requires=['transformers>=4.38'],
    support_flash_attn=True,
    support_vllm=True,
    support_lmdeploy=True,
)


def get_model_tokenizer_internlm_xcomposer2(model_dir: str,
                                            torch_dtype: torch.dtype,
                                            model_kwargs: Dict[str, Any],
                                            load_model: bool = True,
                                            **kwargs):
    version = kwargs.pop('version', 'v2')
    model_config = None
    use_flash_attn = kwargs.pop('use_flash_attn', False)
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
        model_config._flash_attn_2_enabled = use_flash_attn

    model, tokenizer = get_model_tokenizer_internlm2(
        model_dir, torch_dtype, model_kwargs, load_model, model_config=model_config, **kwargs)
    if model is not None:
        if version == 'v2' and use_flash_attn:
            # fix AttributeError: no attribute 'attention_dropout'
            model.model.layers[0].attention.__class__.attention_dropout = 0.

        if version == 'v2.5':
            patch_output_to_input_device(model.vit)
            patch_output_to_input_device(model.vision_proj)

    return model, tokenizer


# TODO:xcomposer

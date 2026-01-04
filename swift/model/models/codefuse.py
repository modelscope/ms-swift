# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

from transformers import AutoTokenizer

from swift.template import TemplateType
from ..constant import LLMModelType
from ..model_arch import ModelArch
from ..model_meta import Model, ModelGroup, ModelMeta
from ..register import register_model
from .glm import ChatGLMLoader
from .qwen import QwenLoader

register_model(
    ModelMeta(
        LLMModelType.codefuse_qwen, [
            ModelGroup([
                Model('codefuse-ai/CodeFuse-QWen-14B', 'codefuse-ai/CodeFuse-QWen-14B'),
            ]),
        ],
        QwenLoader,
        template=TemplateType.codefuse,
        architectures=['QWenLMHeadModel'],
        model_arch=ModelArch.qwen,
        tags=['coding']))

register_model(
    ModelMeta(
        LLMModelType.codefuse_codegeex2,
        [
            ModelGroup([Model('codefuse-ai/CodeFuse-CodeGeeX2-6B', 'codefuse-ai/CodeFuse-CodeGeeX2-6B')], ),
        ],
        ChatGLMLoader,
        template=TemplateType.codefuse,
        architectures=['ChatGLMModel', 'ChatGLMForConditionalGeneration'],
        model_arch=ModelArch.chatglm,
        tags=['coding'],
        requires=['transformers<4.34'],
    ))


def get_model_tokenizer_codellama(model_dir: str,
                                  model_info,
                                  model_kwargs: Dict[str, Any],
                                  load_model: bool = True,
                                  **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False, legacy=False)
    return get_model_tokenizer_with_flash_attn(
        model_dir, model_info, model_kwargs, load_model, tokenizer=tokenizer, **kwargs)


register_model(
    ModelMeta(
        LLMModelType.codefuse_codellama,
        [
            ModelGroup(
                [
                    Model('codefuse-ai/CodeFuse-CodeLlama-34B', 'codefuse-ai/CodeFuse-CodeLlama-34B'),
                ],
                tags=['coding'],
            ),
        ],
        template=TemplateType.codefuse_codellama,
        model_arch=ModelArch.llama,
        hf_model_type=['llama'],
    ))

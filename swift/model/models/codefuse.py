# Copyright (c) ModelScope Contributors. All rights reserved.
from transformers import AutoTokenizer, PretrainedConfig

from swift.template import TemplateType
from swift.utils import Processor
from ..constant import LLMModelType
from ..model_arch import ModelArch
from ..model_meta import Model, ModelGroup, ModelMeta
from ..register import ModelLoader, register_model
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


class CodeLlamaLoader(ModelLoader):

    def get_processor(self, model_dir: str, config: PretrainedConfig) -> Processor:
        return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False, legacy=False)


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
        CodeLlamaLoader,
        template=TemplateType.codefuse_codellama,
        model_arch=ModelArch.llama,
        architectures=['LlamaForCausalLM'],
    ))

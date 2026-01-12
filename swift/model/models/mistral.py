# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

from transformers import AutoProcessor, AutoTokenizer, PretrainedConfig, PreTrainedModel

from swift.template import TemplateType
from swift.utils import Processor, safe_snapshot_download
from ..constant import LLMModelType, MLLMModelType
from ..model_arch import ModelArch
from ..model_meta import Model, ModelGroup, ModelMeta
from ..register import ModelLoader, register_model

register_model(
    ModelMeta(
        LLMModelType.mistral,
        [
            ModelGroup([
                Model('AI-ModelScope/Mistral-7B-Instruct-v0.1', 'mistralai/Mistral-7B-Instruct-v0.1'),
                Model('AI-ModelScope/Mistral-7B-Instruct-v0.2', 'mistralai/Mistral-7B-Instruct-v0.2'),
                Model('LLM-Research/Mistral-7B-Instruct-v0.3', 'mistralai/Mistral-7B-Instruct-v0.3'),
                Model('AI-ModelScope/Mistral-7B-v0.1', 'mistralai/Mistral-7B-v0.1'),
                Model('AI-ModelScope/Mistral-7B-v0.2-hf', 'alpindale/Mistral-7B-v0.2-hf'),
            ]),
            ModelGroup([
                Model('swift/Codestral-22B-v0.1', 'mistralai/Codestral-22B-v0.1'),
            ]),
        ],
        template=TemplateType.llama,
        architectures=['MistralForCausalLM'],
        model_arch=ModelArch.llama,
        requires=['transformers>=4.34'],
    ))

register_model(
    ModelMeta(
        LLMModelType.mixtral, [
            ModelGroup([
                Model('AI-ModelScope/Mixtral-8x7B-Instruct-v0.1', 'mistralai/Mixtral-8x7B-Instruct-v0.1'),
                Model('AI-ModelScope/Mixtral-8x7B-v0.1', 'mistralai/Mixtral-8x7B-v0.1'),
                Model('AI-ModelScope/Mixtral-8x22B-v0.1', 'mistral-community/Mixtral-8x22B-v0.1'),
            ],
                       requires=['transformers>=4.36']),
            ModelGroup([
                Model('AI-ModelScope/Mixtral-8x7b-AQLM-2Bit-1x16-hf', 'ISTA-DASLab/Mixtral-8x7b-AQLM-2Bit-1x16-hf'),
            ],
                       requires=['transformers>=4.38', 'aqlm', 'torch>=2.2.0']),
        ],
        template=TemplateType.llama,
        architectures=['MixtralForCausalLM'],
        model_arch=ModelArch.llama))

register_model(
    ModelMeta(
        LLMModelType.mistral_nemo, [
            ModelGroup([
                Model('AI-ModelScope/Mistral-Small-Instruct-2409', 'mistralai/Mistral-Small-Instruct-2409'),
                Model('LLM-Research/Mistral-Large-Instruct-2407', 'mistralai/Mistral-Large-Instruct-2407'),
                Model('AI-ModelScope/Mistral-Nemo-Base-2407', 'mistralai/Mistral-Nemo-Base-2407'),
                Model('AI-ModelScope/Mistral-Nemo-Instruct-2407', 'mistralai/Mistral-Nemo-Instruct-2407'),
            ],
                       requires=['transformers>=4.43']),
            ModelGroup([
                Model('AI-ModelScope/Ministral-8B-Instruct-2410', 'mistralai/Ministral-8B-Instruct-2410'),
            ],
                       requires=['transformers>=4.46']),
        ],
        template=TemplateType.mistral_nemo,
        architectures=['MistralForCausalLM'],
        model_arch=ModelArch.llama))

register_model(
    ModelMeta(
        LLMModelType.mistral_2501, [
            ModelGroup([
                Model('mistralai/Mistral-Small-24B-Base-2501', 'mistralai/Mistral-Small-24B-Base-2501'),
                Model('mistralai/Mistral-Small-24B-Instruct-2501', 'mistralai/Mistral-Small-24B-Instruct-2501'),
            ]),
        ],
        template=TemplateType.mistral_2501,
        architectures=['MistralForCausalLM'],
        model_arch=ModelArch.llama))

register_model(
    ModelMeta(
        LLMModelType.zephyr,
        [
            ModelGroup([
                Model('modelscope/zephyr-7b-beta', 'HuggingFaceH4/zephyr-7b-beta'),
            ]),
        ],
        template=TemplateType.zephyr,
        model_arch=ModelArch.llama,
        architectures=['MistralForCausalLM'],
        requires=['transformers>=4.34'],
    ))

register_model(
    ModelMeta(
        LLMModelType.wizardlm2_moe,
        [ModelGroup([
            Model('AI-ModelScope/WizardLM-2-8x22B', 'alpindale/WizardLM-2-8x22B'),
        ])],
        template=TemplateType.wizardlm2_moe,
        architectures=['MixtralForCausalLM'],
        requires=['transformers>=4.36'],
    ))

register_model(
    ModelMeta(
        LLMModelType.wizardlm2,
        [ModelGroup([
            Model('AI-ModelScope/WizardLM-2-7B-AWQ', 'MaziyarPanahi/WizardLM-2-7B-AWQ'),
        ])],
        template=TemplateType.wizardlm2,
        architectures=['MistralForCausalLM'],
        requires=['transformers>=4.34'],
    ))


class DevstralLoader(ModelLoader):

    def get_processor(self, model_dir: str, config: PretrainedConfig) -> Processor:
        # src: sglang did the same (https://github.com/sgl-project/sglang/pull/6547)
        tokenizer_dir = safe_snapshot_download('mistralai/Mistral-Small-3.1-24B-Instruct-2503', download_model=False)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        return tokenizer


register_model(
    ModelMeta(
        LLMModelType.devstral, [
            ModelGroup([
                Model('mistralai/Devstral-Small-2505', 'mistralai/Devstral-Small-2505'),
            ],
                       requires=['transformers>=4.43', 'mistral-common>=1.5.5'])
        ],
        DevstralLoader,
        template=TemplateType.devstral,
        architectures=['MistralForCausalLM'],
        model_arch=ModelArch.llama))


class Mistral3Loader(ModelLoader):

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        from transformers import Mistral3ForConditionalGeneration
        self.auto_model_cls = self.auto_model_cls or Mistral3ForConditionalGeneration
        return super().get_model(model_dir, *args, **kwargs)


register_model(
    ModelMeta(
        MLLMModelType.mistral3,
        [
            ModelGroup([
                Model('mistralai/Mistral-Small-3.1-24B-Base-2503', 'mistralai/Mistral-Small-3.1-24B-Base-2503'),
                Model('mistralai/Mistral-Small-3.1-24B-Instruct-2503', 'mistralai/Mistral-Small-3.1-24B-Instruct-2503'),
            ],
                       requires=['transformers>=4.49']),
            ModelGroup([
                Model('mistralai/Ministral-3-3B-Base-2512', 'mistralai/Ministral-3-3B-Base-2512'),
                Model('mistralai/Ministral-3-3B-Instruct-2512', 'mistralai/Ministral-3-3B-Instruct-2512'),
                Model('mistralai/Ministral-3-3B-Instruct-2512-BF16', 'mistralai/Ministral-3-3B-Instruct-2512-BF16'),
                Model('mistralai/Ministral-3-8B-Base-2512', 'mistralai/Ministral-3-8B-Base-2512'),
                Model('mistralai/Ministral-3-8B-Instruct-2512', 'mistralai/Ministral-3-8B-Instruct-2512'),
                Model('mistralai/Ministral-3-8B-Instruct-2512-BF16', 'mistralai/Ministral-3-8B-Instruct-2512-BF16'),
                Model('mistralai/Ministral-3-14B-Base-2512', 'mistralai/Ministral-3-14B-Base-2512'),
                Model('mistralai/Ministral-3-14B-Instruct-2512', 'mistralai/Ministral-3-14B-Instruct-2512'),
                Model('mistralai/Ministral-3-14B-Instruct-2512-BF16', 'mistralai/Ministral-3-14B-Instruct-2512-BF16'),
            ],
                       TemplateType.mistral_2512,
                       requires=['transformers>=5.0.0.dev0', 'mistral-common>=1.8.6']),
            ModelGroup([
                Model('mistralai/Ministral-3-3B-Reasoning-2512', 'mistralai/Ministral-3-3B-Reasoning-2512'),
                Model('mistralai/Ministral-3-8B-Reasoning-2512', 'mistralai/Ministral-3-8B-Reasoning-2512'),
                Model('mistralai/Ministral-3-14B-Reasoning-2512', 'mistralai/Ministral-3-14B-Reasoning-2512'),
            ],
                       TemplateType.mistral_2512_thinking,
                       requires=['transformers>=5.0.0.dev0', 'mistral-common>=1.8.6']),
        ],
        Mistral3Loader,
        template=TemplateType.mistral_2503,
        model_arch=ModelArch.llava_hf,
        architectures=['Mistral3ForConditionalGeneration'],
        tags=['vision'],
        ignore_patterns=[],
    ))


class Mistral3_2506Loader(Mistral3Loader):

    def get_processor(self, model_dir: str, config: PretrainedConfig) -> Processor:
        tokenizer_dir = safe_snapshot_download('mistralai/Mistral-Small-3.1-24B-Instruct-2503', download_model=False)
        processor = AutoProcessor.from_pretrained(tokenizer_dir)
        return processor


register_model(
    ModelMeta(
        MLLMModelType.mistral3_2506,
        [
            ModelGroup([
                Model('mistralai/Mistral-Small-3.2-24B-Instruct-2506', 'mistralai/Mistral-Small-3.2-24B-Instruct-2506'),
            ]),
        ],
        Mistral3_2506Loader,
        template=TemplateType.mistral_2506,
        architectures=['Mistral3ForConditionalGeneration'],
        model_arch=ModelArch.llava_hf,
        requires=['transformers>=4.49'],
    ))

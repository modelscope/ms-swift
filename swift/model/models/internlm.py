# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

from transformers import AutoTokenizer, PretrainedConfig, PreTrainedModel
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from swift.template import TemplateType
from swift.utils import Processor, safe_snapshot_download
from ..constant import LLMModelType, MLLMModelType, RMModelType
from ..model_arch import ModelArch
from ..model_meta import Model, ModelGroup, ModelMeta
from ..patcher import patch_output_clone, patch_output_to_input_device
from ..register import ModelLoader, RewardModelLoader, register_model
from ..utils import use_submodel_func
from .qwen import Qwen2AudioLoader

register_model(
    ModelMeta(
        LLMModelType.internlm,
        [
            ModelGroup([
                Model('Shanghai_AI_Laboratory/internlm-chat-7b', 'internlm/internlm-chat-7b'),
                Model('Shanghai_AI_Laboratory/internlm-7b', 'internlm/internlm-7b'),
                Model('Shanghai_AI_Laboratory/internlm-chat-7b-8k'),
                Model('Shanghai_AI_Laboratory/internlm-20b', 'internlm/internlm-20b'),
                Model('Shanghai_AI_Laboratory/internlm-chat-20b', 'internlm/internlm-chat-20b'),
            ])
        ],
        template=TemplateType.internlm,
        architectures=['InternLMForCausalLM'],
        model_arch=ModelArch.llama,
    ))

register_model(
    ModelMeta(
        LLMModelType.internlm2,
        [
            ModelGroup([
                Model('Shanghai_AI_Laboratory/internlm2-chat-1_8b', 'internlm/internlm2-chat-1_8b'),
                Model('Shanghai_AI_Laboratory/internlm2-1_8b', 'internlm/internlm2-1_8b'),
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
                Model('Shanghai_AI_Laboratory/internlm2-math-7b', 'internlm/internlm2-math-7b'),
                Model('Shanghai_AI_Laboratory/internlm2-math-base-7b', 'internlm/internlm2-math-base-7b'),
                Model('Shanghai_AI_Laboratory/internlm2-math-base-20b', 'internlm/internlm2-math-base-20b'),
                Model('Shanghai_AI_Laboratory/internlm2-math-20b', 'internlm/internlm2-math-20b'),
            ],
                       tags=['math']),
            ModelGroup([
                Model('Shanghai_AI_Laboratory/internlm2_5-1_8b-chat', 'internlm/internlm2_5-1_8b-chat'),
                Model('Shanghai_AI_Laboratory/internlm2_5-1_8b', 'internlm/internlm2_5-1_8b'),
                Model('Shanghai_AI_Laboratory/internlm2_5-7b', 'internlm/internlm2_5-7b'),
                Model('Shanghai_AI_Laboratory/internlm2_5-7b-chat', 'internlm/internlm2_5-7b-chat'),
                Model('Shanghai_AI_Laboratory/internlm2_5-7b-chat-1m', 'internlm/internlm2_5-7b-chat-1m'),
                Model('Shanghai_AI_Laboratory/internlm2_5-20b', 'internlm/internlm2_5-20b'),
                Model('Shanghai_AI_Laboratory/internlm2_5-20b-chat', 'internlm/internlm2_5-20b-chat'),
            ])
        ],
        template=TemplateType.internlm2,
        requires=['transformers>=4.38'],
        architectures=['InternLM2ForCausalLM'],
        model_arch=ModelArch.internlm2,
    ))

register_model(
    ModelMeta(
        LLMModelType.internlm3,
        [
            ModelGroup([
                Model('Shanghai_AI_Laboratory/internlm3-8b-instruct', 'internlm/internlm3-8b-instruct'),
            ]),
        ],
        template=TemplateType.internlm2,
        requires=['transformers>=4.48'],
        architectures=['InternLM3ForCausalLM'],
        model_arch=ModelArch.llama,
    ))


class InternVLLoader(ModelLoader):

    def get_processor(self, model_dir: str, config: PretrainedConfig) -> Processor:
        self.auto_tokenizer_cls = AutoTokenizer
        return super().get_processor(model_dir, config)

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        model = super().get_model(model_dir, *args, **kwargs)
        if self.model_info.quant_method == 'bnb':  # 'is_training'
            # patch: bnb backward shape mismatch bug
            if model is not None and model.language_model is not None:
                model.language_model.output.state.force_no_igemmlt = True
        use_submodel_func(model, 'language_model')
        patch_output_clone(model.language_model.get_input_embeddings())
        return model


register_model(
    ModelMeta(
        MLLMModelType.internvl,
        [
            ModelGroup([
                Model('OpenGVLab/Mini-InternVL-Chat-2B-V1-5', 'OpenGVLab/Mini-InternVL-Chat-2B-V1-5'),
                Model('AI-ModelScope/InternVL-Chat-V1-5', 'OpenGVLab/InternVL-Chat-V1-5'),
                Model('AI-ModelScope/InternVL-Chat-V1-5-int8', 'OpenGVLab/InternVL-Chat-V1-5-int8'),
            ], ),
        ],
        InternVLLoader,
        template=TemplateType.internvl,
        architectures=['InternVLChatModel'],
        model_arch=ModelArch.internvl,
        requires=['transformers>=4.35', 'timm'],
        tags=['vision'],
    ))

register_model(
    ModelMeta(
        MLLMModelType.internvl_phi3,
        [
            ModelGroup([
                Model('OpenGVLab/Mini-InternVL-Chat-4B-V1-5', 'OpenGVLab/Mini-InternVL-Chat-4B-V1-5'),
            ], ),
        ],
        InternVLLoader,
        template=TemplateType.internvl_phi3,
        architectures=['InternVLChatModel'],
        model_arch=ModelArch.internvl,
        requires=['transformers>=4.35,<4.42', 'timm'],
        tags=['vision'],
    ))

register_model(
    ModelMeta(
        MLLMModelType.internvl2,
        [
            ModelGroup([
                Model('OpenGVLab/InternVL2-1B', 'OpenGVLab/InternVL2-1B'),
                Model('OpenGVLab/InternVL2-2B', 'OpenGVLab/InternVL2-2B'),
                Model('OpenGVLab/InternVL2-8B', 'OpenGVLab/InternVL2-8B'),
                Model('OpenGVLab/InternVL2-26B', 'OpenGVLab/InternVL2-26B'),
                Model('OpenGVLab/InternVL2-40B', 'OpenGVLab/InternVL2-40B'),
                Model('OpenGVLab/InternVL2-Llama3-76B', 'OpenGVLab/InternVL2-Llama3-76B'),
            ]),
            # (infer use lmdeploy)
            ModelGroup([
                Model('OpenGVLab/InternVL2-2B-AWQ', 'OpenGVLab/InternVL2-2B-AWQ'),
                Model('OpenGVLab/InternVL2-8B-AWQ', 'OpenGVLab/InternVL2-8B-AWQ'),
                Model('OpenGVLab/InternVL2-26B-AWQ', 'OpenGVLab/InternVL2-26B-AWQ'),
                Model('OpenGVLab/InternVL2-40B-AWQ', 'OpenGVLab/InternVL2-40B-AWQ'),
                Model('OpenGVLab/InternVL2-Llama3-76B-AWQ', 'OpenGVLab/InternVL2-Llama3-76B-AWQ'),
            ]),
            ModelGroup([Model('OpenGVLab/InternVL2-8B-MPO', 'OpenGVLab/InternVL2-8B-MPO')]),
            # pretrain
            ModelGroup([
                Model('OpenGVLab/InternVL2-Pretrain-Models:InternVL2-1B-Pretrain',
                      'OpenGVLab/InternVL2-Pretrain-Models:InternVL2-1B-Pretrain'),
                Model('OpenGVLab/InternVL2-Pretrain-Models:InternVL2-2B-Pretrain',
                      'OpenGVLab/InternVL2-Pretrain-Models:InternVL2-2B-Pretrain'),
                Model('OpenGVLab/InternVL2-Pretrain-Models:InternVL2-4B-Pretrain',
                      'OpenGVLab/InternVL2-Pretrain-Models:InternVL2-4B-Pretrain'),
                Model('OpenGVLab/InternVL2-Pretrain-Models:InternVL2-8B-Pretrain',
                      'OpenGVLab/InternVL2-Pretrain-Models:InternVL2-8B-Pretrain'),
                Model('OpenGVLab/InternVL2-Pretrain-Models:InternVL2-26B-Pretrain',
                      'OpenGVLab/InternVL2-Pretrain-Models:InternVL2-26B-Pretrain'),
                Model('OpenGVLab/InternVL2-Pretrain-Models:InternVL2-40B-Pretrain',
                      'OpenGVLab/InternVL2-Pretrain-Models:InternVL2-40B-Pretrain'),
                Model('OpenGVLab/InternVL2-Pretrain-Models:InternVL2-Llama3-76B-Pretrain',
                      'OpenGVLab/InternVL2-Pretrain-Models:InternVL2-Llama3-76B-Pretrain'),
            ])
        ],
        InternVLLoader,
        template=TemplateType.internvl2,
        architectures=['InternVLChatModel'],
        model_arch=ModelArch.internvl,
        requires=['transformers>=4.36', 'timm'],
        tags=['vision', 'video'],
    ))

register_model(
    ModelMeta(
        MLLMModelType.internvl2_phi3,
        [
            ModelGroup([
                Model('OpenGVLab/InternVL2-4B', 'OpenGVLab/InternVL2-4B'),
            ], ),
        ],
        InternVLLoader,
        template=TemplateType.internvl2_phi3,
        architectures=['InternVLChatModel'],
        model_arch=ModelArch.internvl,
        requires=['transformers>=4.36,<4.42', 'timm'],
        tags=['vision', 'video'],
    ))

register_model(
    ModelMeta(
        MLLMModelType.internvl2_5,
        [
            ModelGroup([
                Model('OpenGVLab/InternVL2_5-1B', 'OpenGVLab/InternVL2_5-1B'),
                Model('OpenGVLab/InternVL2_5-2B', 'OpenGVLab/InternVL2_5-2B'),
                Model('OpenGVLab/InternVL2_5-4B', 'OpenGVLab/InternVL2_5-4B'),
                Model('OpenGVLab/InternVL2_5-8B', 'OpenGVLab/InternVL2_5-8B'),
                Model('OpenGVLab/InternVL2_5-26B', 'OpenGVLab/InternVL2_5-26B'),
                Model('OpenGVLab/InternVL2_5-38B', 'OpenGVLab/InternVL2_5-38B'),
                Model('OpenGVLab/InternVL2_5-78B', 'OpenGVLab/InternVL2_5-78B'),
            ]),
            # quant (infer use lmdeploy)
            ModelGroup([
                Model('OpenGVLab/InternVL2_5-4B-AWQ', 'OpenGVLab/InternVL2_5-4B-AWQ'),
                Model('OpenGVLab/InternVL2_5-8B-AWQ', 'OpenGVLab/InternVL2_5-8B-AWQ'),
                Model('OpenGVLab/InternVL2_5-26B-AWQ', 'OpenGVLab/InternVL2_5-26B-AWQ'),
                Model('OpenGVLab/InternVL2_5-38B-AWQ', 'OpenGVLab/InternVL2_5-38B-AWQ'),
                Model('OpenGVLab/InternVL2_5-78B-AWQ', 'OpenGVLab/InternVL2_5-78B-AWQ'),
            ]),
            ModelGroup([
                Model('OpenGVLab/InternVL2_5-1B-MPO', 'OpenGVLab/InternVL2_5-1B-MPO'),
                Model('OpenGVLab/InternVL2_5-2B-MPO', 'OpenGVLab/InternVL2_5-2B-MPO'),
                Model('OpenGVLab/InternVL2_5-4B-MPO', 'OpenGVLab/InternVL2_5-4B-MPO'),
                Model('OpenGVLab/InternVL2_5-8B-MPO', 'OpenGVLab/InternVL2_5-8B-MPO'),
                Model('OpenGVLab/InternVL2_5-26B-MPO', 'OpenGVLab/InternVL2_5-26B-MPO'),
                Model('OpenGVLab/InternVL2_5-38B-MPO', 'OpenGVLab/InternVL2_5-38B-MPO'),
                Model('OpenGVLab/InternVL2_5-78B-MPO', 'OpenGVLab/InternVL2_5-78B-MPO'),
            ]),
        ],
        InternVLLoader,
        template=TemplateType.internvl2_5,
        architectures=['InternVLChatModel'],
        model_arch=ModelArch.internvl,
        requires=['transformers>=4.36', 'timm'],
        tags=['vision', 'video'],
    ))

register_model(
    ModelMeta(
        MLLMModelType.internvl3,
        [
            # pretrain
            ModelGroup([
                Model('OpenGVLab/InternVL3-1B-Pretrained', 'OpenGVLab/InternVL3-1B-Pretrained'),
                Model('OpenGVLab/InternVL3-2B-Pretrained', 'OpenGVLab/InternVL3-2B-Pretrained'),
                Model('OpenGVLab/InternVL3-8B-Pretrained', 'OpenGVLab/InternVL3-8B-Pretrained'),
                Model('OpenGVLab/InternVL3-9B-Pretrained', 'OpenGVLab/InternVL3-9B-Pretrained'),
                Model('OpenGVLab/InternVL3-14B-Pretrained', 'OpenGVLab/InternVL3-14B-Pretrained'),
                Model('OpenGVLab/InternVL3-38B-Pretrained', 'OpenGVLab/InternVL3-38B-Pretrained'),
                Model('OpenGVLab/InternVL3-78B-Pretrained', 'OpenGVLab/InternVL3-78B-Pretrained'),
            ]),
            # instruct
            ModelGroup([
                Model('OpenGVLab/InternVL3-1B-Instruct', 'OpenGVLab/InternVL3-1B-Instruct'),
                Model('OpenGVLab/InternVL3-2B-Instruct', 'OpenGVLab/InternVL3-2B-Instruct'),
                Model('OpenGVLab/InternVL3-8B-Instruct', 'OpenGVLab/InternVL3-8B-Instruct'),
                Model('OpenGVLab/InternVL3-9B-Instruct', 'OpenGVLab/InternVL3-9B-Instruct'),
                Model('OpenGVLab/InternVL3-14B-Instruct', 'OpenGVLab/InternVL3-14B-Instruct'),
                Model('OpenGVLab/InternVL3-38B-Instruct', 'OpenGVLab/InternVL3-38B-Instruct'),
                Model('OpenGVLab/InternVL3-78B-Instruct', 'OpenGVLab/InternVL3-78B-Instruct'),
            ]),
            # mpo
            ModelGroup([
                Model('OpenGVLab/InternVL3-1B', 'OpenGVLab/InternVL3-1B'),
                Model('OpenGVLab/InternVL3-2B', 'OpenGVLab/InternVL3-2B'),
                Model('OpenGVLab/InternVL3-8B', 'OpenGVLab/InternVL3-8B'),
                Model('OpenGVLab/InternVL3-9B', 'OpenGVLab/InternVL3-9B'),
                Model('OpenGVLab/InternVL3-14B', 'OpenGVLab/InternVL3-14B'),
                Model('OpenGVLab/InternVL3-38B', 'OpenGVLab/InternVL3-38B'),
                Model('OpenGVLab/InternVL3-78B', 'OpenGVLab/InternVL3-78B'),
            ]),
            # awq (Use lmdeploy for inference.)
            ModelGroup([
                Model('OpenGVLab/InternVL3-1B-AWQ', 'OpenGVLab/InternVL3-1B-AWQ'),
                Model('OpenGVLab/InternVL3-2B-AWQ', 'OpenGVLab/InternVL3-2B-AWQ'),
                Model('OpenGVLab/InternVL3-8B-AWQ', 'OpenGVLab/InternVL3-8B-AWQ'),
                Model('OpenGVLab/InternVL3-9B-AWQ', 'OpenGVLab/InternVL3-9B-AWQ'),
                Model('OpenGVLab/InternVL3-14B-AWQ', 'OpenGVLab/InternVL3-14B-AWQ'),
                Model('OpenGVLab/InternVL3-38B-AWQ', 'OpenGVLab/InternVL3-38B-AWQ'),
                Model('OpenGVLab/InternVL3-78B-AWQ', 'OpenGVLab/InternVL3-78B-AWQ'),
            ]),
            # SenseNova-SI
            ModelGroup([
                Model('SenseNova/SenseNova-SI-InternVL3-2B', 'sensenova/SenseNova-SI-InternVL3-2B'),
                Model('SenseNova/SenseNova-SI-InternVL3-8B', 'sensenova/SenseNova-SI-InternVL3-8B'),
                Model('SenseNova/SenseNova-SI-1.1-InternVL3-2B', 'sensenova/SenseNova-SI-1.1-InternVL3-2B'),
                Model('SenseNova/SenseNova-SI-1.1-InternVL3-8B', 'sensenova/SenseNova-SI-1.1-InternVL3-8B'),
            ]),
        ],
        InternVLLoader,
        template=TemplateType.internvl2_5,
        architectures=['InternVLChatModel'],
        model_arch=ModelArch.internvl,
        requires=['transformers>=4.37.2', 'timm'],
        tags=['vision', 'video'],
    ))

register_model(
    ModelMeta(
        MLLMModelType.internvl3_5,
        [
            # pretrain
            ModelGroup([
                Model('OpenGVLab/InternVL3_5-1B-Pretrained', 'OpenGVLab/InternVL3_5-1B-Pretrained'),
                Model('OpenGVLab/InternVL3_5-2B-Pretrained', 'OpenGVLab/InternVL3_5-2B-Pretrained'),
                Model('OpenGVLab/InternVL3_5-4B-Pretrained', 'OpenGVLab/InternVL3_5-4B-Pretrained'),
                Model('OpenGVLab/InternVL3_5-8B-Pretrained', 'OpenGVLab/InternVL3_5-8B-Pretrained'),
                Model('OpenGVLab/InternVL3_5-14B-Pretrained', 'OpenGVLab/InternVL3_5-14B-Pretrained'),
                Model('OpenGVLab/InternVL3_5-38B-Pretrained', 'OpenGVLab/InternVL3_5-38B-Pretrained'),
                Model('OpenGVLab/InternVL3_5-30B-A3B-Pretrained', 'OpenGVLab/InternVL3_5-30B-A3B-Pretrained'),
                Model('OpenGVLab/InternVL3_5-241B-A28B-Pretrained', 'OpenGVLab/InternVL3_5-241B-A28B-Pretrained'),
            ]),
            # Instruct
            ModelGroup([
                Model('OpenGVLab/InternVL3_5-1B-Instruct', 'OpenGVLab/InternVL3_5-1B-Instruct'),
                Model('OpenGVLab/InternVL3_5-2B-Instruct', 'OpenGVLab/InternVL3_5-2B-Instruct'),
                Model('OpenGVLab/InternVL3_5-4B-Instruct', 'OpenGVLab/InternVL3_5-4B-Instruct'),
                Model('OpenGVLab/InternVL3_5-8B-Instruct', 'OpenGVLab/InternVL3_5-8B-Instruct'),
                Model('OpenGVLab/InternVL3_5-14B-Instruct', 'OpenGVLab/InternVL3_5-14B-Instruct'),
                Model('OpenGVLab/InternVL3_5-38B-Instruct', 'OpenGVLab/InternVL3_5-38B-Instruct'),
                Model('OpenGVLab/InternVL3_5-30B-A3B-Instruct', 'OpenGVLab/InternVL3_5-30B-A3B-Instruct'),
                Model('OpenGVLab/InternVL3_5-241B-A28B-Instruct', 'OpenGVLab/InternVL3_5-241B-A28B-Instruct'),
            ]),
            # MPO
            ModelGroup([
                Model('OpenGVLab/InternVL3_5-1B-MPO', 'OpenGVLab/InternVL3_5-1B-MPO'),
                Model('OpenGVLab/InternVL3_5-2B-MPO', 'OpenGVLab/InternVL3_5-2B-MPO'),
                Model('OpenGVLab/InternVL3_5-4B-MPO', 'OpenGVLab/InternVL3_5-4B-MPO'),
                Model('OpenGVLab/InternVL3_5-8B-MPO', 'OpenGVLab/InternVL3_5-8B-MPO'),
                Model('OpenGVLab/InternVL3_5-14B-MPO', 'OpenGVLab/InternVL3_5-14B-MPO'),
                Model('OpenGVLab/InternVL3_5-38B-MPO', 'OpenGVLab/InternVL3_5-38B-MPO'),
                Model('OpenGVLab/InternVL3_5-30B-A3B-MPO', 'OpenGVLab/InternVL3_5-30B-A3B-MPO'),
                Model('OpenGVLab/InternVL3_5-241B-A28B-MPO', 'OpenGVLab/InternVL3_5-241B-A28B-MPO'),
            ]),
            ModelGroup([
                Model('OpenGVLab/InternVL3_5-1B', 'OpenGVLab/InternVL3_5-1B'),
                Model('OpenGVLab/InternVL3_5-2B', 'OpenGVLab/InternVL3_5-2B'),
                Model('OpenGVLab/InternVL3_5-4B', 'OpenGVLab/InternVL3_5-4B'),
                Model('OpenGVLab/InternVL3_5-8B', 'OpenGVLab/InternVL3_5-8B'),
                Model('OpenGVLab/InternVL3_5-14B', 'OpenGVLab/InternVL3_5-14B'),
                Model('OpenGVLab/InternVL3_5-38B', 'OpenGVLab/InternVL3_5-38B'),
                Model('OpenGVLab/InternVL3_5-30B-A3B', 'OpenGVLab/InternVL3_5-30B-A3B'),
                Model('OpenGVLab/InternVL3_5-241B-A28B', 'OpenGVLab/InternVL3_5-241B-A28B'),
            ]),
        ],
        InternVLLoader,
        template=TemplateType.internvl3_5,
        architectures=['InternVLChatModel'],
        model_arch=ModelArch.internvl,
        requires=['transformers>=4.37.2', 'timm'],
        tags=['vision', 'video'],
    ))

register_model(
    ModelMeta(
        MLLMModelType.internvl3_5_gpt,
        [
            ModelGroup([
                Model('OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview', 'OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview'),
            ]),
        ],
        InternVLLoader,
        template=TemplateType.internvl3_5_gpt,
        architectures=['InternVLChatModel'],
        model_arch=ModelArch.internvl,
        requires=['transformers>=4.37.2', 'timm'],
        tags=['vision', 'video'],
    ))


class Interns1Loader(ModelLoader):

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        from transformers.modeling_utils import PreTrainedModel
        model = super().get_model(model_dir, *args, **kwargs)
        if not hasattr(PreTrainedModel, '_old_enable_input_require_grads'):
            old_enable_input_require_grads = PreTrainedModel.enable_input_require_grads

            def patched_enable_input_require_grads(self):

                def make_inputs_require_grads(module, input, output):
                    if isinstance(output, tuple):
                        output[0].requires_grad_(True)
                    else:
                        output.requires_grad_(True)

                self._require_grads_hook = self.get_input_embeddings().register_forward_hook(make_inputs_require_grads)

            PreTrainedModel.enable_input_require_grads = patched_enable_input_require_grads
            PreTrainedModel._old_enable_input_require_grads = old_enable_input_require_grads
        return model


class InternVLHfLoader(Interns1Loader):

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        from transformers import AutoModelForImageTextToText
        self.auto_model_cls = self.auto_model_cls or AutoModelForImageTextToText
        return super().get_model(model_dir, *args, **kwargs)


register_model(
    ModelMeta(
        MLLMModelType.internvl_hf,
        [
            ModelGroup([
                Model('OpenGVLab/InternVL3-1B-hf', 'OpenGVLab/InternVL3-1B-hf'),
                Model('OpenGVLab/InternVL3-2B-hf', 'OpenGVLab/InternVL3-2B-hf'),
                Model('OpenGVLab/InternVL3-8B-hf', 'OpenGVLab/InternVL3-8B-hf'),
                Model('OpenGVLab/InternVL3-9B-hf', 'OpenGVLab/InternVL3-9B-hf'),
                Model('OpenGVLab/InternVL3-14B-hf', 'OpenGVLab/InternVL3-14B-hf'),
                Model('OpenGVLab/InternVL3-38B-hf', 'OpenGVLab/InternVL3-38B-hf'),
                Model('OpenGVLab/InternVL3-78B-hf', 'OpenGVLab/InternVL3-78B-hf'),
            ]),
            ModelGroup([
                Model('OpenGVLab/InternVL3_5-1B-HF', 'OpenGVLab/InternVL3_5-1B-HF'),
                Model('OpenGVLab/InternVL3_5-2B-HF', 'OpenGVLab/InternVL3_5-2B-HF'),
                Model('OpenGVLab/InternVL3_5-4B-HF', 'OpenGVLab/InternVL3_5-4B-HF'),
                Model('OpenGVLab/InternVL3_5-8B-HF', 'OpenGVLab/InternVL3_5-8B-HF'),
                Model('OpenGVLab/InternVL3_5-14B-HF', 'OpenGVLab/InternVL3_5-14B-HF'),
                Model('OpenGVLab/InternVL3_5-38B-HF', 'OpenGVLab/InternVL3_5-38B-HF'),
                Model('OpenGVLab/InternVL3_5-30B-A3B-HF', 'OpenGVLab/InternVL3_5-30B-A3B-HF'),
                Model('OpenGVLab/InternVL3_5-241B-A28B-HF', 'OpenGVLab/InternVL3_5-241B-A28B-HF'),
            ]),
        ],
        InternVLHfLoader,
        template=TemplateType.internvl_hf,
        architectures=['InternVLForConditionalGeneration'],
        model_arch=ModelArch.llava_hf,
        requires=['transformers>=4.52.1', 'timm'],
        tags=['vision', 'video'],
    ))

register_model(
    ModelMeta(
        MLLMModelType.internvl_gpt_hf,
        [
            ModelGroup([
                Model('OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview-HF',
                      'OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview-HF'),
            ]),
        ],
        InternVLHfLoader,
        template=TemplateType.internvl_hf,
        architectures=['InternVLForConditionalGeneration'],
        model_arch=ModelArch.llava_hf,
        requires=['transformers>=4.55.0', 'timm'],
        tags=['vision', 'video'],
    ))

register_model(
    ModelMeta(
        MLLMModelType.interns1,
        [
            ModelGroup([
                Model('Shanghai_AI_Laboratory/Intern-S1-mini', 'internlm/Intern-S1-mini'),
                Model('Shanghai_AI_Laboratory/Intern-S1', 'internlm/Intern-S1'),
                Model('Shanghai_AI_Laboratory/Intern-S1-mini-FP8', 'internlm/Intern-S1-mini-FP8'),
                Model('Shanghai_AI_Laboratory/Intern-S1-FP8', 'internlm/Intern-S1-FP8'),
            ]),
        ],
        Interns1Loader,
        template=TemplateType.interns1,
        architectures=['InternS1ForConditionalGeneration'],
        model_arch=ModelArch.interns1,
        requires=['transformers>=4.55.2,<4.56'],
        tags=['vision', 'video'],
    ))


class Xcomposer2Loader(ModelLoader):
    version = 'v2'

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        if self.version == 'v2-4khd':
            from transformers import CLIPVisionModel

            def load_model(self):
                self.vision_tower_name = safe_snapshot_download(
                    'AI-ModelScope/clip-vit-large-patch14-336', check_local=True)
                self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
                self.vision_tower.requires_grad_(False)
                self.is_loaded = True

            CLIPVisionTower = get_class_from_dynamic_module('build_mlp.CLIPVisionTower', model_dir)
            CLIPVisionTower.load_model = load_model
        model = super().get_model(model_dir, *args, **kwargs)
        model.vit.vision_tower.gradient_checkpointing_enable()
        if self.version == 'v2':
            # fix AttributeError: no attribute 'attention_dropout'
            model.model.layers[0].attention.__class__.attention_dropout = 0.

        if self.version == 'v2.5':
            patch_output_to_input_device(model.vit)
            patch_output_to_input_device(model.vision_proj)


register_model(
    ModelMeta(
        MLLMModelType.xcomposer2,
        [
            ModelGroup([
                Model('Shanghai_AI_Laboratory/internlm-xcomposer2-7b', 'internlm/internlm-xcomposer2-7b'),
            ], ),
        ],
        Xcomposer2Loader,
        template=TemplateType.xcomposer2,
        architectures=['InternLMXComposer2ForCausalLM'],
        model_arch=ModelArch.xcomposer,
        tags=['vision'],
    ))


class Xcomposer2_4khdLoader(Xcomposer2Loader):
    version = 'v2-4khd'


register_model(
    ModelMeta(
        MLLMModelType.xcomposer2_4khd,
        [
            ModelGroup([
                Model('Shanghai_AI_Laboratory/internlm-xcomposer2-4khd-7b', 'internlm/internlm-xcomposer2-4khd-7b'),
            ], ),
        ],
        Xcomposer2_4khdLoader,
        template=TemplateType.xcomposer2,
        architectures=['InternLM2ForCausalLM', 'InternLMXComposer2ForCausalLM'],
        model_arch=ModelArch.xcomposer,
        tags=['vision'],
    ))


class Xcomposer2_5Loader(Xcomposer2Loader):
    version = 'v2.5'


register_model(
    ModelMeta(
        MLLMModelType.xcomposer2_5,
        [
            ModelGroup([
                Model('Shanghai_AI_Laboratory/internlm-xcomposer2d5-7b', 'internlm/internlm-xcomposer2d5-7b'),
                Model('Shanghai_AI_Laboratory/internlm-xcomposer2d5-ol-7b:base',
                      'internlm/internlm-xcomposer2d5-ol-7b:base')
            ]),
        ],
        Xcomposer2_5Loader,
        template=TemplateType.xcomposer2_5,
        architectures=['InternLMXComposer2ForCausalLM'],
        model_arch=ModelArch.xcomposer,
        tags=['vision'],
        requires=['decord'],
        # target_modules: attention.wqkv attention.wo feed_forward.w1 feed_forward.w2 feed_forward.w3
    ))

register_model(
    ModelMeta(
        MLLMModelType.xcomposer2_5_ol_audio,
        [
            ModelGroup([
                Model('Shanghai_AI_Laboratory/internlm-xcomposer2d5-ol-7b:audio',
                      'internlm/internlm-xcomposer2d5-ol-7b:audio'),
            ]),
        ],
        Qwen2AudioLoader,
        template=TemplateType.qwen2_audio,
        requires=['transformers>=4.45'],
        architectures=['Qwen2AudioForConditionalGeneration'],
        model_arch=ModelArch.qwen2_audio,
        tags=['audio'],
    ))

register_model(
    ModelMeta(
        RMModelType.internlm2_reward,
        [
            ModelGroup([
                Model('Shanghai_AI_Laboratory/internlm2-1_8b-reward', 'internlm/internlm2-1_8b-reward'),
                Model('Shanghai_AI_Laboratory/internlm2-7b-reward', 'internlm/internlm2-7b-reward'),
                Model('Shanghai_AI_Laboratory/internlm2-20b-reward', 'internlm/internlm2-20b-reward'),
            ]),
        ],
        RewardModelLoader,
        template=TemplateType.internlm2_reward,
        is_reward=True,
        requires=['transformers>=4.38'],
        architectures=['InternLM2ForRewardModel'],
    ))

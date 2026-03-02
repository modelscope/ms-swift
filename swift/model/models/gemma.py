# Copyright (c) ModelScope Contributors. All rights reserved.
from transformers import PreTrainedModel

from swift.template import TemplateType
from ..constant import LLMModelType, MLLMModelType
from ..model_arch import ModelArch
from ..model_meta import Model, ModelGroup, ModelMeta
from ..patcher import patch_output_to_input_device
from ..register import ModelLoader, SentenceTransformersLoader, register_model


class PaligemmaVisionLoader(ModelLoader):

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        from transformers import PaliGemmaForConditionalGeneration
        self.auto_model_cls = self.auto_model_cls or PaliGemmaForConditionalGeneration
        return super().get_model(model_dir, *args, **kwargs)


register_model(
    ModelMeta(
        MLLMModelType.paligemma,
        [
            ModelGroup([
                Model('AI-ModelScope/paligemma-3b-pt-224', 'google/paligemma-3b-pt-224'),
                Model('AI-ModelScope/paligemma-3b-pt-448', 'google/paligemma-3b-pt-448'),
                Model('AI-ModelScope/paligemma-3b-pt-896', 'google/paligemma-3b-pt-896'),
            ]),
            ModelGroup([
                Model('AI-ModelScope/paligemma-3b-mix-224', 'google/paligemma-3b-mix-224'),
                Model('AI-ModelScope/paligemma-3b-mix-448', 'google/paligemma-3b-mix-448'),
            ]),
            ModelGroup([
                Model('AI-ModelScope/paligemma2-3b-pt-224', 'google/paligemma2-3b-pt-224'),
                Model('AI-ModelScope/paligemma2-3b-pt-448', 'google/paligemma2-3b-pt-448'),
                Model('AI-ModelScope/paligemma2-3b-pt-896', 'google/paligemma2-3b-pt-896'),
                Model('AI-ModelScope/paligemma2-10b-pt-224', 'google/paligemma2-10b-pt-224'),
                Model('AI-ModelScope/paligemma2-10b-pt-448', 'google/paligemma2-10b-pt-448'),
                Model('AI-ModelScope/paligemma2-10b-pt-896', 'google/paligemma2-10b-pt-896'),
                Model('AI-ModelScope/paligemma2-28b-pt-224', 'google/paligemma2-28b-pt-224'),
                Model('AI-ModelScope/paligemma2-28b-pt-448', 'google/paligemma2-28b-pt-448'),
                Model('AI-ModelScope/paligemma2-28b-pt-896', 'google/paligemma2-28b-pt-896'),
            ]),
            ModelGroup([
                Model('AI-ModelScope/paligemma2-3b-ft-docci-448', 'google/paligemma2-3b-ft-docci-448'),
                Model('AI-ModelScope/paligemma2-10b-ft-docci-448', 'google/paligemma2-10b-ft-docci-448'),
            ]),
        ],
        PaligemmaVisionLoader,
        template=TemplateType.paligemma,
        architectures=['PaliGemmaForConditionalGeneration'],
        model_arch=ModelArch.llava_hf,
        requires=['transformers>=4.41'],
        tags=['vision'],
    ))

register_model(
    ModelMeta(
        LLMModelType.gemma,
        [
            ModelGroup([
                Model('AI-ModelScope/gemma-2b-it', 'google/gemma-2b-it'),
                Model('AI-ModelScope/gemma-2b', 'google/gemma-2b'),
                Model('AI-ModelScope/gemma-7b', 'google/gemma-7b'),
                Model('AI-ModelScope/gemma-7b-it', 'google/gemma-7b-it'),
            ], ),
        ],
        template=TemplateType.gemma,
        architectures=['GemmaForCausalLM'],
        model_arch=ModelArch.llama,
        requires=['transformers>=4.38'],
    ))

register_model(
    ModelMeta(
        LLMModelType.gemma2,
        [
            ModelGroup([
                Model('LLM-Research/gemma-2-2b-it', 'google/gemma-2-2b-it'),
                Model('LLM-Research/gemma-2-2b', 'google/gemma-2-2b'),
                Model('LLM-Research/gemma-2-9b', 'google/gemma-2-9b'),
                Model('LLM-Research/gemma-2-9b-it', 'google/gemma-2-9b-it'),
                Model('LLM-Research/gemma-2-27b', 'google/gemma-2-27b'),
                Model('LLM-Research/gemma-2-27b-it', 'google/gemma-2-27b-it'),
            ], ),
        ],
        template=TemplateType.gemma,
        architectures=['Gemma2ForCausalLM'],
        model_arch=ModelArch.llama,
        requires=['transformers>=4.42'],
    ))


class Gemma3TextLoader(ModelLoader):

    def get_config(self, model_dir):
        # It is strongly recommended to train Gemma3 models with the `eager` attention implementation instead of `sdpa`.
        self.attn_impl = self.attn_impl or 'eager'
        return super().get_config(model_dir)


register_model(
    ModelMeta(
        LLMModelType.gemma3_text,
        [
            ModelGroup([
                Model('LLM-Research/gemma-3-1b-pt', 'google/gemma-3-1b-pt'),
                Model('LLM-Research/gemma-3-1b-it', 'google/gemma-3-1b-it'),
                Model('google/gemma-3-270m', 'google/gemma-3-270m'),
                Model('google/gemma-3-270m-it', 'google/gemma-3-270m-it'),
                Model('google/medgemma-27b-text-it', 'google/medgemma-27b-text-it'),
            ], ),
        ],
        Gemma3TextLoader,
        template=TemplateType.gemma3_text,
        architectures=['Gemma3ForCausalLM'],
        model_arch=ModelArch.llama,
        requires=['transformers>=4.49'],
    ))


class Gemma3VisionLoader(ModelLoader):

    def get_config(self, model_dir):
        # It is strongly recommended to train Gemma3 models with the `eager` attention implementation instead of `sdpa`.
        self.attn_impl = self.attn_impl or 'eager'
        return super().get_config(model_dir)

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        from transformers import Gemma3ForConditionalGeneration
        self.auto_model_cls = self.auto_model_cls or Gemma3ForConditionalGeneration
        return super().get_model(model_dir, *args, **kwargs)


register_model(
    ModelMeta(
        MLLMModelType.gemma3_vision,
        [
            ModelGroup([
                Model('LLM-Research/gemma-3-4b-pt', 'google/gemma-3-4b-pt'),
                Model('LLM-Research/gemma-3-4b-it', 'google/gemma-3-4b-it'),
                Model('LLM-Research/gemma-3-12b-pt', 'google/gemma-3-12b-pt'),
                Model('LLM-Research/gemma-3-12b-it', 'google/gemma-3-12b-it'),
                Model('LLM-Research/gemma-3-27b-pt', 'google/gemma-3-27b-pt'),
                Model('LLM-Research/gemma-3-27b-it', 'google/gemma-3-27b-it'),
                Model('google/medgemma-4b-pt', 'google/medgemma-4b-pt'),
                Model('google/medgemma-4b-it', 'google/medgemma-4b-it'),
                Model('google/medgemma-27b-it', 'google/medgemma-27b-it'),
            ], ),
        ],
        Gemma3VisionLoader,
        template=TemplateType.gemma3_vision,
        architectures=['Gemma3ForConditionalGeneration'],
        model_arch=ModelArch.llava_hf,
        requires=['transformers>=4.49'],
    ))


class Gemma3nLoader(ModelLoader):

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        from transformers import Gemma3nForConditionalGeneration
        self.auto_model_cls = self.auto_model_cls or Gemma3nForConditionalGeneration
        model = super().get_model(model_dir, *args, **kwargs)
        patch_output_to_input_device(model.model.embed_vision)
        patch_output_to_input_device(model.model.embed_audio)
        return model


register_model(
    ModelMeta(
        MLLMModelType.gemma3n,
        [
            ModelGroup([
                Model('google/gemma-3n-E2B', 'google/gemma-3n-E2B'),
                Model('google/gemma-3n-E4B', 'google/gemma-3n-E4B'),
                Model('google/gemma-3n-E2B-it', 'google/gemma-3n-E2B-it'),
                Model('google/gemma-3n-E4B-it', 'google/gemma-3n-E4B-it'),
            ], ),
        ],
        Gemma3nLoader,
        template=TemplateType.gemma3n,
        architectures=['Gemma3nForConditionalGeneration'],
        model_arch=ModelArch.gemma3n,
        requires=['transformers>=4.53.1'],
    ))

register_model(
    ModelMeta(
        LLMModelType.gemma_emb,
        [
            ModelGroup([
                Model('google/embeddinggemma-300m', 'google/embeddinggemma-300m'),
            ], ),
        ],
        SentenceTransformersLoader,
        template=TemplateType.dummy,
        architectures=['Gemma3TextModel'],
    ))

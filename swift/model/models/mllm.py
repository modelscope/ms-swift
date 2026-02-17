# Copyright (c) ModelScope Contributors. All rights reserved.
from types import MethodType

import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from swift.template import TemplateType
from swift.utils import Processor, get_logger
from ..constant import MLLMModelType
from ..model_arch import ModelArch
from ..model_meta import Model, ModelGroup, ModelMeta
from ..patcher import patch_output_clone
from ..register import ModelLoader, register_model
from ..utils import use_submodel_func
from .qwen import Qwen2VLLoader, patch_qwen_vl_utils

logger = get_logger()


class Idefics3Loader(ModelLoader):

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        from transformers import AutoModelForVision2Seq
        self.auto_model_cls = self.auto_model_cls or AutoModelForVision2Seq
        return super().get_model(model_dir, *args, **kwargs)


register_model(
    ModelMeta(
        MLLMModelType.idefics3,
        [
            ModelGroup([
                Model('AI-ModelScope/Idefics3-8B-Llama3', 'HuggingFaceM4/Idefics3-8B-Llama3'),
            ]),
        ],
        Idefics3Loader,
        template=TemplateType.idefics3,
        model_arch=ModelArch.idefics3,
        architectures=['Idefics3ForConditionalGeneration'],
        tags=['vision'],
        requires=['transformers>=4.45'],
    ))


class PixtralLoader(ModelLoader):

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        from transformers import LlavaForConditionalGeneration
        self.auto_model_cls = self.auto_model_cls or LlavaForConditionalGeneration
        return super().get_model(model_dir, *args, **kwargs)


register_model(
    ModelMeta(
        MLLMModelType.pixtral,
        [
            ModelGroup([
                Model('AI-ModelScope/pixtral-12b', 'mistral-community/pixtral-12b'),
            ]),
        ],
        PixtralLoader,
        template=TemplateType.pixtral,
        model_arch=ModelArch.llava_hf,
        architectures=['LlavaForConditionalGeneration'],
        requires=['transformers>=4.45'],
        tags=['vision'],
    ))


class MolMoeLoader(ModelLoader):

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        model = super().get_model(model_dir, *args, **kwargs)

        # fix bug for molmoe-1b
        def to_dict(self, *args, **kwargs):
            res = self._to_dict(*args, **kwargs)
            res['vision_backbone'] = self.vision_backbone.__dict__
            res.pop('to_dict')
            res.pop('_to_dict')
            return res

        model.config._to_dict = model.config.to_dict
        model.config.to_dict = MethodType(to_dict, model.config)
        patch_output_clone(model.model.transformer.wte)
        return model


register_model(
    ModelMeta(
        MLLMModelType.molmoe,
        [
            ModelGroup([
                Model('LLM-Research/MolmoE-1B-0924', 'allenai/MolmoE-1B-0924'),
            ]),
        ],
        MolMoeLoader,
        template=TemplateType.molmo,
        model_arch=ModelArch.molmo,
        torch_dtype=torch.float32,
        architectures=['OLMoForCausalLM'],
        tags=['vision'],
        requires=['transformers>=4.45'],
    ))


class MolmoLoader(ModelLoader):

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        model_cls = get_class_from_dynamic_module('modeling_molmo.MolmoForCausalLM', model_dir)
        model_cls._no_split_modules = ['MolmoSequentialBlock']
        model = super().get_model(model_dir, *args, **kwargs)
        patch_output_clone(model.model.transformer.wte)
        return model


register_model(
    ModelMeta(
        MLLMModelType.molmo,
        [
            ModelGroup([
                Model('LLM-Research/Molmo-7B-O-0924', 'allenai/Molmo-7B-O-0924'),
                Model('LLM-Research/Molmo-7B-D-0924', 'allenai/Molmo-7B-D-0924'),
                Model('LLM-Research/Molmo-72B-0924', 'allenai/Molmo-72B-0924'),
            ]),
        ],
        MolmoLoader,
        template=TemplateType.molmo,
        model_arch=ModelArch.molmo,
        architectures=['MolmoForCausalLM'],
        tags=['vision'],
        requires=['transformers>=4.45'],
    ))


class MegrezOmniLoader(ModelLoader):

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        model_cls = get_class_from_dynamic_module('modeling_megrezo.MegrezO', model_dir)
        model_cls._no_split_modules = ['ResidualAttentionBlock', 'LlamaDecoderLayer']
        model_cls = get_class_from_dynamic_module('modeling_megrezo.SiglipVisionTransformer', model_dir)
        model_cls._no_split_modules = ['SiglipEncoderLayer']
        model = super().get_model(model_dir, *args, **kwargs)
        patch_output_clone(model.llm.model.embed_tokens)
        use_submodel_func(model, 'llm')
        return model

    def _get_model_processor(self, model_dir, config):
        model, processor = super().get_processor(model_dir, config)
        if model:
            processor = model._get_or_init_processor()
        return model, processor


register_model(
    ModelMeta(
        MLLMModelType.megrez_omni,
        [
            ModelGroup([
                Model('InfiniAI/Megrez-3B-Omni', 'Infinigence/Megrez-3B-Omni'),
            ]),
        ],
        MegrezOmniLoader,
        template=TemplateType.megrez_omni,
        model_arch=ModelArch.megrez_omni,
        architectures=['MegrezO'],
        tags=['vision', 'audio'],
    ))

register_model(
    ModelMeta(
        MLLMModelType.qwen2_gme, [
            ModelGroup([
                Model('iic/gme-Qwen2-VL-2B-Instruct', 'Alibaba-NLP/gme-Qwen2-VL-2B-Instruct'),
                Model('iic/gme-Qwen2-VL-7B-Instruct', 'Alibaba-NLP/gme-Qwen2-VL-7B-Instruct'),
            ]),
        ],
        Qwen2VLLoader,
        template=TemplateType.qwen2_gme,
        model_arch=ModelArch.qwen2_vl,
        architectures=['Qwen2VLForConditionalGeneration'],
        tags=['vision']))


class JinaRerankerM0Loader(ModelLoader):

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        # Use AutoModel to respect the model repo's dynamic class mapping
        # and load the custom Jina reranker head via trust_remote_code.
        from transformers import AutoModel
        from transformers.modeling_outputs import SequenceClassifierOutputWithPast
        self.auto_model_cls = self.auto_model_cls or AutoModel
        model = super().get_model(model_dir, *args, **kwargs)
        # Patch forward to return a sequence-classification-style output with `.logits`
        # Use the model's own head (already present in jina-reranker-m0), just wrap outputs.

        if not hasattr(model, '_forward_origin'):
            model._forward_origin = model.forward
            model.logit_bias = 2.65

            def forward(self,
                        input_ids=None,
                        attention_mask=None,
                        position_ids=None,
                        inputs_embeds=None,
                        pixel_values=None,
                        image_grid_thw=None,
                        video_grid_thw=None,
                        output_attentions=None,
                        output_hidden_states=None,
                        return_dict=None,
                        **kwargs):
                # Remove labels to avoid upstream asserts in ranking models
                kwargs.pop('labels', None)
                if return_dict is None:
                    return_dict = True

                out = self._forward_origin(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    inputs_embeds=inputs_embeds,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs)

                logits = out.unsqueeze(-1) - self.logit_bias

                if not return_dict:
                    return (logits, )

                return SequenceClassifierOutputWithPast(logits=logits)

            model.forward = MethodType(forward, model)

            def padding_free_fn(self, output, kwargs, padding_side):
                return_dict = kwargs.get('return_dict', None)

                output.logits = output['last_hidden_state'][:, -1]
                logits = self.score(output.logits)
                logits = logits - self.logit_bias

                if not return_dict:
                    return (logits, )

                return SequenceClassifierOutputWithPast(logits=logits)

            model.padding_free_fn = MethodType(padding_free_fn, model)
            return model


register_model(
    ModelMeta(
        MLLMModelType.jina_reranker_m0,
        [ModelGroup([Model('JinaAI/jina-reranker-m0', 'JinaAI/jina-reranker-m0')])],
        JinaRerankerM0Loader,
        template=TemplateType.jina_reranker_m0,
        model_arch=ModelArch.qwen2_vl,
        architectures=['JinaRerankerM0ForConditionalGeneration'],
        task_type='reranker',
        tags=['reranker', 'vision'],
    ))


class KeyeVLLoader(ModelLoader):

    def get_processor(self, model_dir: str, config: PretrainedConfig) -> Processor:
        processor = super().get_processor(model_dir, config)
        from keye_vl_utils import vision_process
        global_vars = patch_qwen_vl_utils(vision_process)
        processor.global_vars = global_vars
        return processor


register_model(
    ModelMeta(
        MLLMModelType.keye_vl,
        [
            ModelGroup([
                Model('Kwai-Keye/Keye-VL-8B-Preview', 'Kwai-Keye/Keye-VL-8B-Preview'),
            ]),
        ],
        KeyeVLLoader,
        template=TemplateType.keye_vl,
        model_arch=ModelArch.keye_vl,
        architectures=['KeyeForConditionalGeneration'],
        tags=['vision'],
        requires=['keye_vl_utils'],
    ))

register_model(
    ModelMeta(
        MLLMModelType.keye_vl_1_5,
        [
            ModelGroup([
                Model('Kwai-Keye/Keye-VL-1_5-8B', 'Kwai-Keye/Keye-VL-1_5-8B'),
            ]),
        ],
        KeyeVLLoader,
        template=TemplateType.keye_vl_1_5,
        model_arch=ModelArch.keye_vl,
        architectures=['KeyeVL1_5ForConditionalGeneration'],
        tags=['vision'],
        requires=['keye_vl_utils>=1.5.2', 'transformers==4.52.4'],
    ))


class DotsOCRLoader(ModelLoader):

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        model_cls = get_class_from_dynamic_module('modeling_dots_vision.DotsVisionTransformer', model_dir)
        model_cls._no_split_modules = ['DotsVisionBlock']
        return super().get_model(model_dir, *args, **kwargs)


register_model(
    ModelMeta(
        MLLMModelType.dots_ocr,
        [ModelGroup([
            Model('rednote-hilab/dots.ocr', 'rednote-hilab/dots.ocr'),
        ])],
        DotsOCRLoader,
        template=TemplateType.dots_ocr,
        model_arch=ModelArch.dots_ocr,
        architectures=['DotsOCRForCausalLM'],
        requires=['transformers>=4.51.0'],
    ))


class Sail2VLLoader(ModelLoader):

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        model = super().get_model(model_dir, *args, **kwargs)
        use_submodel_func(model, 'language_model')
        return model


register_model(
    ModelMeta(
        MLLMModelType.sail_vl2, [
            ModelGroup([
                Model('BytedanceDouyinContent/SAIL-VL2-2B', 'BytedanceDouyinContent/SAIL-VL2-2B'),
                Model('BytedanceDouyinContent/SAIL-VL2-2B-Thinking', 'BytedanceDouyinContent/SAIL-VL2-2B-Thinking'),
                Model('BytedanceDouyinContent/SAIL-VL2-8B', 'BytedanceDouyinContent/SAIL-VL2-8B'),
                Model('BytedanceDouyinContent/SAIL-VL2-8B-Thinking', 'BytedanceDouyinContent/SAIL-VL2-8B-Thinking'),
            ])
        ],
        Sail2VLLoader,
        template=TemplateType.sail_vl2,
        model_arch=ModelArch.internvl,
        architectures=['SAILVLModel'],
        requires=['transformers<=4.51.3'],
        tags=['vision']))

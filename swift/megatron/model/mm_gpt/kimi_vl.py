# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
from megatron.training import get_args
from PIL import Image
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from swift.llm import ModelType
from ..constant import MegatronModelType
from ..gpt.hf2mcore import convert_hf2mcore
from ..gpt.mcore2hf import convert_mcore2hf
from ..register import register_megatron_model
from .utils import HuggingFaceModule, MMGPTMegatronModelMeta


def convert_hf2mcore_kimi_vl(hf_model, mg_model):
    convert_hf2mcore(hf_model.language_model, mg_model.language_model)
    mg_model.visual.vision_tower.load_state_dict(hf_model.vision_tower.state_dict())
    mg_model.visual.multi_modal_projector.load_state_dict(hf_model.multi_modal_projector.state_dict())


def convert_mcore2hf_kimi_vl(hf_model, mg_model):
    convert_mcore2hf(hf_model.language_model, mg_model.language_model)
    hf_model.vision_tower.load_state_dict(mg_model.visual.vision_tower.state_dict())
    hf_model.multi_modal_projector.load_state_dict(mg_model.visual.multi_modal_projector.state_dict())


class KimiVLVit(HuggingFaceModule):
    module_mapping = {'vision_tower': 'vision_tower', 'multi_modal_projector': 'multi_modal_projector'}
    _vision_tower = ['vision_tower']
    _aligner = ['multi_modal_projector']

    def __init__(self, config):
        args = get_args()
        model_dir = args.model_info.model_dir
        model_cls = get_class_from_dynamic_module('modeling_kimi_vl.DeepseekV3ForCausalLM', model_dir)
        super().__init__(config, [model_cls])

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        model = self._hf_model[0]
        input_ids = kwargs['input_ids']
        pixel_values = kwargs.get('pixel_values')
        if pixel_values is not None and pixel_values.size(0) > 0:
            pixel_values = pixel_values.to(model.vision_tower.dtype)
            image_features: torch.Tensor = model._extract_image_features(pixel_values, kwargs['image_grid_hws'])
            inputs_embeds = inputs_embeds.to(image_features[0].dtype).clone()
            inputs_embeds = model._merge_with_image_features(inputs_embeds, input_ids, image_features)
        else:
            image_processor = self.processor.image_processor
            dummy_image = Image.new('RGB', (32, 32), (0, 0, 0))
            image_inputs = image_processor([dummy_image], return_tensors='pt')
            pixel_values = image_inputs['pixel_values'].to(model.vision_tower.dtype)
            image_features: torch.Tensor = model._extract_image_features(pixel_values, image_inputs['image_grid_hws'])
            inputs_embeds = inputs_embeds + image_features.mean() * 0.
        return inputs_embeds


register_megatron_model(
    MMGPTMegatronModelMeta(
        MegatronModelType.kimi_vl, [
            ModelType.kimi_vl,
        ],
        convert_hf2mcore=convert_hf2mcore_kimi_vl,
        convert_mcore2hf=convert_mcore2hf_kimi_vl,
        visual_cls=KimiVLVit))

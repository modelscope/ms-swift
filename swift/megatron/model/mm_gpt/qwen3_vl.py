# Copyright (c) Alibaba, Inc. and its affiliates.
import torch

from swift.llm import ModelType
from ..constant import MegatronModelType
from ..gpt.hf2mcore import convert_hf2mcore
from ..gpt.mcore2hf import convert_mcore2hf
from ..register import register_megatron_model
from .utils import HuggingFaceModule, MMGPTMegatronModelMeta


def convert_hf2mcore_qwen3_vl(hf_model, mg_model):

    convert_hf2mcore(hf_model.language_model, mg_model.language_model)
    mg_model.visual.vision_model.load_state_dict(hf_model.vision_model.state_dict())
    mg_model.visual.mlp1.load_state_dict(hf_model.mlp1.state_dict())


def convert_mcore2hf_qwen3_vl(hf_model, mg_model):
    convert_mcore2hf(hf_model.language_model, mg_model.language_model)
    hf_model.vision_model.load_state_dict(mg_model.visual.vision_model.state_dict())
    hf_model.mlp1.load_state_dict(mg_model.visual.mlp1.state_dict())


class Qwen3VL_Vit(HuggingFaceModule):
    module_mapping = {'visual': 'visual'}
    vision_tower = ['visual']
    aligner = ['visual.merger', 'visual.deepstack_merger_list']

    def __init__(self, config):
        from transformers.models.qwen3_vl import Qwen3VLTextModel
        from transformers.models.qwen3_vl import Qwen3VLMoeTextModel
        super().__init__(config, [Qwen3VLTextModel, Qwen3VLMoeTextModel])

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        model = self._hf_model[0]
        input_ids = kwargs['input_ids']
        pixel_values = kwargs.get('pixel_values')
        if pixel_values is None:
            dummy_pixel_values = torch.zeros((1, 3, 32, 32), dtype=self.vision_model.dtype, device=inputs_embeds.device)
            vit_embeds = model.extract_feature(dummy_pixel_values)
            inputs_embeds = inputs_embeds + vit_embeds.mean() * 0.
        else:
            vit_embeds = model.extract_feature(pixel_values)
            selected = (input_ids == self.processor.encode('<IMG_CONTEXT>', add_special_tokens=False)[0])
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[selected] = vit_embeds.reshape(-1, vit_embeds.shape[-1]).to(dtype=inputs_embeds.dtype)
        return inputs_embeds


register_megatron_model(
    MMGPTMegatronModelMeta(
        MegatronModelType.qwen3_vl, [
            ModelType.qwen3_vl,
            ModelType.qwen3_moe_vl,
        ],
        convert_hf2mcore=convert_hf2mcore_qwen3,
        convert_mcore2hf=convert_mcore2hf_qwen3,
        visual_cls=Qwen3VL_Vit))

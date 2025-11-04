# Copyright (c) Alibaba, Inc. and its affiliates.
import torch

from swift.llm import ModelType
from ..constant import MegatronModelType
from ..gpt_bridge import GPTBridge, MultimodalGPTBridge
from ..register import MegatronModelMeta, register_megatron_model
from .utils import HuggingFaceModule


class Internvl3Bridge(GPTBridge):
    hf_layers_prefix = 'language_model.model.layers'
    hf_embed_key = 'language_model.model.embed_tokens.weight'
    hf_final_layernorm_key = 'language_model.model.norm.weight'
    hf_lm_head_key = 'language_model.lm_head.weight'
    hf_score_key = 'language_model.score.weight'

    def _init_meta_hf_model(self):
        internvl3_vit = Internvl3Vit(None)
        self.hf_model = internvl3_vit._hf_model[0]
        self.hf_model.vision_model = None
        self.processor = internvl3_vit.processor


class Internvl3Vit(HuggingFaceModule):
    module_mapping = {'vision_model': 'vision_model', 'mlp1': 'mlp1'}
    _vision_tower = ['vision_model']
    _aligner = ['mlp1']

    def __init__(self, config):
        model_cls = []
        from transformers.models.qwen2 import Qwen2ForCausalLM
        model_cls.append(Qwen2ForCausalLM)
        try:
            from transformers.models import Qwen3ForCausalLM
            model_cls.append(Qwen3ForCausalLM)
        except ImportError:
            pass
        try:
            from transformers.models import Qwen3MoeForCausalLM
            model_cls.append(Qwen3MoeForCausalLM)
        except ImportError:
            pass
        super().__init__(config, model_cls)

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
    MegatronModelMeta(
        MegatronModelType.internvl3, [
            ModelType.internvl3,
            ModelType.internvl3_5,
        ],
        bridge_cls=Internvl3Bridge,
        visual_cls=Internvl3Vit))


class InternvlHfBridge(MultimodalGPTBridge):
    hf_state_dict_mapping = {
        'language_model.lm_head': 'lm_head',
        'language_model.model': 'model.language_model',
        'vision_tower': 'model.vision_tower',
        'multi_modal_projector': 'model.multi_modal_projector',
    }


class InternvlHfVit(HuggingFaceModule):
    module_mapping = {'model.vision_tower': 'vision_tower', 'model.multi_modal_projector': 'multi_modal_projector'}
    _vision_tower = ['vision_tower']
    _aligner = ['multi_modal_projector']

    def __init__(self, config):
        model_cls = []
        from transformers.models import Qwen2Model
        model_cls.append(Qwen2Model)
        try:
            from transformers.models import Qwen3Model
            model_cls.append(Qwen3Model)
        except ImportError:
            pass
        try:
            from transformers.models import Qwen3MoeModel
            model_cls.append(Qwen3MoeModel)
        except ImportError:
            pass
        super().__init__(config, model_cls)

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        input_ids = kwargs['input_ids']
        pixel_values = kwargs.get('pixel_values')
        model = self._hf_model[0]
        device = self.vision_tower.device
        if pixel_values is not None:
            pixel_values = pixel_values.to(device=device)
            image_features = model.model.get_image_features(
                pixel_values,
                vision_feature_layer=self.model_config.vision_feature_layer,
                vision_feature_select_strategy=self.model_config.vision_feature_select_strategy,
            )
            special_image_mask = input_ids == self.model_config.image_token_id
            special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
        else:
            dummy_pixel_values = torch.zeros((1, 3, 32, 32), device=device, dtype=self.vision_tower.dtype)
            image_features = model.model.get_image_features(
                dummy_pixel_values,
                vision_feature_layer=self.model_config.vision_feature_layer,
                vision_feature_select_strategy=self.model_config.vision_feature_select_strategy,
            )
            inputs_embeds = inputs_embeds + image_features.mean() * 0.
        return inputs_embeds


register_megatron_model(
    MegatronModelMeta(
        MegatronModelType.internvl_hf, [
            ModelType.internvl_hf,
        ], bridge_cls=InternvlHfBridge, visual_cls=InternvlHfVit))

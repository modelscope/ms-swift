# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
from megatron.training import get_args

from swift.llm import ModelType
from ..constant import MegatronModelType
from ..gpt.hf2mcore import convert_hf2mcore
from ..gpt.hf2mcore import set_layer_state as set_layer_state_hf2mcore
from ..gpt.mcore2hf import convert_mcore2hf
from ..gpt.mcore2hf import set_layer_state as set_layer_state_mcore2hf
from ..register import register_megatron_model
from .utils import HuggingFaceModule, MMGPTMegatronModelMeta


def convert_hf2mcore_internvl3(hf_model, mg_model):

    convert_hf2mcore(hf_model.language_model, mg_model.language_model)
    mg_model.visual.vision_model.load_state_dict(hf_model.vision_model.state_dict())
    mg_model.visual.mlp1.load_state_dict(hf_model.mlp1.state_dict())


def convert_mcore2hf_internvl3(hf_model, mg_model):
    convert_mcore2hf(hf_model.language_model, mg_model.language_model)
    hf_model.vision_model.load_state_dict(mg_model.visual.vision_model.state_dict())
    hf_model.mlp1.load_state_dict(mg_model.visual.mlp1.state_dict())


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
    MMGPTMegatronModelMeta(
        MegatronModelType.internvl3, [
            ModelType.internvl3,
            ModelType.internvl3_5,
        ],
        convert_hf2mcore=convert_hf2mcore_internvl3,
        convert_mcore2hf=convert_mcore2hf_internvl3,
        visual_cls=Internvl3Vit))


def convert_hf2mcore_internvl_hf(hf_model, mg_model):
    language_model = hf_model.language_model
    mg_language_model = mg_model.language_model
    args = get_args()
    mg_language_model.embedding.word_embeddings.weight.data.copy_(language_model.embed_tokens.weight)
    if args.untie_embeddings_and_output_weights:
        mg_language_model.output_layer.weight.data.copy_(hf_model.lm_head.weight)
    mg_language_model.decoder.final_layernorm.weight.data.copy_(language_model.norm.weight)
    for layer_idx in range(args.num_layers):
        set_layer_state_hf2mcore(args, mg_language_model, language_model, layer_idx)
    mg_model.visual.vision_tower.load_state_dict(hf_model.vision_tower.state_dict())
    mg_model.visual.multi_modal_projector.load_state_dict(hf_model.multi_modal_projector.state_dict())


def convert_mcore2hf_internvl_hf(hf_model, mg_model):
    language_model = hf_model.language_model
    mg_language_model = mg_model.language_model
    args = get_args()
    language_model.embed_tokens.weight.data.copy_(mg_language_model.embedding.word_embeddings.weight)
    if args.untie_embeddings_and_output_weights:
        lm_head_weight = hf_model.score.weight if args.task_type == 'seq_cls' else hf_model.lm_head.weight
        lm_head_weight.data.copy_(mg_language_model.output_layer.weight)
    language_model.norm.weight.data.copy_(mg_language_model.decoder.final_layernorm.weight)
    for layer_idx in range(args.num_layers):
        set_layer_state_mcore2hf(args, mg_language_model, language_model, layer_idx)

    hf_model.vision_tower.load_state_dict(mg_model.visual.vision_tower.state_dict())
    hf_model.multi_modal_projector.load_state_dict(mg_model.visual.multi_modal_projector.state_dict())


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
    MMGPTMegatronModelMeta(
        MegatronModelType.internvl_hf, [
            ModelType.internvl_hf,
        ],
        convert_hf2mcore=convert_hf2mcore_internvl_hf,
        convert_mcore2hf=convert_mcore2hf_internvl_hf,
        visual_cls=InternvlHfVit))

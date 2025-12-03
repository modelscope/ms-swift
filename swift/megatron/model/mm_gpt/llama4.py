# Copyright (c) Alibaba, Inc. and its affiliates.
from copy import deepcopy

import megatron.core
import torch
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from megatron.training import get_args
from packaging import version

from swift.llm import ModelType
from ..constant import MegatronModelType
from ..gpt_bridge import GPTBridge
from ..register import MegatronModelMeta, register_megatron_model
from .utils import HuggingFaceModule

mcore_013 = version.parse(megatron.core.__version__) >= version.parse('0.13.0rc0')


class Llama4Vit(HuggingFaceModule):
    module_mapping = {'multi_modal_projector': 'multi_modal_projector', 'vision_model': 'vision_model'}
    _vision_tower = ['vision_model']
    _aligner = ['multi_modal_projector']

    def __init__(self, config):
        from transformers.models.llama4 import Llama4TextModel
        super().__init__(config, Llama4TextModel)

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        pixel_values = kwargs.get('pixel_values')
        input_ids = kwargs.get('input_ids')
        model = self._hf_model[0]
        vision_feature_select_strategy = self.model_config.vision_config.vision_feature_select_strategy
        origin_pixel_values = pixel_values
        if pixel_values is None:
            pixel_values = torch.zeros((1, 3, 336, 336), dtype=self.vision_model.dtype, device=inputs_embeds.device)
        image_features = model.get_image_features(
            pixel_values=pixel_values,
            vision_feature_select_strategy=vision_feature_select_strategy,
        )
        vision_flat = image_features.view(-1, image_features.size(-1))
        projected_vision_flat = model.multi_modal_projector(vision_flat).to(inputs_embeds.device, inputs_embeds.dtype)
        if origin_pixel_values is None:
            inputs_embeds = inputs_embeds + projected_vision_flat.mean() * 0.
        else:
            special_image_mask = model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=projected_vision_flat)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, projected_vision_flat)
        return inputs_embeds


def get_llama4_transformer_layer_spec(config, vp_stage=None):
    args = get_args()
    use_te = args.transformer_impl == 'transformer_engine'
    kwargs = {'qk_l2_norm': args.qk_l2_norm, 'vp_stage': vp_stage} if mcore_013 else {}
    # Define the decoder block spec
    transformer_layer_spec = get_gpt_decoder_block_spec(
        config, use_transformer_engine=use_te, normalization=args.normalization, **kwargs)
    for i, layer_spec in enumerate(transformer_layer_spec.layer_specs):
        global_i = i + get_transformer_layer_offset(config, vp_stage)
        no_rope = config.no_rope_freq[global_i]
        layer_spec = deepcopy(layer_spec)
        if no_rope:
            layer_spec.submodules.self_attention.submodules.q_layernorm = IdentityOp
            layer_spec.submodules.self_attention.submodules.k_layernorm = IdentityOp
            transformer_layer_spec.layer_specs[i] = layer_spec
    return transformer_layer_spec


class Llama4Bridge(GPTBridge):
    hf_layers_prefix = 'language_model.model.layers'
    hf_embed_key = 'language_model.model.embed_tokens.weight'
    hf_final_layernorm_key = 'language_model.model.norm.weight'
    hf_lm_head_key = 'language_model.lm_head.weight'
    hf_score_key = 'language_model.score.weight'


register_megatron_model(
    MegatronModelMeta(
        MegatronModelType.llama4, [
            ModelType.llama4,
        ],
        bridge_cls=Llama4Bridge,
        get_transformer_layer_spec=get_llama4_transformer_layer_spec,
        visual_cls=Llama4Vit))

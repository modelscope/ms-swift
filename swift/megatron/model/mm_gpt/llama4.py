# Copyright (c) Alibaba, Inc. and its affiliates.
from copy import deepcopy

import megatron.core
import torch
from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.training import get_args
from packaging import version

from swift.llm import ModelType, Template
from swift.megatron.utils import get_local_layer_specs
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
    # Define the decoder block spec
    kwargs = {'use_kitchen': config.use_kitchen} if mcore_013 else {}
    moe_layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        qk_layernorm=config.qk_layernorm,
        multi_latent_attention=config.multi_latent_attention,
        moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
        qk_l2_norm=args.qk_l2_norm,
        **kwargs,
    )
    layer_specs = []
    for i in range(args.num_layers):
        no_rope = config.no_rope_freq[i]
        layer_spec = deepcopy(moe_layer_spec)
        if no_rope:
            layer_spec.submodules.self_attention.submodules.q_layernorm = IdentityOp
            layer_spec.submodules.self_attention.submodules.k_layernorm = IdentityOp
        layer_specs.append(layer_spec)
    local_layer_specs = get_local_layer_specs(config, layer_specs, vp_stage=vp_stage)
    block_spec = TransformerBlockSubmodules(layer_specs=local_layer_specs, layer_norm=TENorm)

    return block_spec


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

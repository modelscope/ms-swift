# Copyright (c) ModelScope Contributors. All rights reserved.
from megatron.core.transformer.attention import SelfAttention
from typing import Optional

from swift.model import ModelType
from ..constant import MegatronModelType
from ..gpt_bridge import GPTBridge
from ..modules import GatedDeltaNet, GatedSelfAttention
from ..register import MegatronModelLoader, MegatronModelMeta, register_megatron_model


class Qwen3NextBridge(GPTBridge):
    hf_mtp_prefix = 'mtp.layers'

    def _set_layer_attn(self, mg_layer, hf_state_dict, layer_idx: int, to_mcore: bool):
        mg_attn = None if mg_layer is None else mg_layer.self_attention
        is_linear_attention = (layer_idx + 1) % self.config.linear_attention_freq != 0
        if is_linear_attention:
            hf_state_dict.update(
                self._set_linear_attn_state(mg_attn, hf_state_dict, 'linear_attn.', layer_idx, to_mcore))
            self._set_state_dict(mg_layer, 'self_attention.in_proj.layer_norm_weight', hf_state_dict,
                                 'input_layernorm.weight', to_mcore)
        else:
            hf_state_dict.update(self._set_attn_state(mg_attn, hf_state_dict, 'self_attn.', layer_idx, to_mcore))
            self._set_state_dict(mg_layer, 'self_attention.linear_qkv.layer_norm_weight', hf_state_dict,
                                 'input_layernorm.weight', to_mcore)
        return hf_state_dict

    def _convert_mtp_extra(self, mtp_layer, hf_state_dict, to_mcore, origin_hf_state_dict):
        hf_state_dict = self._remove_prefix(origin_hf_state_dict, 'mtp.')
        for mg_key, key in zip(['enorm.weight', 'hnorm.weight', 'eh_proj.weight'],
                               ['pre_fc_norm_embedding.weight', 'pre_fc_norm_hidden.weight', 'fc.weight']):
            self._set_state_dict(mtp_layer, mg_key, hf_state_dict, key, to_mcore)
        self._set_state_dict(mtp_layer, 'final_layernorm.weight', hf_state_dict, 'norm.weight', to_mcore)
        if not to_mcore:
            origin_hf_state_dict.update(self._add_prefix(hf_state_dict, 'mtp.'))


class Qwen3NextLoader(MegatronModelLoader):

    def get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        from megatron.core.models.gpt.experimental_attention_variant_module_specs import \
            get_transformer_block_with_experimental_attention_variant_spec
        layer_specs = get_transformer_block_with_experimental_attention_variant_spec(self.config, vp_stage)
        for layer_spec in layer_specs.layer_specs:
            attn_module = layer_spec.submodules.self_attention.module
            if issubclass(attn_module, SelfAttention):
                layer_spec.submodules.self_attention.module = GatedSelfAttention
            else:
                layer_spec.submodules.self_attention.module = GatedDeltaNet
        return layer_specs

    def build_model(
        self,
        pre_process=True,
        post_process=True,
        vp_stage: Optional[int] = None,
    ):
        model = super().build_model(pre_process, post_process, vp_stage)
        lm_model = model.language_model if hasattr(model, 'language_model') else model
        for layer in lm_model.decoder.layers:
            if hasattr(layer.self_attention, 'out_norm'):
                assert hasattr(layer.self_attention.out_norm, 'zero_centered_gamma')
                layer.self_attention.out_norm.zero_centered_gamma = False
        return model


register_megatron_model(
    MegatronModelMeta(
        MegatronModelType.qwen3_next,
        [
            ModelType.qwen3_next,
        ],
        bridge_cls=Qwen3NextBridge,
        loader=Qwen3NextLoader,
    ))

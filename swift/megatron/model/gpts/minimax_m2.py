# Copyright (c) ModelScope Contributors. All rights reserved.

import megatron.core
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from packaging import version

from swift.model import ModelType
from ..constant import MegatronModelType
from ..gpt_bridge import GPTBridge
from ..register import MegatronModelMeta, register_megatron_model

mcore_013 = version.parse(megatron.core.__version__) >= version.parse('0.13.0rc0')


class MinimaxM2SelfAttention(SelfAttention):

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        *args,
        **kwargs,
    ):
        q_layernorm = submodules.q_layernorm
        k_layernorm = submodules.k_layernorm
        submodules.q_layernorm = IdentityOp
        submodules.k_layernorm = IdentityOp
        super().__init__(config, submodules, *args, **kwargs)
        submodules.q_layernorm = q_layernorm
        submodules.k_layernorm = k_layernorm
        self.q_norm = build_module(
            submodules.q_layernorm,
            hidden_size=self.hidden_size_per_attention_head * config.num_attention_heads,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )
        self.k_norm = build_module(
            submodules.k_layernorm,
            hidden_size=self.hidden_size_per_attention_head * config.num_query_groups,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

    def get_query_key_value_tensors(self, *args, **kwargs):
        query, key, value = super().get_query_key_value_tensors(*args, **kwargs)
        query_shape = query.shape
        key_shape = key.shape
        query = self.q_norm(query.reshape(*query_shape[:-2], -1)).view(query_shape)
        key = self.k_norm(key.reshape(*key_shape[:-2], -1)).view(key_shape)
        return query, key, value


class MinimaxM2Bridge(GPTBridge):

    def _set_qk_layernorm(self, mg_attn, hf_attn, hf_state_dict, to_mcore):
        hf_q_norm_key = 'q_norm.weight' if hasattr(hf_attn, 'q_norm') else 'query_layernorm.weight'
        hf_k_norm_key = 'k_norm.weight' if hasattr(hf_attn, 'k_norm') else 'key_layernorm.weight'
        self._set_state_dict(mg_attn, 'q_norm.weight', hf_state_dict, hf_q_norm_key, to_mcore)
        self._set_state_dict(mg_attn, 'k_norm.weight', hf_state_dict, hf_k_norm_key, to_mcore)

    def get_hf_mlp_prefix(self, layer_idx):
        return 'block_sparse_moe'

    def get_e_score_correction_bias_key(self, hf_mlp):
        return 'e_score_correction_bias'

    def _set_moe_state(
        self,
        mg_mlp,
        hf_state_dict,
        hf_prefix: str,
        layer_idx: int,
        to_mcore: bool,
    ):
        if to_mcore:
            hf_state_dict = {
                k.replace('.w1.', '.gate_proj.').replace('.w3.', '.up_proj.').replace('.w2.', '.down_proj.'): v
                for k, v in hf_state_dict.items()
            }
        hf_state_dict = super()._set_moe_state(mg_mlp, hf_state_dict, hf_prefix, layer_idx, to_mcore)
        if not to_mcore:
            hf_state_dict = {
                k.replace('.gate_proj.', '.w1.').replace('.up_proj.', '.w3.').replace('.down_proj.', '.w2.'): v
                for k, v in hf_state_dict.items()
            }
        return hf_state_dict


def get_minimax_m2_transformer_layer_spec(config, vp_stage=None):
    kwargs = {'use_kitchen': config.use_kitchen} if mcore_013 else {}
    layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        qk_layernorm=config.qk_layernorm,
        multi_latent_attention=config.multi_latent_attention,
        moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
        **kwargs,
    )
    layer_spec.submodules.self_attention.module = MinimaxM2SelfAttention
    return layer_spec


register_megatron_model(
    MegatronModelMeta(
        MegatronModelType.minimax_m2,
        [
            ModelType.minimax_m2,
        ],
        get_transformer_layer_spec=get_minimax_m2_transformer_layer_spec,
        bridge_cls=MinimaxM2Bridge,
    ))

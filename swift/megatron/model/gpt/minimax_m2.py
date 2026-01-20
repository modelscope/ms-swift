# Copyright (c) Alibaba, Inc. and its affiliates.

import megatron.core
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.tensor_parallel.mappings import (gather_from_tensor_model_parallel_region,
                                                    scatter_to_tensor_model_parallel_region)
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training import get_args
from packaging import version

from swift.llm import ModelType
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

    def get_query_key_value_tensors(self, *_args, **kwargs):
        args = get_args()
        query, key, value = super().get_query_key_value_tensors(*_args, **kwargs)
        query = query.reshape(*query.shape[:-2], -1)
        key = key.reshape(*key.shape[:-2], -1)
        if args.tensor_model_parallel_size > 1:
            query = gather_from_tensor_model_parallel_region(query)
            key = gather_from_tensor_model_parallel_region(key)
        query = self.q_norm(query)
        key = self.k_norm(key)
        if args.tensor_model_parallel_size > 1:
            query = scatter_to_tensor_model_parallel_region(query)
            key = scatter_to_tensor_model_parallel_region(key)
        query = query.view(*query.shape[:2], -1, self.hidden_size_per_attention_head)
        key = key.view(*key.shape[:2], -1, self.hidden_size_per_attention_head)
        return query, key, value


class MinimaxM2Bridge(GPTBridge):

    def _set_qk_layernorm(self, mg_attn, hf_attn, hf_state_dict, to_mcore):
        self._set_state_dict(mg_attn, 'q_norm.weight', hf_state_dict, 'q_norm.weight', to_mcore)
        self._set_state_dict(mg_attn, 'k_norm.weight', hf_state_dict, 'k_norm.weight', to_mcore)

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

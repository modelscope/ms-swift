# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Optional

import megatron.core
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.transformer import transformer_layer
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.mlp import MLP, apply_swiglu_sharded_factory
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import sharded_state_dict_default
from packaging import version

from swift.llm import ModelType
from ..constant import MegatronModelType
from ..gpt_bridge import GPTBridge
from ..register import MegatronModelMeta, register_megatron_model

mcore_013 = version.parse(megatron.core.__version__) >= version.parse('0.13.0rc0')


class Glm4SelfAttention(SelfAttention):

    def __init__(
        self,
        config: TransformerConfig,
        *args,
        **kwargs,
    ):
        super().__init__(config, *args, **kwargs)
        self.post_self_attn_layernorm = build_module(
            TENorm,
            hidden_size=self.config.hidden_size,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

    def forward(self, hidden_states, *args, **kwargs):
        output, bias = super().forward(hidden_states, *args, **kwargs)
        assert bias is None, 'not support'
        output = self.post_self_attn_layernorm(output)
        return output, bias


class Glm4MLP(MLP):

    def __init__(
        self,
        config: TransformerConfig,
        *args,
        **kwargs,
    ):
        super().__init__(config, *args, **kwargs)
        self.post_mlp_layernorm = build_module(
            TENorm,
            hidden_size=self.config.hidden_size,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

    def forward(self, hidden_states, *args, **kwargs):
        output, bias = super().forward(hidden_states, *args, **kwargs)
        assert bias is None, 'not support'
        output = self.post_mlp_layernorm(output)
        return output, bias

    def sharded_state_dict(self,
                           prefix: str = '',
                           sharded_offsets: tuple = (),
                           metadata: Optional[dict] = None) -> ShardedStateDict:
        """Return the sharded state dictionary of the module."""
        sharded_state_dict = {}
        singleton_local_shards = (metadata or {}).get('singleton_local_shards', False)
        for name, module in self._modules.items():
            sub_sd = sharded_state_dict_default(module, f'{prefix}{name}.', sharded_offsets, metadata)
            if self.config.gated_linear_unit and name == 'linear_fc1':
                for k, v in sub_sd.items():
                    if k in (f'{prefix}{name}.weight', f'{prefix}{name}.bias'):
                        sub_sd[k] = apply_swiglu_sharded_factory(v, sharded_offsets, singleton_local_shards)
            sharded_state_dict.update(sub_sd)
        return sharded_state_dict


class Glm4Bridge(GPTBridge):

    def _set_layer_attn(self, mg_layer, hf_state_dict, layer_idx: int, to_mcore: bool):
        hf_state_dict.update(super()._set_layer_attn(mg_layer, hf_state_dict, layer_idx, to_mcore))
        self._set_state_dict(mg_layer, 'self_attention.post_self_attn_layernorm.weight', hf_state_dict,
                             'post_self_attn_layernorm.weight', to_mcore)
        self._set_state_dict(mg_layer, 'mlp.post_mlp_layernorm.weight', hf_state_dict, 'post_mlp_layernorm.weight',
                             to_mcore)
        return hf_state_dict


def get_glm4_transformer_layer_spec(config, vp_stage=None):
    kwargs = {'use_kitchen': config.use_kitchen} if mcore_013 else {}
    layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        qk_layernorm=config.qk_layernorm,
        multi_latent_attention=config.multi_latent_attention,
        moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
        **kwargs,
    )
    layer_spec.submodules.self_attention.module = Glm4SelfAttention
    layer_spec.submodules.mlp.module = Glm4MLP
    transformer_layer.MLP = Glm4MLP  # patch
    return layer_spec


register_megatron_model(
    MegatronModelMeta(
        MegatronModelType.glm4,
        [
            ModelType.glm4_0414,
            ModelType.glm4_z1_rumination,
        ],
        get_transformer_layer_spec=get_glm4_transformer_layer_spec,
        bridge_cls=Glm4Bridge,
    ))

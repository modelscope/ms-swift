from copy import deepcopy
from typing import Optional

import megatron.core
import torch
import torch.distributed as dist
from megatron.core.extensions.transformer_engine import SplitAlongDim, TENorm
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.transformer.attention import SelfAttention as SelfAttentionBase
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules, get_num_layers_to_build
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from packaging import version

from swift.megatron.tuners import LoraParallelLinear
from swift.model import ModelType
from ..constant import MegatronModelType
from ..gpt_bridge import GPTBridge
from ..register import MegatronModelMeta, register_megatron_model

mcore_013 = version.parse(megatron.core.__version__) >= version.parse('0.13.0rc0')


class OLMoESelfAttention(SelfAttentionBase):

    def __init__(self, config: TransformerConfig, submodules: SelfAttentionSubmodules, *args, **kwargs):
        super().__init__(config, submodules, *args, **kwargs)
        self.q_layernorm = build_module(
            submodules.q_layernorm,
            hidden_size=self.hidden_size_per_attention_head * self.num_attention_heads_per_partition,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )
        self.k_layernorm = build_module(
            submodules.k_layernorm,
            hidden_size=self.hidden_size_per_attention_head * self.num_query_groups_per_partition,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None, *args, **kwargs):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        mixed_qkv, _ = self.linear_qkv(hidden_states)

        # [sq, b, ng * (np/ng + 2) * hn] -> [sq, b, np * hn], [sq, b, ng * hn], [sq, b, ng * hn]
        split_arg_list = [
            self.hidden_size_per_attention_head * self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head * self.num_query_groups_per_partition,
            self.hidden_size_per_attention_head * self.num_query_groups_per_partition
        ]

        if SplitAlongDim is not None:
            (query, key, value) = SplitAlongDim(mixed_qkv, 2, split_arg_list)
        else:
            (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=2)
        if self.q_layernorm is not None:
            query = self.q_layernorm(query)

        if self.k_layernorm is not None:
            key = self.k_layernorm(key)
        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)
        key = key.reshape(key.size(0), key.size(1), -1, self.hidden_size_per_attention_head)
        value = value.reshape(value.size(0), value.size(1), -1, self.hidden_size_per_attention_head)

        if self.config.test_mode:
            self.run_realtime_tests()

        return query, key, value


def get_olmoe_decoder_block_spec(
    config: TransformerConfig,
    vp_stage: Optional[int] = None,
) -> TransformerBlockSubmodules:
    """GPT block spec."""
    layer_norm_impl = TENorm
    kwargs = {'use_kitchen': config.use_kitchen} if mcore_013 else {}
    moe_layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        qk_layernorm=True,
        multi_latent_attention=False,
        moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
        **kwargs,
    )
    layer_specs = []
    for _ in range(config.num_layers):
        layer_spec = deepcopy(moe_layer_spec)
        layer_spec.submodules.self_attention.module = OLMoESelfAttention
        layer_specs.append(layer_spec)

    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)

    if config.pipeline_model_parallel_layout is not None:
        from megatron.core.transformer.enums import LayerType
        local_layer_specs = [
            layer_specs[layer_id] for layer_id in config.pipeline_model_parallel_layout.get_layer_id_list(
                layer_type=LayerType.decoder, vp_stage=vp_stage)
        ]
    else:
        offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
        local_layer_specs = layer_specs[offset:offset + num_layers_to_build]

    # Block spec.
    block_spec = TransformerBlockSubmodules(layer_specs=local_layer_specs, layer_norm=layer_norm_impl)

    return block_spec


class OLMoEBridge(GPTBridge):

    def _set_attn_state(self, mg_attn, hf_state_dict, hf_prefix: str, layer_idx: int, to_mcore: bool):
        if to_mcore:
            hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
        else:
            hf_state_dict = {}
        hf_attn = self.hf_layers[layer_idx].self_attn
        args = self.args
        if to_mcore:
            if isinstance(mg_attn.linear_qkv, LoraParallelLinear):
                lora_A = hf_state_dict['q_proj.lora_A.weight'].load()
                assert (lora_A == hf_state_dict['k_proj.lora_A.weight'].load()).all() and (
                    lora_A == hf_state_dict['v_proj.lora_A.weight'].load()
                ).all(), 'Need to ensure QKV\'s lora_A are consistent'
                lora_B = torch.cat([
                    hf_state_dict['q_proj.lora_B.weight'].load(),
                    hf_state_dict['k_proj.lora_B.weight'].load(),
                    hf_state_dict['v_proj.lora_B.weight'].load(),
                ],
                                   dim=0)
                self._set_weight(mg_attn.linear_qkv.lora_A[self._adapter_name].weight, lora_A,
                                 'linear_qkv.lora_A.weight')
                self._set_weight(mg_attn.linear_qkv.lora_B[self._adapter_name].weight, lora_B,
                                 'linear_qkv.lora_B.weight')
            elif not self._is_peft_format:
                linear_qkv_weight = torch.cat([
                    hf_state_dict['q_proj.weight'].load(),
                    hf_state_dict['k_proj.weight'].load(),
                    hf_state_dict['v_proj.weight'].load(),
                ],
                                              dim=0)
                qkv_scale_inv = None
                if 'q_proj.weight_scale_inv' in hf_state_dict:
                    qkv_scale_inv = torch.cat([
                        hf_state_dict['q_proj.weight_scale_inv'].load(),
                        hf_state_dict['k_proj.weight_scale_inv'].load(),
                        hf_state_dict['v_proj.weight_scale_inv'].load(),
                    ],
                                              dim=0)
                self._set_weight(
                    mg_attn.linear_qkv.weight, linear_qkv_weight, 'linear_qkv.weight', hf_scale_inv=qkv_scale_inv)
        else:
            q_dim, kv_dim = hf_attn.q_proj.weight.shape[0], hf_attn.k_proj.weight.shape[0]
            q_block = q_dim // self.fp8_block_size
            kv_block = kv_dim // self.fp8_block_size
            is_lora = False if mg_attn is None else isinstance(mg_attn.linear_qkv,
                                                               LoraParallelLinear) and self._is_peft_format
            is_lora = torch.tensor([is_lora], dtype=torch.bool, device='cuda')
            if self.pp_size > 1:
                dist.all_reduce(is_lora, group=self.pp_group)
            if is_lora:
                lora_A, _ = self._get_weight(
                    None if mg_attn is None else mg_attn.linear_qkv.lora_A[self._adapter_name].weight.data,
                    f'linear_qkv.lora_A.{self._adapter_name}.weight')
                lora_B, _ = self._get_weight(
                    None if mg_attn is None else mg_attn.linear_qkv.lora_B[self._adapter_name].weight.data,
                    f'linear_qkv.lora_B.{self._adapter_name}.weight')
                if lora_A is not None:
                    self._peft_target_modules.update({'q_proj', 'k_proj', 'v_proj'})
                    for key in ['q_proj', 'k_proj', 'v_proj']:
                        hf_state_dict[f'{key}.lora_A.weight'] = lora_A.clone()
                    hf_state_dict['q_proj.lora_B.weight'] = lora_B[:q_dim, :].clone()
                    hf_state_dict['k_proj.lora_B.weight'] = lora_B[q_dim:-kv_dim, :].clone()
                    hf_state_dict['v_proj.lora_B.weight'] = lora_B[-kv_dim:, :].clone()
            elif not self._is_peft_format:
                mg_attn_weight, scale_inv = self._get_weight(
                    None if mg_attn is None else mg_attn.linear_qkv.weight.data, 'linear_qkv.weight')
                if mg_attn_weight is not None:
                    hf_state_dict['q_proj.weight'] = mg_attn_weight[:q_dim, :].clone()
                    hf_state_dict['k_proj.weight'] = mg_attn_weight[q_dim:-kv_dim, :].clone()
                    hf_state_dict['v_proj.weight'] = mg_attn_weight[-kv_dim:, :].clone()
                if scale_inv is not None:
                    hf_state_dict['q_proj.weight_scale_inv'] = scale_inv[:q_block, :].clone()
                    hf_state_dict['k_proj.weight_scale_inv'] = scale_inv[q_block:-kv_block, :].clone()
                    hf_state_dict['v_proj.weight_scale_inv'] = scale_inv[-kv_block:, :].clone()
                del mg_attn_weight
        self._set_state_dict(mg_attn, 'linear_proj.weight', hf_state_dict, 'o_proj.weight', to_mcore)
        if args.add_qkv_bias and not self._is_peft_format:
            if to_mcore:
                linear_qkv_bias = torch.cat([
                    hf_state_dict['q_proj.bias'].load(),
                    hf_state_dict['k_proj.bias'].load(),
                    hf_state_dict['v_proj.bias'].load(),
                ],
                                            dim=0)
                self._set_weight(mg_attn.linear_qkv.bias, linear_qkv_bias, 'linear_qkv.bias')
            else:
                mg_attn_bias, _ = self._get_weight(None if mg_attn is None else mg_attn.linear_qkv.bias.data,
                                                   'linear_qkv.bias')
                if mg_attn_bias is not None:
                    hf_state_dict['q_proj.bias'] = mg_attn_bias[:q_dim].clone()
                    hf_state_dict['k_proj.bias'] = mg_attn_bias[q_dim:-kv_dim].clone()
                    hf_state_dict['v_proj.bias'] = mg_attn_bias[-kv_dim:].clone()
        hf_q_norm_key = 'q_norm.weight' if hasattr(hf_attn, 'q_norm') else 'query_layernorm.weight'
        hf_k_norm_key = 'k_norm.weight' if hasattr(hf_attn, 'k_norm') else 'key_layernorm.weight'
        self._set_state_dict(mg_attn, 'q_layernorm.weight', hf_state_dict, hf_q_norm_key, to_mcore)
        self._set_state_dict(mg_attn, 'k_layernorm.weight', hf_state_dict, hf_k_norm_key, to_mcore)
        if to_mcore:
            hf_state_dict = {}
        else:
            hf_state_dict = self._add_prefix(hf_state_dict, hf_prefix)
        return hf_state_dict


register_megatron_model(
    MegatronModelMeta(
        MegatronModelType.olmoe,
        [ModelType.olmoe],
        get_transformer_layer_spec=get_olmoe_decoder_block_spec,
        bridge_cls=OLMoEBridge,
    ))

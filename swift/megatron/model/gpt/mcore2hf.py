# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Optional

import torch
from megatron.training import get_args
from torch import nn
from tqdm import tqdm


def set_mla_attn_state(args, state_dict, prefix: str):
    hf_state_dict = {}
    hf_state_dict['o_proj.weight'] = state_dict['linear_proj.weight']
    if args.q_lora_rank is None:
        hf_state_dict['q_proj.weight'] = state_dict['linear_q_proj.weight']
    else:
        hf_state_dict['q_a_proj.weight'] = state_dict['linear_q_down_proj.weight']
        hf_state_dict['q_b_proj.weight'] = state_dict['linear_q_up_proj.weight']
    hf_state_dict['kv_a_proj_with_mqa.weight'] = state_dict['linear_kv_down_proj.weight']
    hf_state_dict['kv_b_proj.weight'] = state_dict['linear_kv_up_proj.weight']
    if args.qk_layernorm:
        hf_state_dict['kv_a_layernorm.weight'] = state_dict['linear_kv_up_proj.layer_norm_weight']
    return _add_prefix(hf_state_dict, prefix)


def set_attn_state(args, state_dict, prefix: str):
    hf_state_dict = {}
    num_query_groups = (args.num_query_groups if args.group_query_attention else args.num_attention_heads)
    mg_attn_weight = state_dict['linear_qkv.weight'].reshape((num_query_groups, -1, args.hidden_size))
    q_dim = args.kv_channels * args.num_attention_heads
    kv_dim = args.kv_channels * args.num_query_groups
    hf_state_dict['q_proj.weight'] = mg_attn_weight[:, :q_dim, :].reshape(-1, args.hidden_size)
    hf_state_dict['k_proj.weight'] = mg_attn_weight[:, q_dim:-kv_dim, :].reshape(-1, args.hidden_size)
    hf_state_dict['v_proj.weight'] = mg_attn_weight[:, -kv_dim:, :].reshape(-1, args.hidden_size)
    hf_state_dict['o_proj.weight'] = state_dict['linear_proj.weight']

    # Copy bias
    if args.add_qkv_bias:
        mg_attn_bias = state_dict['linear_qkv.bias'].reshape((num_query_groups, -1))
        state_dict['q_proj.bias'] = mg_attn_bias[:, :q_dim].reshape(-1)
        state_dict['k_proj.bias'] = mg_attn_bias[:, q_dim:-kv_dim].reshape(-1)
        state_dict['v_proj.bias'] = mg_attn_bias[:, -kv_dim:].reshape(-1)
    if args.qk_layernorm:
        hf_state_dict['q_norm.weight'] = state_dict['q_layernorm.weight']
        hf_state_dict['query_layernorm.weight'] = state_dict['q_layernorm.weight']
        hf_state_dict['k_norm.weight'] = hf_state_dict['k_layernorm.weight']
        hf_state_dict['key_layernorm.weight'] = hf_state_dict['k_layernorm.weight']

    return _add_prefix(mg_state_dict, prefix)


def _set_mlp_state(
    args,
    state_dict,
    prefix: str,
    group_idx: Optional[int] = None,
    hf_grouped: bool = False,
):
    mg_state_dict = {}
    if group_idx is None:
        fc1_key = 'linear_fc1.weight'
        fc2_key = 'linear_fc2.weight'
    else:
        fc1_key = f'linear_fc1.weight{group_idx}'
        fc2_key = f'linear_fc2.weight{group_idx}'
    if hf_grouped:
        mg_state_dict[fc1_key] = state_dict['gate_up_proj'][group_idx].t()
        mg_state_dict[fc2_key] = state_dict['down_proj'][group_idx].t()
    else:
        if 'gate_up_proj.weight' in state_dict:
            mg_state_dict[fc1_key] = state_dict['gate_up_proj.weight']
        else:
            mg_state_dict[fc1_key] = torch.cat([
                state_dict['gate_proj.weight'],
                state_dict['up_proj.weight'],
            ], dim=0)
        mg_state_dict[fc2_key] = state_dict['down_proj.weight']
    return _add_prefix(mg_state_dict, prefix)


def _set_moe_state(args, state_dict, prefix: str):
    mg_state_dict = {}
    if 'gate.wg.weight' in state_dict:
        mg_state_dict['router.weight'] = state_dict['gate.wg.weight']
    else:
        mg_state_dict['router.weight'] = state_dict['gate.weight']
    if args.moe_router_enable_expert_bias:
        mg_state_dict['router.expert_bias'] = state_dict['gate.e_score_correction_bias']

    if args.moe_shared_expert_intermediate_size:
        shared_expert_sd = _remove_prefix(state_dict, 'shared_expert.')
        if not shared_expert_sd:
            shared_expert_sd = _remove_prefix(state_dict, 'shared_experts.')
        if not shared_expert_sd:
            shared_expert_sd = _remove_prefix(state_dict, 'shared_mlp.')
        mg_state_dict.update(_set_mlp_state(args, shared_expert_sd, 'shared_experts.'))
        if 'shared_expert_gate.weight' in state_dict:
            mg_state_dict['shared_experts.gate_weight'] = state_dict['shared_expert_gate.weight']
    for expert_idx in range(args.num_experts):
        expert_sd = _remove_prefix(state_dict, 'experts.')
        hf_grouped = expert_sd is not None
        if expert_sd is None:
            expert_sd = _remove_prefix(state_dict, f'experts.{expert_idx}.')
        mg_state_dict.update(_set_mlp_state(args, expert_sd, 'experts.', group_idx=expert_idx, hf_grouped=hf_grouped))
    return _add_prefix(mg_state_dict, prefix)


def _is_moe(state_dict):
    for k, v in state_dict.items():
        if 'experts.' in k:
            return True
    return False


def set_layer_state(args, state_dict, prefix: str):
    hf_state_dict = {}
    if args.multi_latent_attention:
        hf_state_dict.update(set_mla_attn_state(args, _remove_prefix(state_dict, ), 'self_attn.'))
        hf_state_dict['input_layernorm.weight'] = mg_state_dict['input_layernorm.weight']

    else:
        hf_state_dict.update(set_attn_state(args, _remove_prefix(state_dict, 'self_attention.'), 'self_attn.'))
        hf_state_dict['input_layernorm.weight'] = state_dict['self_attention.linear_qkv.layer_norm_weight']

    mlp_state_dict = _remove_prefix(state_dict, 'mlp.')
    is_moe = _is_moe(mlp_state_dict)
    if is_moe:
        hf_state_dict.update(_set_moe_state(args, mlp_state_dict, 'mlp.'))
    else:
        hf_state_dict.update(_set_mlp_state(args, mlp_state_dict, 'mlp.'))

    if is_moe:
        hf_state_dict['post_attention_layernorm.weight'] = state_dict['pre_mlp_layernorm.weight']
    else:
        hf_state_dict['post_attention_layernorm.weight'] = state_dict['mlp.linear_fc1.layer_norm_weight']
    return _add_prefix(hf_state_dict, prefix)


def _remove_prefix(state_dict, prefix: str):
    if not prefix:
        return state_dict
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}


def _add_prefix(state_dict, prefix: str):
    if not prefix:
        return state_dict
    return {f'{prefix}{k}': v for k, v in state_dict.items()}


def convert_mcore2hf(state_dict, prefix=''):
    args = get_args()
    hf_state_dict = {}
    hf_state_dict['model.embed_tokens.weight'] = state_dict['embedding.word_embeddings.weight']
    if args.untie_embeddings_and_output_weights:
        hf_state_dict['lm_head.weight'] = state_dict['output_layer.weight']
    hf_state_dict['model.norm.weight'] = state_dict['decoder.final_layernorm.weight']
    for layer_idx in tqdm(range(args.num_layers), dynamic_ncols=True, desc='Converting: '):
        hf_state_dict.update(
            set_layer_state(args, _remove_prefix(state_dict, f'decoder.layers.{layer_idx}.'),
                            f'model.layers.{layer_idx}.'))
    return _add_prefix(hf_state_dict, prefix)

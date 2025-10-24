from typing import Any, Dict, Optional

import torch
from megatron.training import get_args
from tqdm import tqdm

from swift.llm import deep_getattr, get_model_tokenizer
from swift.utils import disable_safe_ddp_context_use_barrier


class GPTBridge:
    lm_layers_prefix = 'model.layers'  # hf

    def __init__(self):
        self.args = get_args()
        model_info = self.args.model_info
        with torch.device('meta'), disable_safe_ddp_context_use_barrier():
            self.hf_model, _ = get_model_tokenizer(
                model_info.model_dir, model_type=model_info.model_type, return_dummy_model=True)
        self.hf_layers = deep_getattr(self.hf_model, self.lm_layers_prefix)

    def _set_state_dict(self, state_dict, res_state_dict, hf_key: str, mg_key: str, reverse: bool):
        src_key, tgt_key = hf_key, mg_key
        if reverse:
            src_key, tgt_key = tgt_key, src_key
        res_state_dict[tgt_key] = state_dict[src_key]

    @staticmethod
    def _remove_prefix(state_dict, prefix: str):
        if not prefix:
            return state_dict
        return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

    @staticmethod
    def _add_prefix(state_dict, prefix: str):
        if not prefix:
            return state_dict
        return {f'{prefix}{k}': v for k, v in state_dict.items()}

    @staticmethod
    def _is_moe(state_dict):
        for k, v in state_dict.items():
            if 'experts.' in k:
                return True
        return False

    def _set_attn_state(self, state_dict, hf_prefix: str, mg_prefix: str, hf_attn, reverse: bool):
        src_prefix, tgt_prefix = hf_prefix, mg_prefix
        if reverse:
            src_prefix, tgt_prefix = tgt_prefix, src_prefix
        state_dict = self._remove_prefix(state_dict, src_prefix)
        args = self.args
        res = {}
        num_query_groups = (args.num_query_groups if args.group_query_attention else args.num_attention_heads)
        if reverse:
            pass
        else:
            res['linear_qkv.weight'] = torch.cat([
                state_dict['q_proj.weight'].reshape((num_query_groups, -1, args.hidden_size)),
                state_dict['k_proj.weight'].reshape((num_query_groups, -1, args.hidden_size)),
                state_dict['v_proj.weight'].reshape((num_query_groups, -1, args.hidden_size)),
            ],
                                                 dim=1).reshape((-1, args.hidden_size))
        self._set_state_dict(state_dict, res, 'o_proj.weight', 'linear_proj.weight', reverse)

        # Copy bias
        if args.add_qkv_bias:
            if reverse:
                pass
            else:
                res['linear_qkv.bias'] = torch.cat([
                    state_dict['q_proj.bias'].reshape((num_query_groups, -1)),
                    state_dict['k_proj.bias'].reshape((num_query_groups, -1)),
                    state_dict['v_proj.bias'].reshape((num_query_groups, -1)),
                ],
                                                   dim=1).reshape(-1)
        if args.qk_layernorm:
            if 'q_norm.weight' in state_dict:
                res['q_layernorm.weight'] = state_dict['q_norm.weight']
            else:
                res['q_layernorm.weight'] = state_dict['query_layernorm.weight']
            if 'k_norm.weight' in state_dict:
                res['k_layernorm.weight'] = state_dict['k_norm.weight']
            else:
                res['k_layernorm.weight'] = state_dict['key_layernorm.weight']

        return self._add_prefix(res, tgt_prefix)

    def _set_moe_state(self, state_dict, prefix: str):
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
            mg_state_dict.update(
                _set_mlp_state(args, expert_sd, 'experts.', group_idx=expert_idx, hf_grouped=hf_grouped))
        return _add_prefix(mg_state_dict, prefix)

    def _set_mlp_state(
        self,
        state_dict,
        hf_prefix: str,
        mg_prefix: str,
        hf_mlp,
        reverse: bool,
        group_idx: Optional[int] = None,
        hf_grouped: bool = False,
    ):
        src_prefix, tgt_prefix = hf_prefix, mg_prefix
        if reverse:
            src_prefix, tgt_prefix = tgt_prefix, src_prefix
        state_dict = self._remove_prefix(state_dict, src_prefix)
        res = {}
        # Determines the keys for fc1 and fc2 in megatron
        if group_idx is None:
            fc1_key = 'linear_fc1.weight'
            fc2_key = 'linear_fc2.weight'
        else:
            fc1_key = f'linear_fc1.weight{group_idx}'
            fc2_key = f'linear_fc2.weight{group_idx}'
        if hf_grouped:
            res[fc1_key] = state_dict['gate_up_proj'][group_idx].t()
            res[fc2_key] = state_dict['down_proj'][group_idx].t()
        else:
            if hasattr(hf_mlp, 'gate_up_proj'):
                self._set_state_dict(state_dict, res, 'gate_up_proj.weight', fc1_key, reverse)
            else:
                if reverse:
                    pass
                else:
                    res[fc1_key] = torch.cat([
                        state_dict['gate_proj.weight'],
                        state_dict['up_proj.weight'],
                    ], dim=0)
            self._set_state_dict(state_dict, res, 'down_proj.weight', fc2_key, reverse)
        return self._add_prefix(res, tgt_prefix)

    def _set_layer_state(self, state_dict, layer_idx: int, hf_prefix: str, mg_prefix: str, reverse: bool):
        hf_prefix = f'{hf_prefix}{layer_idx}.'
        mg_prefix = f'{mg_prefix}{layer_idx}.'
        hf_layer = self.hf_layers[layer_idx]
        hf_attn, hf_mlp = hf_layer.self_attn, hf_layer.mlp
        src_prefix, tgt_prefix = hf_prefix, mg_prefix
        if reverse:
            src_prefix, tgt_prefix = tgt_prefix, src_prefix
        state_dict = self._remove_prefix(state_dict, src_prefix)
        res = {}
        if self.args.multi_latent_attention:
            res.update(self._set_mla_attn_state(state_dict, 'self_attn.', 'self_attention.', reverse))
            self._set_state_dict(state_dict, res, 'input_layernorm.weight', 'input_layernorm.weight', reverse)
        else:
            res.update(self._set_attn_state(state_dict, 'self_attn.', 'self_attention.', hf_attn, reverse))
            self._set_state_dict(state_dict, res, 'input_layernorm.weight',
                                 'self_attention.linear_qkv.layer_norm_weight', reverse)

        is_moe = self._is_moe(hf_mlp.state_dict())
        if is_moe:
            res.update(self._set_moe_state(state_dict, 'mlp.'))
        else:
            res.update(self._set_mlp_state(state_dict, 'mlp.', 'mlp.', hf_mlp, reverse))

        if is_moe:
            res['pre_mlp_layernorm.weight'] = state_dict['post_attention_layernorm.weight']
        else:
            res['mlp.linear_fc1.layer_norm_weight'] = state_dict['post_attention_layernorm.weight']
        return self._add_prefix(res, tgt_prefix)

    def _convert(self, state_dict, hf_prefix: str, mg_prefix: str, reverse: bool):
        src_prefix, tgt_prefix = hf_prefix, mg_prefix
        if reverse:
            src_prefix, tgt_prefix = tgt_prefix, src_prefix
        state_dict = self._remove_prefix(state_dict, src_prefix)
        res = {}
        self._set_state_dict(state_dict, res, 'model.embed_tokens.weight', 'embedding.word_embeddings.weight', reverse)
        if self.args.untie_embeddings_and_output_weights:
            self._set_state_dict(state_dict, res, 'lm_head.weight', 'output_layer.weight', reverse)
        self._set_state_dict(state_dict, res, 'model.norm.weight', 'decoder.final_layernorm.weight', reverse)
        for layer_idx in tqdm(range(self.args.num_layers), dynamic_ncols=True, desc='Converting: '):
            res.update(self._set_layer_state(state_dict, layer_idx, 'model.layers.', 'decoder.layers.', reverse))
        return self._add_prefix(res, tgt_prefix)

    def convert_hf2mcore(self, state_dict):
        return self._convert(state_dict, '', '', False)

    def convert_mcore2hf(self, state_dict):
        return self._convert(state_dict, '', '', True)

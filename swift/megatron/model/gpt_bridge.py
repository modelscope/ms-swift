from typing import Dict, Optional

import torch
from megatron.training import get_args
from tqdm import tqdm

from swift.llm import deep_getattr, get_model_tokenizer
from swift.utils import disable_safe_ddp_context_use_barrier


class GPTBridge:
    lm_layers_prefix = 'model.layers'  # HF model

    def __init__(self):
        self.args = get_args()
        model_info = self.args.model_info
        with torch.device('meta'), disable_safe_ddp_context_use_barrier():
            self.hf_model, _ = get_model_tokenizer(
                model_info.model_dir, model_type=model_info.model_type, return_dummy_model=True)
        self.hf_layers = deep_getattr(self.hf_model, self.lm_layers_prefix)

    def _set_state_dict(self, state_dict, res_state_dict, hf_key: str, mg_key: str, reverse: bool, offset: float = 0):
        src_key, tgt_key = hf_key, mg_key
        if reverse:
            src_key, tgt_key = tgt_key, src_key
        res_state_dict[tgt_key] = state_dict[src_key]
        if offset:
            res_state_dict[tgt_key] = res_state_dict[tgt_key] + offset

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
    def _filter_prefix(state_dict, prefix: str):
        if not prefix:
            return state_dict
        return {k: v for k, v in state_dict.items() if k.startswith(prefix)}

    @staticmethod
    def _replace_prefix(state_dict, hf_prefix: str, mg_prefix: str, reverse: bool):
        src_prefix, tgt_prefix = hf_prefix, mg_prefix
        if reverse:
            src_prefix, tgt_prefix = tgt_prefix, src_prefix
        res = GPTBridge._remove_prefix(state_dict, src_prefix)
        return GPTBridge._add_prefix(res, tgt_prefix)

    @staticmethod
    def _is_moe(state_dict):
        for k, v in state_dict.items():
            if 'experts.' in k:
                return True
        return False

    def _set_attn_state(self, state_dict, hf_prefix: str, mg_prefix: str, layer_idx: int, reverse: bool):
        src_prefix, tgt_prefix = hf_prefix, mg_prefix
        if reverse:
            src_prefix, tgt_prefix = tgt_prefix, src_prefix
        state_dict = self._remove_prefix(state_dict, src_prefix)
        hf_attn = self.hf_layers[layer_idx].self_attn
        args = self.args
        res = {}
        num_query_groups = (args.num_query_groups if args.group_query_attention else args.num_attention_heads)
        if reverse:
            mg_attn_weight = state_dict['linear_qkv.weight'].reshape((num_query_groups, -1, args.hidden_size))
            q_dim, kv_dim = hf_attn.q_proj.weight.shape[0] // num_query_groups, hf_attn.k_proj.weight.shape[
                0] // num_query_groups
            res['q_proj.weight'] = mg_attn_weight[:, :q_dim, :].reshape(-1, args.hidden_size)
            res['k_proj.weight'] = mg_attn_weight[:, q_dim:-kv_dim, :].reshape(-1, args.hidden_size)
            res['v_proj.weight'] = mg_attn_weight[:, -kv_dim:, :].reshape(-1, args.hidden_size)
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
                mg_attn_bias = state_dict['linear_qkv.bias'].reshape((num_query_groups, -1))
                res['q_proj.bias'] = mg_attn_bias[:, :q_dim].reshape(-1)
                res['k_proj.bias'] = mg_attn_bias[:, q_dim:-kv_dim].reshape(-1)
                res['v_proj.bias'] = mg_attn_bias[:, -kv_dim:].reshape(-1)
            else:
                res['linear_qkv.bias'] = torch.cat([
                    state_dict['q_proj.bias'].reshape((num_query_groups, -1)),
                    state_dict['k_proj.bias'].reshape((num_query_groups, -1)),
                    state_dict['v_proj.bias'].reshape((num_query_groups, -1)),
                ],
                                                   dim=1).reshape(-1)
        if args.qk_layernorm:
            hf_q_norm_key = 'q_norm.weight' if hasattr(hf_attn, 'q_norm') else 'query_layernorm.weight'
            hf_k_norm_key = 'k_norm.weight' if hasattr(hf_attn, 'k_norm') else 'key_layernorm.weight'
            self._set_state_dict(state_dict, res, hf_q_norm_key, 'q_layernorm.weight', reverse)
            self._set_state_dict(state_dict, res, hf_k_norm_key, 'k_layernorm.weight', reverse)

        return self._add_prefix(res, tgt_prefix)

    def _set_moe_state(self, state_dict, hf_prefix: str, mg_prefix: str, layer_idx: int, reverse: bool):
        src_prefix, tgt_prefix = hf_prefix, mg_prefix
        if reverse:
            src_prefix, tgt_prefix = tgt_prefix, src_prefix
        state_dict = self._remove_prefix(state_dict, src_prefix)
        hf_mlp = self.hf_layers[layer_idx].mlp
        res = {}
        hf_gate_key = 'gate.wg.weight' if hasattr(hf_mlp.gate, 'wg') else 'gate.weight'
        self._set_state_dict(state_dict, res, hf_gate_key, 'router.weight', reverse)
        if self.args.moe_router_enable_expert_bias:
            self._set_state_dict(state_dict, res, 'gate.e_score_correction_bias', 'router.expert_bias', reverse)

        if self.args.moe_shared_expert_intermediate_size:
            for key in ['shared_expert', 'shared_experts', 'shared_mlp']:
                if hasattr(hf_mlp, key):
                    hf_shared_expert_prefix = f'{key}.'
            res.update(self._set_mlp_state(state_dict, hf_shared_expert_prefix, 'shared_experts.', layer_idx, reverse))
            if hasattr(hf_mlp, 'shared_expert_gate'):
                self._set_state_dict(state_dict, res, 'shared_expert_gate.weight', 'shared_experts.gate_weight',
                                     reverse)
        for expert_idx in range(self.args.num_experts):
            hf_expert_prefix = f'experts.{expert_idx}.' if hasattr(hf_mlp.experts, '__len__') else 'experts.'
            res.update(
                self._set_mlp_state(state_dict, hf_expert_prefix, 'experts.', layer_idx, reverse, group_idx=expert_idx))
        return self._add_prefix(res, tgt_prefix)

    def _set_mlp_state(
        self,
        state_dict,
        hf_prefix: str,
        mg_prefix: str,
        layer_idx: int,
        reverse: bool,
        group_idx: Optional[int] = None,
    ):
        src_prefix, tgt_prefix = hf_prefix, mg_prefix
        if reverse:
            src_prefix, tgt_prefix = tgt_prefix, src_prefix
        state_dict = self._remove_prefix(state_dict, src_prefix)
        hf_mlp = self.hf_layers[layer_idx].mlp
        hf_grouped = False
        if group_idx is not None and not hasattr(hf_mlp.experts, '__len__'):
            hf_grouped = True
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
                    ffn_hidden_size = state_dict[fc1_key].shape[0] // 2
                    res['gate_proj.weight'] = state_dict[fc1_key][:ffn_hidden_size]
                    res['up_proj.weight'] = state_dict[fc1_key][ffn_hidden_size:]
                else:
                    res[fc1_key] = torch.cat([
                        state_dict['gate_proj.weight'],
                        state_dict['up_proj.weight'],
                    ], dim=0)
            self._set_state_dict(state_dict, res, 'down_proj.weight', fc2_key, reverse)
        return self._add_prefix(res, tgt_prefix)

    def _set_mla_attn_state(
        self,
        state_dict,
        hf_prefix: str,
        mg_prefix: str,
        layer_idx: int,
        reverse: bool,
    ):
        src_prefix, tgt_prefix = hf_prefix, mg_prefix
        if reverse:
            src_prefix, tgt_prefix = tgt_prefix, src_prefix
        state_dict = self._remove_prefix(state_dict, src_prefix)
        res = {}
        self._set_state_dict(state_dict, res, 'o_proj.weight', 'linear_proj.weight', reverse)
        if self.args.q_lora_rank is None:
            self._set_state_dict(state_dict, res, 'q_proj.weight', 'linear_q_proj.weight', reverse)
        else:
            self._set_state_dict(state_dict, res, 'q_a_proj.weight', 'linear_q_down_proj.weight', reverse)
            self._set_state_dict(state_dict, res, 'q_b_proj.weight', 'linear_q_up_proj.weight', reverse)
        self._set_state_dict(state_dict, res, 'kv_a_proj_with_mqa.weight', 'linear_kv_down_proj.weight', reverse)
        self._set_state_dict(state_dict, res, 'kv_b_proj.weight', 'linear_kv_up_proj.weight', reverse)
        if self.args.qk_layernorm:
            self._set_state_dict(state_dict, res, 'kv_a_layernorm.weight', 'linear_kv_up_proj.layer_norm_weight',
                                 reverse)
        return self._add_prefix(res, tgt_prefix)

    def _set_layer_attn(self, state_dict, layer_idx: int, reverse: bool):
        res = {}
        if self.args.multi_latent_attention:
            res.update(self._set_mla_attn_state(state_dict, 'self_attn.', 'self_attention.', layer_idx, reverse))
            self._set_state_dict(state_dict, res, 'input_layernorm.weight', 'input_layernorm.weight', reverse)
        else:
            res.update(self._set_attn_state(state_dict, 'self_attn.', 'self_attention.', layer_idx, reverse))
            self._set_state_dict(state_dict, res, 'input_layernorm.weight',
                                 'self_attention.linear_qkv.layer_norm_weight', reverse)
        return res

    def _set_layer_mlp(self, state_dict, layer_idx: int, reverse: bool):
        hf_mlp = self.hf_layers[layer_idx].mlp
        res = {}
        is_moe = self._is_moe(hf_mlp.state_dict())
        if is_moe:
            res.update(self._set_moe_state(state_dict, 'mlp.', 'mlp.', layer_idx, reverse))
            self._set_state_dict(state_dict, res, 'post_attention_layernorm.weight', 'pre_mlp_layernorm.weight',
                                 reverse)
        else:
            res.update(self._set_mlp_state(state_dict, 'mlp.', 'mlp.', layer_idx, reverse))
            self._set_state_dict(state_dict, res, 'post_attention_layernorm.weight', 'mlp.linear_fc1.layer_norm_weight',
                                 reverse)
        return res

    def _set_layer_state(self, state_dict, layer_idx: int, hf_prefix: str, mg_prefix: str, reverse: bool):
        hf_prefix = f'{hf_prefix}{layer_idx}.'
        mg_prefix = f'{mg_prefix}{layer_idx}.'
        src_prefix, tgt_prefix = hf_prefix, mg_prefix
        if reverse:
            src_prefix, tgt_prefix = tgt_prefix, src_prefix
        state_dict = self._remove_prefix(state_dict, src_prefix)
        res = self._set_layer_attn(state_dict, layer_idx, reverse)
        res.update(self._set_layer_mlp(state_dict, layer_idx, reverse))
        return self._add_prefix(res, tgt_prefix)

    def _convert(self, state_dict, hf_prefix: str, mg_prefix: str, reverse: bool):
        """reverse: False: hf -> mg; True: mg -> hf"""
        src_prefix, tgt_prefix = hf_prefix, mg_prefix
        if reverse:
            src_prefix, tgt_prefix = tgt_prefix, src_prefix
        state_dict = self._remove_prefix(state_dict, src_prefix)
        res = {}
        self._set_state_dict(state_dict, res, 'model.embed_tokens.weight', 'embedding.word_embeddings.weight', reverse)
        if self.args.untie_embeddings_and_output_weights:
            hf_lm_head_key = 'lm_head.weight'
            if reverse and self.args.task_type == 'seq_cls':
                hf_lm_head_key = 'score.weight'
            self._set_state_dict(state_dict, res, hf_lm_head_key, 'output_layer.weight', reverse)
        self._set_state_dict(state_dict, res, 'model.norm.weight', 'decoder.final_layernorm.weight', reverse)
        for layer_idx in tqdm(range(self.args.num_layers), dynamic_ncols=True, desc='Converting: '):
            res.update(self._set_layer_state(state_dict, layer_idx, 'model.layers.', 'decoder.layers.', reverse))
        return self._add_prefix(res, tgt_prefix)

    def convert_hf2mcore(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self._convert(state_dict, '', '', False)

    def convert_mcore2hf(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self._convert(state_dict, '', '', True)

    def load_state_dict(self, model, state_dict) -> None:
        """The model can be either hf_model or mg_model"""
        incompatible_keys = model.load_state_dict(state_dict, strict=False)
        missing_keys = [k for k in incompatible_keys.missing_keys if not k.endswith('._extra_state')]
        assert len(incompatible_keys.unexpected_keys) == 0, f'unexpected_keys: {incompatible_keys.unexpected_keys}'
        assert len(missing_keys) == 0, f'missing_keys: {missing_keys}'

    def load_from_hf_checkpoint(self, mg_model, hf_model_dir: str) -> None:
        """按照mg_model的模型结构, 加载需要的参数，并进行scatter"""
        print()

    def get_hf_state_dict(self, mg_models) -> Dict[str, torch.Tensor]:
        """获取完整的hf state_dict"""
        print()

    def save_hf_checkpoint(self, mg_models, output_dir: str) -> None:
        """保存mg_model的hf格式checkpoint"""
        state_dict = get_hf_state_dict(mg_models)
        # rank0 save()


from typing import Any, Dict
import torch
from megatron.training import get_args
from swift.llm import get_model_tokenizer, deep_getattr
from swift.utils import disable_safe_ddp_context_use_barrier

class GPTBridge:
    lm_layers_prefix = 'model.layers'  # hf

    def __init__(self):
        self.args = get_args()
        model_info = self.args.model_info
        with torch.device('meta'), disable_safe_ddp_context_use_barrier():
            self.hf_model, _ = get_model_tokenizer(
                model_info.model_dir,
                model_type=model_info.model_type,
                return_dummy_model=True)
        self.hf_layers = deep_getattr(self.hf_model, self.lm_layers_prefix)


    def _set_state_dict(self, state_dict, res_state_dict, hf_key: str, mg_key: str, reverse: bool = False):
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

    def _set_layer_state(state_dict, hf_prefix: str, mg_prefix: str, reverse: bool = False):
        src_prefix, tgt_prefix = hf_prefix, mg_prefix
        if reverse:
            src_prefix, tgt_prefix = tgt_prefix, src_prefix
        state_dict = self._remove_prefix(state_dict)
        res = {}
        if args.multi_latent_attention:
            mg_state_dict.update(set_mla_attn_state(args, _remove_prefix(state_dict, 'self_attn.'), 'self_attention.'))
            mg_state_dict['input_layernorm.weight'] = state_dict['input_layernorm.weight']

        else:
            mg_state_dict.update(set_attn_state(args, _remove_prefix(state_dict, 'self_attn.'), 'self_attention.'))
            mg_state_dict['self_attention.linear_qkv.layer_norm_weight'] = state_dict['input_layernorm.weight']

        mlp_state_dict = _remove_prefix(state_dict, 'mlp.')
        is_moe = _is_moe(mlp_state_dict)
        if is_moe:
            mg_state_dict.update(_set_moe_state(args, mlp_state_dict, 'mlp.'))
        else:
            mg_state_dict.update(_set_mlp_state(args, mlp_state_dict, 'mlp.'))

        if is_moe:
            mg_state_dict['pre_mlp_layernorm.weight'] = state_dict['post_attention_layernorm.weight']
        else:
            mg_state_dict['mlp.linear_fc1.layer_norm_weight'] = state_dict['post_attention_layernorm.weight']
        return _add_prefix(mg_state_dict, prefix)


    def convert_hf2mcore(self, state_dict, prefix: str = '', reverse: bool = False):
        res = {}
        self._set_state_dict(state_dict, res, 'model.embed_tokens.weight', 'embedding.word_embeddings.weight', reverse)
        if args.untie_embeddings_and_output_weights:
            self._set_state_dict(state_dict, res, 'lm_head.weight', 'output_layer.weight', reverse)
        self._set_state_dict(state_dict, res, 'model.norm.weight', 'decoder.final_layernorm.weight', reverse)
        for layer_idx in tqdm(range(args.num_layers), dynamic_ncols=True, desc='Converting: '):
            mg_state_dict.update(
                self._set_layer_state(state_dict, f'model.layers.{layer_idx}.',
                                f'decoder.layers.{layer_idx}.', reverse))



    def convert_mcore2hf(self, state_dict, prefix: str = ''):
        return self.convert_hf2mcore(state_dict, prefix, True)

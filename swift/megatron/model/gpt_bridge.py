from functools import partial
from typing import Dict, Literal, Optional, Union

import torch
import torch.distributed as dist
from megatron.core import mpu
from megatron.training import get_args
from tqdm import tqdm

from swift.llm import deep_getattr, get_model_tokenizer, save_checkpoint
from swift.utils import disable_safe_ddp_context_use_barrier, is_last_rank
from ..utils import LazyTensor, SafetensorLazyLoader, StreamingSafetensorSaver


class GPTBridge:
    lm_layers_prefix = 'model.layers'  # HF model

    def __init__(self):
        self.args = get_args()
        model_info = self.args.model_info
        with torch.device('meta'), disable_safe_ddp_context_use_barrier():
            self.hf_model, self.processor = get_model_tokenizer(
                model_info.model_dir, model_type=model_info.model_type, return_dummy_model=True)
        self.hf_layers = deep_getattr(self.hf_model, self.lm_layers_prefix)
        self.tp_size = self.args.tensor_model_parallel_size
        self.pp_size = self.args.pipeline_model_parallel_size
        self.etp_size = self.args.expert_tensor_parallel_size
        self.ep_size = self.args.expert_model_parallel_size

        self.tp_group = mpu.get_tensor_model_parallel_group()
        self.pp_group = mpu.get_pipeline_model_parallel_group()
        self.etp_group = mpu.get_expert_tensor_parallel_group()
        self.ep_group = mpu.get_expert_model_parallel_group()

        self.tp_rank = mpu.get_tensor_model_parallel_rank()
        self.pp_rank = mpu.get_pipeline_model_parallel_rank()
        self.etp_rank = mpu.get_expert_tensor_parallel_rank()
        self.ep_rank = mpu.get_expert_model_parallel_rank()

    @staticmethod
    def _get_tp_split_dim(mg_key: str) -> Optional[int]:
        key, suffix = mg_key.rsplit('.', 2)[-2:]
        if suffix == 'layer_norm_weight':
            return
        if key in {'word_embeddings', 'output_layer', 'linear_qkv', 'linear_fc1'}:
            return 0
        elif key in {'linear_proj', 'linear_fc2'}:
            return 1

    def _set_weights(self, mg_param, hf_weight, mg_key: str, offset: float = 0, is_expert: bool = False, mg_slices=()):
        # tp/etp
        tp_dim = self._get_tp_split_dim(mg_key)
        hf_weight = hf_weight.to(mg_param.device)
        tp_size = self.etp_size if is_expert else self.tp_size
        tp_rank = self.etp_rank if is_expert else self.tp_rank
        tp_group = self.etp_group if is_expert else self.tp_group
        if tp_dim is not None and tp_size > 1:
            if tp_rank == 0:
                splited_weights = [t.contiguous() for t in hf_weight.chunk(tp_size, dim=tp_dim)]
            else:
                splited_weights = None
            tensor = torch.empty_like(mg_param.data[mg_slices])
            dist.scatter(
                tensor,
                splited_weights,
                src=dist.get_global_rank(tp_group, 0),
                group=tp_group,
            )
            del splited_weights
        else:
            tensor = hf_weight
        if offset:
            tensor = tensor + offset
        mg_param.data[mg_slices].copy_(tensor)

    def _get_weights(self, mg_weight, mg_key, offset: int = 0, is_expert: bool = False):
        # tp/etp
        tp_dim = self._get_tp_split_dim(mg_key)
        tensor = mg_weight
        tp_size = self.etp_size if is_expert else self.tp_size
        tp_group = self.etp_group if is_expert else self.tp_group
        if tensor is not None and tp_dim is not None and tp_size > 1:
            if tp_dim == 0:
                # save memory
                tensor_shape = list(tensor.shape)
                tensor_shape[0] *= tp_size
                output = tensor.new_empty(tensor_shape)
                dist.all_gather_into_tensor(
                    output,
                    tensor,
                    group=tp_group,
                )
                tensor = output
            else:
                output = [torch.empty_like(tensor) for _ in range(tp_size)]
                dist.all_gather(
                    output,
                    tensor,
                    group=tp_group,
                )
                tensor = torch.cat(output, dim=tp_dim)
            del output
        # pp
        if self.pp_size > 1:
            if tensor is None:
                output = [None] * self.pp_size
                dist.all_gather_object(output, None, group=self.pp_group)
                src_idx = self._find_not_none_index(output)
                assert len(src_idx) == 1, f'src_idx: {src_idx}'
                src_idx = src_idx[0]
                shape, dtype = output[src_idx]
                tensor = torch.empty(shape, device='cuda', dtype=dtype)
                dist.broadcast(tensor, src=dist.get_global_rank(self.pp_group, src_idx), group=self.pp_group)
            else:
                output = [None] * self.pp_size
                meta_data = (tensor.shape, tensor.dtype)
                dist.all_gather_object(output, meta_data, group=self.pp_group)
                dist.broadcast(tensor, src=dist.get_global_rank(self.pp_group, self.pp_rank), group=self.pp_group)
        if offset:
            tensor = tensor + offset
        return tensor

    @staticmethod
    def _find_not_none_index(lst):
        res = []
        for i, x in enumerate(lst):
            if x is not None:
                res.append(i)
        return res

    def _set_state_dict(self,
                        mg_module,
                        mg_key: str,
                        hf_state_dict,
                        hf_key: str,
                        reverse: bool,
                        offset: float = 0,
                        is_expert: bool = False):
        mg_param = deep_getattr(mg_module, mg_key)
        if reverse:
            hf_state_dict[hf_key] = self._get_weights(None if mg_param is None else mg_param.data, mg_key, offset,
                                                      is_expert)
        else:
            assert mg_param is not None, f'mg_module: {mg_module}, mg_key: {mg_key}'
            hf_weight = hf_state_dict[hf_key].load()
            self._set_weights(mg_param, hf_weight, mg_key, offset, is_expert)

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

    def _set_attn_state(self, mg_attn, hf_state_dict, hf_prefix: str, layer_idx: int, reverse: bool):
        if reverse:
            hf_state_dict = {}
        else:
            hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
        hf_attn = self.hf_layers[layer_idx].self_attn
        args = self.args
        num_query_groups = (args.num_query_groups if args.group_query_attention else args.num_attention_heads)
        if reverse:
            mg_attn_weight = self._get_weights(None if mg_attn is None else mg_attn.linear_qkv.weight.data,
                                               'linear_qkv.weight')
            mg_attn_weight = mg_attn_weight.reshape((num_query_groups, -1, args.hidden_size))
            q_dim, kv_dim = hf_attn.q_proj.weight.shape[0] // num_query_groups, hf_attn.k_proj.weight.shape[
                0] // num_query_groups
            hf_state_dict['q_proj.weight'] = mg_attn_weight[:, :q_dim, :].reshape(-1, args.hidden_size)
            hf_state_dict['k_proj.weight'] = mg_attn_weight[:, q_dim:-kv_dim, :].reshape(-1, args.hidden_size)
            hf_state_dict['v_proj.weight'] = mg_attn_weight[:, -kv_dim:, :].reshape(-1, args.hidden_size)
        else:
            linear_qkv_weight = torch.cat([
                hf_state_dict['q_proj.weight'].load().reshape((num_query_groups, -1, args.hidden_size)),
                hf_state_dict['k_proj.weight'].load().reshape((num_query_groups, -1, args.hidden_size)),
                hf_state_dict['v_proj.weight'].load().reshape((num_query_groups, -1, args.hidden_size)),
            ],
                                          dim=1).reshape((-1, args.hidden_size))
            self._set_weights(mg_attn.linear_qkv.weight, linear_qkv_weight, 'linear_qkv.weight')
        self._set_state_dict(mg_attn, 'linear_proj.weight', hf_state_dict, 'o_proj.weight', reverse)

        # Copy bias
        if args.add_qkv_bias:
            if reverse:
                mg_attn_bias = self._get_weights(None if mg_attn is None else mg_attn.linear_qkv.bias.data,
                                                 'linear_qkv.bias')
                mg_attn_bias = mg_attn_bias.reshape((num_query_groups, -1))
                hf_state_dict['q_proj.bias'] = mg_attn_bias[:, :q_dim].reshape(-1)
                hf_state_dict['k_proj.bias'] = mg_attn_bias[:, q_dim:-kv_dim].reshape(-1)
                hf_state_dict['v_proj.bias'] = mg_attn_bias[:, -kv_dim:].reshape(-1)
            else:
                linear_qkv_bias = torch.cat([
                    hf_state_dict['q_proj.bias'].load().reshape((num_query_groups, -1)),
                    hf_state_dict['k_proj.bias'].load().reshape((num_query_groups, -1)),
                    hf_state_dict['v_proj.bias'].load().reshape((num_query_groups, -1)),
                ],
                                            dim=1).reshape(-1)
                self._set_weights(mg_attn.linear_qkv.bias, linear_qkv_bias, 'linear_qkv.bias')

        if args.qk_layernorm:
            hf_q_norm_key = 'q_norm.weight' if hasattr(hf_attn, 'q_norm') else 'query_layernorm.weight'
            hf_k_norm_key = 'k_norm.weight' if hasattr(hf_attn, 'k_norm') else 'key_layernorm.weight'
            self._set_state_dict(mg_attn, 'q_layernorm.weight', hf_state_dict, hf_q_norm_key, reverse)
            self._set_state_dict(mg_attn, 'k_layernorm.weight', hf_state_dict, hf_k_norm_key, reverse)
        if reverse:
            hf_state_dict = self._add_prefix(hf_state_dict, hf_prefix)
        return hf_state_dict

    def _set_moe_state(
        self,
        mg_mlp,
        hf_state_dict,
        hf_prefix: str,
        layer_idx: int,
        reverse: bool,
    ):
        if reverse:
            hf_state_dict = {}
        else:
            hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
        hf_mlp = self.hf_layers[layer_idx].mlp
        hf_gate_key = 'gate.wg.weight' if hasattr(hf_mlp.gate, 'wg') else 'gate.weight'
        self._set_state_dict(mg_mlp, 'router.weight', hf_state_dict, hf_gate_key, reverse)
        if self.args.moe_router_enable_expert_bias:
            self._set_state_dict(mg_mlp, 'router.expert_bias', hf_state_dict, 'gate.e_score_correction_bias', reverse)

        if self.args.moe_shared_expert_intermediate_size:
            for key in ['shared_expert', 'shared_experts', 'shared_mlp']:
                if hasattr(hf_mlp, key):
                    hf_shared_expert_prefix = f'{key}.'
            hf_state_dict.update(
                self._set_mlp_state(mg_mlp.shared_experts, hf_state_dict, hf_shared_expert_prefix, layer_idx, reverse))
            if hasattr(hf_mlp, 'shared_expert_gate'):
                self._set_state_dict(mg_mlp, 'shared_experts.gate_weight', hf_state_dict, 'shared_expert_gate.weight',
                                     reverse)
        for expert_idx in range(self.args.num_experts):
            mg_experts = mg_mlp.experts
            start_idx = mg_experts.num_local_experts * self.ep_rank
            expert_available = (start_idx <= expert_idx < start_idx + mg_experts.num_local_experts)
            if expert_available:
                group_idx = expert_idx - start_idx
            else:
                group_idx = None
                if reverse:
                    mg_experts = None
                else:
                    continue
            if hasattr(hf_mlp.experts, '__len__'):
                hf_expert_prefix = f'experts.{expert_idx}.'
                hf_group_idx = None
            else:
                hf_expert_prefix = 'experts.'
                hf_group_idx = expert_idx
            hf_state_dict.update(
                self._set_mlp_state(
                    mg_experts,
                    hf_state_dict,
                    hf_expert_prefix,
                    layer_idx,
                    reverse,
                    group_idx=group_idx,
                    hf_group_idx=hf_group_idx))
        if reverse:
            hf_state_dict = self._add_prefix(hf_state_dict, hf_prefix)
        return hf_state_dict

    def _set_mlp_state(
        self,
        mg_mlp,
        hf_state_dict,
        hf_prefix: str,
        layer_idx: int,
        reverse: bool,
        group_idx: Optional[int] = None,
        hf_group_idx: Optional[int] = None,
    ):
        if reverse:
            hf_state_dict = {}
        else:
            hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
        hf_mlp = self.hf_layers[layer_idx].mlp
        is_expert = group_idx is not None
        if group_idx is not None:
            fc1_key = f'linear_fc1.weight{group_idx}'
            fc2_key = f'linear_fc2.weight{group_idx}'
        else:
            fc1_key = 'linear_fc1.weight'
            fc2_key = 'linear_fc2.weight'
        if hf_group_idx is not None:
            res[fc1_key] = hf_state_dict['gate_up_proj'][hf_group_idx].t()
            res[fc2_key] = hf_state_dict['down_proj'][hf_group_idx].t()
        else:
            if reverse:
                fc1_weight = deep_getattr(mg_mlp, fc1_key)
                gate_proj_weight = self._get_weights(
                    None if fc1_weight is None else fc1_weight[:fc1_weight.shape[0] // 2], fc1_key, is_expert=is_expert)
                up_proj_weight = self._get_weights(
                    None if fc1_weight is None else fc1_weight[fc1_weight.shape[0] // 2:], fc1_key, is_expert=is_expert)
                if hasattr(hf_mlp, 'gate_up_proj'):
                    hf_state_dict['gate_up_proj'] = torch.concat([gate_proj_weight, up_proj_weight], dim=0)
                else:
                    hf_state_dict['gate_proj.weight'] = gate_proj_weight
                    hf_state_dict['up_proj.weight'] = up_proj_weight
            else:
                linear_fc1_weight = deep_getattr(mg_mlp, fc1_key)
                gate_slices = (slice(None, linear_fc1_weight.shape[0] // 2), )
                up_slices = (slice(linear_fc1_weight.shape[0] // 2, None), )
                if hasattr(hf_mlp, 'gate_up_proj'):
                    gate_up_proj_weight = hf_state_dict['gate_up_proj.weight'].load()
                    gate_proj_weight = gate_up_proj_weight[gate_slices]
                    up_proj_weight = gate_up_proj_weight[up_slices]
                else:
                    gate_proj_weight = hf_state_dict['gate_proj.weight'].load()
                    up_proj_weight = hf_state_dict['up_proj.weight'].load()
                self._set_weights(
                    linear_fc1_weight, gate_proj_weight, fc1_key, is_expert=is_expert, mg_slices=gate_slices)
                self._set_weights(linear_fc1_weight, up_proj_weight, fc1_key, is_expert=is_expert, mg_slices=up_slices)

            self._set_state_dict(mg_mlp, fc2_key, hf_state_dict, 'down_proj.weight', reverse, is_expert=is_expert)
        if reverse:
            hf_state_dict = self._add_prefix(hf_state_dict, hf_prefix)
        return hf_state_dict

    def _set_mla_attn_state(
        self,
        mg_model,
        mg_prefix: str,
        state_dict,
        hf_prefix: str,
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

    def _set_layer_attn(self, mg_layer, hf_state_dict, layer_idx: int, reverse: bool):
        mg_attn = None if mg_layer is None else mg_layer.self_attention
        if self.args.multi_latent_attention:
            hf_state_dict.update(self._set_mla_attn_state(mg_attn, hf_state_dict, 'self_attn.', layer_idx, reverse))
            self._set_state_dict(mg_layer, 'input_layernorm.weight', hf_state_dict, 'input_layernorm.weight', reverse)
        else:
            hf_state_dict.update(self._set_attn_state(mg_attn, hf_state_dict, 'self_attn.', layer_idx, reverse))
            self._set_state_dict(mg_layer, 'self_attention.linear_qkv.layer_norm_weight', hf_state_dict,
                                 'input_layernorm.weight', reverse)
        return hf_state_dict

    def _set_layer_mlp(self, mg_layer, hf_state_dict, layer_idx: int, reverse: bool):
        hf_mlp = self.hf_layers[layer_idx].mlp
        is_moe = self._is_moe(hf_mlp.state_dict())
        mg_mlp = None if mg_layer is None else mg_layer.mlp
        if is_moe:
            hf_state_dict.update(self._set_moe_state(mg_mlp, hf_state_dict, 'mlp.', layer_idx, reverse))
            self._set_state_dict(mg_layer, 'pre_mlp_layernorm.weight', hf_state_dict, 'post_attention_layernorm.weight',
                                 reverse)
        else:
            hf_state_dict.update(self._set_mlp_state(mg_mlp, hf_state_dict, 'mlp.', layer_idx, reverse))
            self._set_state_dict(mg_layer, 'mlp.linear_fc1.layer_norm_weight', hf_state_dict,
                                 'post_attention_layernorm.weight', reverse)
        return hf_state_dict

    def _set_layer_state(self, mg_layer, hf_state_dict, hf_prefix: str, layer_idx: int, reverse: bool):
        hf_prefix = f'{hf_prefix}{layer_idx}.'
        if reverse:
            hf_state_dict = {}
        else:
            hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
        hf_state_dict.update(self._set_layer_attn(mg_layer, hf_state_dict, layer_idx, reverse))
        hf_state_dict.update(self._set_layer_mlp(mg_layer, hf_state_dict, layer_idx, reverse))
        if reverse:
            hf_state_dict = self._add_prefix(hf_state_dict, hf_prefix)
        return hf_state_dict

    def _convert(self, mg_model, hf_state_dict, hf_prefix: str, reverse: bool):
        """reverse: False: hf -> mg; True: mg -> hf"""
        if reverse:
            hf_state_dict = {}
        else:
            hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
        if reverse or mpu.is_pipeline_first_stage(ignore_virtual=False, vp_stage=mg_model.vp_stage):
            self._set_state_dict(mg_model, 'embedding.word_embeddings.weight', hf_state_dict,
                                 'model.embed_tokens.weight', reverse)
        if reverse:
            yield from list(self._add_prefix(hf_state_dict, hf_prefix).items())
            hf_state_dict = {}
        else:
            yield
        for layer_idx in tqdm(range(self.args.num_layers), dynamic_ncols=True, desc='Converting: '):
            start_idx = mg_model.decoder.layers[0].layer_number - 1
            mg_layer_available = (start_idx <= layer_idx < mg_model.decoder.layers[-1].layer_number)
            if mg_layer_available:
                mg_layer = mg_model.decoder.layers[layer_idx - start_idx]
            else:
                if reverse:
                    mg_layer = None
                else:
                    continue
            res = self._set_layer_state(mg_layer, hf_state_dict, 'model.layers.', layer_idx, reverse)
            if reverse:
                yield from list(self._add_prefix(res, hf_prefix).items())
                hf_state_dict = {}
            else:
                yield
        if reverse or mpu.is_pipeline_last_stage(ignore_virtual=False, vp_stage=mg_model.vp_stage):
            if self.args.untie_embeddings_and_output_weights:
                hf_lm_head_key = 'lm_head.weight'
                if reverse and self.args.task_type == 'seq_cls':
                    hf_lm_head_key = 'score.weight'
                self._set_state_dict(mg_model, 'output_layer.weight', hf_state_dict, hf_lm_head_key, reverse)
            self._set_state_dict(mg_model, 'decoder.final_layernorm.weight', hf_state_dict, 'model.norm.weight',
                                 reverse)
        if reverse:
            yield from list(self._add_prefix(hf_state_dict, hf_prefix).items())
            hf_state_dict = {}
        else:
            yield

    def load_weights(self, mg_model, hf_model_dir: str) -> None:
        with SafetensorLazyLoader(hf_model_dir) as loader:
            state_dict = loader.get_state_dict()
            list(self._convert(mg_model, state_dict, '', False))

    def export_weights(self, mg_models):
        state_dict = {}
        for mg_model in mg_models:
            yield from self._convert(mg_model, state_dict, '', True)

    def save_weights(self, mg_models, output_dir: str) -> None:
        """Save the mg_model checkpoint in HF format"""
        saver = StreamingSafetensorSaver(save_dir=output_dir, max_shard_size=self.args.max_shard_size)
        for k, v in self.export_weights(mg_models):
            saver.add_tensor(k, v)
        saver.finalize()
        if is_last_rank():
            # TODO: new_special_tokens
            self.hf_model.config.save_pretrained(output_dir)
            save_checkpoint(
                None,
                self.processor,
                output_dir,
                model_dirs=[self.hf_model.model_info.model_dir],
                additional_saved_files=self.hf_model.model_meta.additional_saved_files)

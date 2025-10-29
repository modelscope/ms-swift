from typing import Dict, Literal, Optional, Union

import torch
import torch.distributed as dist
from megatron.core import mpu
from megatron.training import get_args
from tqdm import tqdm

from swift.llm import deep_getattr, get_model_tokenizer, save_checkpoint
from swift.utils import disable_safe_ddp_context_use_barrier, is_last_rank, get_logger
from ..utils import LazyTensor, SafetensorLazyLoader, StreamingSafetensorSaver

logger = get_logger()

class GPTBridge:
    lm_layers_prefix = 'model.layers'  # HF model

    def __init__(self, disable_tqmd: bool = False):
        self.args = get_args()
        self.disable_tqmd = disable_tqmd
        self._target_device = None
        self.only_last_rank = False
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
        if key in {'word_embeddings', 'output_layer', 'linear_qkv'}:
            return 0
        elif key in {'linear_proj', 'linear_fc1', 'linear_fc2'}:
            # linear_fc1 shape [2, X, Y]
            return 1

    def _set_weights(
        self,
        mg_param: torch.Tensor,
        hf_weight: torch.Tensor,
        mg_key: str,
        offset: float = 0,
        is_expert: bool = False,
    ):
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
            tensor = torch.empty_like(mg_param.data)
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
        mg_param.data.copy_(tensor)

    def _get_weights(self, mg_weight: torch.Tensor, mg_key: str, offset: int = 0, is_expert: bool = False):
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
        # pp/ep
        for parallel_state in ['ep', 'pp']:
            if parallel_state == 'pp' and self.pp_size > 1:
                parallel_group = self.pp_group
                parallel_rank = self.pp_rank
            elif parallel_state == 'ep' and is_expert and self.ep_size > 1:
                parallel_group = self.ep_group
                parallel_rank = self.ep_rank
            else:
                continue
            src_rank = torch.tensor([0 if tensor is None else parallel_rank], dtype=torch.int64, device='cuda')
            dist.all_reduce(src_rank, group=parallel_group)
            src_rank = dist.get_global_rank(parallel_group, src_rank.item())
            meta_data = torch.zeros(10, dtype=torch.int64, device='cuda')
            dtype_mapping = {torch.float64: 0, torch.float32: 1, torch.float16: 2, torch.bfloat16: 3}
            dtype_mapping_r = {v: k for k, v in dtype_mapping.items()}
            if tensor is None:
                dist.broadcast(meta_data, src=src_rank, group=parallel_group)
                shape = meta_data[1:1 + meta_data[0]].tolist()
                dtype = dtype_mapping_r[meta_data[-1].item()]
                tensor = torch.empty(shape, device='cuda', dtype=dtype)
                dist.broadcast(tensor, src=src_rank, group=parallel_group)
            else:
                meta_data[0] = tensor.ndim
                meta_data[1:1 + tensor.ndim] = torch.tensor(tensor.shape, dtype=torch.int64, device='cuda')
                meta_data[-1] = dtype_mapping[tensor.dtype]
                dist.broadcast(meta_data, src=src_rank, group=parallel_group)
                dist.broadcast(tensor, src=src_rank, group=parallel_group)
        assert tensor is not None, f'mg_key: {mg_key}'
        if offset:
            tensor = tensor + offset
        if self._target_device is not None:
            tensor = tensor.to(device=self._target_device)
        if self._only_last_rank and not is_last_rank():
            tensor = None
        return tensor

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
            if mg_attn_weight is not None:
                mg_attn_weight = mg_attn_weight.reshape((num_query_groups, -1, args.hidden_size))
                q_dim, kv_dim = hf_attn.q_proj.weight.shape[0] // num_query_groups, hf_attn.k_proj.weight.shape[
                    0] // num_query_groups
                hf_state_dict['q_proj.weight'] = mg_attn_weight[:, :q_dim, :].reshape(-1, args.hidden_size).clone()
                hf_state_dict['k_proj.weight'] = mg_attn_weight[:, q_dim:-kv_dim, :].reshape(-1,
                                                                                             args.hidden_size).clone()
                hf_state_dict['v_proj.weight'] = mg_attn_weight[:, -kv_dim:, :].reshape(-1, args.hidden_size).clone()
                del mg_attn_weight
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
                if mg_attn_bias is not None:
                    mg_attn_bias = mg_attn_bias.reshape((num_query_groups, -1))
                    hf_state_dict['q_proj.bias'] = mg_attn_bias[:, :q_dim].reshape(-1).clone()
                    hf_state_dict['k_proj.bias'] = mg_attn_bias[:, q_dim:-kv_dim].reshape(-1).clone()
                    hf_state_dict['v_proj.bias'] = mg_attn_bias[:, -kv_dim:].reshape(-1).clone()
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
                    shared_expert = getattr(hf_mlp, key)
            hf_state_dict.update(
                self._set_mlp_state(
                    mg_mlp.shared_experts,
                    hf_state_dict,
                    hf_shared_expert_prefix,
                    layer_idx,
                    reverse,
                    hf_mlp=shared_expert))
            if hasattr(hf_mlp, 'shared_expert_gate'):
                self._set_state_dict(mg_mlp, 'shared_experts.gate_weight', hf_state_dict, 'shared_expert_gate.weight',
                                     reverse)
        for ep_rank in range(self.ep_size):
            mg_experts = mg_mlp.experts
            expert_available = ep_rank == self.ep_rank
            if not expert_available:
                if reverse:
                    mg_experts = None
                else:
                    continue
            hf_state_dict.update(
                self._set_expert_state(mg_experts, hf_state_dict, 'experts.', layer_idx, reverse, ep_rank))
        if reverse:
            hf_state_dict = self._add_prefix(hf_state_dict, hf_prefix)
        return hf_state_dict

    def _set_expert_state(
        self,
        mg_mlp,
        hf_state_dict,
        hf_prefix: str,
        layer_idx: int,
        reverse: bool,
        ep_rank: int,
    ):
        if reverse:
            hf_state_dict = {}
        else:
            hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
        hf_experts = self.hf_layers[layer_idx].mlp.experts
        num_local_experts = self.args.num_experts // self.ep_size
        # if hf_group_idx is not None:
        #     res[fc1_key] = hf_state_dict['gate_up_proj'][hf_group_idx].t()
        #     res[fc2_key] = hf_state_dict['down_proj'][hf_group_idx].t()
        # else:
        if reverse:
            if mg_mlp is None:
                fc1_weight = None
            else:
                fc1_weight = torch.concat([getattr(mg_mlp.linear_fc1, f'weight{i}') for i in range(num_local_experts)],
                                          dim=0)
                fc1_weight = fc1_weight.view(num_local_experts * 2, -1, fc1_weight.shape[1])
            gate_up_proj_weight = self._get_weights(fc1_weight, 'linear_fc1.weight', is_expert=True)
            del fc1_weight
            if gate_up_proj_weight is not None:
                gate_up_proj_weight = gate_up_proj_weight.view(num_local_experts, 2, -1, gate_up_proj_weight.shape[-1])
                for i in range(num_local_experts):
                    hf_i = i + ep_rank * num_local_experts
                    if hasattr(hf_experts[i], 'gate_up_proj'):
                        hf_state_dict[f'{hf_i}.gate_up_proj.weight'] = gate_up_proj_weight[i].view(
                            -1, gate_up_proj_weight.shape[-1]).clone()
                    else:
                        hf_state_dict[f'{hf_i}.gate_proj.weight'] = gate_up_proj_weight[i][0].clone()
                        hf_state_dict[f'{hf_i}.up_proj.weight'] = gate_up_proj_weight[i][1].clone()
            del gate_up_proj_weight
        else:
            fc1_weight = mg_mlp.linear_fc1.weight0
            fc1_weight = fc1_weight.new_empty(num_local_experts * 2, fc1_weight.shape[0] // 2, fc1_weight.shape[1])
            if hasattr(hf_experts[0], 'gate_up_proj'):
                gate_up_proj_weight = torch.concat([
                    hf_state_dict[f'{i + ep_rank * num_local_experts}.gate_up_proj.weight'].load()
                    for i in range(num_local_experts)
                ],
                                                   dim=0)
            else:
                weight_list = []
                start_idx = ep_rank * num_local_experts
                for i in range(num_local_experts):
                    gate_proj_weight = hf_state_dict[f'{start_idx + i}.gate_proj.weight'].load()
                    up_proj_weight = hf_state_dict[f'{start_idx + i}.up_proj.weight'].load()
                    weight_list.append(torch.stack([gate_proj_weight, up_proj_weight], dim=0))
                gate_up_proj_weight = torch.concat(weight_list, dim=0)
                del weight_list
            self._set_weights(fc1_weight, gate_up_proj_weight, 'linear_fc1.weight', is_expert=True)
            fc1_weight = fc1_weight.view(num_local_experts, -1, fc1_weight.shape[-1])
            for i in range(num_local_experts):
                getattr(mg_mlp.linear_fc1, f'weight{i}').data.copy_(fc1_weight[i].view(-1, fc1_weight.shape[-1]))
            del fc1_weight
        if reverse:
            if mg_mlp is None:
                fc2_weight = None
            else:
                fc2_weight = torch.concat([getattr(mg_mlp.linear_fc2, f'weight{i}') for i in range(num_local_experts)],
                                          dim=0)
                fc2_weight = fc2_weight.view(num_local_experts * 2, -1, fc2_weight.shape[1])
            down_proj_weight = self._get_weights(fc2_weight, 'linear_fc2.weight', is_expert=True)
            del fc2_weight
            if down_proj_weight is not None:
                down_proj_weight = down_proj_weight.view(num_local_experts, -1, down_proj_weight.shape[-1])
                for i in range(num_local_experts):
                    hf_i = i + ep_rank * num_local_experts
                    hf_state_dict[f'{hf_i}.down_proj.weight'] = down_proj_weight[i].view(
                        -1, down_proj_weight.shape[-1]).clone()
        else:
            fc2_weight = mg_mlp.linear_fc2.weight0
            fc2_weight = fc2_weight.new_empty(num_local_experts * fc2_weight.shape[0], fc2_weight.shape[1])
            down_proj_weight = torch.concat([
                hf_state_dict[f'{i + ep_rank * num_local_experts}.down_proj.weight'].load()
                for i in range(num_local_experts)
            ],
                                            dim=0)
            self._set_weights(fc2_weight, down_proj_weight, 'linear_fc2.weight', is_expert=True)
            fc2_weight = fc2_weight.view(num_local_experts, -1, fc2_weight.shape[-1])
            for i in range(num_local_experts):
                getattr(mg_mlp.linear_fc2, f'weight{i}').data.copy_(fc2_weight[i])
        if reverse:
            hf_state_dict = self._add_prefix(hf_state_dict, hf_prefix)
        return hf_state_dict

    def _set_mlp_state(self, mg_mlp, hf_state_dict, hf_prefix: str, layer_idx: int, reverse: bool, hf_mlp=None):
        if reverse:
            hf_state_dict = {}
        else:
            hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
        if hf_mlp is None:
            hf_mlp = self.hf_layers[layer_idx].mlp
        if reverse:
            if mg_mlp is None:
                fc1_weight = None
            else:
                fc1_weight = mg_mlp.linear_fc1.weight
                fc1_weight = fc1_weight.view(2, fc1_weight.shape[0] // 2, fc1_weight.shape[1])
            gate_up_proj_weight = self._get_weights(None if fc1_weight is None else fc1_weight, 'linear_fc1.weight')
            if gate_up_proj_weight is not None:
                if hasattr(hf_mlp, 'gate_up_proj'):
                    hf_state_dict['gate_up_proj'] = gate_up_proj_weight.view(-1, gate_up_proj_weight.shape[-1]).clone()
                else:
                    hf_state_dict['gate_proj.weight'] = gate_up_proj_weight[0].clone()
                    hf_state_dict['up_proj.weight'] = gate_up_proj_weight[1].clone()
        else:
            fc1_weight = mg_mlp.linear_fc1.weight
            fc1_weight = fc1_weight.new_empty(2, fc1_weight.shape[0] // 2, fc1_weight.shape[1])
            if hasattr(hf_mlp, 'gate_up_proj'):
                gate_up_proj_weight = hf_state_dict['gate_up_proj.weight'].load()
            else:
                gate_proj_weight = hf_state_dict['gate_proj.weight'].load()
                up_proj_weight = hf_state_dict['up_proj.weight'].load()
                gate_up_proj_weight = torch.concat([gate_proj_weight, up_proj_weight], dim=0)
            gate_up_proj_weight = gate_up_proj_weight.view(2, -1, gate_up_proj_weight.shape[-1])
            self._set_weights(fc1_weight, gate_up_proj_weight, 'linear_fc1.weight')
            mg_mlp.linear_fc1.weight.data.copy_(fc1_weight.view(-1, fc1_weight.shape[-1]))
        self._set_state_dict(mg_mlp, 'linear_fc2.weight', hf_state_dict, 'down_proj.weight', reverse)
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

    def _convert(self, mg_models, hf_state_dict, hf_prefix: str, reverse: bool):
        """reverse: False: hf -> mg; True: mg -> hf"""
        if reverse:
            hf_state_dict = {}
        else:
            hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
        mg_models = iter(mg_models)
        mg_model = next(mg_models)
        if reverse or mpu.is_pipeline_first_stage(ignore_virtual=False, vp_stage=mg_model.vp_stage):
            self._set_state_dict(mg_model, 'embedding.word_embeddings.weight', hf_state_dict,
                                 'model.embed_tokens.weight', reverse)
        if reverse:
            yield from list(self._add_prefix(hf_state_dict, hf_prefix).items())
            hf_state_dict = {}
        else:
            yield
        for layer_idx in tqdm(
                range(self.args.num_layers), dynamic_ncols=True, desc='Converting: ', disable=self.disable_tqmd):
            start_idx = mg_model.decoder.layers[0].layer_number - 1
            mg_layer_available = (start_idx <= layer_idx < mg_model.decoder.layers[-1].layer_number)
            if mg_layer_available:
                mg_layer = mg_model.decoder.layers[layer_idx - start_idx]
            else:
                if reverse:
                    mg_layer = None
                else:
                    continue
            if reverse and self.pp_size > 1:
                has_model = torch.tensor([mg_layer is not None], dtype=torch.bool, device='cuda')
                dist.all_reduce(has_model, group=self.pp_group)
                if not has_model:
                    mg_model = next(mg_models)
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
            list(self._convert([mg_model], state_dict, '', False))

    def export_weights(self, mg_models, target_device=None, only_last_rank: bool = False):
        self._target_device = target_device
        self._only_last_rank = only_last_rank
        yield from self._convert(mg_models, {}, '', True)

    def save_weights(self, mg_models, output_dir: str) -> None:
        """Save the mg_model checkpoint in HF format"""
        saver = StreamingSafetensorSaver(save_dir=output_dir, max_shard_size=self.args.max_shard_size)
        for k, v in self.export_weights(mg_models, target_device='cpu', only_last_rank=True):
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
            logger.info_info(f'Successfully saved HF model weights in `{output_dir}`.', cond=is_last_rank())

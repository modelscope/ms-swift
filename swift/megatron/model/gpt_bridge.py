from copy import copy
from typing import Dict, Literal, Optional, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import mpu
from megatron.training import get_args
from peft.utils import ModulesToSaveWrapper
from tqdm import tqdm

from swift.llm import deep_getattr, get_model_tokenizer, save_checkpoint
from swift.utils import disable_safe_ddp_context_use_barrier, get_logger, is_last_rank
from ..tuners import LoraParallelLinear
from ..utils import LazyTensor, SafetensorLazyLoader, StreamingSafetensorSaver

logger = get_logger()


class GPTBridge:
    lm_layers_prefix = 'model.layers'  # HF model

    def __init__(self, disable_tqmd: bool = False):
        self.args = get_args()
        self.disable_tqmd = disable_tqmd
        self._target_device = None
        self._only_last_rank = False
        self._peft_target_modules = set()
        self._peft_modules_to_save = set()
        self._is_peft_format = False
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
        # ColumnLinear
        dim0_keys = {
            'word_embeddings',
            'output_layer',
            'linear_qkv',
            # mla
            'linear_q_proj',
            'linear_kv_down_proj',
            'linear_kv_up_proj'
        }
        # RowLinear
        dim1_keys = {'linear_proj', 'linear_fc2'}
        if 'lora_A' not in mg_key and 'lora_B' not in mg_key:
            key, suffix = mg_key.rsplit('.', 2)[-2:]
            if suffix == 'layer_norm_weight':
                return
            if key in dim0_keys:
                return 0
            elif key in {'linear_fc1'} | dim1_keys:
                # linear_fc1 shape [2, X, Y]
                return 1
        else:
            mg_key_splited = mg_key.rsplit('.', 3)
            key, lora_name = mg_key_splited[:2]
            if lora_name == 'lora_A':
                if key in dim1_keys:
                    return 1
            elif lora_name == 'lora_B':
                if key in dim0_keys:
                    return 0
                elif key in {'linear_fc1'}:
                    return 1

    def _set_weight(
        self,
        mg_param: torch.Tensor,
        hf_weight: torch.Tensor,
        mg_key: str,
        offset: float = 0,
        is_expert: bool = False,
    ):
        # tp/etp
        tp_dim = self._get_tp_split_dim(mg_key)
        hf_weight = hf_weight.to(device=mg_param.device, dtype=mg_param.dtype)
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

    def _get_weight(self, mg_weight: torch.Tensor, mg_key: str, offset: int = 0, is_expert: bool = False):
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
                        to_mcore: bool,
                        offset: float = 0,
                        is_expert: bool = False):
        module_key, param_key = mg_key.rsplit('.', 1)
        hf_module_key, hf_param_key = hf_key.rsplit('.', 1)
        sub_module = deep_getattr(mg_module, module_key)
        if isinstance(sub_module, LoraParallelLinear) and self._is_peft_format and param_key != 'layer_norm_weight':
            if to_mcore:
                lora_A_key = f'{module_key}.lora_A.default.{param_key}'
                lora_B_key = f'{module_key}.lora_B.default.{param_key}'
                mg_lora_A = deep_getattr(mg_module, f'{lora_A_key}')
                mg_lora_B = deep_getattr(mg_module, f'{lora_B_key}')
                hf_lora_A = hf_state_dict[f'{hf_module_key}.lora_A.{hf_param_key}'].load()
                hf_lora_B = hf_state_dict[f'{hf_module_key}.lora_B.{hf_param_key}'].load()
                self._set_weight(mg_lora_A, hf_lora_A, lora_A_key, offset, is_expert)
                self._set_weight(mg_lora_B, hf_lora_B, lora_B_key, offset, is_expert)
            else:
                lora_A_key = f'{module_key}.lora_A.default.{param_key}'
                lora_B_key = f'{module_key}.lora_B.default.{param_key}'
                lora_A_tensor = deep_getattr(mg_module, f'{lora_A_key}.data')
                lora_B_tensor = deep_getattr(mg_module, f'{lora_B_key}.data')
                hf_lora_A_key = f'{hf_module_key}.lora_A.{hf_param_key}'
                hf_lora_B_key = f'{hf_module_key}.lora_B.{hf_param_key}'
                lora_A = self._get_weight(lora_A_tensor, lora_A_key, offset, is_expert)
                lora_B = self._get_weight(lora_B_tensor, lora_B_key, offset, is_expert)
                if lora_A is not None:
                    self._peft_target_modules.add(hf_module_key)
                    hf_state_dict[hf_lora_A_key] = lora_A
                    hf_state_dict[hf_lora_B_key] = lora_B
        elif not self._is_peft_format or isinstance(sub_module, ModulesToSaveWrapper):
            if isinstance(sub_module, LoraParallelLinear):
                mg_param = deep_getattr(sub_module, f'base_layer.{param_key}')
            else:
                mg_param = deep_getattr(sub_module, param_key)
            if to_mcore:
                assert mg_param is not None, f'mg_module: {mg_module}, mg_key: {mg_key}'
                hf_weight = hf_state_dict[hf_key].load()
                if module_key in {'embedding.word_embeddings', 'output_layer'
                                  } and hf_weight.shape[0] < self.args.padded_vocab_size:
                    hf_weight = F.pad(hf_weight, (0, 0, 0, self.args.padded_vocab_size - hf_weight.shape[0]))
                self._set_weight(mg_param, hf_weight, mg_key, offset, is_expert)
            else:
                if isinstance(sub_module, ModulesToSaveWrapper):
                    self._peft_modules_to_save.add(hf_module_key)
                weight = self._get_weight(None if mg_param is None else mg_param.data, mg_key, offset, is_expert)
                if weight is not None:
                    hf_state_dict[hf_key] = weight

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
    def _is_moe(state_dict):
        for k, v in state_dict.items():
            if 'experts.' in k:
                return True
        return False

    def _set_attn_state(self, mg_attn, hf_state_dict, hf_prefix: str, layer_idx: int, to_mcore: bool):
        if to_mcore:
            hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
        else:
            hf_state_dict = {}
        hf_attn = self.hf_layers[layer_idx].self_attn
        args = self.args
        num_query_groups = (args.num_query_groups if args.group_query_attention else args.num_attention_heads)
        if to_mcore:
            if isinstance(mg_attn.linear_qkv, LoraParallelLinear):
                lora_A = hf_state_dict['q_proj.lora_A.weight'].load()
                assert (lora_A == hf_state_dict['k_proj.lora_A.weight'].load()).all() and (
                    lora_A == hf_state_dict['v_proj.lora_A.weight'].load()
                ).all(), 'Need to ensure QKV\'s lora_A are consistent'
                q_lora_B = hf_state_dict['q_proj.lora_B.weight'].load()
                lora_B = torch.cat([
                    q_lora_B.reshape((num_query_groups, -1, q_lora_B.shape[-1])),
                    hf_state_dict['k_proj.lora_B.weight'].load().reshape((num_query_groups, -1, q_lora_B.shape[-1])),
                    hf_state_dict['v_proj.lora_B.weight'].load().reshape((num_query_groups, -1, q_lora_B.shape[-1])),
                ],
                                   dim=1).reshape((-1, q_lora_B.shape[-1]))
                self._set_weight(mg_attn.linear_qkv.lora_A['default'].weight, lora_A, 'linear_qkv.lora_A.weight')
                self._set_weight(mg_attn.linear_qkv.lora_B['default'].weight, lora_B, 'linear_qkv.lora_B.weight')
            else:
                linear_qkv_weight = torch.cat([
                    hf_state_dict['q_proj.weight'].load().reshape((num_query_groups, -1, args.hidden_size)),
                    hf_state_dict['k_proj.weight'].load().reshape((num_query_groups, -1, args.hidden_size)),
                    hf_state_dict['v_proj.weight'].load().reshape((num_query_groups, -1, args.hidden_size)),
                ],
                                              dim=1).reshape((-1, args.hidden_size))
                self._set_weight(mg_attn.linear_qkv.weight, linear_qkv_weight, 'linear_qkv.weight')
        else:
            q_dim, kv_dim = hf_attn.q_proj.weight.shape[0] // num_query_groups, hf_attn.k_proj.weight.shape[
                0] // num_query_groups
            is_lora = False if mg_attn is None else isinstance(mg_attn.linear_qkv,
                                                               LoraParallelLinear) and self._is_peft_format
            if self.pp_size > 1:
                dist.all_reduce(is_lora, group=self.pp_group)
            if is_lora:
                lora_A = self._get_weight(None if mg_attn is None else mg_attn.linear_qkv.lora_A['default'].weight.data,
                                          'linear_qkv.lora_A.default.weight')
                lora_B = self._get_weight(None if mg_attn is None else mg_attn.linear_qkv.lora_B['default'].weight.data,
                                          'linear_qkv.lora_B.default.weight')
                if lora_A is not None:
                    self._peft_target_modules.update({'q_proj', 'k_proj', 'v_proj'})
                    for key in ['q_proj', 'k_proj', 'v_proj']:
                        hf_state_dict[f'{key}.lora_A.weight'] = lora_A.clone()
                    lora_B = lora_B.reshape((num_query_groups, -1, lora_B.shape[-1]))
                    hf_state_dict['q_proj.lora_B.weight'] = lora_B[:, :q_dim, :].reshape(-1, lora_B.shape[-1]).clone()
                    hf_state_dict['k_proj.lora_B.weight'] = lora_B[:,
                                                                   q_dim:-kv_dim, :].reshape(-1,
                                                                                             lora_B.shape[-1]).clone()
                    hf_state_dict['v_proj.lora_B.weight'] = lora_B[:, -kv_dim:, :].reshape(-1, lora_B.shape[-1]).clone()
            else:
                mg_attn_weight = self._get_weight(None if mg_attn is None else mg_attn.linear_qkv.weight.data,
                                                  'linear_qkv.weight')
                if mg_attn_weight is not None:
                    mg_attn_weight = mg_attn_weight.reshape((num_query_groups, -1, args.hidden_size))
                    hf_state_dict['q_proj.weight'] = mg_attn_weight[:, :q_dim, :].reshape(-1, args.hidden_size).clone()
                    hf_state_dict['k_proj.weight'] = mg_attn_weight[:,
                                                                    q_dim:-kv_dim, :].reshape(-1,
                                                                                              args.hidden_size).clone()
                    hf_state_dict['v_proj.weight'] = mg_attn_weight[:, -kv_dim:, :].reshape(-1,
                                                                                            args.hidden_size).clone()
                del mg_attn_weight
        self._set_state_dict(mg_attn, 'linear_proj.weight', hf_state_dict, 'o_proj.weight', to_mcore)

        # Copy bias
        if args.add_qkv_bias and not self._is_peft_format:
            if to_mcore:
                linear_qkv_bias = torch.cat([
                    hf_state_dict['q_proj.bias'].load().reshape((num_query_groups, -1)),
                    hf_state_dict['k_proj.bias'].load().reshape((num_query_groups, -1)),
                    hf_state_dict['v_proj.bias'].load().reshape((num_query_groups, -1)),
                ],
                                            dim=1).reshape(-1)
                self._set_weight(mg_attn.linear_qkv.bias, linear_qkv_bias, 'linear_qkv.bias')
            else:
                mg_attn_bias = self._get_weight(None if mg_attn is None else mg_attn.linear_qkv.bias.data,
                                                'linear_qkv.bias')
                if mg_attn_bias is not None:
                    mg_attn_bias = mg_attn_bias.reshape((num_query_groups, -1))
                    hf_state_dict['q_proj.bias'] = mg_attn_bias[:, :q_dim].reshape(-1).clone()
                    hf_state_dict['k_proj.bias'] = mg_attn_bias[:, q_dim:-kv_dim].reshape(-1).clone()
                    hf_state_dict['v_proj.bias'] = mg_attn_bias[:, -kv_dim:].reshape(-1).clone()
        if args.qk_layernorm:
            hf_q_norm_key = 'q_norm.weight' if hasattr(hf_attn, 'q_norm') else 'query_layernorm.weight'
            hf_k_norm_key = 'k_norm.weight' if hasattr(hf_attn, 'k_norm') else 'key_layernorm.weight'
            self._set_state_dict(mg_attn, 'q_layernorm.weight', hf_state_dict, hf_q_norm_key, to_mcore)
            self._set_state_dict(mg_attn, 'k_layernorm.weight', hf_state_dict, hf_k_norm_key, to_mcore)
        if not to_mcore:
            hf_state_dict = self._add_prefix(hf_state_dict, hf_prefix)
        return hf_state_dict

    def _set_moe_state(
        self,
        mg_mlp,
        hf_state_dict,
        hf_prefix: str,
        layer_idx: int,
        to_mcore: bool,
    ):
        if to_mcore:
            hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
        else:
            hf_state_dict = {}

        hf_mlp = self.hf_layers[layer_idx].mlp
        hf_gate_key = 'gate.wg.weight' if hasattr(hf_mlp.gate, 'wg') else 'gate.weight'
        self._set_state_dict(mg_mlp, 'router.weight', hf_state_dict, hf_gate_key, to_mcore)
        if self.args.moe_router_enable_expert_bias:
            self._set_state_dict(mg_mlp, 'router.expert_bias', hf_state_dict, 'gate.e_score_correction_bias', to_mcore)

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
                    to_mcore,
                    hf_mlp=shared_expert))
            if hasattr(hf_mlp, 'shared_expert_gate'):
                self._set_state_dict(mg_mlp, 'shared_experts.gate_weight', hf_state_dict, 'shared_expert_gate.weight',
                                     to_mcore)
        for ep_rank in range(self.ep_size):
            mg_experts = mg_mlp.experts
            expert_available = ep_rank == self.ep_rank
            if not expert_available:
                if to_mcore:
                    continue
                else:
                    mg_experts = None
            hf_state_dict.update(
                self._set_mlp_state(mg_experts, hf_state_dict, 'experts.', layer_idx, to_mcore, ep_rank=ep_rank))
        if not to_mcore:
            hf_state_dict = self._add_prefix(hf_state_dict, hf_prefix)
        return hf_state_dict

    def _set_mlp_state(self,
                       mg_mlp,
                       hf_state_dict,
                       hf_prefix: str,
                       layer_idx: int,
                       to_mcore: bool,
                       ep_rank: Optional[int] = None,
                       hf_mlp=None):
        if to_mcore:
            hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
        else:
            hf_state_dict = {}
        if hf_mlp is None:
            hf_mlp = self.hf_layers[layer_idx].mlp
        is_expert = ep_rank is not None
        num_local_experts = 1
        if is_expert:
            hf_mlp = hf_mlp.experts[0]
            num_local_experts = self.args.num_experts // self.ep_size
        # linear_fc1
        if to_mcore:
            if isinstance(mg_mlp.linear_fc1, LoraParallelLinear):
                mg_lora_B = mg_mlp.linear_fc1.lora_B['default'].weight0 if is_expert else mg_mlp.linear_fc1.lora_B[
                    'default'].weight
                mg_lora_B = mg_lora_B.new_empty(num_local_experts * 2, mg_lora_B.shape[0] // 2, mg_lora_B.shape[-1])
                if hasattr(hf_mlp, 'gate_up_proj'):
                    if is_expert:
                        lora_A = torch.stack([
                            hf_state_dict[f'{i + ep_rank * num_local_experts}.gate_up_proj.lora_A.weight'].load()
                            for i in range(num_local_experts)
                        ])
                        lora_B = torch.concat([
                            hf_state_dict[f'{i + ep_rank * num_local_experts}.gate_up_proj.lora_B.weight'].load()
                            for i in range(num_local_experts)
                        ])
                    else:
                        lora_A = hf_state_dict['gate_up_proj.lora_A.weight'].load()
                        lora_B = hf_state_dict['gate_up_proj.lora_B.weight'].load()
                else:
                    if is_expert:
                        lora_A = torch.concat([
                            hf_state_dict[f'{i + ep_rank * num_local_experts}.gate_proj.lora_A.weight'].load()
                            for i in range(num_local_experts)
                        ])
                        up_lora_A = torch.concat([
                            hf_state_dict[f'{i + ep_rank * num_local_experts}.up_proj.lora_A.weight'].load()
                            for i in range(num_local_experts)
                        ])
                        weight_list = []
                        for i in range(num_local_experts):
                            gate_lora_B = hf_state_dict[
                                f'{i + ep_rank * num_local_experts}.gate_proj.lora_B.weight'].load()
                            up_lora_B = hf_state_dict[f'{i + ep_rank * num_local_experts}.up_proj.lora_B.weight'].load()
                            weight_list.append(torch.stack([gate_lora_B, up_lora_B], dim=0))
                        lora_B = torch.concat(weight_list, dim=0)
                    else:
                        lora_A = hf_state_dict['gate_proj.lora_A.weight'].load()
                        up_lora_A = hf_state_dict['up_proj.lora_A.weight'].load()
                        gate_lora_B = hf_state_dict['gate_proj.lora_B.weight'].load()
                        up_lora_B = hf_state_dict['up_proj.lora_B.weight'].load()
                        lora_B = torch.stack([gate_lora_B, up_lora_B], dim=0)
                    assert (
                        lora_A == up_lora_A).all(), 'Need to ensure lora_A consistency between gate_proj and up_proj'
                if is_expert:
                    mg_lora_A = mg_mlp.linear_fc1.lora_A['default'].weight0
                    mg_lora_A = mg_lora_A.new_empty(num_local_experts * mg_lora_A.shape[0], mg_lora_A.shape[1])
                else:
                    mg_lora_A = mg_mlp.linear_fc1.lora_A['default'].weight
                self._set_weight(mg_lora_A, lora_A, 'linear_fc1.lora_A.default.weight', is_expert=is_expert)
                self._set_weight(mg_lora_B, lora_B, 'linear_fc1.lora_B.default.weight', is_expert=is_expert)
                if is_expert:
                    mg_lora_A = mg_lora_A.view(num_local_experts, -1, mg_lora_A.shape[-1])
                    mg_lora_B = mg_lora_B.view(num_local_experts, -1, mg_lora_B.shape[-1])
                    for i in range(num_local_experts):
                        getattr(mg_mlp.linear_fc1.lora_A['default'], f'weight{i}').data.copy_(mg_lora_A[i])
                        getattr(mg_mlp.linear_fc1.lora_B['default'], f'weight{i}').data.copy_(mg_lora_B[i])
                else:
                    mg_mlp.linear_fc1.lora_B['default'].weight.data.copy_(mg_lora_B.view(-1, mg_lora_B.shape[-1]))
            else:
                fc1_weight = mg_mlp.linear_fc1.weight0 if is_expert else mg_mlp.linear_fc1.weight
                fc1_weight = fc1_weight.new_empty(2 * num_local_experts, fc1_weight.shape[0] // 2, fc1_weight.shape[1])
                if hasattr(hf_mlp, 'gate_up_proj'):
                    if is_expert:
                        gate_up_proj_weight = torch.concat([
                            hf_state_dict[f'{i + ep_rank * num_local_experts}.gate_up_proj.weight'].load()
                            for i in range(num_local_experts)
                        ],
                                                           dim=0)
                    else:
                        gate_up_proj_weight = hf_state_dict['gate_up_proj.weight'].load().view(
                            2, -1, fc1_weight.shape[-1])
                else:
                    if is_expert:
                        weight_list = []
                        start_idx = ep_rank * num_local_experts
                        for i in range(num_local_experts):
                            gate_proj_weight = hf_state_dict[f'{start_idx + i}.gate_proj.weight'].load()
                            up_proj_weight = hf_state_dict[f'{start_idx + i}.up_proj.weight'].load()
                            weight_list.append(torch.stack([gate_proj_weight, up_proj_weight], dim=0))
                        gate_up_proj_weight = torch.concat(weight_list, dim=0)
                        del weight_list
                    else:
                        gate_proj_weight = hf_state_dict['gate_proj.weight'].load()
                        up_proj_weight = hf_state_dict['up_proj.weight'].load()
                        gate_up_proj_weight = torch.stack([gate_proj_weight, up_proj_weight], dim=0)
                self._set_weight(fc1_weight, gate_up_proj_weight, 'linear_fc1.weight', is_expert=is_expert)
                if is_expert:
                    fc1_weight = fc1_weight.view(num_local_experts, -1, fc1_weight.shape[-1])
                    for i in range(num_local_experts):
                        getattr(mg_mlp.linear_fc1,
                                f'weight{i}').data.copy_(fc1_weight[i].view(-1, fc1_weight.shape[-1]))
                    del fc1_weight
                else:
                    mg_mlp.linear_fc1.weight.data.copy_(fc1_weight.view(-1, fc1_weight.shape[-1]))
        else:
            is_lora = False if mg_mlp is None else isinstance(mg_mlp.linear_fc1,
                                                              LoraParallelLinear) and self._is_peft_format
            is_lora = torch.tensor([is_lora], dtype=torch.bool, device='cuda')
            if self.pp_size > 1:
                dist.all_reduce(is_lora, group=self.pp_group)
            if is_expert and self.ep_size > 1:
                dist.all_reduce(is_lora, group=self.ep_group)
            if is_lora:
                if mg_mlp is None:
                    lora_A = None
                    lora_B = None
                else:
                    if is_expert:
                        lora_A = torch.concat([
                            getattr(mg_mlp.linear_fc1.lora_A['default'], f'weight{i}') for i in range(num_local_experts)
                        ],
                                              dim=0)
                        lora_B = torch.concat([
                            getattr(mg_mlp.linear_fc1.lora_B['default'], f'weight{i}') for i in range(num_local_experts)
                        ],
                                              dim=0)
                    else:
                        lora_A = mg_mlp.linear_fc1.lora_A['default'].weight
                        lora_B = mg_mlp.linear_fc1.lora_B['default'].weight
                    lora_B = lora_B.view(num_local_experts * 2, -1, lora_B.shape[1])
                lora_A = self._get_weight(lora_A, 'linear_fc1.lora_A.default.weight', is_expert=is_expert)
                lora_B = self._get_weight(lora_B, 'linear_fc1.lora_B.default.weight', is_expert=is_expert)
                if lora_A is not None:
                    if hasattr(hf_mlp, 'gate_up_proj'):
                        self._peft_target_modules.update({'gate_up_proj'})
                        if is_expert:
                            lora_A = lora_A.view(num_local_experts, -1, lora_A.shape[-1])
                            lora_B = lora_B.view(num_local_experts, -1, lora_B.shape[-1])
                            for i in range(num_local_experts):
                                hf_i = i + ep_rank * num_local_experts
                                hf_state_dict[f'{hf_i}.gate_up_proj.lora_A.weight'] = lora_A[i].clone()
                                hf_state_dict[f'{hf_i}.gate_up_proj.lora_B.weight'] = lora_B[i].clone()

                        else:
                            hf_state_dict['gate_up_proj.lora_A.weight'] = lora_A.clone()
                            hf_state_dict['gate_up_proj.lora_B.weight'] = lora_B.view(-1, lora_B.shape[-1]).clone()
                    else:
                        self._peft_target_modules.update({'gate_proj', 'up_proj'})
                        if is_expert:
                            lora_A = lora_A.view(num_local_experts, -1, lora_A.shape[-1])
                            lora_B = lora_B.view(num_local_experts, 2, -1, lora_B.shape[-1])
                            for i in range(num_local_experts):
                                hf_i = i + ep_rank * num_local_experts
                                hf_state_dict[f'{hf_i}.gate_proj.lora_A.weight'] = lora_A[i].clone()
                                hf_state_dict[f'{hf_i}.up_proj.lora_A.weight'] = lora_A[i].clone()
                                hf_state_dict[f'{hf_i}.gate_proj.lora_B.weight'] = lora_B[i][0].clone()
                                hf_state_dict[f'{hf_i}.up_proj.lora_B.weight'] = lora_B[i][1].clone()
                        else:
                            hf_state_dict['gate_proj.lora_A.weight'] = lora_A.clone()
                            hf_state_dict['up_proj.lora_A.weight'] = lora_A.clone()
                            hf_state_dict['gate_proj.lora_B.weight'] = lora_B[0].clone()
                            hf_state_dict['up_proj.lora_B.weight'] = lora_B[1].clone()
            else:
                if mg_mlp is None:
                    fc1_weight = None
                else:
                    if is_expert:
                        fc1_weight = torch.concat(
                            [getattr(mg_mlp.linear_fc1, f'weight{i}') for i in range(num_local_experts)], dim=0)
                    else:
                        fc1_weight = mg_mlp.linear_fc1.weight
                    fc1_weight = fc1_weight.view(num_local_experts * 2, -1, fc1_weight.shape[1])
                gate_up_proj_weight = self._get_weight(fc1_weight, 'linear_fc1.weight', is_expert=is_expert)
                del fc1_weight
                if gate_up_proj_weight is not None:
                    if hasattr(hf_mlp, 'gate_up_proj'):
                        if is_expert:
                            gate_up_proj_weight = gate_up_proj_weight.view(num_local_experts, -1,
                                                                           gate_up_proj_weight.shape[-1])
                            for i in range(num_local_experts):
                                hf_i = i + ep_rank * num_local_experts
                                hf_state_dict[f'{hf_i}.gate_up_proj.weight'] = gate_up_proj_weight[i].clone()
                            del gate_up_proj_weight
                        else:
                            hf_state_dict['gate_up_proj.weight'] = gate_up_proj_weight.view(
                                -1, gate_up_proj_weight.shape[-1]).clone()
                    else:
                        if is_expert:
                            gate_up_proj_weight = gate_up_proj_weight.view(num_local_experts, 2, -1,
                                                                           gate_up_proj_weight.shape[-1])
                            for i in range(num_local_experts):
                                hf_i = i + ep_rank * num_local_experts
                                hf_state_dict[f'{hf_i}.gate_proj.weight'] = gate_up_proj_weight[i][0].clone()
                                hf_state_dict[f'{hf_i}.up_proj.weight'] = gate_up_proj_weight[i][1].clone()
                            del gate_up_proj_weight
                        else:
                            hf_state_dict['gate_proj.weight'] = gate_up_proj_weight[0].clone()
                            hf_state_dict['up_proj.weight'] = gate_up_proj_weight[1].clone()
        # linear_fc2
        if is_expert:
            if to_mcore:
                if isinstance(mg_mlp.linear_fc2, LoraParallelLinear):
                    mg_lora_A = mg_mlp.linear_fc2.lora_A.default.weight0
                    mg_lora_A = mg_lora_A.new_empty(num_local_experts * mg_lora_A.shape[0], mg_lora_A.shape[1])
                    mg_lora_B = mg_mlp.linear_fc2.lora_B.default.weight0
                    mg_lora_B = mg_lora_B.new_empty(num_local_experts * mg_lora_B.shape[0], mg_lora_B.shape[1])
                    lora_A = torch.concat([
                        hf_state_dict[f'{i + ep_rank * num_local_experts}.down_proj.lora_A.weight'].load()
                        for i in range(num_local_experts)
                    ],
                                          dim=0)
                    lora_B = torch.concat([
                        hf_state_dict[f'{i + ep_rank * num_local_experts}.down_proj.lora_B.weight'].load()
                        for i in range(num_local_experts)
                    ],
                                          dim=0)
                    self._set_weight(mg_lora_A, lora_A, 'linear_fc2.lora_A.default.weight', is_expert=is_expert)
                    self._set_weight(mg_lora_B, lora_B, 'linear_fc2.lora_B.default.weight', is_expert=is_expert)
                    mg_lora_A = mg_lora_A.view(num_local_experts, -1, mg_lora_A.shape[-1])
                    mg_lora_B = mg_lora_B.view(num_local_experts, -1, mg_lora_B.shape[-1])
                    for i in range(num_local_experts):
                        getattr(mg_mlp.linear_fc2.lora_A['default'], f'weight{i}').data.copy_(mg_lora_A[i])
                        getattr(mg_mlp.linear_fc2.lora_B['default'], f'weight{i}').data.copy_(mg_lora_B[i])
                else:
                    fc2_weight = mg_mlp.linear_fc2.weight0
                    fc2_weight = fc2_weight.new_empty(num_local_experts * fc2_weight.shape[0], fc2_weight.shape[1])
                    down_proj_weight = torch.concat([
                        hf_state_dict[f'{i + ep_rank * num_local_experts}.down_proj.weight'].load()
                        for i in range(num_local_experts)
                    ],
                                                    dim=0)
                    self._set_weight(fc2_weight, down_proj_weight, 'linear_fc2.weight', is_expert=is_expert)
                    fc2_weight = fc2_weight.view(num_local_experts, -1, fc2_weight.shape[-1])
                    for i in range(num_local_experts):
                        getattr(mg_mlp.linear_fc2, f'weight{i}').data.copy_(fc2_weight[i])
            else:
                is_lora = False if mg_mlp is None else isinstance(mg_mlp.linear_fc2,
                                                                  LoraParallelLinear) and self._is_peft_format
                is_lora = torch.tensor([is_lora], dtype=torch.bool, device='cuda')
                if self.pp_size > 1:
                    dist.all_reduce(is_lora, group=self.pp_group)
                if is_expert and self.ep_size > 1:
                    dist.all_reduce(is_lora, group=self.ep_group)
                if is_lora:
                    if mg_mlp is None:
                        lora_A = None
                        lora_B = None
                    else:
                        lora_A = torch.concat([
                            getattr(mg_mlp.linear_fc2.lora_A['default'], f'weight{i}') for i in range(num_local_experts)
                        ],
                                              dim=0)
                        lora_B = torch.concat([
                            getattr(mg_mlp.linear_fc2.lora_B['default'], f'weight{i}') for i in range(num_local_experts)
                        ],
                                              dim=0)
                    lora_A = self._get_weight(lora_A, 'linear_fc2.lora_A.default.weight', is_expert=is_expert)
                    lora_B = self._get_weight(lora_B, 'linear_fc2.lora_B.default.weight', is_expert=is_expert)
                    if lora_A is not None:
                        self._peft_target_modules.update({'down_proj'})
                        lora_A = lora_A.view(num_local_experts, -1, lora_A.shape[-1])
                        lora_B = lora_B.view(num_local_experts, -1, lora_B.shape[-1])
                        for i in range(num_local_experts):
                            hf_i = i + ep_rank * num_local_experts
                            hf_state_dict[f'{hf_i}.down_proj.lora_A.weight'] = lora_A[i].clone()
                            hf_state_dict[f'{hf_i}.down_proj.lora_B.weight'] = lora_B[i].clone()
                else:
                    if mg_mlp is None:
                        fc2_weight = None
                    else:
                        fc2_weight = torch.concat(
                            [getattr(mg_mlp.linear_fc2, f'weight{i}') for i in range(num_local_experts)], dim=0)
                    down_proj_weight = self._get_weight(fc2_weight, 'linear_fc2.weight', is_expert=is_expert)
                    del fc2_weight
                    if down_proj_weight is not None:
                        down_proj_weight = down_proj_weight.view(num_local_experts, -1, down_proj_weight.shape[-1])
                        for i in range(num_local_experts):
                            hf_i = i + ep_rank * num_local_experts
                            hf_state_dict[f'{hf_i}.down_proj.weight'] = down_proj_weight[i].clone()
        else:
            self._set_state_dict(
                mg_mlp, 'linear_fc2.weight', hf_state_dict, 'down_proj.weight', to_mcore, is_expert=is_expert)
        if not to_mcore:
            hf_state_dict = self._add_prefix(hf_state_dict, hf_prefix)
        return hf_state_dict

    def _set_mla_attn_state(
        self,
        mg_attn,
        hf_state_dict,
        hf_prefix: str,
        layer_idx: int,
        to_mcore: bool,
    ):
        if to_mcore:
            hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
        else:
            hf_state_dict = {}
        self._set_state_dict(mg_attn, 'linear_proj.weight', hf_state_dict, 'o_proj.weight', to_mcore)
        if self.args.q_lora_rank is None:
            self._set_state_dict(mg_attn, 'linear_q_proj.weight', hf_state_dict, 'q_proj.weight', to_mcore)
        else:
            self._set_state_dict(mg_attn, 'linear_q_down_proj.weight', hf_state_dict, 'q_a_proj.weight', to_mcore)
            self._set_state_dict(mg_attn, 'linear_q_up_proj.weight', hf_state_dict, 'q_b_proj.weight', to_mcore)
        self._set_state_dict(mg_attn, 'linear_kv_down_proj.weight', hf_state_dict, 'kv_a_proj_with_mqa.weight',
                             to_mcore)
        self._set_state_dict(mg_attn, 'linear_kv_up_proj.weight', hf_state_dict, 'kv_b_proj.weight', to_mcore)
        if self.args.qk_layernorm:
            self._set_state_dict(mg_attn, 'linear_kv_up_proj.layer_norm_weight', hf_state_dict, 'kv_a_layernorm.weight',
                                 to_mcore)
        if not to_mcore:
            hf_state_dict = self._add_prefix(hf_state_dict, hf_prefix)
        return hf_state_dict

    def _set_layer_attn(self, mg_layer, hf_state_dict, layer_idx: int, to_mcore: bool):
        mg_attn = None if mg_layer is None else mg_layer.self_attention
        if self.args.multi_latent_attention:
            hf_state_dict.update(self._set_mla_attn_state(mg_attn, hf_state_dict, 'self_attn.', layer_idx, to_mcore))
            self._set_state_dict(mg_layer, 'input_layernorm.weight', hf_state_dict, 'input_layernorm.weight', to_mcore)
        else:
            hf_state_dict.update(self._set_attn_state(mg_attn, hf_state_dict, 'self_attn.', layer_idx, to_mcore))
            self._set_state_dict(mg_layer, 'self_attention.linear_qkv.layer_norm_weight', hf_state_dict,
                                 'input_layernorm.weight', to_mcore)
        return hf_state_dict

    def _set_layer_mlp(self, mg_layer, hf_state_dict, layer_idx: int, to_mcore: bool):
        hf_mlp = self.hf_layers[layer_idx].mlp
        is_moe = self._is_moe(hf_mlp.state_dict())
        mg_mlp = None if mg_layer is None else mg_layer.mlp
        if is_moe:
            hf_state_dict.update(self._set_moe_state(mg_mlp, hf_state_dict, 'mlp.', layer_idx, to_mcore))
            self._set_state_dict(mg_layer, 'pre_mlp_layernorm.weight', hf_state_dict, 'post_attention_layernorm.weight',
                                 to_mcore)
        else:
            hf_state_dict.update(self._set_mlp_state(mg_mlp, hf_state_dict, 'mlp.', layer_idx, to_mcore))
            self._set_state_dict(mg_layer, 'mlp.linear_fc1.layer_norm_weight', hf_state_dict,
                                 'post_attention_layernorm.weight', to_mcore)
        return hf_state_dict

    def _set_layer_state(self, mg_layer, hf_state_dict, hf_prefix: str, layer_idx: int, to_mcore: bool):
        hf_prefix = f'{hf_prefix}{layer_idx}.'
        if to_mcore:
            hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
        else:
            hf_state_dict = {}
        hf_state_dict.update(self._set_layer_attn(mg_layer, hf_state_dict, layer_idx, to_mcore))
        hf_state_dict.update(self._set_layer_mlp(mg_layer, hf_state_dict, layer_idx, to_mcore))
        if not to_mcore:
            hf_state_dict = self._add_prefix(hf_state_dict, hf_prefix)
        return hf_state_dict

    def _convert(self, mg_models, hf_state_dict, hf_prefix: str, to_mcore: bool):
        if to_mcore:
            hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
        else:
            hf_state_dict = {}
        mg_models = iter(mg_models)
        mg_model = next(mg_models)
        if not to_mcore or mpu.is_pipeline_first_stage(ignore_virtual=False, vp_stage=mg_model.vp_stage):
            self._set_state_dict(mg_model, 'embedding.word_embeddings.weight', hf_state_dict,
                                 'model.embed_tokens.weight', to_mcore)
        if to_mcore:
            yield
        else:
            yield from list(self._add_prefix(hf_state_dict, hf_prefix).items())
            hf_state_dict = {}
        for layer_idx in tqdm(
                range(self.args.num_layers), dynamic_ncols=True, desc='Converting: ', disable=self.disable_tqmd):
            start_idx = mg_model.decoder.layers[0].layer_number - 1
            mg_layer_available = (start_idx <= layer_idx < mg_model.decoder.layers[-1].layer_number)
            if mg_layer_available:
                mg_layer = mg_model.decoder.layers[layer_idx - start_idx]
            else:
                if to_mcore:
                    continue
                else:
                    mg_layer = None
            if not to_mcore and self.pp_size > 1:
                has_model = torch.tensor([mg_layer is not None], dtype=torch.bool, device='cuda')
                dist.all_reduce(has_model, group=self.pp_group)
                if not has_model:
                    mg_model = next(mg_models)
                    continue
            res = self._set_layer_state(mg_layer, hf_state_dict, 'model.layers.', layer_idx, to_mcore)
            if to_mcore:
                yield
            else:
                yield from list(self._add_prefix(res, hf_prefix).items())
                hf_state_dict = {}
        if not to_mcore or mpu.is_pipeline_last_stage(ignore_virtual=False, vp_stage=mg_model.vp_stage):
            if self.args.untie_embeddings_and_output_weights:
                hf_lm_head_key = 'lm_head.weight'
                if not to_mcore and self.args.task_type == 'seq_cls':
                    hf_lm_head_key = 'score.weight'
                self._set_state_dict(mg_model, 'output_layer.weight', hf_state_dict, hf_lm_head_key, to_mcore)
            else:
                self._set_state_dict(mg_model, 'output_layer.weight', hf_state_dict, 'model.embed_tokens.weight',
                                     to_mcore)
            self._set_state_dict(mg_model, 'decoder.final_layernorm.weight', hf_state_dict, 'model.norm.weight',
                                 to_mcore)
        if to_mcore:
            yield
        else:
            yield from list(self._add_prefix(hf_state_dict, hf_prefix).items())

    def load_weights(self, mg_model, hf_model_dir: str, is_peft_format: bool = False):
        self._is_peft_format = is_peft_format
        with SafetensorLazyLoader(hf_model_dir, is_peft_format=is_peft_format) as loader:
            state_dict = loader.get_state_dict()
            hf_prefix = 'base_model.model.' if is_peft_format else ''
            list(self._convert([mg_model], state_dict, hf_prefix, True))

    def export_weights(self, mg_models, target_device=None, only_last_rank: bool = False, is_peft_format: bool = False):
        # TODO: modules_to_save
        self._target_device = target_device
        self._only_last_rank = only_last_rank
        self._is_peft_format = is_peft_format
        self._peft_target_modules = set()
        self._peft_modules_to_save = set()
        hf_prefix = 'base_model.model.' if is_peft_format else ''
        yield from self._convert(mg_models, {}, hf_prefix, False)

    def save_weights(self, mg_models, output_dir: str, is_peft_format: bool = False) -> None:
        """Save the mg_model checkpoint in HF format"""
        saver = StreamingSafetensorSaver(
            save_dir=output_dir, max_shard_size=self.args.max_shard_size, is_peft_format=is_peft_format)
        for k, v in self.export_weights(
                mg_models, target_device='cpu', only_last_rank=True, is_peft_format=is_peft_format):
            saver.add_tensor(k, v)
        saver.finalize()
        if is_last_rank():
            if is_peft_format:
                peft_config = copy(mg_models[0].peft_config['default'])
                peft_config.target_modules = self._peft_target_modules
                peft_config.modules_to_save = self._peft_modules_to_save
                peft_config.save_pretrained(output_dir)
            else:
                # TODO: new_special_tokens
                self.hf_model.config.save_pretrained(output_dir)
                save_checkpoint(
                    None,
                    self.processor,
                    output_dir,
                    model_dirs=[self.hf_model.model_info.model_dir],
                    additional_saved_files=self.hf_model.model_meta.additional_saved_files)
            logger.info_if(f'Successfully saved `safetensors` model weights in `{output_dir}`.', cond=is_last_rank())

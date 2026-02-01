# Copyright (c) ModelScope Contributors. All rights reserved.
import math
from copy import copy
from typing import List, Optional, Union

import megatron.core
import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers
from megatron.core import mpu
from megatron.training import get_args
from packaging import version
from peft.utils import ModulesToSaveWrapper
from tqdm import tqdm
from transformers.modeling_utils import custom_object_save

from swift.model import get_model_processor, save_checkpoint
from swift.utils import (MxFp4Dequantizer, SafetensorLazyLoader, StreamingSafetensorSaver, deep_getattr, get_logger,
                         get_modules_to_not_convert, get_multimodal_target_regex, is_last_rank, safe_snapshot_download)
from ..tuners import LoraParallelLinear

logger = get_logger()

mcore_013 = version.parse(megatron.core.__version__) >= version.parse('0.13.0rc0')


# Some ideas for LoRA conversion are referenced from: https://github.com/modelscope/ms-swift/pull/6225
class GPTBridge:
    fp8_block_size = 128
    hf_layers_prefix = 'model.layers'
    hf_mtp_prefix = 'model.layers'
    hf_embed_key = 'model.embed_tokens.weight'
    hf_final_layernorm_key = 'model.norm.weight'
    hf_lm_head_key = 'lm_head.weight'
    hf_score_key = 'score.weight'
    hf_state_dict_mapping = {}

    def __init__(self, disable_tqmd: bool = False):
        from .register import get_megatron_model_meta
        self.args = get_args()
        self.disable_tqmd = disable_tqmd or not is_last_rank()
        self._target_device = None
        self._only_last_rank = False
        self._peft_target_modules = set()
        self._peft_modules_to_save = set()
        self._is_peft_format = False
        self._adapter_name = 'default'
        self._init_meta_hf_model()
        self.hf_layers = deep_getattr(self.hf_model, self.hf_layers_prefix)
        self.module_mapping = {}
        self.mcore_014 = version.parse(megatron.core.__version__) >= version.parse('0.14.0rc0')
        megatron_model_meta = get_megatron_model_meta(self.args.hf_model_type)
        if self.args.is_multimodal and megatron_model_meta.visual_cls is not None:
            self.module_mapping = megatron_model_meta.visual_cls.module_mapping
        self.tp_size = self.args.tensor_model_parallel_size
        self.pp_size = self.args.pipeline_model_parallel_size
        self.etp_size = self.args.expert_tensor_parallel_size
        self.ep_size = self.args.expert_model_parallel_size

        self.tp_group = mpu.get_tensor_model_parallel_group()
        self.pp_group = mpu.get_pipeline_model_parallel_group()
        self.etp_group = mpu.get_expert_tensor_parallel_group()
        self.ep_group = mpu.get_expert_model_parallel_group()
        self.is_transformers_5 = version.parse(transformers.__version__) >= version.parse('5.0.0.dev')
        if self.is_transformers_5 and self.hf_model.model_info.is_moe_model and not self.args.merge_lora:
            logger.warning('In transformers 5.0, the weight organization of MoE model experts differs from Megatron. '
                           'It is recommended to use `--merge_lora true`, otherwise the trained model may not be '
                           'usable for inference with transformers.')
        self.tp_rank = mpu.get_tensor_model_parallel_rank()
        self.pp_rank = mpu.get_pipeline_model_parallel_rank()
        self.etp_rank = mpu.get_expert_tensor_parallel_rank()
        self.ep_rank = mpu.get_expert_model_parallel_rank()

        self._fp8_quantizer = None
        self.mxfp4_quantizer = MxFp4Dequantizer()

        dp_size = dist.get_world_size() // self.etp_size // self.ep_size // self.pp_size
        expert_decoder_rank_generator = mpu.RankGenerator(
            tp=self.etp_size,
            ep=self.ep_size,
            dp=dp_size,
            pp=self.pp_size,
            cp=1,
            order='tp-cp-ep-dp-pp',
            rank_offset=0,
        )
        rank = dist.get_rank()
        for ranks in expert_decoder_rank_generator.get_ranks('ep-pp'):
            group = mpu.create_group(
                ranks,
                group_desc='EP-PP-GROUP',
            )
            if rank in ranks:
                self.ep_pp_size = self.ep_size * self.pp_size
                self.ep_pp_group = group
                self.ep_pp_rank = dist.get_rank(group)

    def get_hf_mlp_prefix(self, layer_idx):
        if hasattr(self.hf_layers[layer_idx], 'feed_forward'):
            return 'feed_forward'
        else:
            return 'mlp'

    def _get_hf_mlp(self, layer_idx):
        return getattr(self.hf_layers[layer_idx], self.get_hf_mlp_prefix(layer_idx))

    def _init_meta_hf_model(self):
        with torch.device('meta'):
            self.hf_model, self.processor = get_model_processor(
                self.args.model_dir, model_type=self.args.hf_model_type, return_dummy_model=True)

    def _get_tp_split_dim(self, mg_key: Optional[str]) -> Optional[int]:
        if mg_key is None:
            return
        # ColumnLinear
        dim0_keys = {
            'word_embeddings',
            'linear_qkv',
            # mla
            'linear_q_proj',
            'linear_q_up_proj',
            'linear_kv_up_proj',
            # mtp
            'eh_proj',
        }
        if self.args.task_type in {'causal_lm', 'generative_reranker'}:
            dim0_keys.add('output_layer')
        if not self.mcore_014:
            # https://github.com/NVIDIA/Megatron-LM/commit/720c8b40d8e7e2de1dd303d792f29093101c5e72
            dim0_keys.update({'linear_q_down_proj', 'linear_kv_down_proj'})
        # RowLinear
        dim1_keys = {'linear_proj', 'linear_fc2'}
        if 'lora_A' not in mg_key and 'lora_B' not in mg_key:
            key, suffix = mg_key.rsplit('.', 2)[-2:]
            if suffix == 'layer_norm_weight':
                return
            elif mg_key == 'core_attention.softmax_offset':
                return 0
            elif key in dim0_keys:
                return 0
            elif key in {'linear_fc1'} | dim1_keys and suffix != 'bias':
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

    def _split_tp(self, hf_weight, tp_dim, is_expert, is_embedding: bool):
        tp_size = self.etp_size if is_expert else self.tp_size
        tp_rank = self.etp_rank if is_expert else self.tp_rank
        if is_embedding:
            padding_size = math.ceil(hf_weight.shape[0] / tp_size) * tp_size - hf_weight.shape[0]
            if padding_size > 0:
                new_size = hf_weight.shape[0] + padding_size
                logger.warning(
                    f'Padding embedding from {hf_weight.shape[0]} to {new_size} (padding size: {padding_size})')
                hf_weight = F.pad(hf_weight, (0, 0, 0, padding_size))
        if tp_dim is not None and tp_size > 1:
            tensor = hf_weight.chunk(tp_size, dim=tp_dim)[tp_rank]
        else:
            tensor = hf_weight
        return tensor

    def _set_weight(
        self,
        mg_param: Union[torch.Tensor, List[torch.Tensor]],
        hf_weight: torch.Tensor,
        mg_key: str,
        offset: float = 0,
        is_expert: bool = False,
        *,
        hf_scale_inv: Optional[torch.Tensor] = None,
    ):
        # tp/etp
        tp_dim = self._get_tp_split_dim(mg_key)
        is_embedding = mg_key in {'embedding.word_embeddings.weight', 'output_layer.weight'}
        tensor = self._split_tp(hf_weight, tp_dim, is_expert, is_embedding=is_embedding)
        del hf_weight
        if not isinstance(mg_param, (list, tuple)):
            mg_param = [mg_param]
        if hf_scale_inv is not None:
            hf_scale_inv = self._split_tp(hf_scale_inv, tp_dim, is_expert, is_embedding=is_embedding)
            hf_scale_inv = hf_scale_inv.chunk(len(mg_param), dim=0)
        if offset:
            assert hf_scale_inv is None, f'mg_key: {mg_key}'
            tensor = tensor + offset
        tensor_list = tensor.chunk(len(mg_param), dim=0)
        for i, param in enumerate(mg_param):
            tensor = tensor_list[i].reshape(*param.shape)
            if self._is_fp8_param(param):
                if hf_scale_inv is None:
                    param.data.copy_(tensor)
                    param._high_precision_init_val.copy_(tensor)
                else:
                    tensor = tensor.view(torch.uint8)
                    param._rowwise_data.data.copy_(tensor)
                    self._copy_scale_inv(param, hf_scale_inv[i])
                    del param.get_high_precision_init_val
            else:
                if hf_scale_inv is not None:
                    fp8_tensor = self.fp8_quantizer.make_empty(tensor.shape)
                    fp8_tensor._rowwise_data.copy_(tensor.view(torch.uint8))
                    self._copy_scale_inv(fp8_tensor, hf_scale_inv[i])
                    tensor = fp8_tensor
                param.data.copy_(tensor)

    @staticmethod
    def _copy_scale_inv(tensor, scale_inv):
        scale_inv = scale_inv.reshape(-1, scale_inv.shape[-1])
        if scale_inv.shape[-1] < tensor._rowwise_scale_inv.shape[-1]:
            scale_inv = torch.concat([
                scale_inv,
                scale_inv.new_zeros((scale_inv.shape[0], tensor._rowwise_scale_inv.shape[-1] - scale_inv.shape[1]))
            ],
                                     dim=-1)
        tensor._rowwise_scale_inv.data.copy_(scale_inv)

    @property
    def fp8_quantizer(self):
        if self._fp8_quantizer is None:
            from transformer_engine_torch import DType as TE_DType
            from transformer_engine.pytorch import Float8BlockQuantizer
            self._fp8_quantizer = Float8BlockQuantizer(TE_DType.kFloat8E4M3, rowwise=True, columnwise=True)
        return self._fp8_quantizer

    @staticmethod
    def _is_fp8_param(param):
        try:
            from transformer_engine.pytorch import Float8BlockwiseQTensor
            return isinstance(param, Float8BlockwiseQTensor)
        except ImportError:
            return False

    def _set_module(self, mg_module, hf_state_dict, hf_prefix: str, to_mcore: bool):
        if to_mcore:
            if mg_module is None:
                return {}
            hf_state_dict = {k: v.load() for k, v in self._remove_prefix(hf_state_dict, hf_prefix).items()}
            if self._is_peft_format:
                new_state_dict = {}
                for k, v in hf_state_dict.items():
                    k = k.replace('.lora_A.', f'.lora_A.{self._adapter_name}.')
                    k = k.replace('.lora_B.', f'.lora_B.{self._adapter_name}.')
                    k = k.replace('.modules_to_save.', f'.modules_to_save.{self._adapter_name}.')
                    new_state_dict[k] = v
                hf_state_dict = new_state_dict
            incompatible_keys = mg_module.load_state_dict(hf_state_dict, strict=False)
            missing_keys = incompatible_keys.missing_keys
            if self._is_peft_format:
                missing_keys = [
                    k for k in incompatible_keys.missing_keys
                    if '.lora_A.' in k or '.lora_B.' in k or '.modules_to_save.' in k
                ]
            assert len(missing_keys) == 0, f'incompatible_keys.missing_keys: {missing_keys}'
            return {}
        else:
            hf_state_dict = None if mg_module is None else mg_module.state_dict()
            if hf_state_dict is not None:
                new_state_dict = {}
                for k, v in hf_state_dict.items():
                    if self._is_peft_format:
                        if '.lora_A.' in k or '.lora_B.' in k or '.modules_to_save.' in k:
                            k = k.replace(f'{self._adapter_name}.', '')
                            new_state_dict[k] = v
                    else:
                        if '.lora_A.' in k or '.lora_B.' in k or 'original_module.' in k:
                            continue
                        k = k.replace('base_layer.', '')
                        k = k.replace(f'modules_to_save.{self._adapter_name}.', '')
                        new_state_dict[k] = v
                hf_state_dict = new_state_dict
            if self.pp_size > 1:
                src_rank = torch.tensor([0 if hf_state_dict is None else self.pp_rank],
                                        dtype=torch.int64,
                                        device='cuda')
                dist.all_reduce(src_rank, group=self.pp_group)
                src_rank = dist.get_global_rank(self.pp_group, src_rank.item())
                meta_data = [None] if hf_state_dict is None else [list(hf_state_dict.keys())]
                dist.broadcast_object_list(meta_data, src=src_rank, group=self.pp_group)
                if meta_data[0] is None:
                    return {}
                hf_state_dict = hf_state_dict or {k: None for k in meta_data[0]}
                for k, v in hf_state_dict.items():
                    v, _ = self._get_weight(v, None)
                    hf_state_dict[k] = v
            elif hf_state_dict is None:
                return {}
            else:
                if self._target_device is not None:
                    for k, v in hf_state_dict.items():
                        hf_state_dict[k] = v.to(self._target_device)
            return self._add_prefix(hf_state_dict, hf_prefix)

    def _all_gather_tp(self, tensor, tp_dim, is_expert):
        tensor = None if tensor is None else tensor.to('cuda')
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
        return tensor

    def _broadcast_ep_pp(self, tensor, is_expert):
        pp_group = self.ep_pp_group if is_expert else self.pp_group
        pp_size = self.ep_pp_size if is_expert else self.pp_size
        pp_rank = self.ep_pp_rank if is_expert else self.pp_rank
        # pp/ep
        if pp_size > 1:
            src_rank = torch.tensor([0 if tensor is None else pp_rank], dtype=torch.int64, device='cuda')
            dist.all_reduce(src_rank, group=pp_group)
            src_rank = dist.get_global_rank(pp_group, src_rank.item())
            meta_data = torch.zeros(10, dtype=torch.int64, device='cuda')
            dtype_mapping = {torch.float64: 0, torch.float32: 1, torch.float16: 2, torch.bfloat16: 3, torch.uint8: 4}
            dtype_mapping_r = {v: k for k, v in dtype_mapping.items()}
            if tensor is None:
                dist.broadcast(meta_data, src=src_rank, group=pp_group)
                assert meta_data[0].item() > 0, f'meta_data: {meta_data}'
                shape = meta_data[1:1 + meta_data[0]].tolist()
                dtype = dtype_mapping_r[meta_data[-1].item()]
                tensor = torch.empty(shape, device='cuda', dtype=dtype)
                dist.broadcast(tensor, src=src_rank, group=pp_group)
            else:
                meta_data[0] = tensor.ndim
                meta_data[1:1 + tensor.ndim] = torch.tensor(tensor.shape, dtype=torch.int64, device='cuda')
                meta_data[-1] = dtype_mapping[tensor.dtype]
                dist.broadcast(meta_data, src=src_rank, group=pp_group)
                dist.broadcast(tensor, src=src_rank, group=pp_group)
        return tensor

    def _get_weight(
        self,
        mg_weight: Union[torch.Tensor, List[torch.Tensor]],
        mg_key: Optional[str],
        offset: float = 0,
        is_expert: bool = False,
    ):
        # tp/etp
        mg_scale_inv = None
        tensor = mg_weight
        if tensor is not None:
            if not isinstance(tensor, (list, tuple)):
                tensor = [tensor]
            if self._is_fp8_param(tensor[0]):
                mg_scale_inv = [t._rowwise_scale_inv for t in tensor]
                tensor = [t._rowwise_data for t in tensor]
        del mg_weight
        if tensor is not None:
            assert isinstance(tensor, (list, tuple)), f'mg_key: {mg_key}'
            tensor = torch.concat(tensor, dim=0)
            if mg_scale_inv is not None:
                mg_scale_inv = torch.concat(mg_scale_inv, dim=0)
        num_local_experts = self.args.num_experts // self.ep_size if is_expert else 1
        tp_dim = self._get_tp_split_dim(mg_key)
        is_linear_fc1 = (mg_key is not None and mg_key.split('.', 1)[0] == 'linear_fc1' and tp_dim is not None)
        if tensor is not None and is_linear_fc1:
            tensor = tensor.view(num_local_experts * 2, -1, tensor.shape[-1])
            if mg_scale_inv is not None:
                mg_scale_inv = mg_scale_inv.view(num_local_experts * 2, -1, mg_scale_inv.shape[-1])

        tensor = self._all_gather_tp(tensor, tp_dim, is_expert)
        tensor = self._broadcast_ep_pp(tensor, is_expert)
        if tensor.dtype == torch.uint8:
            mg_scale_inv = self._all_gather_tp(mg_scale_inv, tp_dim, is_expert)
            mg_scale_inv = self._broadcast_ep_pp(mg_scale_inv, is_expert)
            tensor = tensor.view(torch.float8_e4m3fn)
            mg_scale_inv = mg_scale_inv[..., :math.ceil(tensor.shape[-1] / self.fp8_block_size)].contiguous()
        assert tensor is not None, f'mg_key: {mg_key}'
        if offset:
            assert mg_scale_inv is None, f'mg_key: {mg_key}'
            tensor = tensor + offset
        is_embedding = mg_key in {'embedding.word_embeddings.weight', 'output_layer.weight'}
        if is_embedding and self.args.padded_vocab_size < tensor.shape[0]:
            tensor = tensor[:self.args.padded_vocab_size]
        if self._target_device is not None:
            tensor = tensor.to(device=self._target_device)
            if mg_scale_inv is not None:
                mg_scale_inv = mg_scale_inv.to(device=self._target_device)
        if self._only_last_rank and not is_last_rank():
            tensor = None
            mg_scale_inv = None
        if is_expert and tensor is not None:
            if mg_key.endswith('bias'):
                tensor = tensor.view(num_local_experts, -1)
            else:
                tensor = tensor.view(num_local_experts, -1, tensor.shape[-1])
                if mg_scale_inv is not None:
                    mg_scale_inv = mg_scale_inv.view(num_local_experts, -1, mg_scale_inv.shape[-1])
        return tensor, mg_scale_inv

    def _set_state_dict(self,
                        mg_module,
                        mg_key: str,
                        hf_state_dict,
                        hf_key: str,
                        to_mcore: bool,
                        *,
                        offset: float = 0,
                        is_expert: bool = False):
        module_key, param_key = mg_key.rsplit('.', 1)
        if '.' in hf_key:
            hf_module_key, hf_param_key = hf_key.rsplit('.', 1)
        else:
            hf_module_key, hf_param_key = hf_key, None
        sub_module = deep_getattr(mg_module, module_key)
        is_lora = isinstance(sub_module, LoraParallelLinear)
        is_modules_to_save = isinstance(sub_module, ModulesToSaveWrapper)
        if not to_mcore:
            state = torch.tensor([is_lora, is_modules_to_save], dtype=torch.bool, device='cuda')
            if is_expert and self.ep_pp_size > 1:
                dist.all_reduce(state, group=self.ep_pp_group)
            elif not is_expert and self.pp_size > 1:
                dist.all_reduce(state, group=self.pp_group)
            is_lora, is_modules_to_save = state
        if is_lora and self._is_peft_format and param_key != 'layer_norm_weight':
            if to_mcore:
                lora_A_key = f'{module_key}.lora_A.{self._adapter_name}.{param_key}'
                lora_B_key = f'{module_key}.lora_B.{self._adapter_name}.{param_key}'
                mg_lora_A = deep_getattr(mg_module, f'{lora_A_key}')
                mg_lora_B = deep_getattr(mg_module, f'{lora_B_key}')
                hf_lora_A = hf_state_dict[f'{hf_module_key}.lora_A.{hf_param_key}'].load()
                hf_lora_B = hf_state_dict[f'{hf_module_key}.lora_B.{hf_param_key}'].load()
                self._set_weight(mg_lora_A, hf_lora_A, lora_A_key, offset, is_expert)
                self._set_weight(mg_lora_B, hf_lora_B, lora_B_key, offset, is_expert)
            else:
                lora_A_key = f'{module_key}.lora_A.{self._adapter_name}.{param_key}'
                lora_B_key = f'{module_key}.lora_B.{self._adapter_name}.{param_key}'
                lora_A_tensor = deep_getattr(mg_module, f'{lora_A_key}.data')
                lora_B_tensor = deep_getattr(mg_module, f'{lora_B_key}.data')
                hf_lora_A_key = f'{hf_module_key}.lora_A.{hf_param_key}'
                hf_lora_B_key = f'{hf_module_key}.lora_B.{hf_param_key}'
                lora_A, _ = self._get_weight(lora_A_tensor, lora_A_key, offset, is_expert)
                lora_B, _ = self._get_weight(lora_B_tensor, lora_B_key, offset, is_expert)
                if lora_A is not None:
                    self._peft_target_modules.add(hf_module_key)
                    hf_state_dict[hf_lora_A_key] = lora_A
                    hf_state_dict[hf_lora_B_key] = lora_B
        elif not self._is_peft_format or is_modules_to_save:
            if is_lora:
                mg_param = deep_getattr(sub_module, f'base_layer.{param_key}')
            else:
                mg_param = deep_getattr(sub_module, param_key)
            if to_mcore:
                assert mg_param is not None, f'mg_module: {mg_module}, mg_key: {mg_key}'
                hf_weight = hf_state_dict[hf_key].load()
                if module_key in {
                        'embedding.word_embeddings', 'output_layer'
                } and hf_weight.shape[0] < self.args.padded_vocab_size and self.args.task_type != 'seq_cls':
                    hf_weight = F.pad(hf_weight, (0, 0, 0, self.args.padded_vocab_size - hf_weight.shape[0]))
                hf_scale_inv = None
                if f'{hf_key}_scale_inv' in hf_state_dict:
                    hf_scale_inv = hf_state_dict[f'{hf_key}_scale_inv'].load()
                self._set_weight(mg_param, hf_weight, mg_key, offset, is_expert, hf_scale_inv=hf_scale_inv)
            else:
                if is_modules_to_save:
                    self._peft_modules_to_save.add(hf_module_key)
                weight, scale_inv = self._get_weight(None if mg_param is None else mg_param.data, mg_key, offset,
                                                     is_expert)
                if weight is not None:
                    hf_state_dict[hf_key] = weight
                if scale_inv is not None:
                    hf_state_dict[f'{hf_key}_scale_inv'] = scale_inv

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
        hidden_size_block = args.hidden_size // self.fp8_block_size
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
                self._set_weight(mg_attn.linear_qkv.lora_A[self._adapter_name].weight, lora_A,
                                 'linear_qkv.lora_A.weight')
                self._set_weight(mg_attn.linear_qkv.lora_B[self._adapter_name].weight, lora_B,
                                 'linear_qkv.lora_B.weight')
            elif not self._is_peft_format:
                linear_qkv_weight = torch.cat([
                    hf_state_dict['q_proj.weight'].load().reshape((num_query_groups, -1, args.hidden_size)),
                    hf_state_dict['k_proj.weight'].load().reshape((num_query_groups, -1, args.hidden_size)),
                    hf_state_dict['v_proj.weight'].load().reshape((num_query_groups, -1, args.hidden_size)),
                ],
                                              dim=1).reshape((-1, args.hidden_size))
                qkv_scale_inv = None
                if 'q_proj.weight_scale_inv' in hf_state_dict:
                    qkv_scale_inv = torch.cat([
                        hf_state_dict['q_proj.weight_scale_inv'].load().reshape(
                            (num_query_groups, -1, hidden_size_block)),
                        hf_state_dict['k_proj.weight_scale_inv'].load().reshape(
                            (num_query_groups, -1, hidden_size_block)),
                        hf_state_dict['v_proj.weight_scale_inv'].load().reshape(
                            (num_query_groups, -1, hidden_size_block)),
                    ],
                                              dim=1).reshape((-1, hidden_size_block))
                self._set_weight(
                    mg_attn.linear_qkv.weight, linear_qkv_weight, 'linear_qkv.weight', hf_scale_inv=qkv_scale_inv)
        else:
            q_dim, kv_dim = hf_attn.q_proj.weight.shape[0] // num_query_groups, hf_attn.k_proj.weight.shape[
                0] // num_query_groups
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
                    lora_B = lora_B.reshape((num_query_groups, -1, lora_B.shape[-1]))
                    hf_state_dict['q_proj.lora_B.weight'] = lora_B[:, :q_dim, :].reshape(-1, lora_B.shape[-1]).clone()
                    hf_state_dict['k_proj.lora_B.weight'] = lora_B[:,
                                                                   q_dim:-kv_dim, :].reshape(-1,
                                                                                             lora_B.shape[-1]).clone()
                    hf_state_dict['v_proj.lora_B.weight'] = lora_B[:, -kv_dim:, :].reshape(-1, lora_B.shape[-1]).clone()
            elif not self._is_peft_format:
                mg_attn_weight, scale_inv = self._get_weight(
                    None if mg_attn is None else mg_attn.linear_qkv.weight.data, 'linear_qkv.weight')
                if mg_attn_weight is not None:
                    mg_attn_weight = mg_attn_weight.reshape((num_query_groups, -1, args.hidden_size))
                    hf_state_dict['q_proj.weight'] = mg_attn_weight[:, :q_dim, :].reshape(-1, args.hidden_size).clone()
                    hf_state_dict['k_proj.weight'] = mg_attn_weight[:,
                                                                    q_dim:-kv_dim, :].reshape(-1,
                                                                                              args.hidden_size).clone()
                    hf_state_dict['v_proj.weight'] = mg_attn_weight[:, -kv_dim:, :].reshape(-1,
                                                                                            args.hidden_size).clone()
                if scale_inv is not None:
                    scale_inv = scale_inv.reshape((num_query_groups, -1, hidden_size_block))
                    hf_state_dict['q_proj.weight_scale_inv'] = scale_inv[:, :q_block, :].reshape(
                        -1, hidden_size_block).clone()
                    hf_state_dict['k_proj.weight_scale_inv'] = scale_inv[:, q_block:-kv_block, :].reshape(
                        -1, hidden_size_block).clone()
                    hf_state_dict['v_proj.weight_scale_inv'] = scale_inv[:, -kv_block:, :].reshape(
                        -1, hidden_size_block).clone()
                del mg_attn_weight
        self._set_state_dict(mg_attn, 'linear_proj.weight', hf_state_dict, 'o_proj.weight', to_mcore)
        if args.add_bias_linear:
            self._set_state_dict(mg_attn, 'linear_proj.bias', hf_state_dict, 'o_proj.bias', to_mcore)

        # Copy bias
        if (args.add_bias_linear or args.add_qkv_bias) and not self._is_peft_format:
            if to_mcore:
                linear_qkv_bias = torch.cat([
                    hf_state_dict['q_proj.bias'].load().reshape((num_query_groups, -1)),
                    hf_state_dict['k_proj.bias'].load().reshape((num_query_groups, -1)),
                    hf_state_dict['v_proj.bias'].load().reshape((num_query_groups, -1)),
                ],
                                            dim=1).reshape(-1)
                self._set_weight(mg_attn.linear_qkv.bias, linear_qkv_bias, 'linear_qkv.bias')
            else:
                mg_attn_bias, _ = self._get_weight(None if mg_attn is None else mg_attn.linear_qkv.bias.data,
                                                   'linear_qkv.bias')
                if mg_attn_bias is not None:
                    mg_attn_bias = mg_attn_bias.reshape((num_query_groups, -1))
                    hf_state_dict['q_proj.bias'] = mg_attn_bias[:, :q_dim].reshape(-1).clone()
                    hf_state_dict['k_proj.bias'] = mg_attn_bias[:, q_dim:-kv_dim].reshape(-1).clone()
                    hf_state_dict['v_proj.bias'] = mg_attn_bias[:, -kv_dim:].reshape(-1).clone()
        if getattr(args, 'softmax_type', 'vanilla') == 'learnable':
            self._set_state_dict(mg_attn, 'core_attention.softmax_offset', hf_state_dict, 'sinks', to_mcore)
        if args.qk_layernorm:
            self._set_qk_layernorm(mg_attn, hf_attn, hf_state_dict, to_mcore)
        if to_mcore:
            hf_state_dict = {}
        else:
            hf_state_dict = self._add_prefix(hf_state_dict, hf_prefix)
        return hf_state_dict

    def _set_qk_layernorm(self, mg_attn, hf_attn, hf_state_dict, to_mcore):
        hf_q_norm_key = 'q_norm.weight' if hasattr(hf_attn, 'q_norm') else 'query_layernorm.weight'
        hf_k_norm_key = 'k_norm.weight' if hasattr(hf_attn, 'k_norm') else 'key_layernorm.weight'
        self._set_state_dict(mg_attn, 'q_layernorm.weight', hf_state_dict, hf_q_norm_key, to_mcore)
        self._set_state_dict(mg_attn, 'k_layernorm.weight', hf_state_dict, hf_k_norm_key, to_mcore)

    def get_e_score_correction_bias_key(self, hf_mlp):
        if hasattr(hf_mlp, 'moe_statics'):
            hf_bias_key = 'moe_statics.e_score_correction_bias'
        else:
            hf_bias_key = 'gate.e_score_correction_bias'
        return hf_bias_key

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
        args = self.args
        hf_mlp = self._get_hf_mlp(layer_idx)
        if hasattr(hf_mlp, 'router'):
            hf_gate_key = 'router.weight'
        elif hasattr(hf_mlp.gate, 'wg'):
            hf_gate_key = 'gate.wg.weight'
        else:
            hf_gate_key = 'gate.weight'
        self._set_state_dict(mg_mlp, 'router.weight', hf_state_dict, hf_gate_key, to_mcore)
        if args.add_bias_linear:
            self._set_state_dict(mg_mlp, 'router.bias', hf_state_dict, hf_gate_key.replace('weight', 'bias'), to_mcore)
        if args.moe_router_enable_expert_bias:
            hf_bias_key = self.get_e_score_correction_bias_key(hf_mlp)
            self._set_state_dict(mg_mlp, 'router.expert_bias', hf_state_dict, hf_bias_key, to_mcore)

        if args.moe_shared_expert_intermediate_size:
            for key in ['shared_expert', 'shared_experts', 'shared_mlp']:
                if hasattr(hf_mlp, key):
                    hf_shared_expert_prefix = f'{key}.'
                    shared_expert = getattr(hf_mlp, key)
            hf_state_dict.update(
                self._set_mlp_state(
                    None if mg_mlp is None else mg_mlp.shared_experts,
                    hf_state_dict,
                    hf_shared_expert_prefix,
                    layer_idx,
                    to_mcore,
                    hf_mlp=shared_expert))
            if hasattr(hf_mlp, 'shared_expert_gate'):
                self._set_state_dict(mg_mlp, 'shared_experts.gate_weight', hf_state_dict, 'shared_expert_gate.weight',
                                     to_mcore)
        for ep_rank in range(self.ep_size):
            mg_experts = None if mg_mlp is None else mg_mlp.experts
            expert_available = ep_rank == self.ep_rank
            if not expert_available:
                if to_mcore:
                    continue
                else:
                    mg_experts = None
            hf_state_dict.update(
                self._set_mlp_state(mg_experts, hf_state_dict, 'experts.', layer_idx, to_mcore, ep_rank=ep_rank))
        if to_mcore:
            hf_state_dict = {}
        else:
            hf_state_dict = self._add_prefix(hf_state_dict, hf_prefix)
        return hf_state_dict

    def _get_hf_grouped(self):
        if self.args.hf_model_type in {
                'qwen2_moe', 'qwen3_moe', 'deepseek_v2', 'deepseek_v3', 'dots1', 'ernie4_5_moe', 'glm4_moe',
                'glm4_moe_lite', 'glm4v_moe', 'minimax_m2', 'olmoe', 'qwen3_next', 'kimi_vl', 'qwen3_omni_moe',
                'qwen3_vl_moe'
        }:
            return False, False
        return None, None

    def _set_mlp_state(self,
                       mg_mlp,
                       hf_state_dict,
                       hf_prefix: str,
                       layer_idx: int,
                       to_mcore: bool,
                       ep_rank: Optional[int] = None,
                       hf_mlp=None):
        if hf_mlp is None:
            hf_mlp = self._get_hf_mlp(layer_idx)
        is_expert = ep_rank is not None
        num_local_experts = 1
        hf_grouped = False
        args = self.args
        if is_expert:
            hf_grouped = not hasattr(hf_mlp.experts, '__len__')
            hf_mlp = hf_mlp.experts if hf_grouped else hf_mlp.experts[0]
            num_local_experts = args.num_experts // self.ep_size
        is_gate_up = hasattr(hf_mlp, 'gate_up_proj')
        # transformers 5.0 compatibility
        if self.is_transformers_5:
            _hf_grouped, _is_gate_up = self._get_hf_grouped()
            if _hf_grouped is not None:
                hf_grouped = _hf_grouped
            if _is_gate_up is not None:
                is_gate_up = _is_gate_up

        if to_mcore or hf_grouped:
            hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
        else:
            hf_state_dict = {}
        # linear_fc1
        if to_mcore:
            has_scale_inv = any('_scale_inv' in k for k in hf_state_dict.keys())
            if isinstance(mg_mlp.linear_fc1, LoraParallelLinear):
                mg_lora_B = mg_mlp.linear_fc1.lora_B[self._adapter_name]
                mg_lora_B = [getattr(mg_lora_B, f'weight{i}')
                             for i in range(num_local_experts)] if is_expert else mg_lora_B.weight
                if is_gate_up:
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
                mg_lora_A = mg_mlp.linear_fc1.lora_A[self._adapter_name]
                mg_lora_A = [getattr(mg_lora_A, f'weight{i}')
                             for i in range(num_local_experts)] if is_expert else mg_lora_A.weight
                self._set_weight(
                    mg_lora_A, lora_A, f'linear_fc1.lora_A.{self._adapter_name}.weight', is_expert=is_expert)
                self._set_weight(
                    mg_lora_B, lora_B, f'linear_fc1.lora_B.{self._adapter_name}.weight', is_expert=is_expert)
            elif not self._is_peft_format:
                fc1_weight = [getattr(mg_mlp.linear_fc1, f'weight{i}')
                              for i in range(num_local_experts)] if is_expert else mg_mlp.linear_fc1.weight
                fc1_bias = None
                if args.add_bias_linear:
                    assert is_expert and not has_scale_inv, 'not support'  # TODO
                    fc1_bias = [getattr(mg_mlp.linear_fc1, f'bias{i}') for i in range(num_local_experts)]
                gate_up_scale_inv = None
                if is_gate_up:
                    if is_expert:
                        if hf_grouped:
                            if 'gate_up_proj_blocks' in hf_state_dict:
                                blocks = hf_state_dict['gate_up_proj_blocks'].load()
                                scales = hf_state_dict['gate_up_proj_scales'].load()
                                gate_up_proj_weight = self.mxfp4_quantizer.convert(blocks, scales)
                            else:
                                gate_up_proj_weight = hf_state_dict['gate_up_proj'].load()
                            gate_up_proj_weight = gate_up_proj_weight.transpose(1, 2)
                            gate_up_proj_weight = gate_up_proj_weight[ep_rank * num_local_experts:(ep_rank + 1)
                                                                      * num_local_experts]
                            if has_scale_inv:
                                gate_up_scale_inv = hf_state_dict['gate_up_proj_scale_inv'].load().transpose(1, 2)
                                gate_up_scale_inv = gate_up_scale_inv[ep_rank * num_local_experts:(ep_rank + 1)
                                                                      * num_local_experts]
                            if fc1_bias is not None:
                                gate_up_proj_bias = hf_state_dict['gate_up_proj_bias'].load()
                                gate_up_proj_bias = gate_up_proj_bias[ep_rank * num_local_experts:(ep_rank + 1)
                                                                      * num_local_experts]
                            if args.llm_model_type == 'gpt_oss':
                                gate_proj_weight = gate_up_proj_weight[:, ::2]
                                up_proj_weight = gate_up_proj_weight[:, 1::2]
                                gate_proj_bias, up_proj_bias = gate_up_proj_bias[:, ::2], gate_up_proj_bias[:, 1::2]
                                gate_up_proj_weight = torch.concat([gate_proj_weight, up_proj_weight], dim=1)
                                gate_up_proj_bias = torch.concat([gate_proj_bias, up_proj_bias], dim=1)
                                del gate_proj_weight, up_proj_weight, gate_proj_bias, up_proj_bias
                        else:
                            gate_up_proj_weight = torch.concat([
                                hf_state_dict[f'{i + ep_rank * num_local_experts}.gate_up_proj.weight'].load()
                                for i in range(num_local_experts)
                            ],
                                                               dim=0)
                            if has_scale_inv:
                                gate_up_scale_inv = torch.concat([
                                    hf_state_dict[f'{i + ep_rank * num_local_experts}.gate_up_proj.weight_scale_inv'].
                                    load() for i in range(num_local_experts)
                                ],
                                                                 dim=0)

                        gate_up_proj_weight = gate_up_proj_weight.reshape(num_local_experts * 2, -1,
                                                                          gate_up_proj_weight.shape[-1])
                        if has_scale_inv:
                            gate_up_scale_inv = gate_up_scale_inv.reshape(num_local_experts * 2, -1,
                                                                          gate_up_scale_inv.shape[-1])
                    else:
                        gate_up_proj_weight = hf_state_dict['gate_up_proj.weight'].load()
                        gate_up_proj_weight = gate_up_proj_weight.view(2, -1, gate_up_proj_weight.shape[-1])
                        if has_scale_inv:
                            gate_up_scale_inv = hf_state_dict['gate_up_proj.weight_scale_inv'].load()
                            gate_up_scale_inv = gate_up_scale_inv.view(2, -1, gate_up_scale_inv.shape[-1])
                else:
                    if is_expert:
                        weight_list = []
                        start_idx = ep_rank * num_local_experts
                        for i in range(num_local_experts):
                            gate_proj_weight = hf_state_dict[f'{start_idx + i}.gate_proj.weight'].load()
                            up_proj_weight = hf_state_dict[f'{start_idx + i}.up_proj.weight'].load()
                            weight_list.append(torch.stack([gate_proj_weight, up_proj_weight], dim=0))
                        gate_up_proj_weight = torch.concat(weight_list, dim=0)
                        if has_scale_inv:
                            scale_inv_list = []
                            for i in range(num_local_experts):
                                gate_scale_inv = hf_state_dict[f'{start_idx + i}.gate_proj.weight_scale_inv'].load()
                                up_scale_inv = hf_state_dict[f'{start_idx + i}.up_proj.weight_scale_inv'].load()
                                scale_inv_list.append(torch.stack([gate_scale_inv, up_scale_inv], dim=0))
                            gate_up_scale_inv = torch.concat(scale_inv_list, dim=0)
                        del weight_list
                    else:
                        gate_proj_weight = hf_state_dict['gate_proj.weight'].load()
                        up_proj_weight = hf_state_dict['up_proj.weight'].load()
                        gate_up_proj_weight = torch.stack([gate_proj_weight, up_proj_weight], dim=0)
                        if has_scale_inv:
                            gate_scale_inv = hf_state_dict['gate_proj.weight_scale_inv'].load()
                            up_scale_inv = hf_state_dict['up_proj.weight_scale_inv'].load()
                            gate_up_scale_inv = torch.stack([gate_scale_inv, up_scale_inv], dim=0)
                self._set_weight(
                    fc1_weight,
                    gate_up_proj_weight,
                    'linear_fc1.weight',
                    is_expert=is_expert,
                    hf_scale_inv=gate_up_scale_inv)
                if fc1_bias is not None:
                    self._set_weight(
                        fc1_bias, gate_up_proj_bias, 'linear_fc1.bias', is_expert=is_expert, hf_scale_inv=None)
        else:
            is_lora = False if mg_mlp is None else isinstance(mg_mlp.linear_fc1,
                                                              LoraParallelLinear) and self._is_peft_format
            is_lora = torch.tensor([is_lora], dtype=torch.bool, device='cuda')
            if is_expert and self.ep_pp_size > 1:
                dist.all_reduce(is_lora, group=self.ep_pp_group)
            elif not is_expert and self.pp_size > 1:
                dist.all_reduce(is_lora, group=self.pp_group)
            if is_lora:
                if hf_grouped:
                    raise ValueError('Since this model\'s transformers and megatron have different expert '
                                     'weight organization methods, LoRA weight conversion is not supported. '
                                     'You can solve this issue by setting `--merge_lora true`.')
                if mg_mlp is None:
                    lora_A = None
                    lora_B = None
                else:
                    if is_expert:
                        lora_A = [
                            getattr(mg_mlp.linear_fc1.lora_A[self._adapter_name], f'weight{i}')
                            for i in range(num_local_experts)
                        ]
                        lora_B = [
                            getattr(mg_mlp.linear_fc1.lora_B[self._adapter_name], f'weight{i}')
                            for i in range(num_local_experts)
                        ]
                    else:
                        lora_A = mg_mlp.linear_fc1.lora_A[self._adapter_name].weight
                        lora_B = mg_mlp.linear_fc1.lora_B[self._adapter_name].weight
                lora_A, _ = self._get_weight(
                    lora_A, f'linear_fc1.lora_A.{self._adapter_name}.weight', is_expert=is_expert)
                lora_B, _ = self._get_weight(
                    lora_B, f'linear_fc1.lora_B.{self._adapter_name}.weight', is_expert=is_expert)
                if lora_A is not None:
                    if is_gate_up:
                        self._peft_target_modules.update({'gate_up_proj'})
                        if is_expert:
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
                            lora_B = lora_B.view(num_local_experts, 2, -1, lora_B.shape[-1])
                            for i in range(num_local_experts):
                                hf_i = i + ep_rank * num_local_experts
                                hf_state_dict[f'{hf_i}.gate_proj.lora_A.weight'] = lora_A[i].clone()
                                hf_state_dict[f'{hf_i}.up_proj.lora_A.weight'] = lora_A[i].clone()
                                hf_state_dict[f'{hf_i}.gate_proj.lora_B.weight'] = lora_B[i][0].clone()
                                hf_state_dict[f'{hf_i}.up_proj.lora_B.weight'] = lora_B[i][1].clone()
                        else:
                            lora_B = lora_B.view(2, -1, lora_B.shape[-1])
                            hf_state_dict['gate_proj.lora_A.weight'] = lora_A.clone()
                            hf_state_dict['up_proj.lora_A.weight'] = lora_A.clone()
                            hf_state_dict['gate_proj.lora_B.weight'] = lora_B[0].clone()
                            hf_state_dict['up_proj.lora_B.weight'] = lora_B[1].clone()
            elif not self._is_peft_format:
                fc1_bias = None
                if mg_mlp is None:
                    fc1_weight = None
                else:
                    if is_expert:
                        linear_fc1 = mg_mlp.linear_fc1
                        if isinstance(linear_fc1, LoraParallelLinear):
                            linear_fc1 = linear_fc1.base_layer
                        fc1_weight = [getattr(linear_fc1, f'weight{i}') for i in range(num_local_experts)]
                        if args.add_bias_linear:
                            fc1_bias = [getattr(linear_fc1, f'bias{i}') for i in range(num_local_experts)]
                    else:
                        fc1_weight = mg_mlp.linear_fc1.weight
                gate_up_proj_weight, scale_inv = self._get_weight(fc1_weight, 'linear_fc1.weight', is_expert=is_expert)
                gate_up_proj_bias = None
                if args.add_bias_linear:
                    gate_up_proj_bias, _ = self._get_weight(fc1_bias, 'linear_fc1.bias', is_expert=is_expert)
                del fc1_weight
                if gate_up_proj_weight is not None:
                    if is_gate_up:
                        if is_expert:
                            if hf_grouped:
                                gate_up_proj_weight = gate_up_proj_weight.transpose(1, 2)
                                if 'gate_up_proj' in hf_state_dict:
                                    gate_up_proj_weight = torch.concat(
                                        [hf_state_dict['gate_up_proj'], gate_up_proj_weight], dim=0)
                                is_last_ckpt = gate_up_proj_weight.shape[0] == args.num_experts
                                if args.llm_model_type == 'gpt_oss' and is_last_ckpt:
                                    gate_proj_weight, up_proj_weight = gate_up_proj_weight.chunk(2, dim=2)
                                    new_gate_up_proj_weight = torch.empty_like(gate_up_proj_weight)
                                    new_gate_up_proj_weight[..., ::2] = gate_proj_weight
                                    new_gate_up_proj_weight[..., 1::2] = up_proj_weight
                                    gate_up_proj_weight = new_gate_up_proj_weight
                                    del new_gate_up_proj_weight, gate_proj_weight, up_proj_weight
                                hf_state_dict['gate_up_proj'] = gate_up_proj_weight.clone()
                                if scale_inv is not None:
                                    scale_inv = scale_inv.transpose(1, 2)
                                    if 'gate_up_proj_scale_inv' in hf_state_dict:
                                        scale_inv = torch.concat([hf_state_dict['gate_up_proj_scale_inv'], scale_inv],
                                                                 dim=0)
                                    hf_state_dict['gate_up_proj_scale_inv'] = scale_inv.clone()

                                if gate_up_proj_bias is not None:
                                    if 'gate_up_proj_bias' in hf_state_dict:
                                        gate_up_proj_bias = torch.concat(
                                            [hf_state_dict['gate_up_proj_bias'], gate_up_proj_bias], dim=0)
                                    if args.llm_model_type == 'gpt_oss' and is_last_ckpt:
                                        gate_proj_bias, up_proj_bias = gate_up_proj_bias.chunk(2, dim=1)
                                        new_gate_up_proj_bias = torch.empty_like(gate_up_proj_bias)
                                        new_gate_up_proj_bias[:, ::2] = gate_proj_bias
                                        new_gate_up_proj_bias[:, 1::2] = up_proj_bias
                                        gate_up_proj_bias = new_gate_up_proj_bias
                                        del new_gate_up_proj_bias, gate_proj_bias, up_proj_bias
                                    hf_state_dict['gate_up_proj_bias'] = gate_up_proj_bias.clone()
                            else:
                                for i in range(num_local_experts):
                                    hf_i = i + ep_rank * num_local_experts
                                    hf_state_dict[f'{hf_i}.gate_up_proj.weight'] = gate_up_proj_weight[i].clone()
                                    if scale_inv is not None:
                                        hf_state_dict[f'{hf_i}.gate_up_proj.weight_scale_inv'] = scale_inv[i].clone()
                            del gate_up_proj_weight
                        else:
                            gate_up_proj_weight = gate_up_proj_weight.view(-1, gate_up_proj_weight.shape[-1])
                            hf_state_dict['gate_up_proj.weight'] = gate_up_proj_weight.clone()
                            if scale_inv is not None:
                                scale_inv = scale_inv.view(-1, scale_inv.shape[-1])
                                hf_state_dict['gate_up_proj.weight_scale_inv'] = scale_inv.clone()
                    else:
                        if is_expert:
                            gate_up_proj_weight = gate_up_proj_weight.view(num_local_experts, 2, -1,
                                                                           gate_up_proj_weight.shape[-1])
                            if scale_inv is not None:
                                scale_inv = scale_inv.view(num_local_experts, 2, -1, scale_inv.shape[-1])
                            for i in range(num_local_experts):
                                hf_i = i + ep_rank * num_local_experts
                                hf_state_dict[f'{hf_i}.gate_proj.weight'] = gate_up_proj_weight[i][0].clone()
                                hf_state_dict[f'{hf_i}.up_proj.weight'] = gate_up_proj_weight[i][1].clone()
                                if scale_inv is not None:
                                    hf_state_dict[f'{hf_i}.gate_proj.weight_scale_inv'] = scale_inv[i][0].clone()
                                    hf_state_dict[f'{hf_i}.up_proj.weight_scale_inv'] = scale_inv[i][1].clone()
                            del gate_up_proj_weight
                        else:
                            gate_up_proj_weight = gate_up_proj_weight.view(2, -1, gate_up_proj_weight.shape[-1])
                            hf_state_dict['gate_proj.weight'] = gate_up_proj_weight[0].clone()
                            hf_state_dict['up_proj.weight'] = gate_up_proj_weight[1].clone()
                            if scale_inv is not None:
                                scale_inv = scale_inv.view(2, -1, scale_inv.shape[-1])
                                hf_state_dict['gate_proj.weight_scale_inv'] = scale_inv[0].clone()
                                hf_state_dict['up_proj.weight_scale_inv'] = scale_inv[1].clone()

        # linear_fc2
        if is_expert:
            if to_mcore:
                if isinstance(mg_mlp.linear_fc2, LoraParallelLinear):
                    mg_lora_A = mg_mlp.linear_fc2.lora_A[self._adapter_name]
                    mg_lora_A = [getattr(mg_lora_A, f'weight{i}')
                                 for i in range(num_local_experts)] if is_expert else mg_lora_A.weight
                    mg_lora_B = mg_mlp.linear_fc2.lora_B[self._adapter_name]
                    mg_lora_B = [getattr(mg_lora_B, f'weight{i}')
                                 for i in range(num_local_experts)] if is_expert else mg_lora_B.weight
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
                    self._set_weight(
                        mg_lora_A, lora_A, f'linear_fc2.lora_A.{self._adapter_name}.weight', is_expert=is_expert)
                    self._set_weight(
                        mg_lora_B, lora_B, f'linear_fc2.lora_B.{self._adapter_name}.weight', is_expert=is_expert)
                elif not self._is_peft_format:
                    fc2_weight = [getattr(mg_mlp.linear_fc2, f'weight{i}')
                                  for i in range(num_local_experts)] if is_expert else mg_mlp.linear_fc2.weight
                    fc2_bias = None
                    if args.add_bias_linear:
                        fc2_bias = [getattr(mg_mlp.linear_fc2, f'bias{i}') for i in range(num_local_experts)]
                    down_scale_inv = None
                    if hf_grouped:
                        if 'down_proj_blocks' in hf_state_dict:
                            blocks = hf_state_dict['down_proj_blocks'].load()
                            scales = hf_state_dict['down_proj_scales'].load()
                            down_proj_weight = self.mxfp4_quantizer.convert(blocks, scales)
                        else:
                            down_proj_weight = hf_state_dict['down_proj'].load()
                        down_proj_weight = down_proj_weight.transpose(1, 2)
                        down_proj_weight = down_proj_weight[ep_rank * num_local_experts:(ep_rank + 1)
                                                            * num_local_experts].reshape(
                                                                -1, down_proj_weight.shape[-1])
                        if has_scale_inv:
                            down_scale_inv = hf_state_dict['down_proj_scale_inv'].load().transpose(1, 2)
                            down_scale_inv = down_scale_inv[ep_rank * num_local_experts:(ep_rank + 1)
                                                            * num_local_experts].reshape(-1, down_scale_inv.shape[-1])
                        if fc2_bias is not None:
                            down_proj_bias = hf_state_dict['down_proj_bias'].load()
                            down_proj_bias = down_proj_bias[ep_rank * num_local_experts:(ep_rank + 1)
                                                            * num_local_experts]
                    else:
                        down_proj_weight = torch.concat([
                            hf_state_dict[f'{i + ep_rank * num_local_experts}.down_proj.weight'].load()
                            for i in range(num_local_experts)
                        ],
                                                        dim=0)
                        if has_scale_inv:
                            down_scale_inv = torch.concat([
                                hf_state_dict[f'{i + ep_rank * num_local_experts}.down_proj.weight_scale_inv'].load()
                                for i in range(num_local_experts)
                            ],
                                                          dim=0)
                    self._set_weight(
                        fc2_weight,
                        down_proj_weight,
                        'linear_fc2.weight',
                        is_expert=is_expert,
                        hf_scale_inv=down_scale_inv)
                    if fc2_bias is not None:
                        self._set_weight(
                            fc2_bias, down_proj_bias, 'linear_fc2.bias', is_expert=is_expert, hf_scale_inv=None)
            else:
                is_lora = False if mg_mlp is None else isinstance(mg_mlp.linear_fc2,
                                                                  LoraParallelLinear) and self._is_peft_format
                is_lora = torch.tensor([is_lora], dtype=torch.bool, device='cuda')
                if is_expert and self.ep_pp_size > 1:
                    dist.all_reduce(is_lora, group=self.ep_pp_group)
                elif not is_expert and self.pp_size > 1:
                    dist.all_reduce(is_lora, group=self.pp_group)
                if is_lora:
                    if hf_grouped:
                        raise ValueError('Since this model\'s transformers and megatron have different expert '
                                         'weight organization methods, LoRA weight conversion is not supported. '
                                         'You can solve this issue by setting `--merge_lora true`.')
                    if mg_mlp is None:
                        lora_A = None
                        lora_B = None
                    else:
                        lora_A = [
                            getattr(mg_mlp.linear_fc2.lora_A[self._adapter_name], f'weight{i}')
                            for i in range(num_local_experts)
                        ]
                        lora_B = [
                            getattr(mg_mlp.linear_fc2.lora_B[self._adapter_name], f'weight{i}')
                            for i in range(num_local_experts)
                        ]
                    lora_A, _ = self._get_weight(
                        lora_A, f'linear_fc2.lora_A.{self._adapter_name}.weight', is_expert=is_expert)
                    lora_B, _ = self._get_weight(
                        lora_B, f'linear_fc2.lora_B.{self._adapter_name}.weight', is_expert=is_expert)
                    if lora_A is not None:
                        self._peft_target_modules.update({'down_proj'})
                        for i in range(num_local_experts):
                            hf_i = i + ep_rank * num_local_experts
                            hf_state_dict[f'{hf_i}.down_proj.lora_A.weight'] = lora_A[i].clone()
                            hf_state_dict[f'{hf_i}.down_proj.lora_B.weight'] = lora_B[i].clone()
                elif not self._is_peft_format:
                    fc2_bias = None
                    if mg_mlp is None:
                        fc2_weight = None
                    else:
                        linear_fc2 = mg_mlp.linear_fc2
                        if isinstance(linear_fc2, LoraParallelLinear):
                            linear_fc2 = linear_fc2.base_layer
                        fc2_weight = [getattr(linear_fc2, f'weight{i}') for i in range(num_local_experts)]
                        if args.add_bias_linear:
                            fc2_bias = [getattr(linear_fc2, f'bias{i}') for i in range(num_local_experts)]
                    down_proj_weight, scale_inv = self._get_weight(fc2_weight, 'linear_fc2.weight', is_expert=is_expert)
                    if args.add_bias_linear:
                        down_proj_bias, _ = self._get_weight(fc2_bias, 'linear_fc2.bias', is_expert=is_expert)
                    del fc2_weight, fc2_bias
                    if down_proj_weight is not None:
                        if hf_grouped:
                            down_proj_weight = down_proj_weight.transpose(1, 2)
                            if 'down_proj' in hf_state_dict:
                                down_proj_weight = torch.concat([hf_state_dict['down_proj'], down_proj_weight], dim=0)
                            hf_state_dict['down_proj'] = down_proj_weight.clone()
                            if scale_inv is not None:
                                scale_inv = scale_inv.transpose(1, 2)
                                if 'down_proj_scale_inv' in hf_state_dict:
                                    scale_inv = torch.concat([hf_state_dict['down_proj_scale_inv'], scale_inv], dim=0)
                                hf_state_dict['down_proj_scale_inv'] = scale_inv.clone()
                            if args.add_bias_linear:
                                if 'down_proj_bias' in hf_state_dict:
                                    down_proj_bias = torch.concat([hf_state_dict['down_proj_bias'], down_proj_bias],
                                                                  dim=0)
                                hf_state_dict['down_proj_bias'] = down_proj_bias.clone()
                        else:
                            for i in range(num_local_experts):
                                hf_i = i + ep_rank * num_local_experts
                                hf_state_dict[f'{hf_i}.down_proj.weight'] = down_proj_weight[i].clone()
                                if scale_inv is not None:
                                    hf_state_dict[f'{hf_i}.down_proj.weight_scale_inv'] = scale_inv[i].clone()
        else:
            self._set_state_dict(
                mg_mlp, 'linear_fc2.weight', hf_state_dict, 'down_proj.weight', to_mcore, is_expert=is_expert)
        if to_mcore:
            hf_state_dict = {}
        else:
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
            if self.args.q_lora_rank is not None:
                self._set_state_dict(mg_attn, 'linear_q_up_proj.layer_norm_weight', hf_state_dict,
                                     'q_a_layernorm.weight', to_mcore)
            self._set_state_dict(mg_attn, 'linear_kv_up_proj.layer_norm_weight', hf_state_dict, 'kv_a_layernorm.weight',
                                 to_mcore)
        if to_mcore:
            hf_state_dict = {}
        else:
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
        hf_mlp_prefix = self.get_hf_mlp_prefix(layer_idx)
        hf_mlp = self._get_hf_mlp(layer_idx)
        is_moe = self._is_moe(hf_mlp.state_dict())
        mg_mlp = None if mg_layer is None else mg_layer.mlp
        if is_moe:
            hf_state_dict.update(self._set_moe_state(mg_mlp, hf_state_dict, f'{hf_mlp_prefix}.', layer_idx, to_mcore))
            self._set_state_dict(mg_layer, 'pre_mlp_layernorm.weight', hf_state_dict, 'post_attention_layernorm.weight',
                                 to_mcore)
        else:
            hf_state_dict.update(self._set_mlp_state(mg_mlp, hf_state_dict, f'{hf_mlp_prefix}.', layer_idx, to_mcore))
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
        if to_mcore:
            hf_state_dict = {}
        else:
            hf_state_dict = self._add_prefix(hf_state_dict, hf_prefix)
        return hf_state_dict

    def _convert_pre_process(self, mg_model, hf_state_dict, hf_prefix: str, to_mcore):
        if to_mcore:
            hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
        else:
            hf_state_dict = {}
        lm_model = getattr(mg_model, 'language_model') if self.args.is_multimodal else mg_model
        self._set_state_dict(lm_model, 'embedding.word_embeddings.weight', hf_state_dict, self.hf_embed_key, to_mcore)
        if self.args.is_multimodal:
            for prefix, mg_prefix in self.module_mapping.items():
                mg_module = deep_getattr(mg_model, f'visual.{mg_prefix}')
                hf_state_dict.update(self._set_module(mg_module, hf_state_dict, f'{hf_prefix}{prefix}.', to_mcore))
        if to_mcore:
            hf_state_dict = {}
        else:
            hf_state_dict = self._add_prefix(hf_state_dict, hf_prefix)
        return hf_state_dict

    def _convert_post_process(self, mg_model, hf_state_dict, hf_prefix: str, to_mcore):
        if to_mcore:
            hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
        else:
            hf_state_dict = {}
        lm_model = getattr(mg_model, 'language_model') if self.args.is_multimodal else mg_model
        if self.args.task_type != 'embedding':
            if self.args.untie_embeddings_and_output_weights:
                hf_lm_head_key = self.hf_lm_head_key
                if self.args.task_type == 'seq_cls':
                    hf_lm_head_key = self.hf_score_key
                if not to_mcore or hf_lm_head_key in hf_state_dict:
                    self._set_state_dict(lm_model, 'output_layer.weight', hf_state_dict, hf_lm_head_key, to_mcore)
            elif to_mcore and lm_model.output_layer.weight is not None:
                self._set_state_dict(lm_model, 'output_layer.weight', hf_state_dict, self.hf_embed_key, to_mcore)
        self._set_state_dict(lm_model, 'decoder.final_layernorm.weight', hf_state_dict, self.hf_final_layernorm_key,
                             to_mcore)
        if to_mcore:
            hf_state_dict = {}
        else:
            hf_state_dict = self._add_prefix(hf_state_dict, hf_prefix)
        return hf_state_dict

    def _convert_hf_state_dict(self, hf_state_dict, to_mcore):
        res = {}
        for k, v in hf_state_dict.items():
            for old_key, new_key in self.hf_state_dict_mapping.items():
                if not to_mcore:
                    old_key, new_key = new_key, old_key
                if k.startswith(old_key):
                    k = k.replace(old_key, new_key)
                    break
            res[k] = v
        return res

    def _convert(self, mg_models, hf_state_dict, hf_prefix: str, to_mcore: bool, tqdm_desc: str = 'Converting: '):
        if to_mcore:
            hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
            hf_state_dict = self._convert_hf_state_dict(hf_state_dict, to_mcore)
        else:
            hf_state_dict = {}
        mg_models = iter(mg_models)
        mg_model = next(mg_models)
        if mcore_013:
            is_pp_first_stage = mpu.is_pipeline_first_stage(ignore_virtual=False, vp_stage=mg_model.vp_stage)
            is_pp_last_stage = mpu.is_pipeline_last_stage(ignore_virtual=False, vp_stage=mg_model.vp_stage)
        else:
            is_pp_first_stage = mpu.is_pipeline_first_stage()
            is_pp_last_stage = mpu.is_pipeline_last_stage()
        if not to_mcore or is_pp_first_stage:
            hf_state_dict.update(self._convert_pre_process(mg_model, hf_state_dict, '', to_mcore))
        if to_mcore:
            yield
        else:
            yield from list(self._add_prefix(hf_state_dict, hf_prefix).items())
            hf_state_dict = {}
        layer_idx = 0
        prog_bar = tqdm(range(self.args.num_layers), dynamic_ncols=True, desc=tqdm_desc, disable=self.disable_tqmd)
        while layer_idx < self.args.num_layers:
            lm_model = getattr(mg_model, 'language_model') if self.args.is_multimodal else mg_model
            if len(lm_model.decoder.layers) > 0:
                start_idx = lm_model.decoder.layers[0].layer_number - 1
                mg_layer_available = (start_idx <= layer_idx < lm_model.decoder.layers[-1].layer_number)
            else:
                mg_layer_available = False
            if mg_layer_available:
                mg_layer = lm_model.decoder.layers[layer_idx - start_idx]
            else:
                if to_mcore:
                    layer_idx += 1
                    prog_bar.update()
                    continue
                else:
                    mg_layer = None
            if not to_mcore and self.pp_size > 1:
                has_model = torch.tensor([mg_layer is not None], dtype=torch.bool, device='cuda')
                dist.all_reduce(has_model, group=self.pp_group)
                if not has_model:
                    mg_model = next(mg_models)  # compat vpp
                    continue
            res = self._set_layer_state(mg_layer, hf_state_dict, f'{self.hf_layers_prefix}.', layer_idx, to_mcore)
            layer_idx += 1
            prog_bar.update()
            if to_mcore:
                yield
            else:
                yield from list(self._add_prefix(res, hf_prefix).items())
                hf_state_dict = {}

        if (not to_mcore or is_pp_last_stage) and self.args.mtp_num_layers:
            lm_model = getattr(mg_model, 'language_model') if self.args.is_multimodal else mg_model
            if to_mcore and self.pp_rank > 0:
                self._set_state_dict(lm_model, 'embedding.word_embeddings.weight', hf_state_dict, self.hf_embed_key,
                                     to_mcore)
            layer_idx = 0
            while layer_idx < self.args.mtp_num_layers:
                res = self._convert_mtp_layer(lm_model, hf_state_dict, f'{self.hf_mtp_prefix}.', layer_idx, to_mcore)
                layer_idx += 1
                if to_mcore:
                    yield
                else:
                    yield from list(self._add_prefix(res, hf_prefix).items())
                    hf_state_dict = {}
        if not to_mcore or is_pp_last_stage:
            hf_state_dict.update(self._convert_post_process(mg_model, hf_state_dict, '', to_mcore))
        if to_mcore:
            yield
        else:
            hf_state_dict = self._convert_hf_state_dict(hf_state_dict, to_mcore)
            yield from list(self._add_prefix(hf_state_dict, hf_prefix).items())

    def _convert_mtp_extra(self, mtp_layer, hf_state_dict, to_mcore, origin_hf_state_dict):
        for key in ['enorm.weight', 'hnorm.weight', 'eh_proj.weight']:
            self._set_state_dict(mtp_layer, key, hf_state_dict, key, to_mcore)
        self._set_state_dict(mtp_layer, 'final_layernorm.weight', hf_state_dict, 'shared_head.norm.weight', to_mcore)

    def _convert_mtp_layer(self, lm_model, hf_state_dict, hf_prefix: str, layer_idx: int, to_mcore: bool):
        mtp_layer = lm_model.mtp.layers[layer_idx] if hasattr(lm_model, 'mtp') else None
        if self.hf_mtp_prefix == self.hf_layers_prefix:
            hf_layer_idx = layer_idx + self.args.num_layers
        else:
            hf_layer_idx = layer_idx
        hf_prefix = f'{hf_prefix}{hf_layer_idx}.'
        if to_mcore:
            origin_hf_state_dict = hf_state_dict
            hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
            if len(hf_state_dict) == 0:
                logger.info_if(
                    f'MTP Layer {mtp_layer.layer_number} safetensors weights not found, '
                    'this part will be randomly initialized.',
                    cond=is_last_rank())
                for param in mtp_layer.parameters():
                    if param.ndim == 2:
                        mtp_layer.config.init_method(param.data)
                return {}
        else:
            origin_hf_state_dict = {}
            hf_state_dict = {}
        self._convert_mtp_extra(mtp_layer, hf_state_dict, to_mcore, origin_hf_state_dict)
        transformer_layer = None if mtp_layer is None else mtp_layer.transformer_layer
        if not to_mcore and not self.args.hf_model_type.startswith('qwen3_next'):
            self._set_state_dict(lm_model, 'embedding.word_embeddings.weight', hf_state_dict, 'embed_tokens.weight',
                                 to_mcore)
            if self.args.untie_embeddings_and_output_weights:
                self._set_state_dict(lm_model, 'output_layer.weight', hf_state_dict, 'shared_head.head.weight',
                                     to_mcore)
        hf_state_dict.update(self._set_layer_attn(transformer_layer, hf_state_dict, -1, to_mcore))
        hf_state_dict.update(self._set_layer_mlp(transformer_layer, hf_state_dict, -1, to_mcore))
        if to_mcore:
            hf_state_dict = {}
        else:
            hf_state_dict = self._add_prefix(hf_state_dict, hf_prefix)
            hf_state_dict.update(origin_hf_state_dict)
        return hf_state_dict

    def load_weights(self, mg_model, hf_model_dir: str, is_peft_format: bool = False, adapter_name: str = 'default'):
        self._is_peft_format = is_peft_format
        self._adapter_name = adapter_name
        hf_model_dir = safe_snapshot_download(hf_model_dir, use_hf=self.args.use_hf, hub_token=self.args.hub_token)
        with torch.no_grad(), SafetensorLazyLoader(hf_model_dir, is_peft_format=is_peft_format) as loader:
            state_dict = loader.get_state_dict()
            hf_prefix = 'base_model.model.' if is_peft_format else ''
            list(self._convert([mg_model], state_dict, hf_prefix, True, 'Loading: '))

    def export_weights(self,
                       mg_models,
                       target_device=None,
                       only_last_rank: bool = False,
                       is_peft_format: bool = False,
                       tqdm_desc: str = 'Exporting: '):
        self._target_device = target_device
        self._only_last_rank = only_last_rank
        self._is_peft_format = is_peft_format
        self._adapter_name = 'default'
        self._peft_target_modules = set()
        self._peft_modules_to_save = set()
        hf_prefix = 'base_model.model.' if is_peft_format else ''
        with torch.no_grad():
            yield from self._convert(mg_models, {}, hf_prefix, False, tqdm_desc=tqdm_desc)

    def save_weights(self,
                     mg_models,
                     output_dir: str,
                     is_peft_format: bool = False,
                     processor=None,
                     config=None) -> None:
        """Save the mg_model checkpoint in HF format"""
        torch.cuda.empty_cache()
        saver = StreamingSafetensorSaver(
            save_dir=output_dir, max_shard_size=self.args.max_shard_size, is_peft_format=is_peft_format)
        for k, v in self.export_weights(
                mg_models, target_device='cpu', only_last_rank=True, is_peft_format=is_peft_format,
                tqdm_desc='Saving: '):
            saver.add_tensor(k, v)
        saver.finalize()
        args = self.args
        processor = processor if processor is not None else self.processor
        if config is None:
            config = self.hf_model.config
        config = copy(config)
        if is_last_rank():
            if is_peft_format:
                peft_config = copy(mg_models[0].peft_config[self._adapter_name])
                if args.task_type == 'seq_cls':
                    peft_config.task_type = 'SEQ_CLS'
                if args.is_multimodal and 'all-linear' in args.target_modules:
                    peft_config.target_modules = get_multimodal_target_regex(
                        self.hf_model,
                        freeze_llm=args.freeze_llm,
                        freeze_vit=args.freeze_vit,
                        freeze_aligner=args.freeze_aligner,
                        include_embedding='all-embedding' in args.target_modules,
                        exclude_router='all-router' not in args.target_modules)
                else:
                    assert not isinstance(peft_config.target_modules, str), (
                        'target_regex is not currently supported for LoRA conversion. Please set `--merge_lora true`.')
                    peft_config.target_modules = self._peft_target_modules
                peft_config.modules_to_save = self._peft_modules_to_save
                peft_config.save_pretrained(output_dir)
            else:
                if args.mtp_num_layers:
                    config.num_nextn_predict_layers = args.mtp_num_layers
                config.vocab_size = args.padded_vocab_size
                if args.fp8 is not None and args.fp8_recipe == 'blockwise' and args.fp8_param_gather:
                    if getattr(config, 'quantization_config', None) is None:
                        from transformers.utils.quantization_config import FineGrainedFP8Config
                        modules_to_not_convert = get_modules_to_not_convert(self.hf_model)
                        config.quantization_config = FineGrainedFP8Config(modules_to_not_convert=modules_to_not_convert)
                elif hasattr(config, 'quantization_config'):
                    del config.quantization_config
                config.save_pretrained(output_dir)
                if getattr(self.hf_model, '_auto_class') is not None:
                    try:
                        custom_object_save(self.hf_model, output_dir, config=config)
                    except FileNotFoundError as e:
                        logger.error(f'custom_object_save Error: {e}')
                save_checkpoint(
                    None,
                    processor,
                    output_dir,
                    model_dirs=[args.model_dir],
                    additional_saved_files=self.hf_model.model_meta.additional_saved_files)
            logger.info_if(f'Successfully saved `safetensors` model weights in `{output_dir}`.', cond=is_last_rank())
        dist.barrier()  # Ensure all weights are saved completely


class MultimodalGPTBridge(GPTBridge):
    hf_layers_prefix = 'model.language_model.layers'
    hf_embed_key = 'model.language_model.embed_tokens.weight'
    hf_final_layernorm_key = 'model.language_model.norm.weight'

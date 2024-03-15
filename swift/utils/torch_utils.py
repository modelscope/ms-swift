# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import random
import socket
import types
import einops
import math
from bisect import bisect_right
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import Module

from .logger import get_logger, is_master

logger = get_logger()


def is_on_same_device(model: torch.nn.Module) -> bool:
    device_set = set(map(lambda p: p.device, model.parameters()))
    return len(device_set) == 1


def _find_free_port() -> str:
    # Copied from https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/launch.py # noqa: E501
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def seed_everything(seed: Optional[int] = None,
                    gpu_deterministic: bool = False) -> int:
    if seed is None:
        seed_max = np.iinfo(np.int32).max
        seed = random.randint(0, seed_max)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f'Global seed set to {seed}')
    if gpu_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f'Setting deterministic: {True}, benchmark: {False}')
    return seed


def get_model_info(model: Module, name: Optional[str] = None) -> str:
    if name is None:
        name = model.__class__.__name__

    n_params = sum(p.numel() for p in model.parameters())
    n_grads = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_buffers = sum(p.numel() for p in model.buffers())

    n_params /= 1e6
    n_grads /= 1e6
    n_buffers /= 1e6
    s = (f'{name}: '
         f'{n_params:.4f}M Params ({n_grads:.4f}M Trainable '
         f'[{100 * n_grads / n_params:.4f}%]), '
         f'{n_buffers:.4f}M Buffers.')
    return s


def find_sub_module(module: torch.nn.Module,
                    module_name: str) -> List[torch.nn.Module]:
    _modules = list()
    for name, sub_module in module.named_modules():
        if not name:
            continue
        if name.endswith(module_name):
            _modules.append(sub_module)
    return _modules


def get_dist_setting() -> Tuple[int, int, int, int]:
    """return rank, local_rank, world_size, local_world_size"""
    rank = int(os.getenv('RANK', -1))
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    local_world_size = int(os.getenv('LOCAL_WORLD_SIZE', 1))
    return rank, local_rank, world_size, local_world_size


def is_local_master():
    local_rank = get_dist_setting()[1]
    return local_rank in {-1, 0}


def use_torchacc() -> bool:
    return os.getenv('USE_TORCHACC', '0') == '1'


def is_dist():
    """Determine if the training is distributed"""
    if use_torchacc():
        return False
    rank, local_rank, _, _ = get_dist_setting()
    return rank >= 0 and local_rank >= 0


def is_ddp_plus_mp() -> bool:
    if use_torchacc():
        return False
    if not is_dist():
        return False
    n_gpu = torch.cuda.device_count()
    local_world_size = get_dist_setting()[3]
    assert n_gpu % local_world_size == 0
    if n_gpu // local_world_size >= 2:
        logger.info('Using DDP + MP(device_map)')
        return True
    return False


def show_layers(model: Module, max_lines: Optional[int] = 20) -> None:
    named_p = list(model.named_parameters())
    for i, (n, p) in enumerate(named_p):
        if max_lines is not None and i >= max_lines:
            logger.info('...')
            break
        logger.info(
            f'[{n}]: requires_grad={p.requires_grad}, dtype={p.dtype}, device={p.device}'
        )


def freeze_model_parameters(model: Module, freeze_parameters: float) -> None:
    n_parameters = np.array([p.numel() for p in model.parameters()],
                            dtype=np.int64)
    n_freeze_parameters = int(np.sum(n_parameters) * freeze_parameters)
    n_parameters_cs = np.cumsum(n_parameters)
    idx = bisect_right(n_parameters_cs, n_freeze_parameters)
    for _, p in zip(range(idx), model.parameters()):
        p.requires_grad = False


def activate_model_parameters(
        model: Module, additional_trainable_parameters: List[int]) -> None:
    if len(additional_trainable_parameters) == 0:
        return
    has_activate = False
    for n, p in model.named_parameters():
        for additional_tp in additional_trainable_parameters:
            if n.startswith(additional_tp):
                p.requires_grad = True
                has_activate = True
    if not has_activate:
        logger.warning(
            'len(additional_trainable_parameters) > 0 but no parameters are activated. '
            f'additional_trainable_parameters: {additional_trainable_parameters}'
        )


def broadcast_string(string: Optional[str], buffer_size: int = 1024) -> str:
    """String broadcasting in case of DDP
    string: main rank: str
        other rank: None or str(not use)
    return: all rank: str
    """
    assert dist.is_initialized()
    rank, local_rank, _, _ = get_dist_setting()
    assert rank >= 0
    if rank == 0:
        assert string is not None
        tensor = torch.tensor(
            [ord(c) for c in string] + [0] * (buffer_size - len(string)),
            dtype=torch.int64,
            device=local_rank)
    else:
        tensor = torch.zeros(buffer_size, dtype=torch.int64, device=local_rank)
    dist.broadcast(tensor, 0)
    first_zero = (tensor == 0).nonzero()[0].item()
    res = tensor.tolist()[:first_zero]
    return ''.join([chr(x) for x in res])


def time_synchronize() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()  # second

def patch_acc_model(model, args):
    
    # patah qwen
    if args.model_type.startswith('qwen'):
        import torchacc as ta
        model = ta.patch_qwen_model(model)
    elif args.model_type.startswith('baichuan'):
        model = patch_baichuan_model(model)
        # pass
    elif args.model_type.startswith('llama'):
        model = patch_llama_model(model)
    elif args.model_type.startswith('chatglm'):
        model = patah_chatglm_model(model)
    return model


def patch_llama_model(model):

    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            q (`torch.Tensor`): The query tensor.
            k (`torch.Tensor`): The key tensor.
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            position_ids (`torch.Tensor`):
                The position indices of the tokens corresponding to the query and key tensors. For example, this can be
                used to pass offsetted position ids when working with a KV-cache.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
                that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
                k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
                the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        Returns:
            `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    def llama_attn_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        from torchacc.ops import flash_attn_varlen_xla

        bsz, q_len, _ = hidden_states.size()

        query_states = (
            self.q_proj(hidden_states).view(bsz, q_len, self.num_heads,
                                            self.head_dim).transpose(1, 2))
        key_states = (
            self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads,
                                            self.head_dim).transpose(1, 2))
        value_states = (
            self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads,
                                            self.head_dim).transpose(1, 2))

        kv_seq_len = key_states.shape[-2]
        assert past_key_value is None, "past_key_value is not supported"

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                        cos, sin, position_ids)
        assert not output_attentions, "output_attentions is not supported"

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        past_key_value = (key_states, value_states) if use_cache else None

        # See https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attention.py
        # if attention_mask is not None:
        #     value_states = value_states * attention_mask.unsqueeze(1).unsqueeze(-1)
        q = einops.rearrange(query_states, "b h s ... -> (b s) h ...")
        k = einops.rearrange(key_states, "b h s ... -> (b s) h ...")
        v = einops.rearrange(value_states, "b h s ... -> (b s) h ...")
        max_s = q_len
        cu_q_lens = torch.arange(
            0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device=q.device)
        output = flash_attn_varlen_xla(
            q,
            k,
            v,
            cu_q_lens,
            cu_q_lens,
            max_s,
            max_s,
            0.0,
            softmax_scale=None,
            causal=True)
        output = einops.rearrange(output, "(b s) ... -> b s ...", b=bsz)

        return self.o_proj(einops.rearrange(
            output, "b s h d -> b s (h d)")), None, past_key_value
    #     self,
    #     hidden_states: torch.Tensor,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.Tensor] = None,
    #     past_key_value: Optional[Tuple[torch.Tensor]] = None,
    #     output_attentions: bool = False,
    #     use_cache: bool = False,
    # ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    #     from torchacc.ops import flash_attn_varlen_qkvpacked_xla

    #     bsz, q_len, _ = hidden_states.size()

    #     query_states = (
    #         self.q_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads,
    #                                         self.head_dim).transpose(1, 2))
    #     key_states = (
    #         self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads,
    #                                         self.head_dim).transpose(1, 2))
    #     value_states = (
    #         self.v_proj(hidden_states).view(bsz, q_len, self.num_heads,
    #                                         self.head_dim).transpose(1, 2))

    #     kv_seq_len = key_states.shape[-2]
    #     assert past_key_value is None, "past_key_value is not supported"

    #     cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    #     query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
    #                                                     cos, sin, position_ids)
    #     assert not output_attentions, "output_attentions is not supported"

    #     if past_key_value is not None:
    #         key_states = torch.cat([past_key_value[0], key_states], dim=2)
    #         value_states = torch.cat([past_key_value[1], value_states], dim=2)
    #     past_key_value = (key_states, value_states) if use_cache else None

    #     # See https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attention.py
    #     # if attention_mask is not None:
    #     #     value_states = value_states * attention_mask.unsqueeze(1).unsqueeze(-1)
    #     # qkv = torch.stack([query_states, key_states, value_states], dim=2)
    #     # qkv = qkv.transpose(1, 3)

    #     # qkv = einops.rearrange(qkv, "b s ... -> (b s) ...")
    #     # max_s = q_len
    #     # cu_q_lens = torch.arange(
    #     #     0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device=qkv.device)
    #     # output = flash_attn_varlen_qkvpacked_xla(
    #     #     qkv, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True)
    #     # output = einops.rearrange(output, "(b s) ... -> b s ...", b=bsz)

    #     # return self.o_proj(einops.rearrange(
    #     #     output, "b s h d -> b s (h d)")), None, past_key_value

    #     from torchacc.ops import flash_attn_varlen_xla
    #     query_states = query_states.transpose(1, 2)
    #     key_states = key_states.transpose(1, 2)
    #     value_states = value_states.transpose(1, 2)
    #     q, k, v = [einops.rearrange(x, "b s ... -> (b s) ...") for x in [query_states, key_states, value_states]]
    #     cu_q_lens = torch.arange(
    #         0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device=q.device)
    #     output = flash_attn_varlen_xla(q, k, v, cu_q_lens, cu_q_lens, q_len, q_len, 0.0, softmax_scale=None, causal=True)
    #     output = einops.rearrange(output, "(b s) ... -> b s ...", b=bsz)
    #     output = self.o_proj(einops.rearrange(output, "b s h d -> b s (h d)"))
    #     return output, None, past_key_value

    for layer in model.model.layers:
        layer.self_attn.forward = types.MethodType(llama_attn_forward, layer.self_attn)
    
    return model

def patah_chatglm_model(model):
    def apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
        # x: [sq, b, np, hn]
        sq, b, np, hn = x.size(0), x.size(1), x.size(2), x.size(3)
        rot_dim = rope_cache.shape[-2] * 2
        x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
        # truncate to support variable sizes
        rope_cache = rope_cache[:sq]
        xshaped = x.reshape(sq, -1, np, rot_dim // 2, 2)
        rope_cache = rope_cache.view(sq, -1, 1, xshaped.size(3), 2)
        x_out2 = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )
        x_out2 = x_out2.flatten(3)
        return torch.cat((x_out2, x_pass), dim=-1)

    def chatglm_attn_forward(
        self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True
    ):
        # hidden_states: [sq, b, h]

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        # =====================
        # Query, Key, and Value
        # =====================

        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        mixed_x_layer = self.query_key_value(hidden_states)

        if self.multi_query_attention:
            (query_layer, key_layer, value_layer) = mixed_x_layer.split(
                [
                    self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                ],
                dim=-1,
            )
            query_layer = query_layer.view(
                query_layer.size()[:-1] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )
            key_layer = key_layer.view(
                key_layer.size()[:-1] + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
            value_layer = value_layer.view(
                value_layer.size()[:-1]
                + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
        else:
            new_tensor_shape = mixed_x_layer.size()[:-1] + \
                               (self.num_attention_heads_per_partition,
                                3 * self.hidden_size_per_attention_head)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

        # apply relative positional encoding (rotary embedding)
        if rotary_pos_emb is not None:
            query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
            key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)

        # adjust key and value for inference
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            key_layer = torch.cat((cache_k, key_layer), dim=0)
            value_layer = torch.cat((cache_v, value_layer), dim=0)
        if use_cache:
            kv_cache = (key_layer, value_layer)
        else:
            kv_cache = None

        if self.multi_query_attention:
            key_layer = key_layer.unsqueeze(-2)
            key_layer = key_layer.expand(
                -1, -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1
            )
            key_layer = key_layer.contiguous().view(
                key_layer.size()[:2] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )
            value_layer = value_layer.unsqueeze(-2)
            value_layer = value_layer.expand(
                -1, -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1
            )
            value_layer = value_layer.contiguous().view(
                value_layer.size()[:2] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )

        # ==================================
        # core attention computation
        # ==================================

        from torchacc.ops import flash_attn_varlen_qkvpacked_xla
        query_layer, key_layer, value_layer = [k.permute(1, 2, 0, 3) for k in [query_layer, key_layer, value_layer]]
        bsz, _, q_len, _ = query_layer.size()
        qkv = torch.stack([query_layer, key_layer, value_layer], dim=2)
        qkv = qkv.transpose(1, 3)
        qkv = einops.rearrange(qkv, "b s ... -> (b s) ...")
        cu_q_lens = torch.arange(0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device=qkv.device)
        context_layer = flash_attn_varlen_qkvpacked_xla(qkv, cu_q_lens, q_len, 0.0, None, True, False)
        context_layer = einops.rearrange(context_layer, "(b s) ... -> b s ...", b=bsz)
        context_layer = context_layer.permute(1, 0, 2, 3)
        new_context_layer_shape = context_layer.size()[:-2] + (self.core_attention.hidden_size_per_partition,)
        context_layer = context_layer.reshape(*new_context_layer_shape)

        # =================
        # Output. [sq, b, h]
        # =================

        output = self.dense(context_layer)

        return output, kv_cache   

    for layer in model.transformer.encoder.layers:
        layer.self_attention.forward = types.MethodType(chatglm_attn_forward, layer.self_attention)
    
    return model

def patch_baichuan_model(model):

    def baichuan_attn_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        proj = self.W_pack(hidden_states)
        proj = (
            proj.unflatten(-1, (3, self.hidden_size))
            .unsqueeze(0)
            .transpose(0, -2)
            .squeeze(-2)
        )
        query_states = (
            proj[0].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        )
        key_states = (
            proj[1].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        )
        value_states = (
            proj[2].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        )

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        if not use_torchacc():
        # if attention_mask is not None:
        #     if q_len == 1: # inference with cache
        #         if len(attention_mask.size()) == 4:
        #             attention_mask = attention_mask[:, :, -1:, :]   
        #         else:
        #             attention_mask = attention_mask[:, -1:, :]    
        #     attn_weights = attn_weights + attention_mask
        #     attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

            attn_output = torch.matmul(attn_weights, value_states)

            attn_output = attn_output.transpose(1, 2)
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)

            if not output_attentions:
                attn_weights = None

            return attn_output, attn_weights, past_key_value

        else:
        # # try:
        # from torchacc.ops import flash_attn_varlen_qkvpacked_xla
        # qkv = torch.stack([query_states, key_states, value_states], dim=2)
        # qkv = qkv.transpose(1, 3)
        # qkv = einops.rearrange(qkv, "b s ... -> (b s) ...")
        # cu_q_lens = torch.arange(
        #     0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device=qkv.device)
        # output = flash_attn_varlen_qkvpacked_xla(qkv, cu_q_lens, q_len, 0.0, None, True, False)
        # output = einops.rearrange(output, "(b s) ... -> b s ...", b=bsz)
        # output = self.o_proj(einops.rearrange(output, "b s h d -> b s (h d)"))
        # return output, None, past_key_value
        # # except:
        #     # print('import torchacc failed.')
            from torchacc.ops import flash_attn_varlen_xla
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)
            q, k, v = [einops.rearrange(x, "b s ... -> (b s) ...") for x in [query_states, key_states, value_states]]
            cu_q_lens = torch.arange(
                0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device=q.device)
            output = flash_attn_varlen_xla(q, k, v, cu_q_lens, cu_q_lens, q_len, q_len, 0.0, softmax_scale=None, causal=True)
            output = einops.rearrange(output, "(b s) ... -> b s ...", b=bsz)
            output = self.o_proj(einops.rearrange(output, "b s h d -> b s (h d)"))
            return output, None, past_key_value

    for layer in model.base_model.layers:
        layer.self_attn.forward = types.MethodType(baichuan_attn_forward, layer.self_attn)
    
    return model

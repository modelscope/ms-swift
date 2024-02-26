# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import random
import socket
import types
import einops
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

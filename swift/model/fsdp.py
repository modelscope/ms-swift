# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import accelerate.utils.fsdp_utils as fsdp_utils
import torch
from functools import wraps

_original_fsdp2_load_full_state_dict = fsdp_utils.fsdp2_load_full_state_dict


@wraps(_original_fsdp2_load_full_state_dict)
def _patched_fsdp2_load_full_state_dict(accelerator, model: torch.nn.Module, full_sd: dict, cpu_offload: bool = False):
    """Let Accelerate load DTensors while synchronizing ordinary buffers separately."""
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor

    buffers = {}

    def filter_buffers(module, state_dict, prefix, metadata):
        for name, value in list(state_dict.items()):
            if isinstance(value, DTensor):
                continue
            if accelerator.is_main_process:
                tensor = full_sd[name].detach().to(device=accelerator.device, dtype=value.dtype).contiguous()
            else:
                tensor = torch.empty(value.shape, dtype=value.dtype, device=accelerator.device)
            dist.broadcast(tensor, src=0)
            buffers[name] = tensor.cpu() if cpu_offload else tensor
            del state_dict[name]

    def restore_buffers(module, state_dict, prefix, *args):
        state_dict.update(buffers)

    state_hook = model.register_state_dict_post_hook(filter_buffers)
    load_hook = model.register_load_state_dict_pre_hook(restore_buffers)
    try:
        return _original_fsdp2_load_full_state_dict(accelerator, model, full_sd, cpu_offload)
    finally:
        state_hook.remove()
        load_hook.remove()


_APPLIED = False


def apply_patch() -> None:
    global _APPLIED
    if not _APPLIED:
        fsdp_utils.fsdp2_load_full_state_dict = _patched_fsdp2_load_full_state_dict
        _APPLIED = True

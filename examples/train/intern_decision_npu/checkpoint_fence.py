"""Fence CPU checkpoint offload before other ranks enqueue HCCL collectives.

External, task-local workaround for the observed FSDP2 checkpoint-300 stall:
rank zero in Tensor.to(cpu), other ranks in Trainer's post-save HCCL barrier.
No package source files are changed. Enabled only for this recovery run.
"""
import torch.distributed as dist
from datetime import timedelta
from functools import wraps
from transformers import Trainer

_original_save_model = Trainer.save_model
_cpu_group = None


@wraps(_original_save_model)
def save_model_with_cpu_fence(self, *args, **kwargs):
    global _cpu_group
    distributed = dist.is_available() and dist.is_initialized()
    if distributed and _cpu_group is None:
        _cpu_group = dist.new_group(backend='gloo', timeout=timedelta(minutes=20))
    result = _original_save_model(self, *args, **kwargs)
    if distributed:
        dist.monitored_barrier(group=_cpu_group, timeout=timedelta(minutes=20))
    return result


Trainer.save_model = save_model_with_cpu_fence

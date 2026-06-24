# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import torch.distributed as dist

from swift.utils import ShutdownManager, get_device
from .base import TrainerCallback


class DeepspeedElasticCallback(TrainerCallback):
    """Compatibility marker for enabling DeepSpeed elastic setup during argument initialization."""


class GracefulExitCallback(TrainerCallback):

    def __init__(self, args=None, trainer=None):
        if args is not None and trainer is not None:
            super().__init__(args, trainer)
        shutdown_manager = ShutdownManager()
        shutdown_manager.register()
        self.shutdown_manager = shutdown_manager
        self._pending_stop = False

    def on_step_end(self, args, state, control, **kwargs):
        device_type = get_device()

        local_req = 1 if self.shutdown_manager.should_shutdown() else 0
        if dist.is_available() and dist.is_initialized():

            t = torch.tensor([local_req], dtype=torch.uint8, device=device_type)
            # all_reduce with MAX: if any rank has 1 -> result 1 everywhere
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            any_req = bool(int(t.item()))
        else:
            any_req = bool(local_req)

        if any_req:
            control.should_save = True
            self._pending_stop = True
        return control

    def on_save(self, args, state, control, **kwargs):
        if self._pending_stop:
            control.should_training_stop = True
            self._pending_stop = False
        return control

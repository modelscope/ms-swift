# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import torch.distributed as dist
from transformers import TrainerControl, TrainerState, TrainingArguments

from swift.utils import ShutdownManager, get_device
from .base import TrainerCallback


class DeepspeedElasticCallback(TrainerCallback):

    def __init__(self, args=None, trainer=None):
        if args is not None and trainer is not None:
            super().__init__(args, trainer)

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of training.
        """

        if args.deepspeed:
            from deepspeed.elasticity import compute_elastic_config
            from deepspeed.git_version_info import version as __version__
            args.deepspeed['checkpoint'] = {'load_universal': True}
            if 'elasticity' not in args.deepspeed:
                args.deepspeed['elasticity'] = {
                    'ignore_non_elastic_batch_info': True,
                    'enabled': True,
                    'max_train_batch_size': 8,
                    'micro_batch_sizes': [2],
                    'min_gpus': 1,
                    'max_gpus': 4,
                    'min_time': 20,
                    'version': 0.1
                }
                world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
                final_batch_size, _, micro_batch_size = compute_elastic_config(
                    ds_config=args.deepspeed,
                    target_deepspeed_version=__version__,
                    world_size=world_size,
                )
                denom = micro_batch_size * world_size
                gradient_accu_steps = max(1, final_batch_size // denom)
                args.per_device_train_batch_size = micro_batch_size
                args.gradient_accumulation_steps = gradient_accu_steps
                state.train_batch_size = args.per_device_train_batch_size * max(1, args.n_gpu)


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

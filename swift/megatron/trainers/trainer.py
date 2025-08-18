# Copyright (c) Alibaba, Inc. and its affiliates.
from functools import partial
from typing import Optional

import megatron.core
import torch
import torch.distributed as dist
import torch.nn
from megatron.core import mpu
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.training import get_args, get_timers
from packaging import version
from torch.distributed.nn import all_reduce

from swift.utils import get_logger
from .base import BaseMegatronTrainer
from .utils import get_batch

logger = get_logger()


class MegatronTrainer(BaseMegatronTrainer):

    # Code borrowed from NVIDIA/Megatron-LM
    def loss_func(self,
                  output_tensor: torch.Tensor,
                  *,
                  labels: torch.Tensor,
                  loss_scale: Optional[torch.Tensor] = None):
        args = get_args()

        losses = output_tensor.float()
        if args.enable_dft_loss:
            losses = losses * torch.exp(-losses.detach())
        if loss_scale is not None:
            losses = losses * loss_scale
        loss_mask = labels != -100
        loss = torch.cat([torch.sum(losses * loss_mask).view(1), loss_mask.sum().view(1)])

        megatron_core_013 = version.parse(megatron.core.__version__) >= version.parse('0.13.0rc0')
        if args.context_parallel_size > 1 and not megatron_core_013:
            loss = all_reduce(loss, group=mpu.get_context_parallel_group())

        # Check individual rank losses are not NaN prior to DP all-reduce.
        rerun_state_machine = get_rerun_state_machine()
        if args.check_for_nan_in_loss_and_grad:
            rerun_state_machine.validate_result(
                result=loss[0],
                rejection_func=torch.isnan,
                message='found NaN in local forward loss calculation',
                tolerance=0.0,  # forward pass calculations are determinisic
                fatal=True,
            )
            rerun_state_machine.validate_result(
                result=loss[0],
                rejection_func=torch.isinf,
                message='found Inf in local forward loss calculation',
                tolerance=0.0,  # forward pass calculations are determinisic
                fatal=True,
            )
        # Check for spiky loss
        if args.check_for_spiky_loss:
            # define spiky loss as a loss that's 10x the max loss observed
            SPIKY_LOSS_FACTOR = 10
            rerun_state_machine.validate_result(
                result=loss[0],
                rejection_func=partial(
                    rerun_state_machine.is_unexpectedly_large,
                    threshold=SPIKY_LOSS_FACTOR,
                    context='loss',
                ),
                message='Spiky loss',
                tolerance=0.0,  # forward pass calculations are determinisic
                fatal=False,
            )
        # Reduce loss for logging.
        reporting_loss = loss.clone().detach()
        lm_loss = loss[0]
        if not megatron_core_013:
            # fix megatron-lm bug
            # https://github.com/NVIDIA/Megatron-LM/blob/core_r0.12.0/megatron/core/pipeline_parallel/schedules.py#L291
            torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())
            lm_loss = lm_loss / mpu.get_context_parallel_world_size()
            reporting_loss = (reporting_loss[0], reporting_loss[1])
        else:
            lm_loss = lm_loss.clone()
        local_num_tokens = loss[1].clone().detach().to(torch.int)
        return (
            lm_loss,
            local_num_tokens,
            {
                'lm loss': reporting_loss
            },
        )

    def forward_step(self, data_iterator, model):
        timers = get_timers()

        # Get the batch.
        timers('batch-generator', log_level=2).start()
        with self.stimer(bdata=True):
            data = get_batch(data_iterator)
        timers('batch-generator').stop()
        loss_scale = data.pop('loss_scale', None)
        with self.stimer:
            output_tensor = model(**data)
        labels = data.get('labels')
        return output_tensor, partial(self.loss_func, labels=labels, loss_scale=loss_scale)

# Copyright (c) Alibaba, Inc. and its affiliates.
from collections import namedtuple
from functools import partial

import torch
from megatron.core import mpu
from megatron.training import get_args, get_timers
from torch.distributed.nn import all_reduce

from swift.trainers import DPOTrainer
from swift.utils import get_current_device, get_logger
from .base import MegatronRLHFTrainer
from .utils import get_batch

logger = get_logger()


class DummyDPOTrainer(DPOTrainer):
    # For reusing the dpo_loss function in TRL.
    def __init__(self, args):
        from trl.trainer import FDivergenceConstants
        self.accelerator = namedtuple('Accelerator', ['device'])(device=get_current_device())
        self.f_alpha_divergence_coef = 1.
        self.f_divergence_params = {FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY: self.f_alpha_divergence_coef}
        self.reference_free = args.reference_free
        self.label_smoothing = args.label_smoothing
        self.f_divergence_type = args.f_divergence_type
        self.loss_type = args.loss_type
        self.beta = args.beta


class MegatronDPOTrainer(MegatronRLHFTrainer):

    def __init__(self, args, template):
        super().__init__(args, template)
        assert args.padding_free, 'Currently `rlhf_type="dpo"` only supports padding_free.'
        self.dummy_dpo_trainer = DummyDPOTrainer(args)
        self.ref_models = []

    @staticmethod
    def get_logps(output_tensor, labels, packed_seq_params):
        args = get_args()
        per_token_logps = -output_tensor
        loss_mask = labels != -100
        per_token_logps = per_token_logps * loss_mask
        num_samples = packed_seq_params.num_samples
        cu_seqlens = packed_seq_params.cu_seqlens_q[:num_samples * 2 + 1] // args.context_parallel_size
        all_logps = per_token_logps.new_zeros((num_samples * 2, ))
        for i in range(num_samples * 2):
            start, end = cu_seqlens[i], cu_seqlens[i + 1]
            all_logps[i] = per_token_logps[:, start:end].sum()
        if args.context_parallel_size > 1:
            all_logps = all_reduce(all_logps, group=mpu.get_context_parallel_group())
        return all_logps

    def loss_func(self, output_tensor: torch.Tensor, *, labels: torch.Tensor, packed_seq_params):
        ref_output_tensor = output_tensor[:output_tensor.shape[0] // 2].detach()
        output_tensor = output_tensor[output_tensor.shape[0] // 2:]
        args = get_args()
        num_samples = packed_seq_params.num_samples

        logps = self.get_logps(output_tensor, labels, packed_seq_params)
        ref_logps = self.get_logps(ref_output_tensor, labels, packed_seq_params)
        loss, chosen_rewards, rejected_rewards = self.dummy_dpo_trainer.dpo_loss(
            logps[:num_samples],
            logps[num_samples:],
            ref_logps[:num_samples],
            ref_logps[num_samples:],
        )
        if args.rpo_alpha:
            loss_mask = labels != -100
            num_tokens = packed_seq_params.cu_seqlens_q[num_samples] // args.context_parallel_size
            loss_mask[:, num_tokens:] = 0
            nll_loss = torch.concat([torch.sum(output_tensor * loss_mask)[None], loss_mask.sum()[None]])
            if args.context_parallel_size > 1:
                nll_loss = all_reduce(nll_loss, group=mpu.get_context_parallel_group())
            nll_loss = nll_loss[0] / nll_loss[1]
            loss = loss + args.rpo_alpha * nll_loss
        loss = loss.mean()
        metric = {
            'loss': loss.detach().clone(),
            'logps/chosen': logps[:num_samples].mean(),
            'logps/rejected': logps[num_samples:].mean(),
            'rewards/chosen': chosen_rewards.mean(),
            'rewards/rejected': rejected_rewards.mean(),
            'rewards/accuracies': (chosen_rewards > rejected_rewards).float().mean(),
            'rewards/margins': (chosen_rewards - rejected_rewards).mean(),
        }
        if args.rpo_alpha:
            metric['nll_loss'] = nll_loss.detach()
        metric = self._all_reduce_metric(metric)
        # fix megatron-lm bug
        # https://github.com/NVIDIA/Megatron-LM/blob/core_r0.12.0/megatron/core/pipeline_parallel/schedules.py#L291
        loss = loss / mpu.get_context_parallel_world_size()
        return loss, metric

    def forward_step(self, data_iterator, model):
        timers = get_timers()
        # Get the batch.
        unwrapped_model = model.module.module
        input_tensor = unwrapped_model.get_input_tensor()
        if input_tensor is not None:
            unwrapped_model.set_input_tensor(input_tensor[input_tensor.shape[0] // 2:])
        vp_stage = unwrapped_model.vp_stage
        with torch.no_grad(), self.null_ref_context() as ref_models:
            ref_model = ref_models[vp_stage or 0]
            if input_tensor is not None:
                ref_model.set_input_tensor(input_tensor[:input_tensor.shape[0] // 2].detach())
            timers('batch-generator', log_level=2).start()
            with self.stimer(bdata=True):
                data = get_batch(data_iterator, vp_stage)
            timers('batch-generator').stop()
            data.pop('loss_scale', None)
            ref_output_tensor = ref_model(**data)

        with self.stimer:
            output_tensor = model(**data)
        return torch.concat([ref_output_tensor, output_tensor], dim=0), partial(
            self.loss_func, labels=data.get('labels'), packed_seq_params=data.get('packed_seq_params'))

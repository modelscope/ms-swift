# Copyright (c) Alibaba, Inc. and its affiliates.
from collections import namedtuple
from functools import partial
from typing import List, Tuple, Union

import torch
from megatron.core import mpu
from megatron.training import get_args, get_model, get_timers, training
from megatron.training.checkpointing import load_checkpoint
from megatron.training.utils import unwrap_model

from swift.trainers import DPOTrainer
from swift.utils import get_current_device, get_logger
from ..argument import MegatronRLHFArguments
from .sft import MegatronSft
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


class MegatronRLHF(MegatronSft):
    args_class = MegatronRLHFArguments
    args: args_class

    def __init__(self, args: Union[List[str], MegatronRLHFArguments, None] = None) -> None:
        super().__init__(args)
        self.dummy_dpo_trainer = DummyDPOTrainer(self.args)

    def _prepare_template(self) -> None:
        super()._prepare_template()
        self.template.set_mode('rlhf')

    def _patch_setup_model_and_optimizer(self):
        origin_setup_model_and_optimizer = training.setup_model_and_optimizer

        def setup_model_and_optimizer(model_provider_func, model_type, *_args, **kwargs):
            args = get_args()
            ref_model = get_model(model_provider_func, model_type)
            if args.ref_load is None:
                args.ref_load = args.load
            args.iteration, args.num_floating_point_operations_so_far = load_checkpoint(
                ref_model, None, None, load_arg='ref_load')
            self.ref_model = ref_model[0]
            self.ref_model.eval()
            return origin_setup_model_and_optimizer(model_provider_func, model_type, *_args, **kwargs)

        training.setup_model_and_optimizer = setup_model_and_optimizer

    def ref_forward(self, data_iterator):
        ref_model = unwrap_model(self.ref_model)
        timers = get_timers()
        timers('batch-ref-generator', log_level=2).start()
        with self.stimer(bdata=True):
            data = get_batch(data_iterator)
        if not data:
            raise StopIteration
        timers('batch-ref-generator').stop()
        labels = data['labels']
        with torch.no_grad():
            output_tensor = ref_model(**data)
        data['logps'] = self.get_logps(output_tensor, labels, data['packed_seq_params'])
        return data

    @staticmethod
    def get_logps(output_tensor, labels, packed_seq_params):
        args = get_args()
        per_token_logps = -output_tensor
        loss_mask = labels != -100
        per_token_logps = per_token_logps * loss_mask
        cu_seqlens = packed_seq_params.cu_seqlens_q[:args.micro_batch_size * 2 + 1]
        all_logps = per_token_logps.new_zeros((args.micro_batch_size * 2, ))
        for i in range(args.micro_batch_size * 2):
            start, end = cu_seqlens[i], cu_seqlens[i + 1]
            all_logps[i] = per_token_logps[:, start:end].sum()
        return all_logps

    def train_step(self, forward_step_func, data_iterator, model, optimizer, opt_param_scheduler, config):
        args = get_args()
        num_iters_per_step = args.global_batch_size // (args.micro_batch_size * mpu.get_data_parallel_world_size())
        res = []
        for i in range(num_iters_per_step):
            with torch.no_grad():
                res.append(self.ref_forward(data_iterator))
        return super().train_step(forward_step_func, iter(res), model, optimizer, opt_param_scheduler, config)

    def run(self):
        self._patch_setup_model_and_optimizer()
        super().run()

    def loss_func(self, output_tensor: torch.Tensor, *, ref_logps: torch.Tensor, labels: torch.Tensor,
                  packed_seq_params):
        from swift.trainers import DPOTrainer
        args = get_args()
        loss_mask = labels != -100
        num_tokens = packed_seq_params.cu_seqlens_q[args.micro_batch_size]
        loss_mask[:, num_tokens:] = 0
        nll_loss = torch.sum(output_tensor * loss_mask) / loss_mask.sum()
        logps = self.get_logps(output_tensor, labels, packed_seq_params)
        loss, chosen_rewards, rejected_rewards = self.dummy_dpo_trainer.dpo_loss(
            logps[:args.micro_batch_size],
            logps[args.micro_batch_size:],
            ref_logps[:args.micro_batch_size],
            ref_logps[args.micro_batch_size:],
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        if args.rpo_alpha > 0:
            loss = loss + args.rpo_alpha * nll_loss
        loss = loss.mean()
        reporting_loss = loss.clone().detach()
        torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())

        return loss, {
            'loss': reporting_loss,
            'rewards/chosen': chosen_rewards.mean(),
            'rewards/rejected': rejected_rewards.mean(),
            'rewards/accuracies': reward_accuracies.mean(),
            'rewards/margins': (chosen_rewards - rejected_rewards).mean(),
            'logps/chosen': logps[:args.micro_batch_size].mean(),
            'logps/rejected': logps[args.micro_batch_size:].mean(),
            'nll_loss': nll_loss
        }

    def forward_step(self, data_iterator, model):
        timers = get_timers()

        # Get the batch.
        timers('batch-generator', log_level=2).start()
        with self.stimer(bdata=True):
            data = next(data_iterator)
        timers('batch-generator').stop()
        ref_logps = data.pop('logps')
        with self.stimer:
            output_tensor = model(**data)
        return output_tensor, partial(
            self.loss_func, ref_logps=ref_logps, labels=data['labels'], packed_seq_params=data['packed_seq_params'])


def megatron_rlhf_main(args: Union[List[str], MegatronRLHFArguments, None] = None):
    return MegatronRLHF(args).main()

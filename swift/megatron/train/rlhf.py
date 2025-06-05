# Copyright (c) Alibaba, Inc. and its affiliates.
from functools import partial
from typing import List, Tuple, Union

import torch
from megatron.core import mpu
from megatron.training import get_args, get_model, get_timers, training
from megatron.training.checkpointing import load_checkpoint
from megatron.training.utils import unwrap_model

from swift.utils import get_logger
from ..argument import MegatronRLHFArguments
from .sft import MegatronSft
from .utils import get_batch

logger = get_logger()


class MegatronRLHF(MegatronSft):
    args_class = MegatronRLHFArguments
    args: args_class

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
        from swift.trainers import DPOTrainer
        args = get_args()
        ref_model = unwrap_model(self.ref_model)
        timers = get_timers()
        timers('batch-ref-generator', log_level=2).start()
        with self.stimer(bdata=True):
            data = get_batch(data_iterator)
        if not data:
            raise StopIteration
        timers('batch-ref-generator').stop()
        labels = data.pop('labels', None)
        with torch.no_grad():
            ref_logits = ref_model(**data)
        ref_logits = ref_logits.to(torch.float32)
        per_token_logps, _, loss_mask = DPOTrainer.get_per_token_logps(ref_logits, labels)
        cu_seqlens = data['packed_seq_params'].cu_seqlens_q[:args.micro_batch_size * 2 + 1]
        all_logps = per_token_logps.new_zeros((args.micro_batch_size, ))
        for i in range(args.micro_batch_size):
            start, end = cu_seqlens[i], cu_seqlens[i + 1]
            all_logps[i] = per_token_logps[:, start:end].sum()
        num_tokens = cu_seqlens[args.micro_batch_size]
        # output['nll_loss'] = self.get_nll_loss(all_logits[:, :num_tokens], labels[:, :num_tokens])
        data['labels'] = labels
        data['logps'] = all_logps
        return data

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

    def loss_func(self, loss_mask: torch.Tensor, output_tensor: torch.Tensor, *, ref_logps: torch.Tensor):
        from swift.trainers import DPOTrainer
        nll_loss = torch.sum(output_tensor * loss_mask) / loss_mask.sum()
        losses, chosen_rewards, rejected_rewards = DPOTrainer.dpo_loss(model_output['chosen_logps'],
                                                                       model_output['rejected_logps'], ref_chosen_logps,
                                                                       ref_rejected_logps)
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

    def forward_step(self, data_iterator, model):
        timers = get_timers()
        args = get_args()

        # Get the batch.
        timers('batch-generator', log_level=2).start()
        with self.stimer(bdata=True):
            data = next(data_iterator)
        timers('batch-generator').stop()
        ref_logps = data.pop('logps')
        labels = data.pop('labels')
        with self.stimer:
            logits = model(**data)
        if labels is None:
            loss_mask = None
        else:
            loss = model.module.module.compute_language_model_loss(labels, logits)
            cu_seqlens = data['packed_seq_params'].cu_seqlens_q[:args.micro_batch_size * 2 + 1]
            num_tokens = cu_seqlens[args.micro_batch_size]
            loss_mask = (labels != -100).float()
            loss_mask[:, num_tokens:] = 0
        return logits, partial(self.loss_func, loss_mask, ref_logps=ref_logps)


def megatron_rlhf_main(args: Union[List[str], MegatronRLHFArguments, None] = None):
    return MegatronRLHF(args).main()

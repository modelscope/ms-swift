# Copyright (c) Alibaba, Inc. and its affiliates.
from collections import namedtuple
from functools import partial

import torch
from megatron.core import mpu
from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy
from megatron.training import get_args, get_timers
from torch.distributed.nn import all_reduce
from trl import KTOTrainer

from swift.utils import get_current_device, get_logger
from .base import MegatronRLHFTrainer

logger = get_logger()


class DummyKTOTrainer(KTOTrainer):
    # For reusing the dpo_loss function in TRL.
    def __init__(self, args):
        self.accelerator = namedtuple('Accelerator', ['device'])(device=get_current_device())
        self.loss_type = args.loss_type
        self.beta = args.beta
        self.desirable_weight = args.desirable_weight
        self.undesirable_weight = args.undesirable_weight
        self.calculate_KL = args.calculate_KL


class MegatronKTOTrainer(MegatronRLHFTrainer):

    def __init__(self, args, template):
        super().__init__(args, template)
        self.dummy_kto_trainer = DummyKTOTrainer(args)

    @staticmethod
    def get_logps(output_tensor, labels, packed_seq_params=None):
        args = get_args()
        if output_tensor is None:
            return None

        shifted_logits = output_tensor[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()

        logits_for_loss = shifted_logits.transpose(0, 1).contiguous()
        labels_for_loss = shifted_labels.transpose(0, 1).contiguous()

        per_token_cross_entropy_loss = vocab_parallel_cross_entropy(
            logits_for_loss, labels_for_loss, label_smoothing=0.0)

        per_token_logps = -per_token_cross_entropy_loss
        loss_mask = (labels_for_loss != -100)
        masked_logps = per_token_logps * loss_mask

        if args.padding_free and packed_seq_params is not None:
            flattened_logps = masked_logps.squeeze(1)  # [seq-1]

            cu_seqlens = packed_seq_params.cu_seqlens_q
            num_sequences = cu_seqlens.shape[0] - 1
            all_logps = flattened_logps.new_zeros((num_sequences, ))
            for i in range(num_sequences):
                start_index, end_index = cu_seqlens[i], cu_seqlens[i + 1] - 1
                if end_index > start_index:
                    all_logps[i] = flattened_logps[start_index:end_index].sum()
        else:
            all_logps = masked_logps.sum(dim=0)

        if args.context_parallel_size > 1:
            all_logps = all_reduce(all_logps, group=mpu.get_context_parallel_group())

        return all_logps

    def loss_func(self, output_tensor, *, policy_KL_logps, reference_logps, reference_KL_logps, labels, all_labels,
                  packed_seq_params):
        policy_logps = self.get_logps(output_tensor, labels, packed_seq_params)
        is_desirable = all_labels.bool()

        policy_chosen_logps = policy_logps[is_desirable]
        policy_rejected_logps = policy_logps[~is_desirable]
        reference_chosen_logps = reference_logps[is_desirable]
        reference_rejected_logps = reference_logps[~is_desirable]

        loss, chosen_rewards, rejected_rewards, kl = self.dummy_kto_trainer.kto_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            policy_KL_logps=policy_KL_logps,
            reference_chosen_logps=reference_chosen_logps,
            reference_rejected_logps=reference_rejected_logps,
            reference_KL_logps=reference_KL_logps,
        )

        loss = loss.mean() if loss.numel() > 0 else torch.tensor(0.0, device=policy_logps.device)

        with torch.no_grad():
            chosen_rewards_mean = chosen_rewards.mean() if chosen_rewards.numel() > 0 else torch.tensor(
                0.0, device=loss.device)
            rejected_rewards_mean = rejected_rewards.mean() if rejected_rewards.numel() > 0 else torch.tensor(
                0.0, device=loss.device)
            policy_chosen_logps_mean = policy_chosen_logps.mean() if policy_chosen_logps.numel() > 0 else torch.tensor(
                0.0, device=loss.device)
            policy_rejected_logps_mean = policy_rejected_logps.mean(
            ) if policy_rejected_logps.numel() > 0 else torch.tensor(
                0.0, device=loss.device)

        metric = {
            'loss': loss.clone().detach(),
            'logps/chosen': policy_chosen_logps_mean,
            'logps/rejected': policy_rejected_logps_mean,
            'rewards/chosen': chosen_rewards_mean,
            'rewards/rejected': rejected_rewards_mean,
            'rewards/margins': chosen_rewards_mean - rejected_rewards_mean,
            'kl': kl.detach() if kl is not None else torch.tensor(0.0, device=loss.device),
        }

        reporting_metric = loss.new_tensor(list(metric.values()))
        torch.distributed.all_reduce(
            reporting_metric, torch.distributed.ReduceOp.AVG, group=mpu.get_data_parallel_group())
        reporting_metric = {k: reporting_metric[i] for i, k in enumerate(metric.keys())}
        # fix megatron-lm bug
        # https://github.com/NVIDIA/Megatron-LM/blob/core_r0.12.0/megatron/core/pipeline_parallel/schedules.py#L291
        loss = loss / mpu.get_context_parallel_world_size()
        return loss, reporting_metric

    @staticmethod
    def _get_input_tensor(input_tensor, is_ref: bool, is_KL: bool):
        i = (not is_ref) * 2 + is_KL
        return input_tensor[i:i + 1]

    def forward_step(self, data_iterator, model):
        timers = get_timers()
        # Get the batch.
        unwrapped_model = model.module.module
        input_tensor = unwrapped_model.get_input_tensor()
        vp_stage = unwrapped_model.vp_stage
        timers('batch-generator', log_level=2).start()
        with self.stimer(bdata=True):
            data = self.get_batch(data_iterator, vp_stage)
        timers('batch-generator').stop()
        data.pop('loss_scale', None)

        with torch.no_grad(), self.null_ref_context() as ref_models:
            ref_model = ref_models[vp_stage or 0]
            if input_tensor is not None:
                ref_model.set_input_tensor(self._get_input_tensor(True, False).detach())
            ref_output_tensor = ref_model(**data)
            if input_tensor is not None:
                ref_model.set_input_tensor(self._get_input_tensor(True, True).detach())
            ref_KL_output_tensor = ref_model(**data)
        with torch.no_grad():
            if input_tensor is not None:
                unwrapped_model.set_input_tensor(self._get_input_tensor(input_tensor, False, True))
            KL_output_tensor = model(*data)

        if input_tensor is not None:
            unwrapped_model.set_input_tensor(self._get_input_tensor(input_tensor, False, False))
        with self.stimer:
            output_tensor = model(**data)
        return torch.concat([ref_output_tensor, ref_KL_output_tensor, output_tensor, KL_output_tensor], dim=0), partial(
            self.loss_func, labels=data.get('labels'), packed_seq_params=data.get('packed_seq_params'))

# Copyright (c) ModelScope Contributors. All rights reserved.
from collections import namedtuple
from functools import partial
from typing import Any, Dict

import torch
from megatron.core import mpu
from trl import KTOTrainer

from swift.utils import get_current_device, get_logger
from .rlhf_mixin import MegatronRLHFTrainer

logger = get_logger()


class DummyKTOTrainer(KTOTrainer):
    # For reusing the kto_loss function in TRL.

    def gather_for_metrics(self, input_data, *args, **kwargs):
        output_tensors = torch.empty(
            mpu.get_data_parallel_world_size() * input_data.numel(),
            dtype=input_data.dtype,
            device=input_data.device,
        )
        torch.distributed.all_gather_into_tensor(output_tensors, input_data, group=mpu.get_data_parallel_group())
        return output_tensors

    def __init__(self, args):
        self.accelerator = namedtuple('Accelerator', ['device', 'gather_for_metrics'])(
            device=get_current_device(), gather_for_metrics=self.gather_for_metrics)
        self.loss_type = args.loss_type
        self.beta = args.beta
        self.desirable_weight = args.desirable_weight
        self.undesirable_weight = args.undesirable_weight
        self.calculate_KL = args.calculate_KL


class MegatronKTOTrainer(MegatronRLHFTrainer):

    def __init__(self, args, template):
        super().__init__(args, template)
        self.dummy_kto_trainer = DummyKTOTrainer(args)

    def _kto_get_logps(self, output_tensor, data, is_KL: bool, is_ref: bool, length: int):
        labels = data['labels']
        packed_seq_params = data.get('packed_seq_params')
        num_samples = output_tensor.shape[0] if packed_seq_params is None else packed_seq_params.num_samples
        output = self._get_input_tensor(output_tensor, is_KL, is_ref, length, dim=1)
        return self.get_logps(output, labels, packed_seq_params, num_samples)

    def _get_kto_length(self, data: Dict[str, Any]) -> int:
        if 'packed_seq_params' in data:
            return data['packed_seq_params'].cu_seqlens_q[-1] // self.args.context_parallel_size
        else:
            return data['position_ids'].shape[-1]

    def loss_func(self, output_tensor, *, data, kl_data, label):
        length = self._get_kto_length(data)
        policy_logps = self._kto_get_logps(output_tensor, data, False, False, length)
        ref_logps = self._kto_get_logps(output_tensor, data, False, True, length)
        if self.args.calculate_KL:
            policy_KL_logps = self._kto_get_logps(output_tensor, kl_data, True, False, length)
            ref_KL_logps = self._kto_get_logps(output_tensor, kl_data, True, True, length)
        else:
            policy_KL_logps, ref_KL_logps = None, None
        label = output_tensor.new_tensor(label, dtype=torch.bool)
        policy_chosen_logps = policy_logps[label]
        policy_rejected_logps = policy_logps[~label]
        ref_chosen_logps = ref_logps[label]
        ref_rejected_logps = ref_logps[~label]

        loss, chosen_rewards, rejected_rewards, kl = self.dummy_kto_trainer.kto_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            policy_KL_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            ref_KL_logps,
        )

        loss = loss.mean()
        mean_metric = {
            'loss': loss.detach().clone(),
            'kl': kl.squeeze().detach(),
        }
        metric = self._all_reduce_metric(mean_metric)
        chosen_count = chosen_rewards.shape[0]
        rejected_count = rejected_rewards.shape[0]
        sum_metric = {
            'logps/chosen': loss.new_tensor([policy_chosen_logps.detach().nansum(), chosen_count]),
            'logps/rejected': loss.new_tensor([policy_rejected_logps.detach().nansum(), rejected_count]),
            'rewards/chosen': loss.new_tensor([chosen_rewards.nansum(), chosen_count]),
            'rewards/rejected': loss.new_tensor([rejected_rewards.nansum(), rejected_count]),
        }
        metric.update(self._all_reduce_metric(sum_metric, torch.distributed.ReduceOp.SUM))
        # fix megatron-lm bug
        # https://github.com/NVIDIA/Megatron-LM/blob/core_r0.12.0/megatron/core/pipeline_parallel/schedules.py#L291
        loss = loss / mpu.get_context_parallel_world_size()
        return loss, metric

    @staticmethod
    def _get_input_tensor(input_tensor, is_KL: bool, is_ref: bool, length: int, dim: int):
        # policy, ref, policy_KL, ref_KL
        total_length = input_tensor.shape[dim]
        KL_length = (total_length - 2 * length) // 2
        slice_list = [0, length, 2 * length, total_length - KL_length, total_length]
        idx = is_KL * 2 + is_ref
        slice_ = (slice(None), ) * dim + (slice(slice_list[idx], slice_list[idx + 1]), )
        res = input_tensor[slice_]
        if is_KL or is_ref:
            res = res.detach()
        return res

    def forward_step(self, data_iterator, model):
        # Get the batch.
        unwrapped_model = model.module.module
        input_tensor = unwrapped_model.get_input_tensor()
        vp_stage = unwrapped_model.vp_stage
        # not support loss_scale
        data, kl_data = self.get_batch(data_iterator, vp_stage)
        label = data.pop('label')
        data.pop('loss_scale', None)
        kl_data.pop('loss_scale', None)

        length = self._get_kto_length(data)
        if self.args.sequence_parallel:
            length //= mpu.get_tensor_model_parallel_world_size()
        with torch.no_grad(), self.null_ref_context() as ref_models:
            ref_model = ref_models[vp_stage or 0]
            if self.args.calculate_KL:
                if input_tensor is not None:
                    ref_model.set_input_tensor(self._get_input_tensor(input_tensor, True, True, length, 0))
                ref_KL_output_tensor = ref_model(**kl_data)

            if input_tensor is not None:
                ref_model.set_input_tensor(self._get_input_tensor(input_tensor, False, True, length, 0))
            ref_output_tensor = ref_model(**data)

        if self.args.calculate_KL:
            with torch.no_grad():
                if input_tensor is not None:
                    unwrapped_model.set_input_tensor(self._get_input_tensor(input_tensor, True, False, length, 0))
                KL_output_tensor = model(**kl_data)

        if input_tensor is not None:
            unwrapped_model.set_input_tensor(self._get_input_tensor(input_tensor, False, False, length, 0))
        output_tensor = model(**data)
        if self.mcore_013:
            is_pp_last_stage = mpu.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage)
        else:
            is_pp_last_stage = mpu.is_pipeline_last_stage()
        dim = 1 if is_pp_last_stage else 0
        if self.args.calculate_KL:
            res = torch.concat([output_tensor, ref_output_tensor, KL_output_tensor, ref_KL_output_tensor], dim=dim)
        else:
            res = torch.concat([output_tensor, ref_output_tensor], dim=dim)
        return res, partial(self.loss_func, data=data, kl_data=kl_data, label=label)

    def _prepare_batch(self, data, vp_stage=None, num_samples=None):
        res = []
        num_samples = data.pop('num_samples')
        for key in ['completion_', 'KL_completion_']:
            _data = {k[len(key):]: v for k, v in data.items() if k.startswith(key)}
            if not self.args.calculate_KL and key == 'KL_completion_':
                _data = {}
            else:
                _data = super()._prepare_batch(_data, vp_stage, num_samples)
            res.append(_data)
        res[0]['label'] = data['label']
        return res

    def _log_callback(self, logs, n_steps):
        super()._log_callback(logs, n_steps)
        if 'rewards/chosen' in logs and 'rewards/rejected' in logs:
            logs['rewards/margins'] = logs['rewards/chosen'] - logs['rewards/rejected']

# Copyright (c) Alibaba, Inc. and its affiliates.
from collections import namedtuple
from functools import partial
from typing import Literal

import torch
from megatron.core import mpu
from megatron.training import get_args, get_timers
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
        assert args.padding_free, 'Currently `rlhf_type="kto"` only supports padding_free.'
        self.dummy_kto_trainer = DummyKTOTrainer(args)

    def _kto_get_logps(self, output_tensor, data, is_KL: bool, is_ref: bool, length: int):
        labels = data['labels']
        packed_seq_params = data['packed_seq_params']
        output = self._get_input_tensor(output_tensor, is_KL, is_ref, length, dim=1)
        return self.get_logps(output, labels, packed_seq_params, packed_seq_params.num_samples)

    def loss_func(self, output_tensor, *, data, kl_data, label):
        length = data['packed_seq_params'].cu_seqlens_q[-1] // self.args.context_parallel_size
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
            'kl': kl.detach(),
        }
        metric = self._all_reduce_metric(mean_metric)
        sum_metric = {
            'logps/chosen_sum': policy_chosen_logps.nansum(),
            'logps/rejected_sum': policy_rejected_logps.nansum(),
            'rewards/chosen_sum': chosen_rewards.nansum(),
            'rewards/rejected_sum': rejected_rewards.nansum(),
            'count/chosen': loss.new_tensor(chosen_rewards.shape[0]),
            'count/rejected': loss.new_tensor(rejected_rewards.shape[0]),
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
        timers = get_timers()
        # Get the batch.
        unwrapped_model = model.module.module
        input_tensor = unwrapped_model.get_input_tensor()
        vp_stage = unwrapped_model.vp_stage
        timers('batch-generator', log_level=2).start()
        with self.stimer(bdata=True):
            # not support loss_scale
            data, kl_data = self.get_batch(data_iterator, vp_stage)
        timers('batch-generator').stop()
        label = data.pop('label')
        data.pop('loss_scale', None)
        kl_data.pop('loss_scale', None)

        length = data['packed_seq_params'].cu_seqlens_q[-1] // self.args.context_parallel_size
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
        with self.stimer:
            output_tensor = model(**data)
        dim = 1 if mpu.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage) else 0
        if self.args.calculate_KL:
            res = torch.concat([output_tensor, ref_output_tensor, KL_output_tensor, ref_KL_output_tensor], dim=dim)
        else:
            res = torch.concat([output_tensor, ref_output_tensor], dim=dim)
        return res, partial(self.loss_func, data=data, kl_data=kl_data, label=label)

    def _prepare_batch(self, data, vp_stage):
        res = []
        num_samples = data.pop('num_samples')
        for key in ['completion_', 'KL_completion_']:
            _data = {k[len(key):]: v for k, v in data.items() if k.startswith(key)}
            res.append(super()._prepare_batch(_data, vp_stage, num_samples))
        res[0]['label'] = data['label']
        return res

    def custom_log(self, total_loss_dict, mode: Literal['train', 'eval']) -> None:
        super().custom_log(total_loss_dict, mode)
        res = {}
        for k, v in total_loss_dict.items():
            if k.startswith('count/') or k.endswith('_sum'):
                continue
            res[k] = v
        for key in ['chosen', 'rejected']:
            count = total_loss_dict.get(f'count/{key}')
            if count is None or count.item() == 0:
                continue
            res[f'logps/{key}'] = total_loss_dict[f'logps/{key}_sum'] / count
            res[f'rewards/{key}'] = total_loss_dict[f'rewards/{key}_sum'] / count
        if 'rewards/chosen' in res and 'rewards/rejected' in res:
            res['rewards/margins'] = res['rewards/chosen'] - res['rewards/rejected']
        total_loss_dict.clear()
        total_loss_dict.update(res)

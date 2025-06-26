# Copyright (c) Alibaba, Inc. and its affiliates.
from collections import namedtuple
from functools import partial

import torch
from megatron.core import mpu
from megatron.core.inference.communication_utils import recv_from_prev_pipeline_rank_, send_to_next_pipeline_rank
from megatron.training import get_args, get_model, training
from megatron.training.checkpointing import load_checkpoint
from megatron.training.utils import unwrap_model
from torch.distributed.nn import all_reduce

from swift.trainers import DPOTrainer
from swift.utils import get_current_device, get_logger
from ..utils import get_batch
from .trainer import MegatronTrainer

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


class MegatronDPOTrainer(MegatronTrainer):

    def __init__(self, args):
        super().__init__(args)
        self._patch_setup_model_and_optimizer()
        self.dummy_dpo_trainer = DummyDPOTrainer(args)

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

    @staticmethod
    def _forward_step_helper(model, inputs):
        args = get_args()
        if mpu.is_pipeline_first_stage():
            micro_batch_size = 1  # use qkv_format 'thd'
            seq_length = inputs['input_ids'].shape[1]
            if args.sequence_parallel:
                seq_length //= mpu.get_tensor_model_parallel_world_size()
            recv_shape_buffer = torch.tensor([seq_length, micro_batch_size, args.hidden_size],
                                             device=torch.cuda.current_device(),
                                             dtype=torch.int64)
        else:
            recv_shape_buffer = torch.empty((3, ), device=torch.cuda.current_device(), dtype=torch.int64)
            recv_from_prev_pipeline_rank_(recv_shape_buffer)
        if not mpu.is_pipeline_last_stage():
            send_to_next_pipeline_rank(recv_shape_buffer)
        shape = recv_shape_buffer.tolist()

        if not mpu.is_pipeline_first_stage():
            recv_buffer = torch.empty(shape, device=torch.cuda.current_device(), dtype=args.params_dtype)
            recv_from_prev_pipeline_rank_(recv_buffer)
            model.set_input_tensor(recv_buffer)
        output_tensor = model(**inputs)
        if not mpu.is_pipeline_last_stage():
            send_to_next_pipeline_rank(output_tensor)
            output_tensor = None

        return output_tensor

    def ref_forward(self, data_iterator):
        ref_model = unwrap_model(self.ref_model)
        with self.stimer(bdata=True):
            data = get_batch(data_iterator)
        labels = data.get('labels')
        with torch.no_grad():
            output_tensor = self._forward_step_helper(ref_model, data)
        data['logps'] = None if labels is None else self.get_logps(output_tensor, labels, data['packed_seq_params'])
        return data

    @staticmethod
    def get_logps(output_tensor, labels, packed_seq_params):
        args = get_args()
        per_token_logps = -output_tensor
        loss_mask = labels != -100
        per_token_logps = per_token_logps * loss_mask
        cu_seqlens = packed_seq_params.cu_seqlens_q[:args.micro_batch_size * 2 + 1] // args.context_parallel_size
        all_logps = per_token_logps.new_zeros((args.micro_batch_size * 2, ))
        for i in range(args.micro_batch_size * 2):
            start, end = cu_seqlens[i], cu_seqlens[i + 1]
            all_logps[i] = per_token_logps[:, start:end].sum()
        if args.context_parallel_size > 1:
            all_logps = all_reduce(all_logps, group=mpu.get_context_parallel_group())
        return all_logps

    def loss_func(self, output_tensor: torch.Tensor, *, ref_logps: torch.Tensor, labels: torch.Tensor,
                  packed_seq_params):
        args = get_args()
        loss_mask = labels != -100
        num_tokens = packed_seq_params.cu_seqlens_q[args.micro_batch_size] // args.context_parallel_size
        loss_mask[:, num_tokens:] = 0
        nll_loss = torch.concat([torch.sum(output_tensor * loss_mask)[None], loss_mask.sum()[None]])
        if args.context_parallel_size > 1:
            nll_loss = all_reduce(nll_loss, group=mpu.get_context_parallel_group())
        nll_loss = nll_loss[0] / nll_loss[1]

        logps = self.get_logps(output_tensor, labels, packed_seq_params)
        loss, chosen_rewards, rejected_rewards = self.dummy_dpo_trainer.dpo_loss(
            logps[:args.micro_batch_size],
            logps[args.micro_batch_size:],
            ref_logps[:args.micro_batch_size],
            ref_logps[args.micro_batch_size:],
        )
        if args.rpo_alpha > 0:
            loss = loss + args.rpo_alpha * nll_loss
        loss = loss.mean()
        metric = {
            'loss': loss.clone().detach(),
            'nll_loss': nll_loss.detach(),
            'logps/chosen': logps[:args.micro_batch_size].mean(),
            'logps/rejected': logps[args.micro_batch_size:].mean(),
            'rewards/chosen': chosen_rewards.mean(),
            'rewards/rejected': rejected_rewards.mean(),
            'rewards/accuracies': (chosen_rewards > rejected_rewards).float().mean(),
            'rewards/margins': (chosen_rewards - rejected_rewards).mean(),
        }
        reporting_metric = loss.new_tensor(list(metric.values()))
        torch.distributed.all_reduce(
            reporting_metric, torch.distributed.ReduceOp.AVG, group=mpu.get_data_parallel_group())
        reporting_metric = {k: reporting_metric[i] for i, k in enumerate(metric.keys())}
        # fix megatron-lm bug
        # https://github.com/NVIDIA/Megatron-LM/blob/core_r0.12.0/megatron/core/pipeline_parallel/schedules.py#L291
        loss = loss / mpu.get_context_parallel_world_size()
        return (loss, reporting_metric)

    def _replace_data_iterator(self, data_iterator):
        args = get_args()
        num_iters_per_step = args.global_batch_size // (args.micro_batch_size * mpu.get_data_parallel_world_size())
        res = []
        for i in range(num_iters_per_step):
            with torch.no_grad():
                res.append(self.ref_forward(data_iterator))
        return iter(res)

    def forward_step(self, data_iterator, model):
        with torch.no_grad():
            data = next(data_iterator)

        ref_logps = data.pop('logps')
        with self.stimer:
            output_tensor = model(**data)
        return output_tensor, partial(
            self.loss_func,
            ref_logps=ref_logps,
            labels=data.get('labels'),
            packed_seq_params=data.get('packed_seq_params'))

# Copyright (c) Alibaba, Inc. and its affiliates.
from collections import namedtuple
from contextlib import contextmanager, nullcontext
from functools import partial

import torch
from megatron.core import mpu
from megatron.core.inference.communication_utils import recv_from_prev_pipeline_rank_, send_to_next_pipeline_rank
from megatron.training import get_args, get_model, training
from megatron.training.checkpointing import load_checkpoint
from megatron.training.utils import unwrap_model
from torch.distributed.nn import all_reduce

from swift.utils import get_current_device, get_logger
from .trainer import MegatronTrainer
from .utils import get_batch

logger = get_logger()


class MegatronRLHFTrainer(MegatronTrainer):

    @contextmanager
    def _get_iters(self, train_dataset, val_dataset):
        origin_initialize_megatron = training.initialize_megatron

        def initialize_megatron(*_args, **kwargs):
            res = origin_initialize_megatron(*_args, **kwargs)
            args = get_args()
            data_parallel_size = mpu.get_data_parallel_world_size()
            step_batch_size = args.micro_batch_size * data_parallel_size
            if args.train_iters is None and args.max_epochs is not None:
                if hasattr(train_dataset, '__len__'):
                    dataset_sample = len(train_dataset) // step_batch_size * step_batch_size * args.num_generations
                    args.train_iters = dataset_sample * args.max_epochs // args.global_batch_size
                else:
                    raise ValueError(
                        'You are using a streaming training dataset. Please explicitly specify `--train_iters`.')
            if args.eval_iters < 0:
                if val_dataset is None:
                    args.eval_iters = 0
                elif hasattr(val_dataset, '__len__'):
                    dataset_sample = len(val_dataset) // step_batch_size * step_batch_size
                    args.eval_iters = max(dataset_sample // args.global_batch_size, 1)
                else:
                    raise ValueError(
                        'You are using a streaming validation dataset. Please explicitly specify `--eval_iters`.')
            return res

        training.initialize_megatron = initialize_megatron
        try:
            yield
        finally:
            training.initialize_megatron = origin_initialize_megatron

    def setup_model_and_optimizer(self, model_provider_func, model_type, *_args, **kwargs):
        args = get_args()
        if args.train_type == 'full':
            ref_model = get_model(model_provider_func, model_type)
            if args.ref_load is None:
                args.ref_load = args.load
            args.iteration, args.num_floating_point_operations_so_far = load_checkpoint(
                ref_model, None, None, load_arg='ref_load')
            self.ref_model = ref_model[0]
            self.ref_model.eval()
        else:
            self.ref_model = None
        return super().setup_model_and_optimizer(model_provider_func, model_type, *_args, **kwargs)

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

    def model_forward(self, model, data_iterator, no_grad=True):
        # used to calculate model forward (logps)
        with self.stimer(bdata=True):
            data = get_batch(data_iterator)
        data.pop('loss_scale', None)
        labels = data.get('labels')
        context = torch.no_grad() if no_grad else nullcontext()
        with context:
            output_tensor = self._forward_step_helper(model, data)
        data['logps'] = None if labels is None else self.get_logps(
            output_tensor, labels, data['packed_seq_params'], per_token=True)
        return data

    @staticmethod
    def get_logps(output_tensor, labels, packed_seq_params, per_token: bool = False):
        args = get_args()
        per_token_logps = -output_tensor
        loss_mask = labels != -100
        per_token_logps = per_token_logps * loss_mask
        num_samples = packed_seq_params.num_samples
        if args.rlhf_type == 'dpo':
            total_samples = num_samples * 2
        elif args.rlhf_type in 'grpo':
            total_samples = num_samples

        cu_seqlens = packed_seq_params.cu_seqlens_q[:total_samples + 1] // args.context_parallel_size

        if per_token:
            if args.context_parallel_size > 1:
                per_token_logps = all_reduce(per_token_logps, group=mpu.get_context_parallel_group())
            return per_token_logps
        else:
            all_logps = per_token_logps.new_zeros((total_samples, ))
            for i in range(total_samples):
                start, end = cu_seqlens[i], cu_seqlens[i + 1]
                all_logps[i] = per_token_logps[:, start:end].sum()
            if args.context_parallel_size > 1:
                all_logps = all_reduce(all_logps, group=mpu.get_context_parallel_group())
            return all_logps

    @contextmanager
    def null_ref_context(self):
        args = get_args()
        if args.train_type == 'full':
            context = nullcontext()
            ref_model = unwrap_model(self.ref_model)
        else:
            if args.ref_adapter_load is None:
                context = self.peft_model.disable_adapter()
            else:
                context = nullcontext()
            ref_model = self.unwrapped_model
        with context:
            if args.ref_adapter_load:
                self.peft_model.set_adapter('ref_adapter')
            yield ref_model
            if args.ref_adapter_load:
                self.peft_model.set_adapter('default')

    @contextmanager
    def offload_context(self):
        # TODO: offload
        yield

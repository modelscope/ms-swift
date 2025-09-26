# Copyright (c) Alibaba, Inc. and its affiliates.
from contextlib import contextmanager, nullcontext
from functools import partial

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import mpu
from megatron.core.inference.communication_utils import recv_from_prev_pipeline_rank_, send_to_next_pipeline_rank
from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy
from megatron.training import get_args, get_model
from megatron.training.checkpointing import load_checkpoint
from megatron.training.utils import unwrap_model
from torch.distributed.nn import all_reduce

from swift.utils import get_current_device, get_logger
from .trainer import MegatronTrainer
from .utils import get_kto_batch

logger = get_logger()


class MegatronKTOTrainer(MegatronTrainer):

    def __init__(self, args, template):
        super().__init__(args, template)
        self.beta = args.beta
        self.desirable_weight = args.desirable_weight
        self.undesirable_weight = args.undesirable_weight
        self.calculate_KL = args.calculate_KL

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

    @staticmethod
    def kto_loss(policy_chosen_logps, policy_rejected_logps, policy_KL_logps, reference_chosen_logps,
                 reference_rejected_logps, reference_KL_logps, beta, desirable_weight, undesirable_weight, calculate_KL,
                 device):
        if calculate_KL and policy_KL_logps is not None and reference_KL_logps is not None:
            kl = (policy_KL_logps - reference_KL_logps).mean().detach()
            dist.all_reduce(kl, group=mpu.get_data_parallel_group())
            kl = kl / mpu.get_data_parallel_world_size()
            kl = kl.clamp(min=0)
        else:
            kl = torch.tensor(0.0, device=device)

        chosen_rewards = torch.tensor([], device=kl.device)
        if policy_chosen_logps.shape[0] > 0:
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            chosen_losses = 1 - F.sigmoid(beta * (chosen_logratios - kl))
            chosen_rewards = beta * chosen_logratios.detach()
        else:
            chosen_losses = torch.tensor([], device=kl.device)

        rejected_rewards = torch.tensor([], device=kl.device)
        if policy_rejected_logps.shape[0] > 0:
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            rejected_losses = 1 - F.sigmoid(beta * (kl - rejected_logratios))
            rejected_rewards = beta * rejected_logratios.detach()
        else:
            rejected_losses = torch.tensor([], device=kl.device)

        losses = torch.cat((desirable_weight * chosen_losses, undesirable_weight * rejected_losses), 0)
        return losses, chosen_rewards, rejected_rewards, kl

    def loss_func(self, output_tensor, *, policy_KL_logps, reference_logps, reference_KL_logps, labels, all_labels,
                  packed_seq_params):
        policy_logps = self.get_logps(output_tensor, labels, packed_seq_params)
        is_desirable = all_labels.bool()

        policy_chosen_logps = policy_logps[is_desirable]
        policy_rejected_logps = policy_logps[~is_desirable]
        reference_chosen_logps = reference_logps[is_desirable]
        reference_rejected_logps = reference_logps[~is_desirable]

        loss, chosen_rewards, rejected_rewards, kl = self.kto_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            policy_KL_logps=policy_KL_logps,
            reference_chosen_logps=reference_chosen_logps,
            reference_rejected_logps=reference_rejected_logps,
            reference_KL_logps=reference_KL_logps,
            beta=self.beta,
            desirable_weight=self.desirable_weight,
            undesirable_weight=self.undesirable_weight,
            calculate_KL=self.calculate_KL,
            device=policy_logps.device,
        )

        loss = loss.mean() if loss.numel() > 0 else torch.tensor(0.0, device=policy_logps.device)

        with torch.no_grad():
            chosen_rewards_mean = chosen_rewards.mean() if chosen_rewards.numel() > 0 else torch.tensor(0.0).to(
                loss.device)
            rejected_rewards_mean = rejected_rewards.mean() if rejected_rewards.numel() > 0 else torch.tensor(0.0).to(
                loss.device)
            policy_chosen_logps_mean = policy_chosen_logps.mean() if policy_chosen_logps.numel() > 0 else torch.tensor(
                0.0).to(loss.device)
            policy_rejected_logps_mean = policy_rejected_logps.mean(
            ) if policy_rejected_logps.numel() > 0 else torch.tensor(0.0).to(loss.device)

        metric = {
            'loss': loss.clone().detach(),
            'logps/chosen': policy_chosen_logps_mean,
            'logps/rejected': policy_rejected_logps_mean,
            'rewards/chosen': chosen_rewards_mean,
            'rewards/rejected': rejected_rewards_mean,
            'rewards/margins': chosen_rewards_mean - rejected_rewards_mean,
            'kl': kl.detach() if kl is not None else torch.tensor(0.0).to(loss.device),
        }

        reporting_metric = loss.new_tensor(list(metric.values()))
        torch.distributed.all_reduce(
            reporting_metric, torch.distributed.ReduceOp.AVG, group=mpu.get_data_parallel_group())
        reporting_metric = {k: reporting_metric[i] for i, k in enumerate(metric.keys())}
        # fix megatron-lm bug
        # https://github.com/NVIDIA/Megatron-LM/blob/core_r0.12.0/megatron/core/pipeline_parallel/schedules.py#L291
        loss = loss / mpu.get_context_parallel_world_size()
        return loss, reporting_metric

    def _replace_data_iterator_with_model(self, data_iterator, model):
        args = get_args()
        num_iters_per_step = args.global_batch_size // (args.micro_batch_size * mpu.get_data_parallel_world_size())

        processed_data_list = []
        policy_model = unwrap_model(model)[0]

        for _ in range(num_iters_per_step):
            with torch.no_grad(), self.null_ref_context() as ref_model:
                data = self.ref_forward(ref_model, data_iterator)

            if self.calculate_KL:
                with torch.no_grad():
                    kl_inputs = {
                        'input_ids': data.get('KL_completion_input_ids'),
                        'attention_mask': data.get('KL_completion_attention_mask'),
                        'position_ids': data.get('KL_completion_position_ids'),
                    }

                    kl_output_tensor = self._forward_step_helper(policy_model, kl_inputs)

                    policy_KL_logps = self.get_logps(kl_output_tensor, data['KL_completion_labels'],
                                                     data.get('KL_completion_packed_seq_params'))
                    data['policy_KL_logps'] = policy_KL_logps

            processed_data_list.append(data)

        return iter(processed_data_list)

    def ref_forward(self, ref_model, data_iterator):
        with self.stimer(bdata=True):
            data = get_kto_batch(data_iterator)
        data.pop('loss_scale', None)

        ref_inputs = {
            'input_ids': data.get('completion_input_ids'),
            'attention_mask': data.get('completion_attention_mask'),
            'position_ids': data.get('completion_position_ids'),
        }
        with torch.no_grad():
            output_tensor = self._forward_step_helper(ref_model, ref_inputs)
        data['reference_logps'] = self.get_logps(output_tensor, data['completion_labels'],
                                                 data.get('completion_packed_seq_params'))

        if self.calculate_KL:
            kl_inputs = {
                'input_ids': data.get('KL_completion_input_ids'),
                'attention_mask': data.get('KL_completion_attention_mask'),
                'position_ids': data.get('KL_completion_position_ids'),
            }
            with torch.no_grad():
                kl_output_tensor = self._forward_step_helper(ref_model, kl_inputs)
            data['reference_KL_logps'] = self.get_logps(kl_output_tensor, data['KL_completion_labels'],
                                                        data.get('KL_completion_packed_seq_params'))
        else:
            data['reference_KL_logps'] = None
        return data

    def train_step(self, forward_step_func, data_iterator, model, optimizer, opt_param_scheduler, config):
        new_data_iterator = self._replace_data_iterator_with_model(data_iterator, model)
        return self._origin_train_step(forward_step_func, new_data_iterator, model, optimizer, opt_param_scheduler,
                                       config)

    def forward_step(self, data_iterator, model):
        data = next(data_iterator)

        reference_logps = data.pop('reference_logps')
        reference_KL_logps = data.pop('reference_KL_logps', None)
        policy_KL_logps = data.pop('policy_KL_logps', None)
        all_labels = torch.tensor(data.pop('label')).to(get_current_device())
        completion_packed_seq_params = data.get('completion_packed_seq_params')

        main_inputs = {
            'input_ids': data['completion_input_ids'],
            'attention_mask': data.get('completion_attention_mask'),
            'position_ids': data.get('completion_position_ids')
        }
        with self.stimer():
            output_tensor = model(**main_inputs)

        return output_tensor, partial(
            self.loss_func,
            policy_KL_logps=policy_KL_logps,
            reference_logps=reference_logps,
            reference_KL_logps=reference_KL_logps,
            labels=data['completion_labels'],
            all_labels=all_labels,
            packed_seq_params=completion_packed_seq_params)

    def evaluate(self,
                 forward_step_func,
                 data_iterator,
                 model,
                 process_non_loss_data_func,
                 config,
                 verbose=False,
                 non_loss_data_func=None):
        self._replace_data_iterator = partial(self._replace_data_iterator_with_model, model=model)
        return super().evaluate(forward_step_func, data_iterator, model, process_non_loss_data_func, config, verbose,
                                non_loss_data_func)

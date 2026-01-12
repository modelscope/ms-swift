# Copyright (c) Alibaba, Inc. and its affiliates.
from functools import partial
from typing import List, Optional

import torch
import torch.nn
from megatron.core import mpu
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.training import get_args, get_timers
from torch.distributed.nn import all_reduce
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from swift.utils import get_logger
from .base import BaseMegatronTrainer

logger = get_logger()


class MegatronTrainer(BaseMegatronTrainer):

    def seq_cls_loss_func(self, output_tensor, *, labels: torch.Tensor, packed_seq_params=None, attention_mask=None):
        args = self.args
        assert args.context_parallel_size == 1, 'Currently `task_type="seq_cls"` does not support context parallelism.'
        logits = self.get_last_tokens(output_tensor, packed_seq_params, attention_mask)
        num_labels = args.num_labels
        acc = None
        if args.problem_type == 'regression':
            loss_fct = MSELoss()
            if num_labels == 1:
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                loss = loss_fct(logits, labels)
        elif args.problem_type == 'single_label_classification':
            loss_fct = CrossEntropyLoss()
            logits = logits.view(-1, num_labels)
            labels = labels.view(-1)
            loss = loss_fct(logits, labels)
            acc = (logits.detach().argmax(dim=-1) == labels).float().mean()
        elif args.problem_type == 'multi_label_classification':
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
        metric = {'loss': loss.detach().clone()}
        if acc is not None:
            metric['acc'] = acc
        metric = self._all_reduce_metric(metric)
        return loss, metric

    # Code borrowed from NVIDIA/Megatron-LM
    def loss_func(self,
                  output_tensor: torch.Tensor,
                  *,
                  labels: torch.Tensor,
                  loss_scale: Optional[torch.Tensor] = None,
                  channels: Optional[List[str]] = None,
                  packed_seq_params=None,
                  logits: Optional[torch.Tensor] = None):
        args = get_args()

        losses = output_tensor.float()
        loss_mask = labels != -100
        if args.enable_dft_loss:
            losses = losses * torch.exp(-losses.detach())
        if args.enable_eaft_loss and logits is not None:
            with torch.no_grad():
                logits_float = logits.float()
                vocab_size = logits_float.shape[-1]
                
                batch_size = labels.shape[0]
                seq_length = labels.shape[1]
                logits_reshaped = logits_float.view(batch_size * seq_length, vocab_size)

                topk_logits, topk_indices = torch.topk(logits_reshaped, k=20, dim=-1)
                logsumexp_topk = torch.logsumexp(topk_logits, dim=-1, keepdim=True)
                log_probs_topk = topk_logits - logsumexp_topk
                probs_topk = torch.exp(log_probs_topk)
                entropy_approx = -(probs_topk * log_probs_topk).sum(dim=-1)
                normalized_entropy = entropy_approx / 3.0
                eaft_weight = torch.pow(normalized_entropy, args.eaft_alpha)
                eaft_weight = eaft_weight.view(batch_size, seq_length)
                eaft_weight = torch.where(loss_mask, eaft_weight, torch.ones_like(eaft_weight))
                

            losses = losses * eaft_weight
        
        if loss_scale is not None:
            losses = losses * loss_scale
        if args.enable_channel_loss and channels is not None:
            mode = 'train' if self.unwrapped_models[0].training else 'eval'
            metrics = self.custom_metrics[mode]
            if args.padding_free:
                num_samples = packed_seq_params.num_samples
                cu_seqlens = packed_seq_params.cu_seqlens_q[:num_samples + 1] // args.context_parallel_size
                for i in range(cu_seqlens.shape[0] - 1):
                    channel = channels[i]
                    slice_ = slice(cu_seqlens[i], cu_seqlens[i + 1])
                    metrics[f'loss_{channel}'].update(losses[0, slice_][loss_mask[0, slice_]])
            else:
                for i in range(losses.shape[0]):
                    channel = channels[i]
                    metrics[f'loss_{channel}'].update(losses[i][loss_mask[i]])
        loss = torch.cat([torch.sum(losses * loss_mask).view(1), loss_mask.sum().view(1)])

        if args.context_parallel_size > 1 and not self.mcore_013:
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
        reporting_loss = loss.detach().clone()
        lm_loss = loss[0]
        if not self.mcore_013:
            # fix megatron-lm bug
            # https://github.com/NVIDIA/Megatron-LM/blob/core_r0.12.0/megatron/core/pipeline_parallel/schedules.py#L291
            torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())
            lm_loss = lm_loss / mpu.get_context_parallel_world_size()
            reporting_loss = (reporting_loss[0], reporting_loss[1])
        else:
            lm_loss = lm_loss.clone()
        local_num_tokens = loss[1].detach().clone().to(torch.int)
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
        vp_stage = model.module.module.vp_stage
        timers('batch-generator', log_level=2).start()
        with self.stimer(bdata=True):
            data = self.get_batch(data_iterator, vp_stage)
        timers('batch-generator').stop()
        loss_scale = data.pop('loss_scale', None)
        channels = data.pop('channel', None)
        labels = data.get('labels')
        if self.args.task_type == 'seq_cls':
            data.pop('labels', None)
        packed_seq_params = data.get('packed_seq_params')
        
        
        with self.stimer:
            if self.args.task_type != 'seq_cls':
                output_tensor, logits = model(**data, return_logits=True)
            else:
                output_tensor = model(**data)
        
        if self.args.task_type == 'seq_cls':
            loss_func = partial(
                self.seq_cls_loss_func,
                labels=labels,
                packed_seq_params=packed_seq_params,
                attention_mask=data.get('attention_mask'))
        else:
            loss_func = partial(
                self.loss_func,
                labels=labels,
                loss_scale=loss_scale,
                channels=channels,
                packed_seq_params=packed_seq_params,
                logits=logits)
        return output_tensor, loss_func

# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import torch.distributed as dist
import torch.nn
from collections import defaultdict
from functools import partial
from megatron.core import mpu
from torch.distributed.nn import all_reduce
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from typing import List, Optional

from swift.utils import get_logger
from .base import BaseMegatronTrainer

logger = get_logger()


class MegatronTrainer(BaseMegatronTrainer):

    def seq_cls_loss_func(self, output_tensor, *, labels: torch.Tensor, packed_seq_params=None, attention_mask=None):
        args = self.args
        if args.context_parallel_size > 1:
            raise ValueError('Currently `task_type="seq_cls"` does not support context parallelism.')
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
            preds = logits.sigmoid() > 0.5
            acc = (labels == preds).all(dim=-1).float().mean()
        metric = {'loss': loss.detach().clone()}
        if acc is not None:
            metric['acc'] = acc
        metric = self._all_reduce_metric(metric)
        return loss, metric

    def loss_func(self,
                  output_tensor: torch.Tensor,
                  *,
                  labels: torch.Tensor,
                  loss_scale: Optional[torch.Tensor] = None,
                  channels: Optional[List[str]] = None,
                  packed_seq_params=None):
        args = self.args

        losses = output_tensor.float()
        loss_mask = labels != -100
        if args.enable_dft_loss:
            losses = losses * torch.exp(-losses.detach())
        if loss_scale is not None:
            losses = losses * loss_scale
        loss = torch.cat([torch.sum(losses * loss_mask).view(1), loss_mask.sum().view(1)])

        if args.context_parallel_size > 1 and not self.mcore_013:
            loss = all_reduce(loss, group=mpu.get_context_parallel_group())

        # Reduce loss for logging.
        reporting_loss = loss.detach().clone()
        torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())

        lm_loss = loss[0]
        if not self.mcore_013:
            # fix megatron-lm bug
            # https://github.com/NVIDIA/Megatron-LM/blob/core_r0.12.0/megatron/core/pipeline_parallel/schedules.py#L291
            lm_loss = lm_loss / mpu.get_context_parallel_world_size()
        else:
            lm_loss = lm_loss.clone()
        local_num_tokens = loss[1].detach().clone().to(torch.int)
        metrics = {'loss': reporting_loss}
        if args.enable_channel_loss:
            metrics.update(self._compute_channel_loss(losses, loss_mask, channels, packed_seq_params))
        return (lm_loss, local_num_tokens, metrics)

    def _compute_channel_loss(self, losses, loss_mask, channels, packed_seq_params=None):
        args = self.args
        metrics = defaultdict(lambda: torch.tensor([0.0, 0.0], dtype=torch.float32, device=torch.cuda.current_device()))
        if args.padding_free:
            num_samples = packed_seq_params.num_samples
            cu_seqlens = packed_seq_params.cu_seqlens_q[:num_samples + 1] // args.context_parallel_size
            for i in range(cu_seqlens.shape[0] - 1):
                channel = None if channels is None else channels[i]
                slice_ = slice(cu_seqlens[i], cu_seqlens[i + 1])
                c_loss = losses[0, slice_][loss_mask[0, slice_]]
                metrics[f'loss_{channel}'][0] += c_loss.detach().sum()
                metrics[f'loss_{channel}'][1] += c_loss.shape[0]
        else:
            for i in range(losses.shape[0]):
                channel = None if channels is None else channels[i]
                c_loss = losses[i][loss_mask[i]]
                metrics[f'loss_{channel}'][0] += c_loss.detach().sum()
                metrics[f'loss_{channel}'][1] += c_loss.shape[0]

        # Synchronize keys to avoid getting stuck.
        all_keys = [None] * mpu.get_data_parallel_world_size()
        dist.all_gather_object(all_keys, list(metrics.keys()), group=mpu.get_data_parallel_group())
        new_metrics = {}
        for key in sorted(set().union(*all_keys)):
            new_metrics[key] = metrics[key]
        new_metrics = self._all_reduce_metric(new_metrics, torch.distributed.ReduceOp.SUM)
        return new_metrics

    def forward_step(self, data_iterator, model):
        # Get the batch.
        vp_stage = model.module.module.vp_stage
        data = self.get_batch(data_iterator, vp_stage)
        loss_scale = data.pop('loss_scale', None)
        channels = data.pop('channel', None)
        labels = data.get('labels')
        if self.args.task_type == 'seq_cls':
            data.pop('labels', None)
        output_tensor = model(**data)
        packed_seq_params = data.get('packed_seq_params')
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
                packed_seq_params=packed_seq_params)
        return output_tensor, loss_func

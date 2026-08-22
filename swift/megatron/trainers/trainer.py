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

from swift.utils import get_current_device, get_logger
from .base import BaseMegatronTrainer

logger = get_logger()


class MegatronTrainer(BaseMegatronTrainer):

    def seq_cls_loss_func(self, output_tensor, *, labels: torch.Tensor, packed_seq_params=None, attention_mask=None):
        args = self.args
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

        # Keep local loss stats for logging; defer the DP all-reduce to log time
        # (once per logging event) to avoid a global sync point per microbatch.
        reporting_loss = loss.detach().clone()

        lm_loss = loss[0]
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
            num_samples = packed_seq_params.seq_lens.shape[0]
            cu_seqlens = packed_seq_params.cu_seqlens_q[:num_samples + 1] // args.context_parallel_size
            device = losses.device
            total_len = int(cu_seqlens[-1])
            uniq_channels = [None] if channels is None else list(dict.fromkeys(channels))
            ch_index = {c: i for i, c in enumerate(uniq_channels)}
            if channels is None:
                seg_ch = torch.zeros(num_samples, dtype=torch.long, device=device)
            else:
                seg_ch = torch.tensor([ch_index[c] for c in channels], dtype=torch.long, device=device)
            token_ch = torch.repeat_interleave(seg_ch, cu_seqlens[1:] - cu_seqlens[:-1], output_size=total_len)
            mask = loss_mask[0, :total_len].float()
            stats = torch.zeros(len(uniq_channels), 2, dtype=torch.float32, device=device)
            stats.index_add_(0, token_ch, torch.stack([losses[0, :total_len].detach().float() * mask, mask], dim=-1))
            for c, idx in ch_index.items():
                metrics[f'loss_{c}'] = stats[idx]
        else:
            for i in range(losses.shape[0]):
                channel = None if channels is None else channels[i]
                c_loss = losses[i][loss_mask[i]]
                metrics[f'loss_{channel}'][0] += c_loss.detach().sum()
                metrics[f'loss_{channel}'][1] += c_loss.shape[0]

        # Synchronize keys to avoid getting stuck.
        dp_cp_group = mpu.get_data_parallel_group(with_context_parallel=True)
        all_keys = [None] * torch.distributed.get_world_size(group=dp_cp_group)
        dist.all_gather_object(all_keys, list(metrics.keys()), group=dp_cp_group)
        new_metrics = {}
        for key in sorted(set().union(*all_keys)):
            new_metrics[key] = metrics[key]
        new_metrics = self._all_reduce_metric(new_metrics, torch.distributed.ReduceOp.SUM, group=dp_cp_group)
        return new_metrics

    def _log_callback(self, logs, n_steps):
        # loss_func defers the logging-loss DP all-reduce from per-microbatch
        # to here (once per logging event); mathematically equivalent by
        # linearity, modulo floating-point reduction order.
        # All last-stage ranks must enter the collective unconditionally:
        # a rank whose whole logging window had zero valid tokens (e.g. a fully
        # masked CP shard) contributes zeros instead of skipping the call, which
        # would otherwise deadlock the group.
        if self.args.task_type == 'causal_lm' and mpu.is_pipeline_last_stage(ignore_virtual=True):
            v = logs.get('loss')
            if v is None:
                v = torch.zeros(2, dtype=torch.float32, device=get_current_device())
            dist.all_reduce(v, op=dist.ReduceOp.SUM, group=mpu.get_data_parallel_group(with_context_parallel=True))
            if v[1].item() > 0:
                logs['loss'] = v
            else:
                logs.pop('loss', None)
        super()._log_callback(logs, n_steps)

    def forward_step(self, data_iterator, model):
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
                attention_mask=data.get('attention_mask')
                if data.get('attention_mask') is not None else data.get('attention_mask_2d'))
        else:
            loss_func = partial(
                self.loss_func,
                labels=labels,
                loss_scale=loss_scale,
                channels=channels,
                packed_seq_params=packed_seq_params)
        return output_tensor, loss_func

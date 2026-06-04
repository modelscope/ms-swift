# Copyright (c) ModelScope Contributors. All rights reserved.
"""GKD loss for Ray-based Megatron training."""
from __future__ import annotations

import torch
from functools import partial
from megatron.core import mpu
from typing import Any, Dict

from swift.megatron.trainers.gkd_utils import align_vocab_size, cp_reduce, tp_gather_topk
from swift.megatron.trainers.vocab_parallel_utils import vocab_parallel_kl_div, vocab_parallel_log_softmax
from swift.rlhf_trainers.gkd_loss import gkd_loss
from swift.rlhf_trainers.gkd_trainer import TeacherOutput
from .base import Loss


class GKDLoss(Loss):
    """GKD loss: JSD between student and teacher + optional SFT loss."""

    def __init__(self, args):
        self.args = args
        self.beta = getattr(args, 'beta', 0.5)
        self.temperature = getattr(args, 'temperature', 1.0)
        self.sft_alpha = getattr(args, 'sft_alpha', 0.0)

    def forward_step(self, data_iterator, model):
        from swift.megatron.trainers.utils import prepare_batch

        data = next(data_iterator)
        teacher_output = data.pop('teacher_output', TeacherOutput())
        data.pop('opsd_teacher_batch', None)
        data.pop('opsd_teacher_messages', None)
        data = prepare_batch(self.args, data)

        data.pop('loss_scale', None)
        labels = data.pop('labels', None)

        student_output = model(**data)
        return student_output, partial(
            self.loss_func,
            labels=labels,
            teacher_output=teacher_output,
        )

    def loss_func(self, output_tensor, *, labels, teacher_output, data=None):
        args = self.args
        student_logits = output_tensor

        jsd_loss = gkd_loss(
            student_logits,
            teacher_output,
            labels,
            self.beta,
            self.temperature,
            gather_fn=tp_gather_topk,
            align_vocab_fn=align_vocab_size,
            log_softmax_fn=vocab_parallel_log_softmax,
            kl_div_fn=vocab_parallel_kl_div,
            reduce_fn=partial(cp_reduce, cp_size=args.context_parallel_size))

        loss = jsd_loss
        sft_loss = None
        if self.sft_alpha > 0:
            logits_sbv = student_logits.transpose(0, 1).contiguous()
            loss_mask = labels != -100
            per_token_loss = torch.nn.functional.cross_entropy(
                logits_sbv.view(-1, logits_sbv.size(-1)), labels.view(-1), reduction='none').view_as(labels)
            sft_loss_sum = (per_token_loss * loss_mask).sum()
            sft_loss_count = loss_mask.sum().float()
            if args.context_parallel_size > 1:
                sft_stats = torch.stack([sft_loss_sum, sft_loss_count])
                torch.distributed.all_reduce(
                    sft_stats, op=torch.distributed.ReduceOp.SUM, group=mpu.get_context_parallel_group())
                sft_loss_sum, sft_loss_count = sft_stats[0], sft_stats[1]
            sft_loss = sft_loss_sum / sft_loss_count if sft_loss_count > 0 else sft_loss_sum * 0
            loss = loss + self.sft_alpha * sft_loss

        metric = {'loss': loss.detach().clone()}
        if sft_loss is not None:
            metric['jsd_loss'] = jsd_loss.detach().clone()
            metric['sft_loss'] = sft_loss.detach().clone()

        # All-reduce metrics across DP group
        dp_group = mpu.get_data_parallel_group()
        reporting = torch.stack(list(metric.values()), dim=0)
        torch.distributed.all_reduce(reporting, torch.distributed.ReduceOp.AVG, group=dp_group)
        metric = {k: reporting[i] for i, k in enumerate(metric.keys())}

        loss = loss / mpu.get_context_parallel_world_size()
        return loss, metric

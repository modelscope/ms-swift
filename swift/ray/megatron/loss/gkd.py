# Copyright (c) ModelScope Contributors. All rights reserved.
"""GKD loss for Ray-based Megatron training."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from megatron.core import mpu
from typing import Any, Dict, List, Optional, Sequence

from swift.megatron.trainers.gkd_utils import cp_reduce, tp_gather_topk, vocab_parallel_topk
from swift.megatron.trainers.utils import prepare_batch
from swift.megatron.trainers.vocab_parallel_utils import vocab_parallel_kl_div, vocab_parallel_log_softmax
from swift.megatron.utils import forward_step_helper, get_padding_to
from swift.rlhf_trainers.gkd_loss import TeacherOutput, gkd_loss
from swift.utils import gc_collect, get_current_device, to_device
from .base import Loss


class GKDLoss(Loss):
    """GKD loss: JSD between student and teacher + optional SFT loss."""

    def __init__(self, args):
        self.args = args
        self.beta = getattr(args, 'beta', 0.5)
        self.temperature = getattr(args, 'temperature', 1.0)
        self.sft_alpha = getattr(args, 'sft_alpha', 0.0)

    def forward_step(self, data_iterator, model):
        data = next(data_iterator)
        teacher_output = data.pop('teacher_output', TeacherOutput())
        data.pop('opsd_teacher_batch', None)
        data.pop('opsd_teacher_messages', None)
        data = prepare_batch(self.args, data)

        data.pop('loss_scale', None)
        labels = data.pop('labels', None)

        student_output = model(**data)
        from functools import partial
        return student_output, partial(
            self.loss_func,
            labels=labels,
            teacher_output=teacher_output,
        )

    # ------------------------------------------------------------------
    # Teacher logit computation (called from MegatronWorker thin wrapper)
    # ------------------------------------------------------------------

    def compute_teacher_logits(
        self,
        teacher_model: torch.nn.Module,
        batch: List[Dict[str, Any]],
        template,
        args,
    ):
        """Forward teacher model on encoded batch.

        Returns:
            (results, cached) — In topk mode, results is a list of
            TeacherOutput (CPU) and cached is empty. In full_logits mode,
            results is empty and cached is a list of GPU tensors for
            injection during train_step.
        """
        gkd_logits_topk = getattr(args, 'gkd_logits_topk', None)
        device = get_current_device()
        micro_batch_size = getattr(args, 'micro_batch_size', len(batch))
        sample_chunks = [batch[i:i + micro_batch_size] for i in range(0, len(batch), micro_batch_size)]

        results: list = []
        cached: list = []
        with torch.no_grad():
            for chunk in sample_chunks:
                encoded_list = [s['encoded'] for s in chunk]
                collated = template.data_collator(encoded_list, padding_to=get_padding_to(args))
                collated = to_device(collated, device)
                teacher_data = prepare_batch(args, collated)
                teacher_data.pop('loss_scale', None)
                teacher_data.pop('labels', None)
                teacher_logits = forward_step_helper(teacher_model, teacher_data)
                if teacher_logits is not None:
                    teacher_logits = teacher_logits.detach()
                    for i in range(len(chunk)):
                        if gkd_logits_topk is not None:
                            topk_logits, topk_indices = vocab_parallel_topk(teacher_logits[i:i + 1], k=gkd_logits_topk)
                            results.append(
                                TeacherOutput(topk_logprobs=topk_logits.cpu(), topk_indices=topk_indices.cpu()))
                        else:
                            cached.append(teacher_logits[i:i + 1])
                else:
                    if gkd_logits_topk is not None:
                        results.extend(TeacherOutput() for _ in chunk)
                    else:
                        cached.extend(None for _ in chunk)
                del collated

        return results, cached

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    @staticmethod
    def _logp_gap(student_logits, teacher_output, labels, temperature):
        """Debug metric: teacher-probability-weighted expected abs logprob gap between
        student and teacher over the teacher top-k, at covered response positions.
        For an identical student/teacher model (and correct token alignment) this is ~0."""
        if not teacher_output.is_topk_mode:
            return None
        mask = labels != -100
        uncovered = torch.isinf(teacher_output.topk_logprobs).all(dim=-1)
        mask = mask & ~uncovered
        if mask.sum() == 0:
            return None
        s_topk = tp_gather_topk(student_logits, teacher_output.topk_indices)
        s_log = F.log_softmax(s_topk[mask].float() / temperature, dim=-1)
        t_log = F.log_softmax(teacher_output.topk_logprobs[mask].float() / temperature, dim=-1)
        gap = (t_log.exp() * (s_log - t_log).abs()).sum(dim=-1)
        gap = gap[torch.isfinite(gap)]
        return gap.mean() if gap.numel() > 0 else None

    def loss_func(self, output_tensor, *, labels, teacher_output, data=None):
        args = self.args
        student_logits = output_tensor

        jsd_total, jsd_num_valid = gkd_loss(
            student_logits,
            teacher_output,
            labels,
            self.beta,
            self.temperature,
            gather_fn=tp_gather_topk,
            log_softmax_fn=vocab_parallel_log_softmax,
            kl_div_fn=vocab_parallel_kl_div)
        jsd_loss_val = cp_reduce(jsd_total, jsd_num_valid, cp_size=args.context_parallel_size)

        loss = jsd_loss_val
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
            metric['jsd_loss'] = jsd_loss_val.detach().clone()
            metric['sft_loss'] = sft_loss.detach().clone()

        logp_gap = self._logp_gap(student_logits, teacher_output, labels, self.temperature)
        if logp_gap is not None:
            metric['logp_gap'] = logp_gap.detach().to(loss.dtype)

        dp_group = mpu.get_data_parallel_group()
        reporting = torch.stack(list(metric.values()), dim=0)
        torch.distributed.all_reduce(reporting, torch.distributed.ReduceOp.AVG, group=dp_group)
        metric = {k: reporting[i] for i, k in enumerate(metric.keys())}

        loss = loss / mpu.get_context_parallel_world_size()
        return loss, metric

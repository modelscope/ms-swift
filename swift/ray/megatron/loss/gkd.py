# Copyright (c) ModelScope Contributors. All rights reserved.
"""GKD loss for Ray-based Megatron training."""
from __future__ import annotations

import torch
from functools import partial
from megatron.core import mpu
from typing import Any, Dict, List, Optional

from swift.megatron.trainers.gkd_utils import cp_reduce, tp_gather_topk, vocab_parallel_topk
from swift.megatron.trainers.utils import prepare_batch
from swift.megatron.trainers.vocab_parallel_utils import vocab_parallel_kl_div, vocab_parallel_log_softmax
from swift.megatron.utils import forward_step_helper
from swift.rlhf_trainers.gkd_loss import DataSource, TeacherOutput, gkd_loss
from swift.utils import get_current_device, to_device
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
        data_source = data.pop('data_source', None)
        data.pop('grpo_batch', None)  # RL signals packed in GRPOBatch (not used by GKD loss)
        data = prepare_batch(self.args, data)

        data.pop('loss_scale', None)
        data.pop('routed_experts', None)  # MoE routing replay data (not a model forward kwarg)
        labels = data.pop('labels', None)

        # data is now clean model forward kwargs (template.encode guarantees this)
        student_output = model(**data)
        return student_output, partial(
            self.loss_func,
            labels=labels,
            teacher_output=teacher_output,
            data_source=data_source,
            model=model,
        )

    # ------------------------------------------------------------------
    # Teacher logit computation (called from MegatronWorker thin wrapper)
    # ------------------------------------------------------------------

    def compute_teacher_logits(
        self,
        teacher_model: torch.nn.Module,
        teacher_micro_batches: List[Dict[str, Any]],
        args,
    ) -> List[TeacherOutput]:
        gkd_logits_topk = getattr(args, 'gkd_logits_topk', None)
        device = get_current_device()
        outputs: List[TeacherOutput] = []
        with torch.no_grad():
            for teacher_model_inputs in teacher_micro_batches:
                collated = to_device(dict(teacher_model_inputs), device)
                teacher_data = prepare_batch(args, collated)
                teacher_data.pop('loss_scale', None)
                # Labels are the teacher's (OPSD) or == student's (non-OPSD); prepare_batch
                # shifts them so extract_active mask-aligns the shared response.
                labels = teacher_data.pop('labels', None)
                teacher_logits = forward_step_helper(teacher_model, teacher_data)
                if teacher_logits is None:
                    # PP non-last stage: no logits; placeholder keeps micro-batch alignment.
                    outputs.append(TeacherOutput())
                    continue
                teacher_logits = teacher_logits.detach()
                if gkd_logits_topk is not None:
                    topk_logits, topk_indices = vocab_parallel_topk(teacher_logits, k=gkd_logits_topk)
                    outputs.append(TeacherOutput(topk_logprobs=topk_logits, topk_indices=topk_indices, labels=labels))
                else:
                    outputs.append(TeacherOutput(full_logits=teacher_logits, labels=labels))
                del collated
        return outputs

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def loss_func(self, output_tensor, *, labels, teacher_output, data_source=None, model=None):
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
        # SFT loss only applies to ground-truth (dataset) responses; skip it on
        # student-generated (on-policy) responses, matching the non-ray GKD trainer.
        if self.sft_alpha > 0 and data_source != DataSource.STUDENT:
            # Vocab-parallel-aware SFT loss: route through ``model.compute_language_model_loss``
            # (mirrors the non-ray GKD trainer). Naive ``torch.nn.functional.cross_entropy``
            # would index out of bounds on the TP-sharded local vocab when TP>1.
            assert model is not None, 'sft_alpha>0 requires the model handle from forward_step'
            unwrapped = model
            while hasattr(unwrapped, 'module'):
                unwrapped = unwrapped.module
            if hasattr(unwrapped, 'language_model'):
                unwrapped = unwrapped.language_model
            logits_sbv = student_logits.transpose(0, 1).contiguous()
            per_token_loss = unwrapped.compute_language_model_loss(labels, logits_sbv)
            loss_mask = labels != -100
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

        dp_group = mpu.get_data_parallel_group()
        reporting = torch.stack(list(metric.values()), dim=0)
        torch.distributed.all_reduce(reporting, torch.distributed.ReduceOp.AVG, group=dp_group)
        metric = {k: reporting[i] for i, k in enumerate(metric.keys())}

        loss = loss / mpu.get_context_parallel_world_size()
        return loss, metric

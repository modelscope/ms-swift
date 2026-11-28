# Copyright (c) ModelScope Contributors. All rights reserved.
# Training-time metrics for the `decision` task_type, twinkle Metric contract.
from __future__ import annotations
from typing import Any, Dict, List, Union

import numpy as np
import torch
from twinkle.data_format import InputFeature, ModelOutput
from twinkle.metric import Metric

from swift.dev.model.decision_head import masked_argmax, masked_softmax
from swift.dev.utils.scoring import ece_score

__all__ = ["DecisionScoringMetric"]

ECE_BINS = 15


class DecisionScoringMetric(Metric):
    """acc + record_acc + ece over a decision (System-1 scoring) forward"""

    def __init__(self, device_mesh, process_group, ece_bins: int = ECE_BINS, **kwargs):
        super().__init__(device_mesh, process_group, **kwargs)
        self.ece_bins = int(ece_bins)
        self.correct: List[float] = []  # per-question 0/1
        self.confidences: List[float] = []  # per-question top confidence
        self.record_flags: List[float] = []  # per-record all-correct 0/1

    # -- accumulation ---------------------------------------------------------

    def accumulate(self, inputs: Union[InputFeature, List[InputFeature]], outputs: ModelOutput, **kwargs):
        logits = outputs.get("logits")
        # The twinkle Metric contract hands metrics either ONE merged feature dict (transformers
        # backend) or a LIST of per-microbatch dicts (megatron backend: outputs are merged across
        # micro-batches in forward order, so the labels must be merged the same way). Decision
        # labels are per-question target distributions [total_q, max_opt]; the option width can
        # differ per micro-batch, so pad-and-stack (pad 0 keeps argmax on the real positive-mass
        # options; predictions are option_mask-restricted anyway).
        if isinstance(inputs, list):
            labels_list = [inp.get("labels") for inp in inputs]
            labels_list = [l for l in labels_list if l is not None]
            if not labels_list:
                labels = None
            else:
                from twinkle.utils import pad_and_stack_tensors
                labels = pad_and_stack_tensors(labels_list, pad_value=0)
        else:
            labels = inputs.get("labels")
        if logits is None or labels is None:
            return
        option_mask = outputs.get("option_mask")
        if option_mask is None:
            raise ValueError(
                "decision outputs must carry option_mask ([total_q, max_opt], True = real option "
                "slot); a missing mask means the head forward is broken."
            )
        preds = masked_argmax(logits, option_mask)
        if isinstance(labels, torch.Tensor):
            labels = labels.to(device=logits.device)
        target = labels.argmax(dim=-1) if labels.dim() > 1 else labels
        correct = preds == target
        probs = masked_softmax(logits, option_mask)
        conf = probs.max(dim=-1).values

        self.correct.extend(correct.float().tolist())
        self.confidences.extend(conf.float().tolist())

        counts = outputs.get("counts")
        if counts is not None:
            counts_list = counts.tolist() if isinstance(counts, torch.Tensor) else [int(c) for c in counts]
            offset = 0
            for n in counts_list:
                n = int(n)
                if n > 0:
                    self.record_flags.append(float(correct[offset : offset + n].all()))
                offset += n
        else:
            self.record_flags.extend(correct.float().tolist())

    # -- reporting ------------------------------------------------------------

    def reset(self):
        self.correct = []
        self.confidences = []
        self.record_flags = []

    def calculate(self):
        local_results: List[Dict[str, Any]] = [
            {
                "correct": self.correct,
                "confidences": self.confidences,
                "record_flags": self.record_flags,
            }
        ]
        all_results = self.gather_results(local_results)
        correct = [c for r in all_results for c in r["correct"]]
        confidences = [c for r in all_results for c in r["confidences"]]
        record_flags = [f for r in all_results for f in r["record_flags"]]
        self.reset()
        if not correct:
            return {}
        return {
            "acc": f"{float(np.mean(correct)):.4f}",
            "record_acc": f"{float(np.mean(record_flags)):.4f}" if record_flags else "-",
            "ece": f"{ece_score(confidences, correct, n_bins=self.ece_bins):.4f}",
        }

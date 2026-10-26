# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
from peft import PeftModel

from swift.model.decision_head import get_scoring_head
from swift.utils import get_logger
from .trainer import Trainer
from .utils import is_instance_of_ms_model

logger = get_logger()


class ScoringTrainer(Trainer):
    """Trainer for the `decision` task_type (typed-decision System-1 scoring models)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._activate_scoring_head()

    def _activate_scoring_head(self) -> None:
        """Re-activate the attached `scoring_head` under adapter training"""
        model = self.model
        if not (isinstance(model, PeftModel) or is_instance_of_ms_model(model)):
            return
        for name, p in model.named_parameters():
            if 'scoring_head' not in name:
                continue
            if not p.requires_grad:
                p.requires_grad_(True)
            if p.dtype == torch.float16:
                p.data = p.data.to(torch.float32)

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir, state_dict)
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        head = get_scoring_head(self.model)
        if head is not None:
            is_adapter = isinstance(self.model, PeftModel) or is_instance_of_ms_model(self.model)
            head.save_pretrained(output_dir, save_weights=is_adapter)
            if is_adapter:
                logger.info(f'decision: saved scoring_head ({head.HEAD_WEIGHTS_NAME} + {head.HEAD_META_NAME}) '
                            f'to {output_dir}.')
            else:
                logger.info(f'decision: saved scoring_head meta ({head.HEAD_META_NAME}) '
                            f'to {output_dir} (weights in model state_dict).')

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.compute_loss_func is not None:
            labels = inputs.pop('labels', None)
            outputs = model(**inputs)
            if labels is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=None, trainer=self)
            else:
                loss = outputs.loss

            accumulation_steps = getattr(self, 'current_gradient_accumulation_steps',
                                         self.args.gradient_accumulation_steps)
            loss = loss / accumulation_steps

            if labels is not None:
                self._compute_acc(outputs, labels)

            return (loss, outputs) if return_outputs else loss
        else:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

    def _compute_acc(self, outputs, labels, cu_seqlens=None) -> None:
        """Per-question accuracy and per-record accuracy.

        ``acc``: masked argmax over each question's own option set vs the gold option (argmax of
        the padded target distribution) — every question scored independently.

        ``record_acc``: all-or-nothing accuracy at the record level — a record counts as correct
        only when ALL its questions are predicted correctly. The flat ``total_q`` question rows are
        regrouped by ``outputs.counts`` (questions per record, record-major order) so a multi-question
        record (e.g. Clef's joint decode) is scored as a whole.
        """
        from swift.model.decision_head import masked_argmax
        logits = outputs.logits
        option_mask = getattr(outputs, 'option_mask', None)
        if option_mask is None:
            option_mask = torch.ones_like(logits, dtype=torch.bool)
        preds = masked_argmax(logits, option_mask)  # [total_q]
        target = labels.argmax(dim=-1) if labels.dim() > 1 else labels
        correct = (preds == target).float()
        mode = 'train' if self.model.training else 'eval'
        self.custom_metrics[mode]['acc'].update(correct)

        counts = getattr(outputs, 'counts', None)
        if counts is not None:
            counts_list = counts.tolist() if isinstance(counts, torch.Tensor) else [int(c) for c in counts]
            num_records = len(counts_list)
            if num_records > 0:
                record_acc = torch.zeros(num_records, device=correct.device, dtype=correct.dtype)
                offset = 0
                for i, n in enumerate(counts_list):
                    if n > 0:
                        record_acc[i] = correct[offset:offset + n].min()
                    offset += n
                self.custom_metrics[mode]['record_acc'].update(record_acc)

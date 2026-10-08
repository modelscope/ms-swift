# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
from peft import PeftModel

from swift.model.decision_head import get_scoring_head
from swift.utils import get_logger
from .trainer import Trainer
from .utils import is_instance_of_ms_model

logger = get_logger()


class ScoringTrainer(Trainer):
    """Trainer for the `decision` task_type (typed-decision System-1 scoring models).

    The patched model forward returns a `ScoringOutput` whose `logits` are per-option scores
    `[total_q, max_opt]` (padded, read with `option_mask`). `compute_loss_func` (a ScoringLoss
    picked by `--loss_type`) turns those + the padded `target_probs` labels into a loss.

    Mirrors `RerankerTrainer.compute_loss` (pop labels -> forward -> loss fn -> acc) minus the
    generative_reranker last-token special case, and overrides `_compute_acc` for the
    per-question masked-argmax accuracy (the base mixin only knows seq_cls/causal_lm/reranker).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._activate_scoring_head()

    def _activate_scoring_head(self) -> None:
        """Re-activate the attached `scoring_head` under adapter training (plan decision C: base
        frozen + LoRA + head jointly trained).

        Why here: `TunerMixin.prepare_model`'s adapter branch runs `model.requires_grad_(False)` and
        re-applies `trainable_parameters` ONLY in its `full` branch, so a LoRA run silently freezes
        the head. The usual escapes do not work -- `--trainable_parameters scoring_head` cannot match
        (PEFT renames it to `base_model.model.scoring_head.*` while `activate_parameters` matches by
        `startswith`), and `modules_to_save` is unsafe (the head keeps a live lm_head reference that
        PEFT's `deepcopy` would duplicate; plan decision F persists it separately instead). The
        trainer is built after `prepare_model` and before `create_optimizer` (which runs inside
        `train()`), so toggling `requires_grad` here still lands the head in the optimizer. Full
        finetune is left untouched: `prepare_model` already honored freeze/trainable_parameters.
        """
        model = self.model
        if not (isinstance(model, PeftModel) or is_instance_of_ms_model(model)):
            return
        n = 0
        for name, p in model.named_parameters():
            if 'scoring_head' not in name:
                continue
            if not p.requires_grad:
                p.requires_grad_(True)
                n += 1
            if p.dtype == torch.float16:  # mirror prepare_model's fp16->fp32 fix for trainable params
                p.data = p.data.to(torch.float32)
        if n:
            logger.info(f'decision: re-activated {n} `scoring_head` param(s) to train jointly with the adapter.')

    def _save(self, output_dir=None, state_dict=None):
        # Standard path saves the LoRA adapter (or full weights) via `SwiftMixin._save`; the head is a
        # separate non-LoRA submodule that PEFT does NOT persist (plan decision F), so write it beside
        # the adapter here. Model-local: no shared save code is touched.
        super()._save(output_dir, state_dict)
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        head = get_scoring_head(self.model)
        if head is not None:
            head.save_pretrained(output_dir)
            logger.info(f'decision: saved scoring_head ({head.HEAD_WEIGHTS_NAME} + {head.HEAD_META_NAME}) '
                        f'to {output_dir}.')

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.compute_loss_func is not None:
            labels = inputs.pop('labels', None)
            outputs = model(**inputs)
            if labels is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch, trainer=self)
            else:
                loss = outputs.loss

            if num_items_in_batch is not None and self.model_accepts_loss_kwargs:
                accumulation_steps = getattr(self, 'current_gradient_accumulation_steps',
                                             self.args.gradient_accumulation_steps)
                loss = loss / accumulation_steps

            if labels is not None:
                self._compute_acc(outputs, labels)

            return (loss, outputs) if return_outputs else loss
        else:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

    def _compute_acc(self, outputs, labels, cu_seqlens=None) -> None:
        """Per-question accuracy: masked argmax over each question's own option set vs the gold
        option (argmax of the padded target distribution)."""
        from swift.model.decision_head import masked_argmax
        logits = outputs.logits
        option_mask = getattr(outputs, 'option_mask', None)
        if option_mask is None:
            option_mask = torch.ones_like(logits, dtype=torch.bool)
        preds = masked_argmax(logits, option_mask)  # [total_q]
        target = labels.argmax(dim=-1) if labels.dim() > 1 else labels
        acc = (preds == target).float()
        mode = 'train' if self.model.training else 'eval'
        self.custom_metrics[mode]['acc'].update(acc)

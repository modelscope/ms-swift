"""Loss configuration for the SFT path.

Explicit loss assembly so the GA-correctness contract is visible in the dev layer,
not inherited implicitly from twinkle's default.

Why reduction='sum':
  legacy swift computes ``loss = outputs.loss.sum() / num_items_in_batch`` (SUM with a
  single token denominator across the whole gradient-accumulation window). twinkle's SUM
  path matches this: calculate_loss accumulates loss.sum() + num_tokens across micro-batches
  and clip_grad_norm divides the grad by the total token count. This makes GA=k/bs=1
  gradient-equivalent to GA=1/bs=k. reduction='mean' would instead weight each micro-batch
  equally (denominator = gradient_accumulation_steps), which is only correct when every
  micro-batch has the same token count. So SFT must use SUM.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from swift.dev.model import TrainableModel


def configure_loss(model: TrainableModel,
                   *,
                   loss_type: str = 'cross_entropy',
                   reduction: str = 'sum',
                   **kwargs) -> None:
    """Set the SFT loss on ``model`` with an explicit reduction (default 'sum').

    Args:
        model: a twinkle-derived Model (has set_loss).
        loss_type: swift loss name resolved via the unified naming layer (SFT default
            'cross_entropy'). Non-CE losses (grpo/dpo/...) resolve too but belong to
            their own recipes; SFT keeps CE.
        reduction: 'sum' (default; GA-correct, aligns legacy) or 'mean'.
    """
    from swift.dev.naming import resolve_loss

    if loss_type != 'cross_entropy':
        raise NotImplementedError(f"SFT configure_loss only supports 'cross_entropy', got {loss_type!r}")
    loss_cls = resolve_loss(loss_type)
    model.set_loss(loss_cls(reduction=reduction, **kwargs))

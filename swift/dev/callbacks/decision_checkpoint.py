# Copyright (c) ModelScope Contributors. All rights reserved.
# Persist the decision ``scoring_head`` beside every training checkpoint.
from __future__ import annotations
from typing import Optional

from swift.dev.callbacks.base import CALLBACK, CallbackControl, CallbackState, TrainCallback
from swift.dev.model.decision_head import get_scoring_head
from swift.dev.utils import get_logger
from swift.dev.utils.plugin import PluginRegistry

logger = get_logger()

__all__ = ["DecisionCheckpointCallback"]


def _is_adapter_model(model) -> bool:
    """Whether the bound model's live module is a PEFT adapter wrap (LoRA training)."""
    try:
        from peft import PeftModel
    except ImportError:
        return False
    inner = getattr(model, "model", None)
    if isinstance(inner, PeftModel):
        return True
    return isinstance(getattr(inner, "peft_model", None), PeftModel)


@PluginRegistry.register(CALLBACK, "decision_checkpoint")
class DecisionCheckpointCallback(TrainCallback):
    """Write the attached ``scoring_head`` into every checkpoint dir (weights iff adapter run)."""

    def __init__(self, args=None, **kwargs):
        super().__init__(args=args, **kwargs)
        self._model = None
        self._save_weights: Optional[bool] = None

    def bind(self, model, save_weights: Optional[bool] = None) -> None:
        """Attach the model (head discovery); ``save_weights`` forces the adapter/full choice.

        ``None`` (the default) auto-detects from the live module (PEFT wrap -> weights, else
        meta-only); the recipe passes ``save_weights=tuner_config is not None`` so the choice
        follows the run's own configuration.
        """
        self._model = model
        self._save_weights = save_weights

    def on_save(self, state: CallbackState, control: CallbackControl, checkpoint_dir: Optional[str] = None) -> None:
        if self._model is None or not checkpoint_dir:
            return
        head = get_scoring_head(self._model)
        if head is None:
            return
        save_weights = self._save_weights
        if save_weights is None:
            save_weights = _is_adapter_model(self._model)
        try:
            head.save_pretrained(checkpoint_dir, save_weights=save_weights)
        except Exception as e:  # noqa: BLE001 -- a failed sidecar must not kill the main save
            logger.warning(f"decision_checkpoint: failed to save scoring_head to {checkpoint_dir} ({e}).")
            return
        if save_weights:
            logger.info(
                f"decision: saved scoring_head ({head.HEAD_WEIGHTS_NAME} + {head.HEAD_META_NAME}) to {checkpoint_dir}."
            )
        else:
            logger.info(
                f"decision: saved scoring_head meta ({head.HEAD_META_NAME}) to {checkpoint_dir} "
                "(weights in model state_dict)."
            )

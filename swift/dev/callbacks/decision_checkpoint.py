# Copyright (c) ModelScope Contributors. All rights reserved.
# Persist the decision ``scoring_head`` beside every training checkpoint.
from __future__ import annotations
from typing import Optional

from swift.dev.callbacks.base import CALLBACK, CallbackControl, CallbackState, TrainCallback
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
        meta-only); the recipe passes ``save_weights=has_adapter(tuner_config)`` so the choice
        follows the run's own configuration.
        """
        self._model = model
        self._save_weights = save_weights

    def on_save(self, state: CallbackState, control: CallbackControl, checkpoint_dir: Optional[str] = None) -> None:
        if self._model is None or not checkpoint_dir:
            return
        # Capability probe, not a mode branch: a model without the sidecar face (the remote
        # handle -- the server's own save already persists the head into its checkpoint store)
        # has nothing for this callback to write, and no local dir to stage into either.
        if getattr(self._model, "save_scoring_head_sidecar", None) is None:
            return
        save_weights = self._save_weights
        if save_weights is None:
            save_weights = _is_adapter_model(self._model)
        # write_dir yields the real dir locally; an oss:// / s3:// checkpoint_dir
        # stages, uploads and drops the staging copy.
        from swift.dev.utils.storage import write_dir

        try:
            with write_dir(checkpoint_dir) as local:
                # Worker-side: under Ray the bound model is a driver proxy with no _modules
                # tree (the head lives on the workers), so head discovery + save_pretrained
                # run on the workers via this @remote_function handle (master rank writes).
                saved = self._model.save_scoring_head_sidecar(local, save_weights=save_weights)
        except Exception as e:  # noqa: BLE001 -- a failed sidecar must not kill the main save
            logger.warning(f"decision_checkpoint: failed to save scoring_head to {checkpoint_dir} ({e}).")
            return
        if not saved:
            return
        from swift.dev.model.decision_head import ScoringHead
        if save_weights:
            logger.info(
                f"decision: saved scoring_head ({ScoringHead.HEAD_WEIGHTS_NAME} + {ScoringHead.HEAD_META_NAME}) to {checkpoint_dir}."
            )
        else:
            logger.info(
                f"decision: saved scoring_head meta ({ScoringHead.HEAD_META_NAME}) to {checkpoint_dir} "
                "(weights in model state_dict)."
            )

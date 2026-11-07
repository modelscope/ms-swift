# Copyright (c) ModelScope Contributors. All rights reserved.
# Teach twinkle's transformers backend the ``decision`` forward task
from twinkle.patch import Patch
from twinkle.utils import get_logger

logger = get_logger()

_TASK = "decision"


class TwinkleDecisionTaskPatch(Patch):
    """Register the ``decision`` task name across twinkle's transformers forward path."""

    def __call__(self, module=None, *args, **kwargs):
        import contextlib

        import twinkle.model.transformers.transformers as tft
        import twinkle.processor.base as processor_base

        # -- transformers backend ---------------------------------------------------------
        if getattr(tft._resolve_task_context, "_swift_decision", False):
            return module  # already patched (idempotent)

        origin_resolve = tft._resolve_task_context

        def _resolve_task_context(model, task, template=None):
            if task == _TASK:
                # The ScoringLoader already patched the HF model's own forward; no per-forward
                # context (no lm_head swap, no pooling patch) may interfere with it.
                return contextlib.nullcontext()
            return origin_resolve(model, task, template=template)

        _resolve_task_context._swift_decision = True
        _resolve_task_context._swift_origin = origin_resolve
        tft._resolve_task_context = _resolve_task_context
        # Float [total_q, max_opt] target distributions are not token labels: the logp block
        # (and any labels!=-100 masking) must never see them.
        tft._NON_LOGPS_TASKS = frozenset(tft._NON_LOGPS_TASKS) | {_TASK}

        # -- InputProcessor ----------------------------------------------------------------
        if not getattr(processor_base.InputProcessor.token_level_labels, "_swift_decision", False):
            origin_token_labels = processor_base.InputProcessor.token_level_labels

            @staticmethod
            def token_level_labels(task):
                if task == _TASK:
                    return False  # question-level labels, not token-level
                return origin_token_labels(task)

            token_level_labels._swift_decision = True
            token_level_labels._swift_origin = origin_token_labels
            processor_base.InputProcessor.token_level_labels = token_level_labels

        if not getattr(processor_base.InputProcessor.unpack_packed_sequences, "_swift_decision", False):
            origin_unpack = processor_base.InputProcessor.unpack_packed_sequences

            def unpack_packed_sequences(self, inputs, outputs=None, task="causal_lm", **ukwargs):
                if task == _TASK:
                    return inputs, outputs  # already per-question; no packed layout to restore
                return origin_unpack(self, inputs, outputs, task=task, **ukwargs)

            unpack_packed_sequences._swift_decision = True
            unpack_packed_sequences._swift_origin = origin_unpack
            processor_base.InputProcessor.unpack_packed_sequences = unpack_packed_sequences

        logger.info(
            "Patched twinkle: the 'decision' forward task is registered (nullcontext + "
            "non-logps + question-level labels)."
        )
        return module

    def unpatch(self, module=None, *args, **kwargs):
        import twinkle.model.transformers.transformers as tft
        import twinkle.processor.base as processor_base

        origin = getattr(tft._resolve_task_context, "_swift_origin", None)
        if origin is not None:
            tft._resolve_task_context = origin
        tft._NON_LOGPS_TASKS = frozenset(tft._NON_LOGPS_TASKS) - {_TASK}
        for owner, name in (
            (processor_base.InputProcessor, "token_level_labels"),
            (processor_base.InputProcessor, "unpack_packed_sequences"),
        ):
            method = getattr(owner, name, None)
            origin_method = getattr(method, "_swift_origin", None)
            if origin_method is not None:
                setattr(owner, name, origin_method)
        return module

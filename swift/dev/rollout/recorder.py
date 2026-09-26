# Copyright (c) ModelScope Contributors. All rights reserved.
"""Persist rollout tokens to an NPZ sidecar beside the sampling jsonl.

``run_sampling``'s jsonl rows are the index: when ``SamplingConfig.save_rollout_tokens`` is on, each kept
candidate's row carries the relative path of its NPZ, so there is no separate index file to keep consistent
across a resumed run. NPZ filenames are positionally deterministic (``{prompt_id}_c{candidate_index}.npz``),
so replaying a batch on resume overwrites the same files instead of appending duplicates -- the jsonl itself
is rewritten from the resumed checkpoint either way.

Each NPZ holds the flat training feature the rollout produced: ``input_ids`` / ``labels`` (next-token
shifted, the producer records ``SHIFTED_KEY``) / ``completion_mask`` when present, plus the policy-produced
``response_token_ids`` / ``rollout_logprobs`` (old_logps) / ``response_loss_mask``. A message-only backend
(the ``client`` teacher) carries no tokens, so ``record`` returns None for it and the row omits the path.
"""
from __future__ import annotations
import os
from typing import Any, Optional

import numpy as np

__all__ = ['RolloutRecorder']


class RolloutRecorder:
    """Writes one NPZ per kept candidate under ``token_dir``; a no-op when ``enabled`` is False.

    ``base_dir`` is the directory the jsonl lives in, so :meth:`record` returns a path relative to it --
    that relative path is what gets embedded in the row, keeping the jsonl portable alongside its sidecar.
    """

    def __init__(self, token_dir: str, base_dir: str = '', enabled: bool = False):
        self.token_dir = token_dir
        self.base_dir = base_dir
        self.enabled = enabled
        if enabled:
            os.makedirs(token_dir, exist_ok=True)

    def record(self, prompt_id: str, candidate_index: int, candidate: Any) -> Optional[str]:
        """Persist one candidate's rollout tokens; return the NPZ path relative to ``base_dir``.

        Returns None when disabled, or when the candidate carries no token feature (a message-only
        backend), so a path is embedded only for candidates that actually have tokens. ``candidate`` is
        read by attribute (``encoded`` / ``response_token_ids`` / ``rollout_logprobs`` /
        ``response_loss_mask``), so both a ``_Candidate`` and a ``RolloutSample`` work.
        """
        if not self.enabled:
            return None
        encoded = getattr(candidate, 'encoded', None) or {}
        input_ids = encoded.get('input_ids')
        labels = encoded.get('labels')
        if not input_ids or not labels:
            return None
        arrays = {
            'input_ids': np.asarray(input_ids, dtype=np.int64),
            'labels': np.asarray(labels, dtype=np.int64),
            'response_token_ids': np.asarray(getattr(candidate, 'response_token_ids', None) or [], dtype=np.int64),
            'rollout_logprobs': np.asarray(getattr(candidate, 'rollout_logprobs', None) or [], dtype=np.float32),
            'response_loss_mask': np.asarray(getattr(candidate, 'response_loss_mask', None) or [], dtype=np.int8),
        }
        completion_mask = encoded.get('completion_mask')
        if completion_mask is not None:
            arrays['completion_mask'] = np.asarray(completion_mask, dtype=np.int8)
        name = f'{prompt_id}_c{candidate_index}.npz'
        abs_path = os.path.join(self.token_dir, name)
        np.savez_compressed(abs_path, **arrays)
        return os.path.relpath(abs_path, self.base_dir) if self.base_dir else name

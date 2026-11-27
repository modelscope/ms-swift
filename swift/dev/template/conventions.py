# Copyright (c) ModelScope Contributors. All rights reserved.
"""Field ownership and shifted-label conventions for dev encoded trajectories."""
from __future__ import annotations

from twinkle.template.contract.encoding import model_payload_keys

NO_SHIFT_TASK_TYPES = frozenset({'embedding', 'reranker', 'generative_reranker', 'seq_cls', 'decision'})
LENGTH_KEYS = frozenset({'length', 'lengths'})
MEDIA_INPUT_KEYS = ('images', 'audios', 'videos')
RENDER_ONLY_INPUT_KEYS = ('objects', 'tools', 'chat_template_kwargs')
PROMPT_INPUT_KEYS = MEDIA_INPUT_KEYS + RENDER_ONLY_INPUT_KEYS


def is_bookkeeping_key(key: str) -> bool:
    return key.startswith('_') or key in LENGTH_KEYS

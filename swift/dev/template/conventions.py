# Copyright (c) ModelScope Contributors. All rights reserved.
"""Field ownership and shifted-label conventions for dev encoded trajectories."""
from __future__ import annotations

from types import MappingProxyType
from typing import Any, Dict, Mapping, NamedTuple

from twinkle.data_format import InputFeature, Trajectory
from twinkle.template.contract import TOKEN_FILL

IGNORE_INDEX = -100
NO_SHIFT_TASK_TYPES = frozenset({'embedding', 'reranker', 'generative_reranker', 'seq_cls', 'decision'})
LENGTH_KEYS = frozenset({'length', 'lengths'})
MEDIA_INPUT_KEYS = ('images', 'audios', 'videos')
RENDER_ONLY_INPUT_KEYS = ('objects', 'tools', 'chat_template_kwargs')
PROMPT_INPUT_KEYS = MEDIA_INPUT_KEYS + RENDER_ONLY_INPUT_KEYS

TOKEN = 'token'
DERIVED = 'derived'
BATCH = 'batch'
SAMPLE = 'sample'
PROMPT = 'prompt'
LEDGER = 'ledger'


class FieldSpec(NamedTuple):
    kind: str


def _declare(kind: str, *names: str, **spec: Any) -> Dict[str, FieldSpec]:
    return {name: FieldSpec(kind, **spec) for name in names}


FIELDS: Mapping[str, FieldSpec] = MappingProxyType({
    **_declare(TOKEN, 'input_ids', 'inputs_embeds', 'labels', 'completion_mask', 'loss_scale',
               'replay_mask', 'routed_experts', 'token_type_ids', 'mm_token_type_ids'),
    **_declare(DERIVED, 'attention_mask', 'attention_mask_2d', 'position_ids', 'text_position_ids'),
    **_declare(BATCH, 'packed_seq_params', 'cu_seq_lens_q', 'cu_seq_lens_k', 'cu_seqlens_q',
               'cu_seqlens_kv', 'max_length_q', 'max_length_k'),
    **_declare(SAMPLE, 'channel', 'margin', 'label', 'template_inputs'),
    **_declare(PROMPT, 'messages', 'user_data', 'objects', 'tools', 'prompt', 'engine_input_ids',
               'reasoning_started', 'mm_processor_kwargs', 'system', 'extra_kwargs'),
    **_declare(LEDGER, 'logprobs', 'turns', 'stop_reason', 'truncated', 'stuck_stop', 'tool_stop',
               'followups', 'error', 'rollout_infos', 'decoded'),
})

SPLICE_FILL: Mapping[str, int] = TOKEN_FILL


class AppendRole(NamedTuple):
    scored: bool
    sampled: bool


APPEND_ROLES: Mapping[str, AppendRole] = MappingProxyType({
    'completion': AppendRole(scored=True, sampled=True),
    'demonstration': AppendRole(scored=True, sampled=False),
    'context': AppendRole(scored=False, sampled=False),
})


def model_payload_keys(feature: Dict[str, Any]) -> frozenset[str]:
    """Select model-specific payload without forwarding raw conversation or sampling metadata."""
    metadata = Trajectory.__annotations__.keys() - InputFeature.__annotations__.keys()
    return frozenset(
        key for key, value in feature.items()
        if value is not None and key not in FIELDS and key not in metadata
        and key not in RENDER_ONLY_INPUT_KEYS and key not in MEDIA_INPUT_KEYS
        and not is_bookkeeping_key(key)
        and key not in feature.get('_trajectory_columns', ())
    )


def extend_spliced_fields(feature: Dict[str, Any]) -> Dict[str, Any]:
    """Right-pad token type companions to the actual sequence width."""
    length = len(feature.get('input_ids', []))
    for key, fill in SPLICE_FILL.items():
        value = feature.get(key)
        if value is None:
            continue
        values = [int(v) for v in (value.reshape(-1).tolist() if hasattr(value, 'reshape') else value)]
        if len(values) > length:
            raise ValueError(f'{key} has {len(values)} entries for {length} input_ids')
        feature[key] = values + [fill] * (length - len(values))
    return feature


def is_bookkeeping_key(key: str) -> bool:
    return key.startswith('_') or key in LENGTH_KEYS

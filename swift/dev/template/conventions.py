# Copyright (c) ModelScope Contributors. All rights reserved.
"""The label/sequence conventions every dev InputFeature obeys.

A. ``labels[i]`` is the target at position ``i`` (next token): whoever encodes shifts once, and
   ``loss_scale`` shifts with it.
B. Position 0 is never a target (``labels[0] == -100``), so the explicit shift equals twinkle's
   circular ``np.roll(-1)``.
C. ``_``-prefixed keys and the legacy length columns are bookkeeping; the InputProcessor drops them.
D. ``attention_mask`` / ``position_ids`` derive late, from the final sequence, in the InputProcessor.
"""

from __future__ import annotations
from types import MappingProxyType
from typing import Any, Dict, Mapping, NamedTuple

from swift.dev.template_contract import TOKEN_FILL

IGNORE_INDEX = -100

# Task types whose ``labels`` are per-sequence (scores / group markers), not token targets.
# embedding marks its group anchors with 1.0 and InfonceLoss finds them via nonzero -- a shift
# would split groups.
NO_SHIFT_TASK_TYPES = frozenset({"embedding", "reranker", "generative_reranker", "seq_cls"})

# Legacy length columns (``length`` per encode, ``lengths`` for packing); dataset-side only.
LENGTH_KEYS = frozenset({"length", "lengths"})

# Row columns the encoder reads besides ``messages``. Whatever hands a prompt to a sampler must
# carry them, or the model is prompted without its image / tools and nothing downstream notices.
MEDIA_INPUT_KEYS = ("images", "audios", "videos")
PROMPT_INPUT_KEYS = MEDIA_INPUT_KEYS + ("objects", "tools")

# ---- Field registry ------------------------------------------------------------------------------
# Every field a feature may carry, declared once with what it is. A key that is neither registered
# nor bookkeeping is model-specific payload (pixel_values, *_grid_thw, ...), owned by the family's
# collator.

# Kinds are write-ownership classes (who writes it / when it is valid), not one data axis:
# DERIVED fields are token-aligned too, they just derive late (convention D).
TOKEN = "token"  # aligned 1:1 with input_ids -- whichever side writes it (encoder, engine, sampler)
DERIVED = "derived"  # a function of the final ids, derived late (convention D)
BATCH = "batch"  # built by the InputProcessor while packing / splitting for context parallelism
SAMPLE = "sample"  # one value per sample: task targets and pair-level fields
PROMPT = "prompt"  # raw prompt inputs a feature keeps for re-encoding, rewards and the engine
LEDGER = "ledger"  # the rollout's account of a trajectory


class FieldSpec(NamedTuple):
    kind: str


def _declare(kind: str, *names: str, **spec: Any) -> Dict[str, FieldSpec]:
    return {name: FieldSpec(kind, **spec) for name in names}


FIELDS: Mapping[str, FieldSpec] = MappingProxyType(
    {
        **_declare(
            TOKEN,
            "input_ids",
            "inputs_embeds",
            "labels",
            "completion_mask",
            "loss_scale",
            "replay_mask",
            "routed_experts",
        ),
        # Per-token type ids (Qwen-VL's mm_token_type_ids marks media positions for mRoPE); spliced
        # tokens are text, i.e. 0.
        **_declare(TOKEN, "token_type_ids", "mm_token_type_ids"),
        **_declare(DERIVED, "attention_mask", "attention_mask_2d", "position_ids", "text_position_ids"),
        **_declare(
            BATCH,
            "packed_seq_params",
            "cu_seq_lens_q",
            "cu_seq_lens_k",
            "cu_seqlens_q",
            "cu_seqlens_kv",
            "max_length_q",
            "max_length_k",
        ),
        **_declare(SAMPLE, "channel", "margin", "label", "template_inputs"),
        **_declare(PROMPT, "messages", "objects", "tools"),
        **_declare(
            LEDGER,
            "logprobs",
            "turns",
            "stop_reason",
            "truncated",
            "stuck_stop",
            "tool_stop",
            "followups",
            "error",
            "rollout_infos",
            "decoded",
        ),
    }
)

# Fields that follow the sequence when ids are spliced on; fills live in the contract.
SPLICE_FILL: Mapping[str, int] = TOKEN_FILL


class AppendRole(NamedTuple):
    scored: bool  # its ids become labels (trainable)
    sampled: bool  # policy-produced: completion_mask=1, a logprob exists


# What an appended turn is to a trainer (``appended_as=``), same as twinkle.
APPEND_ROLES: Mapping[str, AppendRole] = MappingProxyType(
    {
        "completion": AppendRole(scored=True, sampled=True),
        "demonstration": AppendRole(scored=True, sampled=False),
        "context": AppendRole(scored=False, sampled=False),
    }
)


def model_payload_keys(feature: Dict[str, Any]) -> frozenset[str]:
    """Model-specific, non-text fields in an encoded feature (pixel values, grids, audio, ...).

    Raw prompt media under ``images`` / ``videos`` / ``audios`` (str / bytes / PIL refs) is not
    payload; tensors under the same names are.
    """

    def is_raw_prompt_media(key: str, value: Any) -> bool:
        if key not in MEDIA_INPUT_KEYS:
            return False
        values = value if isinstance(value, (list, tuple)) else [value]
        return not values or all(
            isinstance(item, (str, bytes, dict)) or type(item).__module__.startswith("PIL.") for item in values
        )

    return frozenset(
        key
        for key, value in feature.items()
        if value is not None
        and key not in FIELDS
        and not is_bookkeeping_key(key)
        and not is_raw_prompt_media(key, value)
    )


def extend_spliced_fields(feature: Dict[str, Any]) -> Dict[str, Any]:
    """Right-pad every ``SPLICE_FILL`` field to ``len(input_ids)``, in place: a short mRoPE index
    misassigns positions or fails outright, and over-long input raises."""
    length = len(feature.get("input_ids") or [])
    for key, fill in SPLICE_FILL.items():
        value = feature.get(key)
        if value is None:
            continue
        values = [int(v) for v in (value.reshape(-1).tolist() if hasattr(value, "reshape") else value)]
        if len(values) > length:
            raise ValueError(
                f"{key} has {len(values)} entries for {length} input_ids; it must be aligned with the "
                "sequence it types."
            )
        feature[key] = values + [fill] * (length - len(values))
    return feature


def is_bookkeeping_key(key: str) -> bool:
    return key.startswith("_") or key in LENGTH_KEYS

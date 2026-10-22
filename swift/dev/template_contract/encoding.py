# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
import re
from dataclasses import replace
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, cast

from .types import EncodedRow, InputOrderRow, MediaBundle, RolloutState

SHIFTED_FIELDS: Mapping[str, Any] = MappingProxyType(
    {
        "labels": -100,
        "loss_scale": 0.0,
        "completion_mask": 0,
    }
)
TOKEN_FILL: Mapping[str, int] = MappingProxyType(
    {
        "token_type_ids": 0,
        "mm_token_type_ids": 0,
    }
)
_MEDIA_TAG = re.compile(r"<(?:image|video|audio)>")
_MEDIA_PAYLOAD_KEYS = frozenset(
    {
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "input_features",
        "audio_values",
        "audio_features",
        "feature_attention_mask",
    }
)
_ANCHOR = [{"role": "user", "content": "x"}]


def feature_has_media(feature: Mapping[str, Any]) -> bool:
    return bool(MediaBundle.from_sources(feature)) or any(feature.get(key) is not None for key in _MEDIA_PAYLOAD_KEYS)


def _is_numpy(value):
    return type(value).__module__.startswith("numpy")


def _is_torch(value):
    return type(value).__module__.startswith("torch")


def _boundary_is(value, index, fill):
    if _is_numpy(value) or _is_torch(value):
        result = value[..., index] == fill
        return bool(result.all().item() if hasattr(result.all(), "item") else result.all())
    return value[index] == fill


def shift_values(values, fill):
    if len(values) == 0:
        return values.copy() if hasattr(values, "copy") else type(values)(values)
    if not _boundary_is(values, 0, fill):
        raise ValueError(f"shift expects the first position to be {fill!r}.")
    if _is_numpy(values):
        import numpy as np

        result = np.roll(values, -1, axis=-1)
        result[..., -1] = fill
        return result
    if _is_torch(values):
        result = values.roll(-1, dims=-1)
        result[..., -1] = fill
        return result
    shifted = list(values[1:]) + [fill]
    return tuple(shifted) if isinstance(values, tuple) else shifted


def unshift_values(values, fill):
    if len(values) == 0:
        return values.copy() if hasattr(values, "copy") else type(values)(values)
    if not _boundary_is(values, -1, fill):
        raise ValueError(f"unshift expects the last position to be {fill!r}.")
    if _is_numpy(values):
        import numpy as np

        result = np.roll(values, 1, axis=-1)
        result[..., 0] = fill
        return result
    if _is_torch(values):
        result = values.roll(1, dims=-1)
        result[..., 0] = fill
        return result
    unshifted = [fill] + list(values[:-1])
    return tuple(unshifted) if isinstance(values, tuple) else unshifted


def shift(row: InputOrderRow) -> EncodedRow:
    result: Dict[str, Any] = dict(row)
    for key, fill in SHIFTED_FIELDS.items():
        if result.get(key) is not None:
            result[key] = shift_values(result[key], fill)
    return cast(EncodedRow, result)


def unshift(row: EncodedRow) -> InputOrderRow:
    result: Dict[str, Any] = dict(row)
    for key, fill in SHIFTED_FIELDS.items():
        if result.get(key) is not None:
            result[key] = unshift_values(result[key], fill)
    return cast(InputOrderRow, result)


def _append_fill(value, count, fill):
    if _is_numpy(value):
        import numpy as np

        padding = np.full((*value.shape[:-1], count), fill, dtype=value.dtype)
        return np.concatenate((value, padding), axis=-1)
    if _is_torch(value):
        import torch

        padding = torch.full((*value.shape[:-1], count), fill, dtype=value.dtype, device=value.device)
        return torch.cat((value, padding), dim=-1)
    values = list(value) + [fill] * count
    return tuple(values) if isinstance(value, tuple) else values


def append_input_order(row: EncodedRow, tokens: Sequence[int], *, scored: bool, sampled: bool) -> InputOrderRow:
    if row.get("loss_scale") is not None:
        raise ValueError("a rollout prefix must not carry loss_scale.")
    result = cast(Dict[str, Any], unshift(row))
    input_ids = list(result["input_ids"])
    new_tokens = list(tokens)
    labels = result.get("labels")
    labels = [-100] * len(input_ids) if labels is None else list(labels)
    completion_mask = result.get("completion_mask")
    if completion_mask is None:
        completion_mask = [0 if label == -100 else 1 for label in labels]
    else:
        completion_mask = list(completion_mask)
    if len(labels) != len(input_ids) or len(completion_mask) != len(input_ids):
        raise ValueError("input_ids, labels and completion_mask must be aligned before splice.")
    result["input_ids"] = input_ids + new_tokens
    result["labels"] = labels + (new_tokens if scored else [-100] * len(new_tokens))
    result["completion_mask"] = completion_mask + [int(sampled)] * len(new_tokens)
    for key, fill in TOKEN_FILL.items():
        if result.get(key) is not None:
            result[key] = _append_fill(result[key], len(new_tokens), fill)
    return cast(InputOrderRow, result)


def append_and_finalize(
    row: EncodedRow,
    tokens: Sequence[int],
    *,
    scored: bool,
    sampled: bool,
    finalize: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]],
) -> Optional[EncodedRow]:
    input_order = append_input_order(row, tokens, scored=scored, sampled=sampled)
    finalized = finalize([input_order])
    if not finalized:
        return None
    if len(finalized) != 1:
        raise ValueError("appending one turn must produce exactly one feature.")
    return cast(EncodedRow, finalized[0])


def splice(row: EncodedRow, tokens: Sequence[int], *, scored: bool, sampled: bool) -> EncodedRow:
    return refresh(append_input_order(row, tokens, scored=scored, sampled=sampled))


def refresh(row: InputOrderRow) -> EncodedRow:
    result: Dict[str, Any] = dict(row)
    input_ids = result.get("input_ids")
    if input_ids is None:
        return shift(cast(InputOrderRow, result))
    result["attention_mask"] = [1] * len(input_ids)
    result["length"] = len(input_ids)
    return shift(cast(InputOrderRow, result))


def splice_state(
    state: RolloutState, feature_tokens: Sequence[int], engine_tokens: Sequence[int], *, scored: bool, sampled: bool
) -> RolloutState:
    feature = splice(state.feature, feature_tokens, scored=scored, sampled=sampled)
    return replace(
        state,
        feature=feature,
        engine_input_ids=state.engine_input_ids + tuple(engine_tokens),
        render_cache=None,
    )


def _has_media(message: Dict[str, Any]) -> bool:
    content = message.get("content")
    if isinstance(content, str):
        return bool(_MEDIA_TAG.search(content))
    return isinstance(content, list) and any(
        isinstance(item, dict) and item.get("type", "text") != "text" for item in content
    )


def _without_media(message: Dict[str, Any]) -> Dict[str, Any]:
    content = message.get("content")
    if isinstance(content, str):
        return {**message, "content": _MEDIA_TAG.sub("", content)}
    if isinstance(content, list) and content and isinstance(content[0], dict) and "type" in content[0]:
        return {**message, "content": "".join(item.get("text") or "" for item in content if item.get("type") == "text")}
    return message


def render_delta(
    messages_before: List[Dict[str, Any]],
    appended: List[Dict[str, Any]],
    *,
    gen_prompt_before: bool,
    gen_prompt_after: bool,
    render: Callable[[List[Dict[str, Any]], bool], Sequence[int]],
) -> List[int]:
    if any(_has_media(message) for message in appended):
        raise NotImplementedError("render_delta does not render media in appended turns.")
    history = [_without_media(message) for message in messages_before]
    before = list(render(history, gen_prompt_before))
    after = list(render(history + list(appended), gen_prompt_after))
    if after[: len(before)] == before:
        return after[len(before) :]
    anchors = [_ANCHOR if gen_prompt_before else _ANCHOR + [{"role": "assistant", "content": "x"}]]
    if not gen_prompt_before:
        anchors.append(_ANCHOR)
    for anchor in anchors:
        before = list(render(anchor, gen_prompt_before))
        after = list(render(anchor + list(appended), gen_prompt_after))
        if after[: len(before)] == before:
            return after[len(before) :]
    raise RuntimeError("non-monotonic render: appended turns are not a prefix-extension of the history or anchor.")

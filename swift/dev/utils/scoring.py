# Copyright (c) ModelScope Contributors. All rights reserved.
# Backend-agnostic scoring pieces: the response-token frame, forward-output alignment, reward folds.
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence

import torch

_MULTIMODAL_KEYS = ('images', 'videos', 'audios', 'tools', 'objects', 'chat_template_kwargs')

def should_copy_inputs() -> bool:
    """Whether a driver-side defensive copy of forward inputs is load-bearing in the current mode.

    Under twinkle's ``'ray'`` mode a ``forward_only`` / ``forward_backward`` serializes its inputs to a
    worker process, so the compute already sees an isolated copy and a driver-side deepcopy is redundant.
    Under ``'local'`` the compute shares this process and may mutate the feature dicts in place (padding,
    writing ``routed_experts``), so the copy protects the sample's canonical ``input_feature``. Read
    through twinkle's public ``get_mode()`` rather than reaching into a private global.
    """
    from twinkle import get_mode
    return get_mode() != 'ray'

def copy_inputs(features: Sequence[Any]) -> List[Any]:
    """Defensive-copy a forward's inputs iff the current twinkle mode needs it (see :func:`should_copy_inputs`)."""
    if should_copy_inputs():
        return [copy.deepcopy(feature) for feature in features]
    return list(features)

def response_positions(feature: Dict[str, Any]) -> List[int]:
    """Indices of the trainable (response) tokens in ``feature``, i.e. where ``labels != -100``.

    This is the exact frame the loss mask uses, so scores extracted at these positions (a teacher's
    log-probs, a PRM's per-token rewards) align one-to-one with the student's per-token loss.
    """
    labels = feature.get('labels')
    if labels is None:
        return []
    return [idx for idx, label in enumerate(labels) if int(label) != -100]

def row_token_scores(row: Any, positions: List[int], feature: Dict[str, Any]) -> List[float]:
    """Extract one feature's response-token scores from its (possibly padded) full-sequence row.

    A response-only frame already holds exactly ``len(positions)`` values, so it is returned as-is; a
    full-sequence frame is indexed at the feature's own response positions. A frame shorter than the
    feature cannot be aligned and raises rather than silently mis-scoring.
    """
    if row.numel() == len(positions):
        return row.tolist()  # response-only form: nothing to strip
    labels = feature.get('labels')
    feature_length = len(labels) if labels is not None else 0
    if row.numel() < feature_length:
        raise RuntimeError(f'forward returned {row.numel()} values for a {feature_length}-token feature; '
                           'response-token alignment is impossible.')
    return row[positions].tolist()

def response_token_scores(frame: Any, features: List[dict], *, output_key: str = 'logps',
                          allow_missing: bool = False) -> List[Any]:
    """Align one whole-batch ``forward_only`` frame -> one response-token score list per feature.

    ``frame`` is a single forward's output -- either materialised (local mode, or a caller that
    collected already) or the lazy handle the ray driver receives; the lazy form is collected HERE,
    so both deployment forms feed the same alignment. The dict ``collect_tensor_dict`` pads and
    stacks across the DP ranks, or a bare tensor; ``output_key`` selects the frame (``'logps'`` for
    a causal scorer, ``'logits'`` for a value / seq_cls head). The batch splits are the dispatch's
    business (``slice_dp`` per rank, ``micro_batch_size`` inside the kernel), so this side sees
    exactly one frame for the whole batch. Full-sequence, right-padded rows are indexed at each
    feature's own response positions (see :func:`row_token_scores`), so right-padding never shifts
    them.

    ``allow_missing`` returns ``[None] * len(features)`` when the frame is entirely absent -- PPO's
    reference reads a missing ``logps`` as "no KL penalty" -- while the default raises, so a silent
    no-score never passes for a real one.
    """
    if callable(frame) and getattr(frame, '_is_lazy_collect', False):
        frame = frame()
    values = frame.get(output_key) if isinstance(frame, dict) else frame
    if values is None:
        if allow_missing:
            return [None] * len(features)
        raise RuntimeError(f'forward_only returned no {output_key!r}.')
    values = torch.as_tensor(values).detach().float().cpu()
    if values.dim() == 1:
        values = values.unsqueeze(0)
    if values.shape[0] != len(features):
        raise RuntimeError(f'forward returned {values.shape[0]} rows for {len(features)} features; '
                           'per-row alignment is impossible.')
    return [row_token_scores(values[i].reshape(-1), response_positions(feature), feature)
            for i, feature in enumerate(features)]

def encode_privileged_view(
    template: Any,
    messages: List[dict],
    response_ids: List[int],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Encode a privileged teacher view (``messages`` carrying the shared ``response_ids``) and assert alignment.

    Shared by GRPO's RLSD/SDAR teacher view and OPSD's, which differ only in HOW ``messages`` is built
    (multi-turn reconstruction vs. a single privileged user turn) but not in the encode-and-check tail.
    Encoded in ``train`` mode with ``add_eos=False`` so the assistant turn is exactly ``response_ids`` (no
    extra end-of-sequence token), then checked: the encoded feature must expose the SAME number of trainable
    response positions as ``response_ids``. If a template or tokenizer difference changes that count, teacher
    and student would no longer score identical tokens and the alignment would be silently wrong -- so it is
    a hard error, not a truncation.
    """
    if template is None:
        raise ValueError('teacher_prompt requires a template to build the privileged teacher view.')
    extra = extra or {}
    row: Dict[str, Any] = {'messages': messages, 'add_eos': False}
    for key in _MULTIMODAL_KEYS:
        if extra.get(key) is not None:
            row[key] = extra[key]
    teacher_template = copy.copy(template)
    teacher_template.set_mode('train')
    feature = teacher_template.encode(row)
    if len(response_positions(feature)) != len(response_ids):
        raise RuntimeError('teacher_prompt encoding changed the response-token count; teacher and student must '
                           'share a tokenizer and score exactly the same response tokens.')
    return feature

def encode_privileged_feature(template: Any, prompt_messages: List[dict], teacher_prompt: Any,
                              response_ids: List[int], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Encode one single-turn privileged teacher view (RLSD / OPSD share the exact contract).

    Replaces the FIRST user turn's content with the privileged ``teacher_prompt``, appends the
    assistant turn carrying the student's response token ids verbatim, then delegates the
    encode-and-align-check tail to :func:`encode_privileged_view` -- teacher and student must score
    exactly the same response tokens.
    """
    messages = [dict(message) for message in prompt_messages]
    if messages and messages[-1].get('role') == 'assistant':
        messages = messages[:-1]
    for message in messages:
        if message.get('role') == 'user':
            message['content'] = teacher_prompt
            break
    else:
        raise ValueError('teacher_prompt requires a user turn in the prompt to carry the privileged context.')
    messages.append({'role': 'assistant', 'content': list(response_ids)})
    return encode_privileged_view(template, messages, response_ids, extra)

def build_reward_weights(reward_weights: Optional[Sequence[float]], n_funcs: int) -> torch.Tensor:
    """Build/validate the per-function weight vector (``None`` -> all-ones).

    Public because the advantage API needs the same validation, so the "weights must match the number
    of reward functions" rule has one implementation.

    Raises:
        ValueError: ``len(reward_weights) != n_funcs``.
    """
    if reward_weights is None:
        return torch.ones(n_funcs, dtype=torch.float32)
    if len(reward_weights) != n_funcs:
        raise ValueError(f'reward_weights length {len(reward_weights)} != number of reward funcs {n_funcs}.')
    return torch.tensor(list(reward_weights), dtype=torch.float32)

def weighted_rewards(rewards_per_func: Any, reward_weights: Optional[List[float]]):
    """Combine the ``[N, n_scorers]`` matrix into one ``[N]`` weighted-sum reward per sample.

    The weights are ``reward_weights`` (``--orm_weights``), defaulting to all-ones so an unweighted channel is
    a plain sum; ``nansum`` keeps a judge's parse-failure ``nan`` from poisoning the total.
    """
    weight_tensor = build_reward_weights(reward_weights, rewards_per_func.shape[1]).to(
        device=rewards_per_func.device, dtype=rewards_per_func.dtype)
    return (rewards_per_func * weight_tensor.unsqueeze(0)).nansum(dim=1)

def accumulate_reward_metric(metric: Any,
                             samples: List[Any],
                             rewards_per_func: Any,
                             *,
                             scorer_names: List[str],
                             reward_weights: Optional[List[float]]) -> None:
    """Feed one batch's rewards + completion lengths into a CompletionRewardMetric.

    Reset first, so the metric reflects exactly the batch the upcoming optimizer step(s) train on. Per-scorer
    columns keep their resolved names; a single-callable fallback resolves no names, so only the weighted
    ``total`` is reported there.
    """
    metric.reset()
    names = scorer_names
    rewards: Dict[str, List[float]] = {}
    for col in range(rewards_per_func.shape[1]):
        if col < len(names):
            rewards[names[col]] = [float(v) for v in rewards_per_func[:, col].tolist()]
    rewards['total'] = [float(v) for v in weighted_rewards(rewards_per_func, reward_weights).tolist()]
    completion_lengths = [len(getattr(sample, 'response_token_ids', None) or []) for sample in samples]
    metric.accumulate(rewards=rewards, completion_lengths=completion_lengths)

# Copyright (c) ModelScope Contributors. All rights reserved.
# Backend-agnostic scoring pieces: the response-token frame, chunked no-grad forwards, reward folds.
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

_MULTIMODAL_KEYS = ('images', 'videos', 'audios', 'tools', 'objects', 'chat_template_kwargs')

def split_mini_batches(items: List[Any], mini_batch_size: int) -> List[List[Any]]:
    """Split a rollout into full ``mini_batch_size`` chunks, dropping an undersized tail.

    Every ``forward_backward`` must carry exactly ``mini_batch_size`` rows so slice_dp hands each DP rank
    the same ``per_device_train_batch_size`` count; a partial tail would give ranks uneven batches (and,
    below dp_size rows, no data at all). Dropping it keeps each optimizer step uniform -- the same
    exact-global-batch invariant Megatron enforces -- at the cost of a few rollout samples per round.
    """
    return [items[i:i + mini_batch_size] for i in range(0, len(items) - mini_batch_size + 1, mini_batch_size)]

def score_chunks(features: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split ``features`` into ``chunk_size`` blocks, folding a ragged tail into the last one.

    Every block must hold >= dp_size rows or slice_dp starves a rank. ``chunk_size`` (the recipes'
    ``train_batch_size``) is a multiple of dp_size, so full blocks qualify, and folding the remainder into
    the last full block keeps it >= dp_size too. A rollout smaller than one block yields a single short
    block that ``forward_only`` rejects loudly -- the same "rollout too small" condition training guards.
    """
    chunks = split_mini_batches(features, chunk_size)
    tail = features[len(chunks) * chunk_size:]
    if tail:
        if chunks:
            chunks[-1] = chunks[-1] + tail
        else:
            chunks = [tail]
    return chunks

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

def batched_response_scores(scorer: Any,
                            features: List[dict],
                            *,
                            train_batch_size: int,
                            output_key: str = 'logps',
                            forward_kwargs: Optional[Dict[str, Any]] = None,
                            allow_missing: bool = False) -> List[Any]:
    """Score ``features`` with chunked ``forward_only`` -> one response-token score list per feature.

    One forward per ``train_batch_size`` chunk (not one per feature) is what makes DP>1 work:
    ``forward_only`` is a slice_dp method, so a length-1 ``inputs`` leaves all but one rank with no data
    ("Batch too small"). Chunks also bound the no-grad scoring memory (the materialised frame) the same
    way design B bounds a training mini-batch. ``forward_only`` returns full-sequence, right-padded frames
    of shape ``[n, seq_len]`` in submission order (``collect_tensor_dict`` pads + stacks the per-rank
    shards), so each row is indexed by that feature's own response positions -- right-padding never
    shifts them.

    ``output_key`` selects the frame (``'logps'`` for a causal scorer, ``'logits'`` for a value / seq_cls
    head); ``forward_kwargs`` are passed through to ``forward_only`` (e.g. ``task='value'``,
    ``disable_lora=True``). ``allow_missing`` returns ``[None] * len(features)`` when the frame is
    entirely absent -- PPO's reference reads a missing ``logps`` as "no KL penalty" -- while the default
    raises, so a silent no-score never passes for a real one.
    """
    forward_kwargs = forward_kwargs or {}
    positions = [response_positions(feature) for feature in features]
    results: List[Any] = []
    offset = 0
    for chunk in score_chunks(features, train_batch_size):
        out = scorer.forward_only(inputs=copy_inputs(chunk), **forward_kwargs)
        frame = out.get(output_key) if isinstance(out, dict) else None
        if frame is None:
            if allow_missing:
                return [None] * len(features)
            raise RuntimeError(f'forward_only returned no {output_key!r}.')
        values = torch.as_tensor(frame).detach().float().cpu()
        if values.dim() == 1:
            values = values.unsqueeze(0)
        for i, row in enumerate(values):
            index = offset + i
            results.append(row_token_scores(row.reshape(-1), positions[index], features[index]))
        offset += len(chunk)
    return results

def overlap_by_placement(tasks: List[Tuple[Any, Callable[[], Any]]]) -> List[Any]:
    """Run ``(placement_key, thunk)`` tasks: concurrent across distinct keys, serial within one key.

    A placement key is the DeviceGroup a scorer forwards on. Two scorers on the SAME group must not
    forward concurrently (they contend for one device / OOM), so their thunks chain serially; scorers on
    DISJOINT groups sit on independent Ray actors and overlap on threads -- a blocking ``forward_only``
    releases the GIL while it waits on its actor, so threads (not coroutines) are the right lever, the
    pattern ``run_infer`` already uses. A single distinct key (or a single task)
    has nothing to overlap and runs inline, so no thread pool is spun up for the common serial case.
    Results come back in task order; the first worker exception propagates.
    """
    if len(tasks) <= 1:
        return [thunk() for _, thunk in tasks]
    order: List[Any] = []
    chains: Dict[Any, List[int]] = {}
    for index, (key, _) in enumerate(tasks):
        if key not in chains:
            chains[key] = []
            order.append(key)
        chains[key].append(index)
    if len(order) == 1:
        return [thunk() for _, thunk in tasks]  # one group: nothing to overlap
    results: List[Any] = [None] * len(tasks)

    def run_chain(key: Any) -> None:
        for index in chains[key]:
            results[index] = tasks[index][1]()

    with ThreadPoolExecutor(max_workers=len(order)) as pool:
        futures = [pool.submit(run_chain, key) for key in order]
        for future in futures:
            future.result()
    return results

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

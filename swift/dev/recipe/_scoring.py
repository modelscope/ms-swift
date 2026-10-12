"""Shared scoring utilities for the online-RL recipes (GRPO / PPO / distillation).

A recipe-private module (peer of ``_teacher`` / ``_distill`` / ``_streaming_loop``) holding the pieces
two or more recipes need to score a rollout with ``forward_only``: mini-batch splitting, chunked
no-grad scoring, disjoint-group overlap, and the driver-side defensive-copy policy. Keeping them here
means PPO scores a rollout exactly the way GRPO does -- one batched forward per chunk, not one per
sample -- instead of re-deriving (and re-fixing) the same DP>1 batching bug in each recipe.
"""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


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
    import torch

    from swift.dev.recipe._teacher import response_positions

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
    pattern ``_streaming_loop`` and ``run_infer`` already use. A single distinct key (or a single task)
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

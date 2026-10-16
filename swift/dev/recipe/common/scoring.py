"""Shared scoring utilities for the online-RL recipes (GRPO / PPO / distillation).

A recipe-shared module (peer of ``distill.assembler`` / ``rl.auxiliary``) holding the pieces
two or more recipes need to score a rollout with ``forward_only``: mini-batch splitting, chunked
no-grad scoring, disjoint-group overlap, the response-token frame, and the driver-side defensive-copy
policy. Keeping them here means PPO scores a rollout exactly the way GRPO does -- one batched forward
per chunk, not one per sample -- instead of re-deriving (and re-fixing) the same DP>1 batching bug in
each recipe. It also holds the outcome-reward chokepoint every reward-scoring recipe shares
(``reward_rows`` / ``score_samples`` / ``weighted_rewards`` / ``accumulate_reward_metric``), so GRPO,
RFT and PPO rebuild a conversation, score an ``[N, n_scorers]`` matrix and feed the reward metric one
way. The response-frame helpers (``response_positions`` / ``encode_privileged_view``) live here rather
than with the auxiliary-model wrappers in ``rl.auxiliary`` because they are SCORING logic (how a
feature's trainable tokens are located and how a privileged teacher view is encoded), not a model shape.
"""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

#: Dataset columns forwarded to ``template.encode`` when building a privileged teacher view, so a multimodal
#: sample's teacher forward sees the same non-text inputs as the student.
_MULTIMODAL_KEYS = ('images', 'videos', 'audios', 'tools', 'objects', 'chat_template_kwargs')


@dataclass
class TrainingBatch:
    """One rollout's per-row training payload, held column-wise with a construction-time length check.

    Each optional field is a list parallel to ``samples`` (one entry per rollout row), so an off-by-one
    between any two columns is impossible to build silently: :meth:`__post_init__` rejects unequal lengths
    up front. ``ref_logps`` / ``teacher_logps`` are all-or-none across a rollout (both scoring helpers return
    ``None`` or a full list), so they are ``Optional`` columns rather than per-row optionals. ``values`` /
    ``returns`` are PPO's critic columns (a value baseline and its GAE return target per row); the other
    algorithms leave them ``None``. RFT -- which trains the kept completions with plain cross-entropy --
    builds a samples-only batch and leaves every RL column ``None``.

    Slicing returns a ``TrainingBatch`` (every present column sliced consistently), so
    :func:`split_mini_batches` -- which uses only ``len`` and ``items[i:j]`` -- splits a batch into
    mini-batches unchanged.
    """
    samples: List[Any]
    advantages: Optional[List[Any]] = None
    old_logps: Optional[List[List[float]]] = None
    rollout_logps: Optional[List[List[float]]] = None
    ref_logps: Optional[List[List[float]]] = None
    teacher_logps: Optional[List[List[float]]] = None
    #: PPO's critic columns: the value baseline and its GAE return target, one per row. Never published as
    #: the tracker's main loss (the critic is a separate model), only fed to the critic's forward.
    values: Optional[List[Any]] = None
    returns: Optional[List[Any]] = None
    #: Per-row sampler support sets (twinkle ``SamplingMask``), present only under ``enable_sampling_replay``;
    #: assembled into the top-level ``sampling_masks`` forward_backward kwarg (NOT into ``encoded``).
    sampling_masks: Optional[List[Any]] = None

    #: Per-row columns that must align with ``samples`` when present (unannotated, so not a dataclass field).
    _COLUMNS = ('advantages', 'old_logps', 'rollout_logps', 'ref_logps', 'teacher_logps', 'values', 'returns',
                'sampling_masks')

    def __post_init__(self) -> None:
        rows = len(self.samples)
        for name in self._COLUMNS:
            column = getattr(self, name)
            if column is not None and len(column) != rows:
                raise ValueError(f'TrainingBatch column {name!r} has {len(column)} rows but samples has {rows}; '
                                 'per-row rollout fields must align exactly.')

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, key: slice) -> 'TrainingBatch':
        if not isinstance(key, slice):
            raise TypeError(f'TrainingBatch supports slice indexing only, got {type(key).__name__}.')

        def sliced(column):
            return None if column is None else column[key]

        return TrainingBatch(
            samples=self.samples[key],
            advantages=sliced(self.advantages),
            old_logps=sliced(self.old_logps),
            rollout_logps=sliced(self.rollout_logps),
            ref_logps=sliced(self.ref_logps),
            teacher_logps=sliced(self.teacher_logps),
            values=sliced(self.values),
            returns=sliced(self.returns),
            sampling_masks=sliced(self.sampling_masks))


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
    import torch

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


# --- the outcome-reward chokepoint (shared by GRPO / RFT / PPO; the distillation family scores none) -------


def reward_rows(samples: List[Any]) -> List[Dict[str, Any]]:
    """Rebuild each sample's complete conversation, retaining its dataset-side reward columns.

    A scalar reward model or a judge scores the WHOLE conversation, not a bare completion string, so each row
    is ``{'messages': prompt + this completion, **dataset extras}``. When the rollout already carried the
    conversation (``sample.messages``, e.g. a multi-turn trajectory) it is used as-is; otherwise the prompt
    stamped on the sample by the driver's collect step (``sample.prompt``) is used and this completion
    appended as the assistant turn -- the same reconstruction every reward-scoring recipe relies on, so a PPO
    sample (which scores on its own, no group contract) rebuilds its prompt exactly like a GRPO one.
    """
    rows: List[Dict[str, Any]] = []
    for sample in samples:
        messages = getattr(sample, 'messages', None)
        if messages is None:
            prompt = getattr(sample, 'prompt', None)
            if prompt is None:
                raise ValueError('reward model scoring requires the sample prompt to be stamped by the '
                                 'driver collect step, got None.')
            messages = copy.deepcopy(prompt)
            messages.append({'role': 'assistant', 'content': sample.decoded})
        row = copy.deepcopy(getattr(sample, 'extra', None) or {})
        row['messages'] = copy.deepcopy(messages)
        rows.append(row)
    return rows


def reward_columns(samples: List[Any]) -> Dict[str, List[Any]]:
    """Batch the per-sample reward columns: the dataset extras plus the full conversation.

    Every scorer receives all of these and takes the ones it needs -- a rule reads a dataset column like
    ``solution``, a scalar reward model or judge reads ``messages`` -- so one column dict serves the whole
    ordered list (see :func:`~swift.dev.rewards.reward.score_completions`).
    """
    rows = reward_rows(samples)
    keys = {key for row in rows for key in row}
    return {key: [row.get(key) for row in rows] for key in keys}


def score_samples(samples: List[Any],
                  *,
                  reward_scorers: Optional[List[Any]] = None,
                  reward_fn: Optional[Callable[[Any], float]] = None):
    """Score a rollout into an ``[N, n_scorers]`` reward matrix (one column per resolved scorer).

    The single reward chokepoint every reward-scoring recipe shares: the ordered ``reward_scorers`` (rules +
    reward models + judges) are scored together through :func:`~swift.dev.rewards.reward.score_completions`,
    which groups them by ``placement_key`` and overlaps the disjoint groups. With no scorer list resolved it
    falls back to the single ``reward_fn`` callable (a recipe's smoke default), and with neither -- a PPO run
    configured with no ``--orm`` and no ``--reward_model``, whose reward is then the KL penalty alone -- it
    returns a zero column so the downstream weighted sum is a plain zero per sample.
    """
    import torch

    from swift.dev.rewards.reward import score_completions

    if reward_scorers:
        completions = [sample.decoded for sample in samples]
        return score_completions(completions, reward_scorers, reward_columns(samples))
    if reward_fn is not None:
        return torch.tensor([[reward_fn(sample)] for sample in samples], dtype=torch.float32)
    return torch.zeros((len(samples), 1), dtype=torch.float32)


def weighted_rewards(rewards_per_func: Any, reward_weights: Optional[List[float]]):
    """Combine the ``[N, n_scorers]`` matrix into one ``[N]`` weighted-sum reward per sample.

    The weights are ``reward_weights`` (``--orm_weights``), defaulting to all-ones so an unweighted channel is
    a plain sum; ``nansum`` keeps a judge's parse-failure ``nan`` from poisoning the total.
    """
    from swift.dev.rewards.reward import build_reward_weights

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

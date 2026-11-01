# Copyright (c) ModelScope Contributors. All rights reserved.
"""The numeric compute plane for remote-execution stages.

Heavy forward payloads (logps / logits) must never materialise in the session
process: they flow between stages as infra references (LazyCollect), and
every numeric consumer -- reduction, shaping, advantage estimation,
combination -- runs HERE, on this worker, where the references are resolved.
The driver only carries small control scalars back.

Placement notes:

- The session is ALWAYS ray: a reference passed as an argument is resolved by
  the dispatch layer on this worker; the return value rides back as ONE
  reference (``lazy_collect=True``), which downstream consumers (the next
  numeric call, ``train_step``'s small scalar read-back, the preview
  materialisation) dereference where THEY run.
- ``project`` is the field-projection primitive: a forward return is a dict
  of channels; selecting ONE channel happens worker-side so the other
  channels never cross the boundary. It replaces the driver-side
  ``result.get(field)`` read.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from twinkle import remote_class, remote_function

from .placement import NUMERIC_GROUP


def deref(value: Any) -> Any:
    """Resolve a LazyCollect handle HERE (never in the driver); pass others through.

    The same shape ``ScoreWorker._resolve`` uses: a handle is detected by the
    public ``_is_lazy_collect`` marker and collected by calling it; every
    other value -- a plain dict, list, tensor -- is returned unchanged.
    """
    if callable(value) and getattr(value, '_is_lazy_collect', False):
        return value()
    return value


def _rows(values: Any) -> List[Any]:
    if values is None:
        return []
    if isinstance(values, list):
        return values
    if hasattr(values, 'tolist'):
        return values.tolist()
    return [values]


def _axis_op(seq: Any, op: str) -> Any:
    """One reduction over a flat numeric sequence."""
    flat = [float(v) for v in (seq or []) if v is not None
            and not (isinstance(v, float) and math.isnan(v))]
    if not flat:
        return None
    if op == 'sum':
        return sum(flat)
    if op == 'mean':
        return sum(flat) / len(flat)
    if op == 'variance':
        mean = sum(flat) / len(flat)
        return sum((v - mean) ** 2 for v in flat) / len(flat)
    if op == 'logsumexp':
        top = max(flat)
        return top + math.log(sum(math.exp(v - top) for v in flat))
    raise ValueError(f'unknown reduce op {op!r}')


def _seq_mean(row: Any) -> float:
    """One row's mean over its finite numeric entries (a sequence-level k1 KL term)."""
    flat = [float(v) for v in _rows(row) if v is not None
            and not (isinstance(v, float) and math.isnan(v))]
    return sum(flat) / len(flat) if flat else 0.0


@remote_class(execute='first')
class NumericWorker:
    """The resident numeric engine: field projection, reductions, shaping,
    advantage estimation and lane combination, all over resolved values.

    Every handle-taking method declares ``**kwargs``: when a LazyCollect
    reference arrives as a TOP-LEVEL argument, the dispatch layer forwards the
    call with its internal re-dispatch parameter (``_dispatch_context``) and
    Ray validates kwargs against this signature on the driver -- the plain
    parameter list would reject it. The dispatch wrapper strips that parameter
    before the body runs, so the catch-all stays empty here. The same contract
    ``TrainableModel.forward_only(*, inputs, **kwargs)`` already follows.
    """

    @remote_function(dispatch='all', collect='first', lazy_collect=True)
    def project(self, handle: Any, field: str, **kwargs) -> Any:
        """One channel of a dict-shaped handle, resolved HERE.

        A missing channel is a contract violation and fails loudly; a present
        ``None`` value (an optional channel the computation did not produce)
        passes through as ``None``."""
        value = deref(handle)
        if not hasattr(value, '__contains__') or field not in value:
            raise ValueError(
                f'the forward return carries no {field!r} channel '
                '(this model/task combination does not produce it)')
        return value[field]

    @remote_function(dispatch='all', collect='first', lazy_collect=True)
    def reduce(self, signal: Any, op: str, axis: str, **kwargs) -> Dict[str, Any]:
        """A named reduction over an explicit axis: pure math over resolved rows."""
        value = deref(signal)
        values = value.get('values') if isinstance(value, dict) else value
        rows = _rows(values)
        use = value.get('use') if isinstance(value, dict) else None
        if axis == 'token':
            reduced = [_axis_op(row, op) for row in rows]
        else:
            reduced = [_axis_op([r for r in rows if r is not None], op) if rows else []]
        return {'out': {'values': reduced, 'use': use}}

    @remote_function(dispatch='all', collect='first', lazy_collect=True)
    def reward_shape(self, reward: Any, baseline: Any, logps: Any, reference: Any,
                     beta: float, kl: str, **kwargs) -> Dict[str, Any]:
        """r' = r - baseline - beta*KL over resolved reward / baseline / KL lanes.

        The KL term is the sequence-level k1 estimate (mean logp - mean
        ref_logp per row) computed from the two explicit forward lanes; an
        absent lane means the term is absent -- never a hidden default."""
        reward_value = deref(reward)
        values = reward_value.get('values') if isinstance(reward_value, dict) else reward_value
        rows = _rows(values)
        base = deref(baseline)
        base_value = base.get('scalar') if isinstance(base, dict) else base
        kl_rows: List[Any] = []
        if kl != 'none':
            if kl != 'k1':
                raise ValueError(f'kl estimator {kl!r} is not bound on this worker yet')
            if logps is None or reference is None:
                raise ValueError('a k1 KL term needs both logps and reference lanes')
            lp_rows = _rows(deref(logps))
            rf_rows = _rows(deref(reference))
            kl_rows = [_seq_mean(lp) - _seq_mean(rf)
                       for lp, rf in zip(lp_rows, rf_rows)]
        shaped = []
        for i, row in enumerate(rows):
            value = row[0] if isinstance(row, list) and row else row
            if base_value is not None:
                value = (value or 0.0) - float(base_value)
            if beta and i < len(kl_rows):
                value = (value or 0.0) - beta * float(kl_rows[i])
            shaped.append([value])
        return {'reward': {'values': shaped, 'scorer_ids': ['shaped']}}

    @remote_function(dispatch='all', collect='first', lazy_collect=True)
    def advantage(self, samples: List[Any], reward: Any, values_signal: Any,
                  num_generations: int, estimator: str, gamma: float, lam: float,
                  extra: Optional[dict] = None, **kwargs) -> Dict[str, Any]:
        """Baseline-relative credit over resolved rewards / values; no forward here.

        The GRPO family delegates to ``swift.dev.rewards.advantage.compute_advantages``
        (rewards as the ``[N, C]`` matrix the score stage produces, grouped by
        ``num_generations``); GAE goes to ``twinkle.advantage.GAEAdvantage`` with
        token rewards built from the terminal rewards and the per-sample response
        lengths. This worker never touches a model. ``returns`` is filled by GAE
        and stays ``None`` for the others (an optional output channel)."""
        reward_value = deref(reward)
        rewards = None
        if reward_value is not None:
            rewards = reward_value.get('values') if isinstance(reward_value, dict) else reward_value
        if estimator == 'gae':
            from twinkle.advantage import GAEAdvantage

            if values_signal is None:
                raise ValueError('gae needs a wired values lane')
            resolved = deref(values_signal)
            values = resolved.get('values') if isinstance(resolved, dict) else resolved
            # The length authority is the ENCODED labels' trainable segment -- the
            # same segment the loss mask scores -- not response_token_ids: a
            # multi-turn trajectory counts turn-boundary tokens in labels, so the
            # two disagree and a GAE row aligned to the wrong one fails the loss's
            # pad-and-align check.
            def _response_length(sample: Any) -> int:
                labels = (getattr(sample, 'encoded', None) or {}).get('labels')
                if labels is None:
                    return len(getattr(sample, 'response_token_ids', None) or ())
                if hasattr(labels, 'tolist'):  # a torch tensor: elementwise != would
                    labels = labels.tolist()   # yield a bool tensor, uncountable
                if labels and isinstance(labels[0], list):
                    # a single-sample batched lane [[...]] (the ledger's collated
                    # form): the batch dimension holds exactly this one row
                    labels = labels[0]
                return sum(1 for label in labels if label != -100)

            lengths = [_response_length(sample) for sample in samples]
            flat_rewards = [sum(float(v) for v in row) if isinstance(row, list) and row
                            else float(row) for row in _rows(rewards)]
            # A model_forward value lane is whole-sequence (prompt+completion);
            # GAE works on the response segment, which is the sequence TAIL --
            # trim each value row to its sample's response length. Rows already
            # response-length pass through unchanged.
            value_rows = [list(row) if isinstance(row, list) else [row]
                          for row in _rows(values)]
            aligned_values = []
            for i, row in enumerate(value_rows):
                n = lengths[i] if i < len(lengths) else 0
                if n and len(row) > n:
                    row = row[-n:]
                aligned_values.append(row)
            token_rewards = GAEAdvantage.build_token_rewards(
                flat_rewards, lengths, kl_coef=0.0)
            advantages, returns = GAEAdvantage(
                gamma=gamma, gae_lambda=lam)(token_rewards, aligned_values)
            return {'adv': {'values': advantages, 'estimator': estimator},
                    'returns': returns}
        from swift.dev.rewards.advantage import compute_advantages

        advantages = compute_advantages(
            rewards, num_generations, advantage_estimator=estimator,
            **(extra or {}))
        return {'adv': {'values': advantages, 'estimator': estimator}, 'returns': None}

    @remote_function(dispatch='all', collect='first', lazy_collect=True)
    def signal_combine(self, lanes: List[Tuple[str, Any]], mode: str,
                       weights: List[float]) -> Dict[str, Any]:
        """Weighted sum or probability mixture over named, aligned lanes.

        logps lanes mix in probability space -- logsumexp_k(log w_k + logp_k) --
        never as an arithmetic mean of logps. A row is either a SCALAR or a
        TOKEN SEQUENCE (a logps lane): token rows blend element-wise over the
        lanes' aligned positions; mixing a token row with a scalar row in one
        blend is refused (no silent broadcasting). Lane payloads resolve HERE."""
        lane_values = [deref(lane_value) for _slot, lane_value in lanes]
        n_rows = min(len(_rows(v)) for v in lane_values) if lane_values else 0
        fused: List[Any] = []
        for i in range(n_rows):
            terms = []
            for lane, weight in zip(lane_values, weights):
                lane_rows = _rows(lane)
                value = lane_rows[i] if i < len(lane_rows) else None
                if value is None:
                    continue
                terms.append((value, weight))
            if not terms:
                fused.append(None)
                continue
            token_rows = [v for v, _w in terms if isinstance(v, list)]
            scalars = [v for v, _w in terms if not isinstance(v, list)]
            if token_rows and scalars:
                raise ValueError(
                    'signal_combine cannot blend a token-sequence row with a scalar row; '
                    'wire lanes of one row shape per combine node')
            if token_rows:
                fused.append(self._blend_token_rows(terms, mode))
            elif mode == 'probability_mixture':
                logs = [math.log(max(w, 1e-12)) + float(v) for v, w in terms]
                top = max(logs)
                fused.append(top + math.log(sum(math.exp(v - top) for v in logs)))
            else:  # weighted_sum over scalars
                total_w = sum(w for _v, w in terms)
                fused.append(sum(float(v) for v, _w in terms) / total_w if total_w else None)
        return {'out': {'values': fused, 'mode': mode}}

    @staticmethod
    def _blend_token_rows(terms: List[Tuple[Any, float]], mode: str) -> List[float]:
        """Element-wise blend over aligned token positions (the logps-lane form).

        weighted_sum: position j is sum_k(w_k * v_k[j]) / sum_k(w_k) over the
        lanes that HAVE position j. probability_mixture: position j is
        logsumexp_k(log w_k + v_k[j]) -- mixing in probability space, the same
        rule the scalar form applies, per token."""
        rows = [(list(v), w) for v, w in terms if isinstance(v, list)]
        length = max(len(v) for v, _w in rows)
        out: List[float] = []
        for j in range(length):
            present = [(v[j], w) for v, w in rows if j < len(v) and v[j] is not None]
            if not present:
                out.append(0.0)
                continue
            if mode == 'probability_mixture':
                logs = [math.log(max(w, 1e-12)) + float(v) for v, w in present]
                top = max(logs)
                out.append(top + math.log(sum(math.exp(v - top) for v in logs)))
            else:
                total_w = sum(w for _v, w in present)
                out.append(sum(float(v) for v, _w in present) / total_w if total_w else 0.0)
        return out


@lru_cache(maxsize=1)
def get_numeric_worker(remote_group: str = NUMERIC_GROUP) -> NumericWorker:
    """The process-resident numeric engine (one per session process).

    The session is ALWAYS ray: the worker pool is created on first use under
    ``remote_group`` -- the CPU group the session bootstrap declared at
    ``twinkle.initialize`` time (the placement module's NUMERIC_GROUP, the
    single spelling of that name). A session started without the group fails
    on the first remote-execution stage."""
    return NumericWorker(remote_group=remote_group)

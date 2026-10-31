# Copyright (c) ModelScope Contributors. All rights reserved.
"""The numeric compute plane for remote-execution stages.

Heavy forward payloads (logps / logits) must never materialise in the session
process: they flow between stages as infra references (LazyCollect under Ray,
plain values locally), and every numeric consumer -- reduction, shaping,
advantage estimation, combination -- runs HERE, on this worker, where the
references are resolved. The driver only carries small control scalars back.

Placement notes:

- ``local`` mode: the methods are plain calls returning real values -- the
  same binding code path, zero remoting cost.
- ``ray`` mode: a reference passed as an argument is resolved by the dispatch
  layer on this worker; the return value rides back as ONE reference
  (``lazy_collect=True``), which downstream consumers (the next numeric call,
  ``train_step``'s small scalar read-back, the preview materialisation)
  dereference where THEY run.
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
    advantage estimation and lane combination, all over resolved values."""

    @remote_function(dispatch='all', collect='first', lazy_collect=True)
    def project(self, handle: Any, field: str) -> Any:
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
    def reduce(self, signal: Any, op: str, axis: str) -> Dict[str, Any]:
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
                     beta: float, kl: str) -> Dict[str, Any]:
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
                  kwargs: Optional[dict] = None) -> Dict[str, Any]:
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
            lengths = [len(getattr(sample, 'response_token_ids', None) or ())
                       for sample in samples]
            flat_rewards = [sum(float(v) for v in row) if isinstance(row, list) and row
                            else float(row) for row in _rows(rewards)]
            token_rewards = GAEAdvantage.build_token_rewards(
                flat_rewards, lengths, kl_coef=0.0)
            advantages, returns = GAEAdvantage(
                gamma=gamma, gae_lambda=lam)(token_rewards, values)
            return {'adv': {'values': advantages, 'estimator': estimator},
                    'returns': returns}
        from swift.dev.rewards.advantage import compute_advantages

        advantages = compute_advantages(
            rewards, num_generations, advantage_estimator=estimator,
            **(kwargs or {}))
        return {'adv': {'values': advantages, 'estimator': estimator}, 'returns': None}

    @remote_function(dispatch='all', collect='first', lazy_collect=True)
    def signal_combine(self, lanes: List[Tuple[str, Any]], mode: str,
                       weights: List[float]) -> Dict[str, Any]:
        """Weighted sum or probability mixture over named, aligned lanes.

        logps lanes mix in probability space -- logsumexp_k(log w_k + logp_k) --
        never as an arithmetic mean of logps. Lane payloads resolve HERE."""
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
                if mode == 'probability_mixture':
                    terms.append(math.log(max(weight, 1e-12)) + float(value))
                else:  # weighted_sum
                    terms.append((float(value), weight))
            if not terms:
                fused.append(None)
                continue
            if mode == 'probability_mixture':
                top = max(terms)
                fused.append(top + math.log(sum(math.exp(v - top) for v in terms)))
            else:
                total_w = sum(w for _v, w in terms)
                fused.append(sum(v for v, _w in terms) / total_w if total_w else None)
        return {'out': {'values': fused, 'mode': mode}}


@lru_cache(maxsize=1)
def get_numeric_worker(remote_group: str = 'webui_numeric') -> NumericWorker:
    """The process-resident numeric engine (one per session process).

    ``local`` mode: a plain object, methods are direct calls. ``ray`` mode:
    the worker pool is created on first use under ``remote_group`` -- the
    group is declared by the session bootstrap
    (``python -m swift.dev.webui.session graph.json --mode ray``); a session
    started without it fails on the first remote-execution stage."""
    return NumericWorker(remote_group=remote_group)

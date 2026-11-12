# Copyright (c) ModelScope Contributors. All rights reserved.
"""Signal-side bindings: forward reads, reductions, shaping, advantage, fusion.

Every stage here is an independent, general math/functional meaning -- no
algorithm name decides behaviour. All five are remote-execution stages: the
binding only wires inputs and parameters and hands references to twinkle's
signal components on the numeric group, so heavy forward payloads never
materialise in the session process.
"""
from __future__ import annotations

from typing import Any, Dict, List

from twinkle.loss import Loss

from ..component import ComponentError, StageComponent
from ..numeric import deref, numeric_component


def _row_scalars(rows: Any) -> List[float]:
    """A score matrix ``values[B, C]`` (or flat ``[B]``) folded to one scalar per row; None / nan count 0."""
    from twinkle_agentic.async_rl import weighted_score
    return [weighted_score(row, None) if isinstance(row, (list, tuple)) else float(row) for row in rows]


class _ForwardProbeLoss(Loss):
    """Field-selection lens for ONE independent forward call -- never an objective.

    The require_* flags ride the existing ``set_loss`` registry so the native
    forward path computes exactly the requested channels. The call body must
    never execute: the transformers ``forward_only`` never calls the loss, and
    the Megatron path is handed ``compute_loss=False``.
    """

    def __init__(self, *, require_logps: bool = False, require_logits: bool = False,
                 require_values: bool = False):
        self.require_logps = require_logps
        self.require_logits = require_logits
        self.require_values = require_values

    def __call__(self, inputs: Any, outputs: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            'the forward probe loss selects output fields; it is never an objective')


class ModelForwardStage(StageComponent):
    """model_forward: one independent forward call with an explicit field set.

    A probe loss is registered ONCE per node on the model's own loss registry
    under a stable key (``forward:<node_id>``); ``forward_only`` then selects
    that key explicitly. The training loss keeps its own separate key and the
    optimizer is never touched -- this call never changes what training uses.
    Each declared output port carries ONE channel of the native return, read
    on the numeric group (``ResponseScores``): ``logps`` / ``values`` aligned to
    each row's response tokens, ``logits`` / ``embeddings`` as the native frame.
    Under Ray the other channels never cross the boundary, and a channel the
    model does not produce fails loudly there. The input is raw data /
    trajectories: encoding happens HERE under the node's own tokenizer param
    (a construction fact, not a wire)."""

    type = 'model_forward'

    def _features(self, data: Any) -> Any:
        """The input encoded under this node's own tokenizer, built once."""
        from .resources import build_template_from, encode_rows
        if getattr(self, '_template', None) is None:
            self._template = build_template_from(
                str(self.params.get('tokenizer') or ''),
                dict(self.params.get('template_config') or {}), self.node_id,
                model_kv=dict(self.params.get('model_config') or {}))
        if data is None:
            raise ComponentError(f'{self.node_id}: model_forward needs a wired traj input')
        return encode_rows(self._template, data)

    def invoke(self, inputs: Dict[str, Any], batch: Any) -> Dict[str, Any]:
        model = inputs.get('model')
        if model is None:
            raise ComponentError(f'{self.node_id}: model_forward needs a wired model')
        forward = getattr(model, 'forward_only', None)
        if not callable(forward):
            raise ComponentError(f'{self.node_id}: model exposes no forward_only()')
        # Port-first: the wired traj lane is the data THIS forward scores
        # (the generated trajectories); the positional batch is the frame
        # SOURCE's batch -- the loader's prompts. Encoding the positional one
        # silently scores the prompt instead of the episode, and every
        # downstream lane (KL, teacher, values) is wrong with no shape alarm.
        data = inputs.get('traj')
        if data is None:
            data = batch
        features = self._features(data)
        if not isinstance(features, list):
            features = [features]
        if not features:
            raise ComponentError(f'{self.node_id}: the encoded input is empty')
        raw_fields = self.params.get('outputs') or ['logps']
        fields = [str(field) for field in (raw_fields if isinstance(raw_fields, list) else [raw_fields])]
        if not fields:
            raise ComponentError(f'{self.node_id}: model_forward needs at least one output field')
        if 'topk' in fields:
            raise ComponentError(
                f'{self.node_id}: the native forward path returns no global top-k channel; '
                'a topk output is rejected rather than silently degraded to truncated logits')
        if 'values' in fields and any(field in fields for field in ('logps', 'logits', 'embeddings')):
            raise ComponentError(
                f'{self.node_id}: values come from the value head (task=value) while '
                'logps/logits/embeddings use the language-model task; wire one '
                'model_forward node per task')
        kwargs: Dict[str, Any] = {'compute_loss': False, 'return_logits': 'logits' in fields}
        if self.params.get('disable_lora'):
            kwargs['disable_lora'] = True
        temperature = self.params.get('temperature')
        if temperature not in (None, ''):
            kwargs['temperature'] = float(temperature)
        micro = self.params.get('micro_batch_size')
        if micro:
            kwargs['micro_batch_size'] = int(micro)
        if 'values' in fields:
            kwargs['task'] = 'value'
        loss_key = f'forward:{self.node_id}'
        if not getattr(self, '_probe_registered', False):
            set_loss = getattr(model, 'set_loss', None)
            if not callable(set_loss):
                raise ComponentError(f'{self.node_id}: model exposes no set_loss()')
            set_loss(_ForwardProbeLoss(
                require_logps='logps' in fields,
                require_logits='logits' in fields,
                require_values='values' in fields), loss_key=loss_key)
            self._probe_registered = True
        result = forward(inputs=features, loss_key=loss_key, **kwargs)
        from twinkle.signals import ResponseScores
        scores = numeric_component(ResponseScores)
        return {field: scores(features if field in ('logps', 'values') else None, result, output_key=field)
                for field in fields}


class ReduceStage(StageComponent):
    """A named reduction over an explicit axis: pure math on the numeric group (``Reduce``)."""

    type = 'reduce'

    def invoke(self, inputs: Dict[str, Any], batch: Any) -> Dict[str, Any]:
        signal = inputs.get('signal')
        if signal is None:
            raise ComponentError(f'{self.node_id}: reduce needs a wired signal')
        from twinkle.signals import Reduce
        return {'out': numeric_component(Reduce)(
            signal, op=str(self.params.get('op', 'mean')), axis=str(self.params.get('axis', 'token')))}


class RewardShapeStage(StageComponent):
    """r' = r - baseline - beta*KL: combine the wired reward, baseline and the
    two explicit KL forward lanes. Absent inputs are absent terms -- never
    hidden defaults. The reward's scorer columns are summed per row; the
    ``outcome`` profile shapes one scalar per row (``values[B, 1]``), the
    ``process`` profile one reward per response token (``KLEstimator`` +
    ``RewardShape`` on the numeric group)."""

    type = 'reward_shape'

    def invoke(self, inputs: Dict[str, Any], batch: Any) -> Dict[str, Any]:
        from twinkle.reward import RewardShape
        from twinkle.signals import KLEstimator

        reward = inputs.get('reward')
        if reward is None:
            raise ComponentError(f'{self.node_id}: reward_shape needs a wired reward')
        beta = float(self.params.get('beta', 0.0) or 0.0)
        kl = str(self.params.get('kl', 'none') or 'none')
        outcome = str(self.params.get('profile', 'outcome') or 'outcome') == 'outcome'
        reward = deref(reward)
        rewards = _row_scalars(reward.get('values') if isinstance(reward, dict) else reward)
        kl_lane = None
        if kl != 'none':
            if inputs.get('logps') is None or inputs.get('reference') is None:
                raise ComponentError(f'{self.node_id}: a {kl} KL term needs both logps and reference lanes')
            kl_lane = numeric_component(KLEstimator)(
                inputs['logps'], inputs['reference'], kind=kl, reduce='mean' if outcome else 'none')
        base = deref(inputs.get('baseline'))
        shaped = numeric_component(RewardShape)(
            rewards, kl_lane, base.get('scalar') if isinstance(base, dict) else base,
            beta=beta, profile='outcome' if outcome else 'token')
        # the outcome rows are one scalar each (read back here, the score lane's [B, C] shape);
        # the per-token process rows stay a reference.
        values = [[value] for value in deref(shaped)] if outcome else shaped
        return {'reward': {'values': values, 'scorer_ids': ['shaped']}}


class AdvantageStage(StageComponent):
    """A = baseline-relative credit over rewards / values. Never touches a
    model: those are inputs. The per-sequence estimators run HERE on the light
    reward matrix (``compute_advantages``); GAE runs on the numeric group over
    the per-token values lane and additionally emits the returns lane (an
    optional output channel).

    ``old_logps`` carries the SAMPLER's per-token logps off the samples
    (``rollout_logprobs``) -- the importance-sampling denominator the loss
    consumes next; it is ``None`` when any sample lacks it (a non-rollout
    batch), and the loss then runs its on-policy form. The ``adv`` channel is
    the BARE per-sample value sequence -- the shape the loss's ``advantages``
    kwarg takes."""

    type = 'advantage'

    def invoke(self, inputs: Dict[str, Any], batch: Any) -> Dict[str, Any]:
        estimator = str(self.params.get('estimator', 'grpo'))
        reward = inputs.get('reward')
        samples = inputs.get('traj') if inputs.get('traj') is not None else batch
        if not isinstance(samples, list):
            samples = []
        kwargs = dict(self.params.get('kwargs') or {})
        # Declared param (registry), matching generate/select -- the estimator's grouping width.
        # kwargs keeps its old pop so a stray num_generations there never leaks into the estimator.
        kwargs.pop('num_generations', None)
        num_generations = int(self.params.get('num_generations', 8) or 8)
        rollout_rows = [list(sample.get('rollout_logprobs') or ()) if isinstance(sample, dict) else []
                        for sample in samples]
        old_logps = rollout_rows if rollout_rows and all(rollout_rows) else None
        reward = deref(reward) if not isinstance(reward, dict) else reward
        lane = reward.get('values') if isinstance(reward, dict) else reward
        if estimator == 'gae':
            from twinkle.advantage import GAEAdvantage

            if inputs.get('values') is None:
                raise ComponentError(f'{self.node_id}: gae needs a wired values lane')
            # a score matrix folds to one terminal reward per row; a process lane (per-token rows,
            # still a reference) is consumed as-is
            rewards = _row_scalars(lane) if isinstance(lane, list) else lane
            # read back HERE like every lane train_step consumes
            adv, returns = deref(numeric_component(GAEAdvantage)(
                rewards, inputs['values'], gamma=float(self.params.get('gamma', 1.0) or 1.0),
                gae_lambda=float(self.params.get('lam', 1.0) or 1.0)))
            return {'adv': adv, 'returns': returns, 'old_logps': old_logps}
        from twinkle.advantage import compute_advantages

        adv = compute_advantages(deref(lane), num_generations, advantage_estimator=estimator, **kwargs)
        return {'adv': adv, 'returns': None, 'old_logps': old_logps}


class SignalCombineStage(StageComponent):
    """An explicit combination over named, aligned lanes: weighted sum or
    probability mixture. logps lanes mix in probability space --
    logsumexp_k(log w_k + logp_k) -- never as an arithmetic mean of logps.
    Weights are non-negative with a positive sum; a lane without an explicit
    per-slot weight defaults to uniform."""

    type = 'signal_combine'

    def invoke(self, inputs: Dict[str, Any], batch: Any) -> Dict[str, Any]:
        lanes = inputs.get('signals')
        if not isinstance(lanes, list) or not lanes:
            raise ComponentError(f'{self.node_id}: signal_combine needs wired signal lanes')
        mode = str(self.params.get('mode', 'weighted_sum'))
        if mode not in ('weighted_sum', 'probability_mixture'):
            raise ComponentError(f'{self.node_id}: unknown combine mode {mode!r}')
        raw_weights = dict(self.params.get('weights') or {})
        slots = [str(slot) for slot, _value in lanes]
        weights = [float(raw_weights.get(slot, 1.0 / len(lanes))) for slot in slots]
        if any(w < 0 for w in weights) or sum(weights) <= 0:
            raise ComponentError(f'{self.node_id}: weights must be non-negative and sum > 0')
        from twinkle.signals import SignalCombine
        # The out port carries the VALUE sequence only; the combine mode stays a node param.
        return {'out': numeric_component(SignalCombine)(
            [value for _slot, value in lanes], weights=weights, mode=mode)}

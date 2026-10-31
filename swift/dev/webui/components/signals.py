# Copyright (c) ModelScope Contributors. All rights reserved.
"""Signal-side bindings: forward reads, reductions, shaping, advantage, fusion.

Every stage here is an independent, general math/functional meaning -- no
algorithm name decides behaviour. All five are remote-execution stages: the
binding only wires inputs and parameters and hands references to the numeric
worker, so heavy forward payloads never materialise in the session process
(local mode: the same calls are plain function calls returning real values).
"""
from __future__ import annotations

from typing import Any, Dict, List

from twinkle.loss import Loss

from ..component import ComponentError, StageComponent
from ..numeric import get_numeric_worker


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
    Each declared output port carries ONE channel of the native return,
    projected on the numeric worker (``project``): under Ray the other
    channels never cross the boundary, and a channel the model does not
    produce fails loudly at the projection. The input is raw data /
    trajectories: encoding happens HERE under the node's own tokenizer param
    (a construction fact, not a wire)."""

    type = 'model_forward'

    def _features(self, data: Any) -> Any:
        """The input encoded under this node's own tokenizer, built once."""
        if getattr(self, '_template', None) is None:
            from .resources import build_template_from, encode_rows
            self._template = build_template_from(
                str(self.params.get('tokenizer') or ''),
                dict(self.params.get('template_config') or {}), self.node_id)
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
        features = self._features(inputs.get('traj'))
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
        worker = get_numeric_worker()
        return {field: worker.project(result, field) for field in fields}


class ReduceStage(StageComponent):
    """A named reduction over an explicit axis and mask: pure math on the worker."""

    type = 'reduce'

    def invoke(self, inputs: Dict[str, Any], batch: Any) -> Dict[str, Any]:
        signal = inputs.get('signal')
        if signal is None:
            raise ComponentError(f'{self.node_id}: reduce needs a wired signal')
        worker = get_numeric_worker()
        result = worker.reduce(
            signal, str(self.params.get('op', 'mean')), str(self.params.get('axis', 'token')))
        return {'out': worker.project(result, 'out')}


class RewardShapeStage(StageComponent):
    """r' = r - baseline - beta*KL: combine the wired reward, baseline and the
    two explicit KL forward lanes. Absent inputs are absent terms -- never
    hidden defaults."""

    type = 'reward_shape'

    def invoke(self, inputs: Dict[str, Any], batch: Any) -> Dict[str, Any]:
        reward = inputs.get('reward')
        if reward is None:
            raise ComponentError(f'{self.node_id}: reward_shape needs a wired reward')
        beta = float(self.params.get('beta', 0.0) or 0.0)
        kl = str(self.params.get('kl', 'none') or 'none')
        worker = get_numeric_worker()
        result = worker.reward_shape(
            reward, inputs.get('baseline'), inputs.get('logps'), inputs.get('reference'),
            beta, kl)
        return {'reward': worker.project(result, 'reward')}


class AdvantageStage(StageComponent):
    """A = baseline-relative credit, computed on the worker over resolved
    rewards / values. Never touches a model: those are inputs. GAE
    additionally emits the returns lane (an optional output channel)."""

    type = 'advantage'

    def invoke(self, inputs: Dict[str, Any], batch: Any) -> Dict[str, Any]:
        estimator = str(self.params.get('estimator', 'grpo'))
        reward = inputs.get('reward')
        samples = inputs.get('traj') if inputs.get('traj') is not None else batch
        if not isinstance(samples, list):
            samples = []
        kwargs = dict(self.params.get('kwargs') or {})
        num_generations = int(kwargs.pop('num_generations', 8))
        worker = get_numeric_worker()
        result = worker.advantage(
            samples, reward, inputs.get('values'), num_generations, estimator,
            float(self.params.get('gamma', 1.0) or 1.0),
            float(self.params.get('lam', 1.0) or 1.0), kwargs)
        return {'adv': worker.project(result, 'adv'),
                'returns': worker.project(result, 'returns')}


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
        worker = get_numeric_worker()
        result = worker.signal_combine(list(lanes), mode, weights)
        return {'out': worker.project(result, 'out')}

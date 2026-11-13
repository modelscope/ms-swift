# Copyright (c) ModelScope Contributors. All rights reserved.
"""Signal-side bindings: the model forward, the privileged view, advantage, fusion.

Every stage here is an independent, general math/functional meaning -- no
algorithm name decides behaviour. All are remote-execution stages: the
binding wires inputs and parameters and hands references to twinkle
components on the numeric group, so heavy forward payloads never materialise
in the session process.
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..component import ComponentError, StageComponent
from ..numeric import deref, numeric_component
from ..placement import NUMERIC_GROUP

#: ModelOutput 的元字段：通道解析时排除，剩下的才是数据通道。
_OUTPUT_META_KEYS = ('field_frames', 'episode_ids', 'response_token_ids', 'tokenizer_fingerprint',
                     'model_id', 'parameter_version')


def output_channels(value: Any) -> Dict[str, Any]:
    """One MODEL_OUTPUT value's data channels, resolved HERE (never on the producer's side).

    A port value is a whole ModelOutput (or its lazy handle); a channel is one
    field's per-row sequence. Resolution is by name -- never by container shape.
    """
    value = deref(value)
    if isinstance(value, dict):
        return {key: item for key, item in value.items() if key not in _OUTPUT_META_KEYS}
    return {'values': value}


def single_channel(value: Any, *, what: str) -> Any:
    """One MODEL_OUTPUT value's data channel: exactly one, the names in the error otherwise."""
    channels = output_channels(value)
    if len(channels) != 1:
        raise ComponentError(f'{what} needs a single data channel; got {sorted(channels) or "none"}')
    return next(iter(channels.values()))


class ModelForwardStage(StageComponent):
    """model_forward: one independent forward call with an explicit field set.

    The forward consumes the node-encoded rows and requests its channels
    explicitly: ``logps`` / ``values`` come back response-framed, ``logits`` /
    ``embeddings`` keep their full input frame -- ONE whole ModelOutput rides
    the ``out`` port, and the forward role (ref / teacher / old / value) is
    the consumer's business, not a payload property. The training loss and
    optimizer are never touched (``compute_loss=False``: no probe loss is
    installed, the explicit request is honoured by the model's own forward).
    The input is raw data / trajectories: encoding happens HERE under the
    node's own tokenizer param (a construction fact, not a wire)."""

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
        # Port-first: the wired traj lane is the data THIS forward scores; the positional
        # batch is the frame SOURCE's batch. Encoding the positional one silently scores
        # the wrong rows with no shape alarm.
        data = inputs.get('traj')
        data = deref(data) if data is not None else batch
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
            # The topk request was never implemented (a truncated-logits degradation): it is
            # rejected rather than silently faked; the old knob is gone from the registry.
            raise ComponentError(
                f'{self.node_id}: the native forward path returns no global top-k channel; '
                'a topk output is rejected rather than silently degraded to truncated logits')
        if 'values' in fields and any(field in fields for field in ('logps', 'logits', 'embeddings')):
            raise ComponentError(
                f'{self.node_id}: values come from the value head (task=value) while '
                'logps/logits/embeddings use the language-model task; wire one '
                'model_forward node per task')
        kwargs: Dict[str, Any] = {'compute_loss': False, 'output_fields': tuple(fields)}
        if {'logps', 'values'} & set(fields):
            # Response projection for the token-score channels; logits/embeddings keep
            # their own declared frame inside the same ModelOutput.
            kwargs['output_frame'] = 'response'
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
        return {'out': forward(inputs=features, **kwargs)}


class PrivilegedViewStage(StageComponent):
    """privileged_view: re-encode each row under its ``teacher_prompt`` column.

    A thin binding over twinkle's :class:`PrivilegedView` -- the SAME transform
    the recipes' teacher branch uses. A row without a ``teacher_prompt`` passes
    through unchanged; a row with one is re-encoded with the policy's exact
    sampled response token ids pinned, so the teacher forward grades the same
    answer hint-augmented. The policy's own trajectories are never touched and
    nothing is written back: this is the teacher branch's input builder."""

    type = 'privileged_view'

    def _view(self) -> Any:
        view = getattr(self, '_view_instance', None)
        if view is None:
            from .resources import build_template_from

            from twinkle.processor import PrivilegedView
            template = build_template_from(
                str(self.params.get('tokenizer') or ''),
                dict(self.params.get('template_config') or {}), self.node_id,
                model_kv=dict(self.params.get('model_config') or {}))
            view = self._view_instance = PrivilegedView(template=template, remote_group=NUMERIC_GROUP)
        return view

    def invoke(self, inputs: Dict[str, Any], batch: Any) -> Dict[str, Any]:
        data = inputs.get('traj') if inputs.get('traj') is not None else batch
        if data is None:
            raise ComponentError(f'{self.node_id}: privileged_view needs a wired traj input')
        return {'traj': self._view()(data)}


class AdvantageStage(StageComponent):
    """advantage: trajectories and forward lanes in, training-target columns out.

    The estimator is twinkle's :class:`Advantage` family member (one remote
    instance per node, built under the node's knobs -- a knob the estimator
    does not read is rejected at construction, not silently ignored). It reads
    the named score columns off the trajectories and the four optional forward
    lanes (``ref_out`` / ``teacher_out`` / ``old_out`` / ``value_out``, each a
    whole ModelOutput or its handle), and writes advantages / returns /
    old_logps / ref_logps / teacher_logps onto copies of the rows: the ``traj``
    port carries the updated rows for training, the ``metric`` port the light
    numeric summary. No model is held and no forward runs here."""

    type = 'advantage'

    def _estimator(self) -> Any:
        name = str(self.params.get('estimator', 'grpo') or 'grpo')
        if name == 'gae':
            from twinkle.advantage import GAEAdvantage

            cls = GAEAdvantage
        else:
            from twinkle.advantage.estimate import ESTIMATORS

            if name not in ESTIMATORS:
                raise ComponentError(
                    f'{self.node_id}: unknown estimator {name!r}; one of {sorted(ESTIMATORS)} or gae')
            cls = ESTIMATORS[name]
        kwargs = dict(num_generations=int(self.params.get('num_generations', 8) or 8),
                      **dict(self.params.get('kwargs') or {}))
        return cls(**kwargs, remote_group=NUMERIC_GROUP)

    def invoke(self, inputs: Dict[str, Any], batch: Any) -> Dict[str, Any]:
        trajs = inputs.get('traj') if inputs.get('traj') is not None else batch
        if not isinstance(trajs, list) or not trajs:
            raise ComponentError(f'{self.node_id}: advantage needs a wired traj batch')
        estimator = getattr(self, '_estimator_instance', None)
        if estimator is None:
            estimator = self._estimator_instance = self._estimator()
        result = estimator(trajs,
                           ref_out=inputs.get('ref_out'), teacher_out=inputs.get('teacher_out'),
                           old_out=inputs.get('old_out'), value_out=inputs.get('value_out'))
        # The tuple (new_trajs, summary) resolves HERE: the rows feed the session-local
        # training loop (train_step encodes them under its own tokenizer), the summary
        # is the light metric port's payload.
        new_trajs, summary = deref(result)
        metric = {'values': {key: float(value) for key, value in summary.items()
                             if isinstance(value, (int, float)) and not isinstance(value, bool)},
                  'invalid_count': 0}
        return {'traj': new_trajs, 'metric': metric}


class SignalCombineStage(StageComponent):
    """An explicit combination over named, aligned lanes: weighted sum or
    probability mixture. logps lanes mix in probability space --
    logsumexp_k(log w_k + logp_k) -- never as an arithmetic mean of logps.
    A lane is one MODEL_OUTPUT value with exactly ONE data channel (the
    explicit field); weights are non-negative with a positive sum; a lane
    without an explicit per-slot weight defaults to uniform."""

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
        rows = [single_channel(value, what=f'{self.node_id} lane {slot!r}') for slot, value in lanes]
        from twinkle.signals import SignalCombine

        # The out port carries the VALUE sequence only; the combine mode stays a node param.
        return {'out': numeric_component(SignalCombine)(rows, weights=weights, mode=mode)}

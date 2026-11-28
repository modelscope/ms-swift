# Copyright (c) ModelScope Contributors. All rights reserved.
"""Signal-side bindings: the model forward, the privileged view, advantage.

Every stage here is an independent, general math/functional meaning -- no
algorithm name decides behaviour. All are remote-execution stages: the
binding wires inputs and parameters and hands references to twinkle
components on the numeric group, so heavy forward payloads never materialise
in the session process.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict

from ..component import ComponentError, StageComponent
from ..numeric import deref
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
        # the wrong rows with no shape alarm. A handle-carrying lane is dereferenced HERE:
        # the batch is materialised once in the session process before it is encoded.
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
            # instance_id feeds the actor NAME (group + class + id): the node id
            # keeps same-type nodes distinct, the random suffix keeps a rebuilt
            # node from colliding with its predecessor -- whose kill may not have
            # landed in the GCS yet, and Ray rejects a taken name at creation.
            view = self._view_instance = PrivilegedView(
                template=template, instance_id=f'{self.node_id}-{uuid.uuid4().hex[:8]}',
                remote_group=NUMERIC_GROUP)
        return view

    def invoke(self, inputs: Dict[str, Any], batch: Any) -> Dict[str, Any]:
        data = inputs.get('traj') if inputs.get('traj') is not None else batch
        if data is None:
            raise ComponentError(f'{self.node_id}: privileged_view needs a wired traj input')
        return {'traj': self._view()(data)}

    def release(self) -> None:
        """Kill the view's remote workers: the session calls this on
        invalidation (param change / graph revision), so the actor never
        outlives the component that built it."""
        view = getattr(self, '_view_instance', None)
        if view is None:
            return
        self._view_instance = None
        try:
            view.close()
        except Exception:
            pass


class AdvantageStage(StageComponent):
    """Forward trajectories, scores and model lanes to the estimator; retain only light metrics locally."""

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
        kwargs = dict(self.params.get('kwargs') or {})
        if name != 'gae':
            # GAE is token-level: its constructor REJECTS num_generations (no group
            # fold to size). The node param still declares the group width for the
            # graph's width check; only the scalar family consumes it here. An
            # explicit kwargs entry always wins over the node param.
            kwargs.setdefault('num_generations', int(self.params.get('num_generations', 8) or 8))
        # instance_id feeds the actor NAME (group + class + id): the node id
        # keeps same-type nodes distinct, the random suffix keeps a rebuilt
        # node from colliding with its predecessor -- whose kill may not have
        # landed in the GCS yet, and Ray rejects a taken name at creation.
        kwargs['instance_id'] = f'{self.node_id}-{uuid.uuid4().hex[:8]}'
        return cls(**kwargs, remote_group=NUMERIC_GROUP)

    def release(self) -> None:
        """Kill the estimator's remote workers: the session calls this on
        invalidation (param change / graph revision), so the actor never
        outlives the component that built it."""
        estimator = getattr(self, '_estimator_instance', None)
        if estimator is None:
            return
        self._estimator_instance = None
        try:
            estimator.close()
        except Exception:
            pass

    def invoke(self, inputs: Dict[str, Any], batch: Any) -> Dict[str, Any]:
        trajs = inputs.get('traj') if inputs.get('traj') is not None else batch
        if trajs is None:
            raise ComponentError(f'{self.node_id}: advantage needs a wired traj batch')
        estimator = getattr(self, '_estimator_instance', None)
        if estimator is None:
            estimator = self._estimator_instance = self._estimator()
        # The estimator resolves data at its compute boundary; only its summary is materialized here.
        new_trajs, summary = estimator(
            trajs, scores=inputs.get('scores'),
            ref_out=inputs.get('ref_out'), teacher_out=inputs.get('teacher_out'),
            old_out=inputs.get('old_out'), value_out=inputs.get('value_out'))
        # The summary's nested scalar dicts (rollout_correction's train-vs-sampler
        # logp diagnostics) flatten to prefixed keys: the metric port carries
        # scalars only, and dropping the dict would silently hide the diagnostics.
        values: Dict[str, float] = {}
        for key, value in summary.items():
            if isinstance(value, dict):
                for sub, item in value.items():
                    if isinstance(item, (int, float)) and not isinstance(item, bool):
                        values[f'{key}/{sub}'] = float(item)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                values[key] = float(value)
        metric = {'values': values, 'invalid_count': 0}
        return {'traj': new_trajs, 'metric': metric}

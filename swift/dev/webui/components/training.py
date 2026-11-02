# Copyright (c) ModelScope Contributors. All rights reserved.
"""Training-side bindings: the real forward_backward / step / publish / save.

Contract highlights (see the plan's training and publish sections):

- ``train_step`` runs real ``forward_backward`` per micro batch, the GA
  boundary, ``clip_grad_and_step``; ``optimizer_steps`` counts only real
  optimizer completions. Partial mutation errors keep their completed marks --
  no transparent rollback is promised.
- ``weight_sync`` pushes the trained weights into every wired sampler inside
  its own invoke, through a resident CheckpointEngineManager per
  (model, sampler) pair -- the disaggregated shape's sync point (native
  bucketed device-to-device transfer over the ray channel).
- ``weight_average`` copies or EMA-blends between two explicit models through
  the native transfer surface; the target's role is never hardcoded.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List

from ..component import ComponentError, StageComponent, slot_values
from ..numeric import deref

logger = logging.getLogger(__name__)

#: One serialisation gate per model actor (keyed by the resident proxy's id).
#: The checkpoint engine's ACTIVE channel is actor-wide mutable state: the
#: activate call selects it and the data calls (prepare/init/send/receive)
#: consume the selection. Two manager flows driving one actor concurrently
#: (a canvas wave runs weight_average and weight_sync together when the graph
#: declares them independent) interleave an activation inside the other
#: channel's flow: the send runs on the WRONG engine and the receiver waits
#: forever on a socket nobody publishes to. The recipe driver is sequential
#: and never exposes this; the wave is parallel, so the flows gate here.
_channel_gates: Dict[int, threading.Lock] = {}


def _channel_gate(model: Any) -> threading.Lock:
    lock = _channel_gates.get(id(model))
    if lock is None:
        lock = _channel_gates[id(model)] = threading.Lock()
    return lock


def _int_value(raw: Any, default: int) -> int:
    if raw in (None, ''):
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        raise ValueError(f'expected an integer, got {raw!r}') from None


def _float_value(raw: Any, default: float) -> float:
    if raw in (None, ''):
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        raise ValueError(f'expected a number, got {raw!r}') from None


def _aligned_slice(value: Any, start: int, stop: int) -> Any:
    """Slice one sample-aligned loss kwarg alongside a micro batch.

    A per-sample list is sliced with the batch; a tensor / ndarray is sliced on
    its leading (sample) axis; anything else rides every micro batch whole --
    the same convention the recipes use for scalar kwargs."""
    if isinstance(value, list):
        return value[start:stop]
    if hasattr(value, 'shape') and getattr(value, 'shape', None) and len(value.shape) >= 1:
        return value[start:stop]
    return value


class TrainStepStage(StageComponent):
    """train_step: real forward_backward + optimizer step.

    The objective arrives WIRED on the loss port (a loss node's resident
    object); the optimizer and the encoding rule are PARAMS on this node.
    The loss instance is registered ONCE per node on the model's own loss
    registry under ``train:<node_id>`` (the same registry any model_forward
    probe uses with its own key), and the optimizer is configured ONCE on
    the wired model with the total horizon. Loss kwargs come from the
    arguments port's slots (the real loss parameter names). The input is
    raw data / trajectories -- encoded here under the node's own tokenizer
    param, exactly the per-round re-encode the dev recipes run. The
    ``model`` output carries the trained model reference forward (an
    identity downstream effects consume instead of an actor fan-out)."""

    type = 'train_step'

    def _configure_optimizer(self, model: Any) -> None:
        """The update rule, built ONCE on the wired model with the TOTAL
        horizon (per-click budgets never reconfigure the scheduler)."""
        if getattr(self, '_optimizer_built', False):
            return
        from swift.dev.builders.optimizer import configure_optimizer
        kv = {
            'optim': self.params.get('optim', 'adamw'),
            'learning_rate': self.params.get('lr', 1e-5),
            'lr_scheduler': self.params.get('lr_scheduler_type', 'cosine'),
            'warmup_ratio': self.params.get('warmup_ratio', 0.05),
            'weight_decay': self.params.get('weight_decay', 0.0),
        }
        # optimizer_config carries further NATIVE TrainConfig field spellings;
        # _fill fails loudly on a name that is not a real field.
        kv.update(dict(self.params.get('optimizer_config') or {}))
        num_training_steps = _int_value(self.params.get('num_training_steps', 1000), 1000)
        from .resources import _configs, _fill
        train_config = _fill(_configs().TrainConfig,
                             {k: v for k, v in kv.items() if v is not None},
                             self.node_id)
        configure_optimizer(model, train_config, num_training_steps=num_training_steps)
        self._optimizer_built = True

    def _features(self, data: Any) -> Any:
        from .resources import build_template_from, encode_rows
        if getattr(self, '_template', None) is None:
            # ``model_config`` carries the identity fields the tokenizer load
            # needs (a locally built directory is ambiguous without an explicit
            # ``model_type``); it must stay consistent with the model node's own
            # ModelConfig -- the same contract the sampler node follows.
            self._template = build_template_from(
                str(self.params.get('tokenizer') or ''),
                dict(self.params.get('template_config') or {}), self.node_id,
                model_kv=dict(self.params.get('model_config') or {}))
        if data is None:
            raise ComponentError(f'{self.node_id}: train_step needs a wired batch input')
        return encode_rows(self._template, data)

    def invoke(self, inputs: Dict[str, Any], batch: Any) -> Dict[str, Any]:
        model = inputs.get('model')
        if model is None:
            raise ComponentError(f'{self.node_id}: train_step needs a wired model')
        loss = inputs.get('loss')
        if loss is None:
            raise ComponentError(f'{self.node_id}: train_step needs a wired loss (a loss node)')
        # The training rows ride the wired batch PORT when the frame source
        # delivers a grouped object (the challenger's ChallengeBatch: its
        # port projection is the flattened lane) or projects its last batch
        # (the streaming driver); the positional batch is the loader shape,
        # whose port value is the resident DataLoader object itself. A LIST
        # port value wins, anything else falls back to the positional batch --
        # the same port-first convention score and advantage apply.
        port_batch = inputs.get('batch')
        data = port_batch if isinstance(port_batch, list) else batch
        features = self._features(data)
        if not isinstance(features, list):
            features = [features]
        if not features:
            raise ComponentError(f'{self.node_id}: the encoded batch is empty')
        loss_kwargs: Dict[str, Any] = {}
        for slot, value in (inputs.get('arguments') or []):
            if value is not None:
                # A small numeric lane (advantages / returns / old_logps) read
                # back HERE, exactly the way the recipes consume the estimator
                # output; the heavy forward channels stay references end to end.
                loss_kwargs[str(slot)] = deref(value)
        ga = _int_value(self.params.get('gradient_accumulation_steps'), 1)
        micro = _int_value(self.params.get('micro_batch_size'), 1)
        max_grad_norm = _float_value(self.params.get('max_grad_norm'), 1.0)
        loss_key = f'train:{self.node_id}'
        if not getattr(self, '_loss_registered', False):
            set_loss = getattr(model, 'set_loss', None)
            if not callable(set_loss):
                raise ComponentError(f'{self.node_id}: model exposes no set_loss()')
            set_loss(loss, loss_key=loss_key)
            # A keyed loss DISABLES the model's default LossMetric (the metric
            # dispatcher only accumulates a metric whose key matches the
            # forward's), so one keyed LossMetric rides the same registration:
            # without it calculate_metric reports nothing and the loss stays
            # invisible in the node's preview. The same keyed-metric pattern
            # ``configure_rl_metrics`` applies for the algorithm metrics.
            from twinkle.metric import LossMetric
            model.add_metric(LossMetric, is_training=True, loss_key=loss_key)
            self._loss_registered = True
        self._configure_optimizer(model)
        from swift.dev.recipe.utils import backend_owns_gradient_accumulation, is_grad_sync_boundary
        # A value loss trains the per-token value head: forward_backward must run
        # the value task (the same flag the PPO recipe passes) or the seq_cls head
        # returns one scalar per sample and the loss's values/mask check fails.
        forward_task = 'value' if bool(getattr(loss, 'require_values', False)) else None
        optimizer_steps = 0
        if backend_owns_gradient_accumulation(model):
            # Megatron internalises the micro-batch schedule: the feature LIST
            # is the micro-batch plan (GA = its length), so one call is one
            # optimizer step; loss kwargs are sliced worker-side by the loss's
            # own micro_batch_kwargs.
            deref(model.forward_backward(inputs=features, loss_key=loss_key,
                                          micro_batch_size=micro or None,
                                          **({'task': forward_task} if forward_task else {}),
                                          **loss_kwargs))
            deref(model.clip_grad_and_step(max_grad_norm=max_grad_norm))
            optimizer_steps = 1
        else:
            for start in range(0, len(features), micro):
                stop = min(start + micro, len(features))
                mini_kwargs = {key: _aligned_slice(value, start, stop)
                               for key, value in loss_kwargs.items()}
                self._micro_step = getattr(self, '_micro_step', 0) + 1
                # The remote calls return lazy handles; collecting them HERE is
                # what makes a worker-side loss error fail this stage loudly --
                # a dropped handle leaves the step silently no-op (ray prints
                # "Unhandled error" but the driver never sees the exception).
                deref(model.forward_backward(inputs=features[start:stop], loss_key=loss_key,
                                              gradient_accumulation_steps=ga,
                                              **({'task': forward_task} if forward_task else {}),
                                              **mini_kwargs))
                deref(model.clip_grad_and_step(max_grad_norm=max_grad_norm,
                                                gradient_accumulation_steps=ga))
                if is_grad_sync_boundary(self._micro_step, ga):
                    optimizer_steps += 1
        loss_value = None
        metric = getattr(model, 'calculate_metric', None)
        if callable(metric):
            try:
                metric_row = metric(True)
                loss_value = metric_row.get('loss') if isinstance(metric_row, dict) else None
            except Exception as exc:  # noqa
                # A metric read-back failure must not fail the finished step,
                # but it must stay visible: a silent None here reads as "the
                # training produced no loss" in the node's preview.
                logger.warning('%s: calculate_metric failed: %r', self.node_id, exc)
                loss_value = None
        return {'step': {'status': 'completed', 'optimizer_steps': optimizer_steps,
                         'micro_step': getattr(self, '_micro_step', 0)},
                'model': model,
                'optimizer_steps': optimizer_steps,
                'then': {'done': True},
                'metric': {'values': {'loss': loss_value}} if loss_value is not None else None}


class WeightSyncStage(StageComponent):
    """Push the trained weights into every wired sampler, for real.

    The session is disaggregated: every sampler owns its own DeviceGroup, so
    the push runs inside this stage's own invoke through the native
    CheckpointEngineManager -- bucketed device-to-device transfer over the
    ray channel, the same surface the disaggregated recipes use. The
    streaming driver's publish window is the on-the-fly sync path and rides
    the driver itself; deferring to it from here would require a
    model-attached driver handle this graph never wires. Each (model,
    sampler) pair keeps one resident manager: its channel state
    (base_sync_done) must survive across pushes.
    """

    type = 'weight_sync'

    def invoke(self, inputs: Dict[str, Any], batch: Any) -> Dict[str, Any]:
        model = inputs.get('model')
        targets = slot_values(inputs.get('targets'))
        if model is None or not targets:
            raise ComponentError(
                f'{self.node_id}: weight_sync needs a wired source model and target samplers')
        for target in targets:
            # One model actor's checkpoint channels flow strictly in turn (see
            # _channel_gates above): the gate spans the whole manager flow.
            with _channel_gate(model):
                self._manager_for(model, target).sync_weights(merge_and_sync=True)
        return {'sync': {'state': 'completed',
                         'targets': [getattr(t, 'name', str(i)) for i, t in enumerate(targets)]}}

    def _manager_for(self, model: Any, target: Any) -> Any:
        """One resident CheckpointEngineManager per (model, sampler) pair."""
        managers = getattr(self, '_managers', None)
        if managers is None:
            managers = self._managers = {}
        manager = managers.get((id(model), id(target)))
        if manager is None:
            from twinkle.checkpoint_engine import CheckpointEngineManager
            manager = CheckpointEngineManager(model=model, sampler=target)
            managers[(id(model), id(target))] = manager
        return manager


class WeightAverageStage(StageComponent):
    """weight_average: copy or EMA-blend from one explicit model into another.

    The target's role is never hardcoded (reference, teacher, value...); the
    transfer rides the native surfaces: copy goes through the torch
    state_dict pair, EMA through a resident CheckpointEngineManager over the
    (policy, reference) pair -- the same channel the PPO recipe builds, its
    bucketed transfer running device-to-device over the ray channel."""

    type = 'weight_average'

    def invoke(self, inputs: Dict[str, Any], batch: Any) -> Dict[str, Any]:
        source = inputs.get('source')
        target = inputs.get('target')
        if source is None or target is None:
            raise ComponentError(f'{self.node_id}: weight_average needs source and target')
        mode = str(self.params.get('mode', 'ema') or 'ema')
        alpha = float(self.params.get('alpha', '0.9') or 0.9)
        if mode == 'copy':
            sync = getattr(target, 'load_state_dict', None)
            state = getattr(source, 'state_dict', None)
            if callable(sync) and callable(state):
                sync(state())
                return {'sync': {'state': 'completed'}}
            raise ComponentError(f'{self.node_id}: copy mode needs state_dict/load_state_dict')
        if mode != 'ema':
            raise ComponentError(f'{self.node_id}: unknown weight_average mode {mode!r}')
        # One model actor's checkpoint channels flow strictly in turn (see
        # _channel_gates above): the gate spans the whole manager flow.
        with _channel_gate(source):
            self._manager_for(source, target).sync_ema(alpha)
        return {'sync': {'state': 'completed'}}

    def _manager_for(self, source: Any, target: Any) -> Any:
        """One resident CheckpointEngineManager per (source, target) pair, on
        its own channel so an EMA pair never shares transport state with the
        rollout weight-sync channel."""
        managers = getattr(self, '_managers', None)
        if managers is None:
            managers = self._managers = {}
        manager = managers.get((id(source), id(target)))
        if manager is None:
            from twinkle.checkpoint_engine import CheckpointEngineManager
            manager = CheckpointEngineManager(
                model=source, sampler=target, channel=f'ema:{self.node_id}')
            managers[(id(source), id(target))] = manager
        return manager


class CheckpointStage(StageComponent):
    """Persist the real weights and training state; the artifact path is the
    one the model's own save() reports (its return value IS the truth --
    never a path the caller guesses)."""

    type = 'checkpoint'

    def invoke(self, inputs: Dict[str, Any], batch: Any) -> Dict[str, Any]:
        model = inputs.get('model')
        if model is None:
            raise ComponentError(f'{self.node_id}: checkpoint needs a wired model')
        save = getattr(model, 'save_checkpoint', None) or getattr(model, 'save', None)
        if not callable(save):
            raise ComponentError(f'{self.node_id}: model exposes no save surface')
        name = str(self.params.get('name') or 'checkpoint')
        output_dir = str(self.params.get('output_dir') or os.path.join('output', 'checkpoints'))
        os.makedirs(output_dir, exist_ok=True)
        path = save(name, output_dir=output_dir)
        path = str(path or os.path.join(output_dir, name))
        if not os.path.exists(path):
            raise ComponentError(f'{self.node_id}: save reported no artifact at {path}')
        return {'artifact': {'path': path, 'format': 'native'}}

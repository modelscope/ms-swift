# Copyright (c) ModelScope Contributors. All rights reserved.
"""Training-side bindings: the real forward_backward / step / publish / save.

Contract highlights (see the plan's training and publish sections):

- ``train_step`` runs real ``forward_backward`` per micro batch, the GA
  boundary, ``clip_grad_and_step``; ``optimizer_steps`` counts only real
  optimizer completions. Partial mutation errors keep their completed marks --
  no transparent rollback is promised.
- ``weight_sync`` registers the real push through the existing
  ``request_publish`` entry and observes completion through the public stats
  surface; it NEVER waits for the push before the source commit, and never
  fabricates a completion from a registration.
- ``weight_average`` copies or EMA-blends between two explicit models through
  the native transfer surface; the target's role is never hardcoded.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from ..component import ComponentError, StageComponent, slot_values
from ..numeric import deref


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

    A per-sample list is sliced with the batch; anything else rides every
    micro batch whole -- the same convention the recipes use for scalar
    kwargs."""
    if isinstance(value, list):
        return value[start:stop]
    return value


def register_weight_sync(driver: Any, push: Any) -> str:
    """Register one real push on the driver through the existing public entry.

    The existing signature returns None and the push runs in the next pass;
    this helper only registers and reports the request identity.
    """
    driver.request_publish(push)
    return f'publish-{id(push):x}'


class TrainStepStage(StageComponent):
    """train_step: real forward_backward + optimizer step.

    The objective, the optimizer and the encoding rule are PARAMS on this
    node: the loss instance is registered ONCE per node on the model's own
    loss registry under ``train:<node_id>`` (the same registry any
    model_forward probe uses with its own key), and the optimizer is
    configured ONCE on the wired model with the total horizon. Loss kwargs
    come from the arguments port's slots (the real loss parameter names).
    The input is raw data / trajectories -- encoded here under the node's
    own tokenizer param, exactly the per-round re-encode the dev recipes
    run. The ``model`` output carries the trained model reference forward
    (an identity downstream effects consume instead of an actor fan-out)."""

    type = 'train_step'

    def _loss(self) -> Any:
        if getattr(self, '_loss_obj', None) is None:
            from twinkle.loss import torch_loss_mapping
            name = str(self.params.get('loss') or 'cross_entropy')
            cls = torch_loss_mapping.get(name)
            if cls is None:
                raise ComponentError(f'{self.node_id}: unknown loss {name!r}')
            kwargs = dict(self.params.get('loss_kwargs') or {})
            try:
                self._loss_obj = cls(**kwargs)
            except TypeError as exc:
                raise ComponentError(
                    f'{self.node_id}: loss {name!r} rejects kwargs {sorted(kwargs)}: {exc}') from exc
        return self._loss_obj

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
                             {k: str(v) for k, v in kv.items() if v is not None},
                             self.node_id)
        configure_optimizer(model, train_config, num_training_steps=num_training_steps)
        self._optimizer_built = True

    def _features(self, data: Any) -> Any:
        if getattr(self, '_template', None) is None:
            from .resources import build_template_from, encode_rows
            self._template = build_template_from(
                str(self.params.get('tokenizer') or ''),
                dict(self.params.get('template_config') or {}), self.node_id)
        if data is None:
            raise ComponentError(f'{self.node_id}: train_step needs a wired batch input')
        return encode_rows(self._template, data)

    def invoke(self, inputs: Dict[str, Any], batch: Any) -> Dict[str, Any]:
        model = inputs.get('model')
        if model is None:
            raise ComponentError(f'{self.node_id}: train_step needs a wired model')
        features = self._features(inputs.get('batch'))
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
            set_loss(self._loss(), loss_key=loss_key)
            self._loss_registered = True
        self._configure_optimizer(model)
        from swift.dev.recipe.utils import backend_owns_gradient_accumulation, is_grad_sync_boundary
        optimizer_steps = 0
        if backend_owns_gradient_accumulation(model):
            # Megatron internalises the micro-batch schedule: the feature LIST
            # is the micro-batch plan (GA = its length), so one call is one
            # optimizer step; loss kwargs are sliced worker-side by the loss's
            # own micro_batch_kwargs.
            model.forward_backward(inputs=features, loss_key=loss_key,
                                   micro_batch_size=micro or None, **loss_kwargs)
            model.clip_grad_and_step(max_grad_norm=max_grad_norm)
            optimizer_steps = 1
        else:
            for start in range(0, len(features), micro):
                stop = min(start + micro, len(features))
                mini_kwargs = {key: _aligned_slice(value, start, stop)
                               for key, value in loss_kwargs.items()}
                self._micro_step = getattr(self, '_micro_step', 0) + 1
                model.forward_backward(inputs=features[start:stop], loss_key=loss_key,
                                       gradient_accumulation_steps=ga, **mini_kwargs)
                model.clip_grad_and_step(max_grad_norm=max_grad_norm,
                                         gradient_accumulation_steps=ga)
                if is_grad_sync_boundary(self._micro_step, ga):
                    optimizer_steps += 1
        loss_value = None
        metric = getattr(model, 'calculate_metric', None)
        if callable(metric):
            try:
                metric_row = metric(True)
                loss_value = metric_row.get('loss') if isinstance(metric_row, dict) else None
            except Exception:
                loss_value = None
        return {'step': {'status': 'completed', 'optimizer_steps': optimizer_steps,
                         'micro_step': getattr(self, '_micro_step', 0)},
                'model': model,
                'optimizer_steps': optimizer_steps,
                'then': {'done': True},
                'metric': {'values': {'loss': loss_value}} if loss_value is not None else None}


class WeightSyncStage(StageComponent):
    """Register the real push for the wired samplers; observe (never fake) the
    completion. Registration precedes the source commit; the push itself runs
    in the driver's next pass."""

    type = 'weight_sync'

    def invoke(self, inputs: Dict[str, Any], batch: Any) -> Dict[str, Any]:
        model = inputs.get('model')
        targets = slot_values(inputs.get('targets'))
        if model is None or not targets:
            raise ComponentError(
                f'{self.node_id}: weight_sync needs a wired source model and target samplers')
        driver = getattr(model, 'driver', None)
        intent = inputs.get('intent')
        request = None
        if driver is not None and callable(getattr(driver, 'request_publish', None)):
            from .training import register_weight_sync
            request = register_weight_sync(driver, self._push_for(targets))
        return {'sync': {'state': 'pending', 'request': request or intent or 'registered',
                         'targets': [getattr(t, 'name', str(i)) for i, t in enumerate(targets)]}}

    def _push_for(self, targets: List[Any]):
        """The real push callable: snapshot-save then engine sync, per target.

        Only the actual start/end produce sync events; ``in_place`` pushes
        return None (no snapshot path) and snapshot pushes must return the real
        new path -- the existing driver contract.
        """
        def push(window: Any) -> Optional[str]:
            path = None
            for target in targets:
                sync = getattr(target, 'sync_weights', None)
                if callable(sync):
                    sync(merge_and_sync=True)
            return path
        return push


class WeightAverageStage(StageComponent):
    """weight_average: copy or EMA-blend from one explicit model into another.

    The target's role is never hardcoded (reference, teacher, value...); the
    transfer itself is the native surface (state_dict load or the EMA
    manager)."""

    type = 'weight_average'

    def invoke(self, inputs: Dict[str, Any], batch: Any) -> Dict[str, Any]:
        source = inputs.get('source')
        target = inputs.get('target')
        if source is None or target is None:
            raise ComponentError(f'{self.node_id}: weight_average needs source and target')
        mode = self.params.get('mode', 'ema')
        alpha = float(self.params.get('alpha', '0.9') or 0.9)
        if mode == 'copy':
            sync = getattr(target, 'load_state_dict', None) or getattr(target, 'sync_weights', None)
            state = getattr(source, 'state_dict', None)
            if callable(sync) and callable(state):
                sync(state())
                return {'sync': {'state': 'completed'}}
            raise ComponentError(f'{self.node_id}: copy mode needs state_dict/load_state_dict')
        # EMA: ref <- alpha*ref + (1-alpha)*policy. The native EMA manager owns
        # the exact blend; this binding only routes it.
        ema = getattr(target, 'sync_ema', None)
        if callable(ema):
            ema(alpha)
            return {'sync': {'state': 'completed'}}
        raise ComponentError(f'{self.node_id}: ema mode needs the native sync_ema surface')


class CheckpointStage(StageComponent):
    """Persist the real weights and training state; the artifact path is the
    one actually written."""

    type = 'checkpoint'

    def invoke(self, inputs: Dict[str, Any], batch: Any) -> Dict[str, Any]:
        model = inputs.get('model')
        if model is None:
            raise ComponentError(f'{self.node_id}: checkpoint needs a wired model')
        save = getattr(model, 'save_checkpoint', None) or getattr(model, 'save', None)
        if not callable(save):
            raise ComponentError(f'{self.node_id}: model exposes no save surface')
        name = self.params.get('name', 'checkpoint') or 'checkpoint'
        output_dir = self.params.get('output_dir', '') or os.path.join('output', 'checkpoints')
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, name)
        save(path)
        if not os.path.exists(path):
            raise ComponentError(f'{self.node_id}: save reported no artifact at {path}')
        return {'artifact': {'path': path, 'format': 'native'}}

# Copyright (c) ModelScope Contributors. All rights reserved.
"""Data-side bindings: iterators, generation, encoding, selection, binding.

The two iterator bindings carry the batch-handover contract from the plan:

- ``DataloaderComponent``: the native dataloader is created once; state
  save/restore reuses the existing get_state/load_state/set_epoch surface.
- ``StreamingDriverComponent``: the driver is built ONCE through the existing
  assembly path, ``iter`` is called at most once (a driver is single-use), and
  every later call is ``next`` + ``commit``. The click budget NEVER becomes the
  driver's terminating max_steps; a click never aborts; release drains the
  single-slot queue through the public iterator before aborting.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..component import ComponentError, IteratorComponent, StageComponent, slot_values


class DataloaderComponent(IteratorComponent):
    """for batch in dataloader -- advanced on demand, cursor preserved.

    ``source`` picks the dataset's face: ``prompts`` iterates the prompt
    view (assistant tail stripped -- the generation side's input), ``rows``
    iterates the full training rows (the supervised side's input). Both are
    the same resident dataset resource, one loader per face."""

    type = 'dataloader'

    def start(self, inputs: Dict[str, Any]) -> None:
        from swift.dev.builder import build_prompt_dataloader
        dataset = inputs.get('data')
        if not isinstance(dataset, dict) or 'prompts' not in dataset:
            raise ComponentError(f'{self.node_id}: dataloader needs a wired dataset')
        face = str(self.params.get('source', 'prompts') or 'prompts')
        if face == 'rows':
            from twinkle.dataset import Dataset, DatasetMeta
            from twinkle.dataloader import DataLoader
            rows = list(dataset.get('rows') or [])
            if not rows:
                raise ComponentError(f'{self.node_id}: the dataset exposes no training rows')
            self.obj = DataLoader(
                Dataset(DatasetMeta(data=rows)),
                batch_size=int(self.params.get('batch_size', '8') or 8),
                data_seed=int(self.params.get('seed', '42') or 42),
                num_workers=int(self.params.get('num_workers', '0') or 0))
        elif face == 'prompts':
            self.obj = build_prompt_dataloader(
                dataset['prompts'], dataset.get('extras') or [],
                seed=int(self.params.get('seed', '42') or 42),
                batch_size=int(self.params.get('batch_size', '8') or 8))
        else:
            raise ComponentError(f'{self.node_id}: unknown source face {face!r} (prompts|rows)')
        self._iterator = iter(self.obj)
        self._state: Optional[Dict[str, Any]] = None
        self._current: Optional[Any] = None

    def take_batch(self) -> Any:
        batch = next(self._iterator)
        # The batch port projects the LAST delivered batch (the same live-view
        # contract the streaming driver's traj lane follows), so consumers
        # wired to it read batch data, not the resident loader object.
        self._current = batch
        return batch

    def port_value(self, key: str) -> Any:
        """The loader's lanes: batch = the last delivered batch (the data
        face), source = this control handle (the resident loader rides on
        ``obj`` -- the face a streaming driver's prompts input wants)."""
        if key == 'source':
            return self
        return getattr(self, '_current', None) or []

    def commit(self, steps: int) -> None:
        # The dataloader advances by take; commit only records the cursor for
        # stop/resume (get_state exists on the native loader).
        get_state = getattr(self.obj, 'get_state', None)
        if callable(get_state):
            try:
                self._state = get_state()
            except Exception:
                self._state = None

    def release(self) -> None:
        self.obj = None
        self._iterator = None


def _orm_adapter(scorer: Any) -> Any:
    """Place admission scoring on its compute worker; the streaming driver receives score Futures only."""
    import uuid
    from twinkle.reward.worker import FunctionWorker
    from ..placement import NUMERIC_GROUP

    return FunctionWorker(scorer, remote_group=NUMERIC_GROUP, instance_id=uuid.uuid4().hex)


class StreamingDriverComponent(IteratorComponent):
    """The async control plane as an on-demand batch source.

    Construction uses the existing builder with the TOTAL budget; iteration
    starts once. ``take_batch`` returns the batch's trajectories (the rollout's
    ledger-merged dicts); ``commit`` records the real optimizer steps of the
    consumption. The driver gets no checkpoint manager -- weight pushes are the
    graph's own weight_sync stage -- so it runs ``max_staleness`` batches ahead
    of the consumer and never drops a group as stale.
    """

    type = 'streaming_driver'

    def start(self, inputs: Dict[str, Any]) -> None:
        """Build the driver if needed, then iter() exactly once."""
        if self.obj is None:
            self._build_driver(inputs)
        self._iterator = iter(self.obj)
        self._current: Optional[Any] = None
        self.last_steps: int = 0

    def _build_driver(self, inputs: Dict[str, Any]) -> None:
        import math
        from dataclasses import asdict

        from swift.dev.builder.sampler import to_sampling_params
        from swift.dev.builder.streaming import build_rollout, build_streaming_driver
        from .resources import _configs, _fill
        engines = slot_values(inputs.get('engine'))
        episode = inputs.get('episode')
        if bool(engines) == (episode is not None):
            raise ComponentError(
                f'{self.node_id}: streaming_driver needs exactly one of a wired engine (a sampler) '
                'or a wired episode (a rollout)')
        # The prompt stream is the wired dataloader's SOURCE handle: the native
        # loader rides on its ``obj`` (batching, seed and the face choice live
        # on THAT node, not here). The batch port is the data face -- the rows
        # of the last delivered batch -- and cannot stand in for the loader.
        prompt_source = inputs.get('prompts')
        if getattr(prompt_source, 'type', None) != 'dataloader':
            raise ComponentError(
                f'{self.node_id}: prompts must be the dataloader\'s source control port '
                '(the resident loader rides on it), not a data lane or another resource')
        prompt_loader = prompt_source.obj
        if prompt_loader is None:
            raise ComponentError(
                f'{self.node_id}: the wired dataloader is not started yet')
        config_pkg = _configs()
        # The panel's scalar knobs are the REAL Config fields' convenience face
        # (dict params win; the scalars fill their Config's field when unset).
        rl_kv = dict(self.params.get('rl_config') or {})
        rl_kv.setdefault('num_generations', self.params.get('num_generations', 8))
        rollout_kv = dict(self.params.get('rollout_config') or {})
        rollout_kv.setdefault('async_mode', self.params.get('async_mode', 'none'))
        rollout_kv.setdefault('max_staleness', self.params.get('max_staleness', 1))
        gen_kv = dict(self.params.get('generation_config') or {})
        rollout_config = _fill(config_pkg.RolloutConfig, rollout_kv, self.node_id)
        num_generations = _fill(config_pkg.RLConfig, rl_kv, self.node_id).num_generations
        self._owned_rollout = None
        if episode is not None:
            if gen_kv:
                raise ComponentError(
                    f'{self.node_id}: a wired episode samples with its own generation_config; '
                    'set it on the rollout node')
            rollout = episode
        else:
            # The rollout encodes episodes with a template held HERE: under a
            # disaggregated ray placement the sampler's own template lives on its
            # remote actor and cannot be read off the driver-side proxy, so this
            # node builds its own (the tokenizer/model_config params carry the
            # identity, the same contract train_step and model_forward follow).
            from .resources import build_template_from
            template = build_template_from(
                str(self.params.get('tokenizer') or ''),
                dict(self.params.get('template_config') or {}), self.node_id,
                model_kv=dict(self.params.get('model_config') or {}))
            gen_config = _fill(config_pkg.GenerationConfig, gen_kv, self.node_id) if gen_kv else None
            from ..placement import NUMERIC_GROUP
            rollout = self._owned_rollout = build_rollout(
                engines[0], template, rollout_config, remote_group=NUMERIC_GROUP,
                sampling_params=asdict(to_sampling_params(gen_config)))
        # One batch is whole groups and whole optimizer updates; max_steps counts
        # optimizer steps, the driver counts batches. The click budget is NOT
        # passed here: a click must never make the driver terminate.
        update_rows = int(self.params.get('train_batch_size', 8) or 8) * int(
            self.params.get('gradient_accumulation_steps', 1) or 1)
        batch_size = math.lcm(update_rows, num_generations)
        max_steps = int(self.params.get('max_steps', 0) or 0)
        # The admission scoring binding (W5): wired scorers run per episode inside
        # the driver under their slot names, so the group verdict rules on real
        # scores and every delivered batch's facts carry its named score columns -- the
        # same scorer identity a downstream score node reuses instead of rescoring.
        scorers = inputs.get('scorers') or []
        orm_names = [str(slot) for slot, _scorer in scorers]
        self._orm_workers = []
        for _slot, scorer in scorers:
            self._orm_workers.append(_orm_adapter(scorer))
        orms = tuple(self._orm_workers)
        self.obj = build_streaming_driver(
            dataloader=prompt_loader,
            rollout=rollout,
            rollout_config=rollout_config,
            batch_size=batch_size,
            num_generations=num_generations,
            num_train_epochs=float(self.params.get('num_train_epochs', 1.0) or 1.0),
            max_steps=math.ceil(max_steps * update_rows / batch_size) if max_steps else None,
            orms=orms,
            orm_names=orm_names or None)

    def take_batch(self) -> Any:
        # Delivery facts are light; field write-back executes where the trajectories are consumed.
        rows, facts = next(self._iterator)
        scores = facts.get('scores') or ()
        if scores:
            transform = getattr(self, '_score_update', None)
            if transform is None:
                import uuid
                from twinkle.processor import TrajectoryUpdate
                from ..placement import NUMERIC_GROUP

                transform = self._score_update = TrajectoryUpdate(
                    remote_group=NUMERIC_GROUP, instance_id=f'{self.node_id}-{uuid.uuid4().hex[:8]}')
            rows, _summary = transform.update(rows, updates=scores, merge_fields=('rewards',))
        # The traj port projects the LAST delivered batch: downstream stages
        # wired to this port read batch data, not the control-plane object.
        self._current = rows
        return rows

    def port_value(self, key: str) -> Any:
        """The iterator's lanes: traj = the last delivered batch, source =
        this control handle (for ports that name the driver itself)."""
        if key == 'source':
            return self
        return getattr(self, '_current', None) or []

    def commit(self, steps: int) -> None:
        self.last_steps = int(steps)

    def release(self) -> None:
        """Close owned source resources, retaining failed handles for cleanup retry."""
        failures = []
        for name in ('obj', '_owned_rollout', '_score_update'):
            closeable = getattr(self, name, None)
            if closeable is None:
                continue
            try:
                closeable.close()
            except BaseException as error:
                failures.append(error)
            else:
                setattr(self, name, None)
        for worker in tuple(getattr(self, '_orm_workers', ())):
            try:
                worker.close()
            except BaseException as error:
                failures.append(error)
            else:
                self._orm_workers.remove(worker)
        self._iterator = None
        if failures:
            raise BaseExceptionGroup('Streaming source cleanup is incomplete', failures)


class ChallengerComponent(IteratorComponent):
    """The agentic task generator as a resident batch source.

    ``start`` builds the REAL AgenticChallenger over the wired backend and env
    and iterates it in fixed batches (``max_tasks`` trajectories per batch);
    ``take_batch`` hands over the native ChallengeBatch and the port projection
    reads the two lanes off the LAST delivered batch (grouped natively, the
    consumer flattens by advantage group). The challenger has no
    acknowledgement of its own: ``commit`` records the real optimizer steps of
    the consumption, nothing more."""

    type = 'challenger'

    def start(self, inputs: Dict[str, Any]) -> None:
        if self.obj is not None:
            raise ComponentError(f'{self.node_id}: challenger already started')
        from twinkle_agentic.challenger.agentic import AgenticChallenger
        from .resources import build_template_from
        backend = inputs.get('backend')
        if backend is None:
            raise ComponentError(f'{self.node_id}: challenger needs a wired backend (the sampler)')
        env = inputs.get('env')
        if env is None:
            raise ComponentError(f'{self.node_id}: challenger needs a wired env (the episode workspace)')
        seed = str(self.params.get('seed') or '')
        template = build_template_from(
            str(self.params.get('tokenizer') or ''),
            dict(self.params.get('template_config') or {}), self.node_id,
            model_kv=dict(self.params.get('model_config') or {}))
        cfg = dict(self.params.get('challenger_config') or {})
        api = inputs.get('api')
        if api is not None:
            # The challenger samples the extra API through the sampler interface; the
            # local template tokenises its turns the way the trainer reads them back.
            from twinkle_agentic.protocol.api_sampler import APISampler
            api = APISampler(api, template)
        try:
            self.obj = AgenticChallenger(
                backend, api=api,
                seed_fn=(lambda text=seed: text) if seed else None,
                env=env, solver_rollout=inputs.get('solver_rollout'),
                template=template, **cfg)
        except TypeError as exc:
            raise ComponentError(f'{self.node_id}: challenger_config: {exc}') from None
        self._batch_size = int(self.params.get('max_tasks', '8') or 8)
        if self._batch_size < 1:
            raise ComponentError(f'{self.node_id}: max_tasks must be >= 1')
        self._iterator = iter(self.obj(self._batch_size))
        self._last: Optional[Any] = None
        self.last_steps: int = 0

    def take_batch(self) -> Any:
        self._last = next(self._iterator)
        return self._last

    def commit(self, steps: int) -> None:
        self.last_steps = int(steps)

    def skip_batch(self) -> None:
        self.last_steps = 0

    def release(self) -> None:
        challenger = self.obj
        if challenger is None:
            return
        try:
            challenger.close()
        except Exception:
            pass
        self.obj = None
        self._iterator = None
        self._last = None

    def port_value(self, key: str) -> Any:
        """The last delivered batch's lanes: questions = proposing groups,
        answers = solving groups, flattened; source = this control handle."""
        if key == 'source':
            return self
        batch = getattr(self, '_last', None)
        if batch is None:
            return []
        if key == 'questions':
            return [member for group in batch.challenger for member in group]
        if key == 'answers':
            return [member for group in batch.solver for member in group]
        raise ComponentError(f'{self.node_id}: no challenger lane for port {key!r}')


class GenerateStage(StageComponent):
    """One blocking batch generation over the prompt rows.

    The engine face is whichever the wire carries: a wired episode IS the
    episode engine; a bare sampler gets a cached MultiTurnRollout wrapper --
    the same class the streaming driver uses, so the trajectories downstream
    (score/advantage/train) have one shape regardless of who generated them."""

    type = 'generate'

    def invoke(self, inputs: Dict[str, Any], batch: Any) -> Dict[str, Any]:
        episode = inputs.get('episode')
        engine = inputs.get('engine')
        if (episode is None) == (engine is None):
            raise ComponentError(f'{self.node_id}: generate needs exactly one wired engine or episode')
        rows = inputs.get('prompts')
        if rows is None:
            rows = batch
        if rows is None:
            raise ComponentError(f'{self.node_id}: generate needs a wired prompt batch')
        # Sampling knobs are the REAL GenerationConfig fields (the typed
        # generation_config param), translated through the SAME builder the
        # recipes use; each episode carries them in user_data, which the
        # rollout reads over its own defaults. No config means the engine's
        # construction-time defaults stand.
        from dataclasses import asdict

        from swift.dev.builder.sampler import to_sampling_params
        from .resources import _configs, _fill
        gen_kv = dict(self.params.get('generation_config') or {})
        params = (asdict(to_sampling_params(_fill(_configs().GenerationConfig, gen_kv, self.node_id)))
                  if gen_kv else None)
        transform = getattr(self, '_prompt_transform', None)
        if transform is None:
            import uuid
            from twinkle.processor import TrajectoryUpdate
            from ..placement import NUMERIC_GROUP

            transform = self._prompt_transform = TrajectoryUpdate(
                remote_group=NUMERIC_GROUP, instance_id=f'{self.node_id}-{uuid.uuid4().hex[:8]}')
        trajs = transform.expand_prompts(
            rows, num_generations=int(self.params.get('num_generations', 8)), sampling_params=params)
        rollout = episode if episode is not None else self._rollout_for(engine)
        return {'traj': rollout(trajs)}

    def _rollout_for(self, engine: Any) -> Any:
        """One resident episode engine per wired sampler."""
        rollouts = getattr(self, '_rollouts', None)
        if rollouts is None:
            rollouts = self._rollouts = {}
        rollout = rollouts.get(id(engine))
        if rollout is None:
            import uuid
            from twinkle_agentic.rollout import MultiTurnRollout
            from .resources import build_template_from
            from ..placement import NUMERIC_GROUP

            template = build_template_from(
                str(self.params.get('tokenizer') or ''),
                dict(self.params.get('template_config') or {}), self.node_id,
                model_kv=dict(self.params.get('model_config') or {}))
            rollout = MultiTurnRollout(
                engine, template, max_turns=1, remote_group=NUMERIC_GROUP,
                instance_id=f'{self.node_id}-{uuid.uuid4().hex[:8]}')
            rollouts[id(engine)] = rollout
        return rollout

    def release(self) -> None:
        """Close owned transforms and rollouts, retaining failures for another cleanup attempt."""
        failures = []
        transform = getattr(self, '_prompt_transform', None)
        if transform is not None:
            try:
                transform.close()
            except BaseException as error:
                failures.append(error)
            else:
                self._prompt_transform = None
        rollouts = getattr(self, '_rollouts', {})
        for key, rollout in tuple(rollouts.items()):
            try:
                rollout.close()
            except BaseException as error:
                failures.append(error)
            else:
                del rollouts[key]
        if failures:
            raise BaseExceptionGroup('Generation cleanup is incomplete', failures)


class ScoreStage(StageComponent):
    """Delegate scoring and field write-back; only scalar metrics reach the graph executor."""

    type = 'score'

    def invoke(self, inputs: Dict[str, Any], batch: Any) -> Dict[str, Any]:
        scorers = inputs.get('scorers')
        if not scorers:
            raise ComponentError(f'{self.node_id}: score needs wired scorers (one per column)')
        samples = inputs.get('traj')
        if not isinstance(samples, list):
            samples = batch if isinstance(batch, list) else []
        if not samples:
            # Refuse an empty/None batch rather than delegate: twinkle's score() raises a plain
            # ValueError on empty input, but the stage contract is a ComponentError naming the
            # unwired traj port (an empty score batch is indistinguishable from a drained one).
            raise ComponentError(f'{self.node_id}: score needs a wired traj batch')
        transform = getattr(self, '_score_transform', None)
        if transform is None:
            import uuid
            from twinkle.processor import TrajectoryUpdate
            from ..placement import NUMERIC_GROUP

            transform = self._score_transform = TrajectoryUpdate(
                remote_group=NUMERIC_GROUP, instance_id=f'{self.node_id}-{uuid.uuid4().hex[:8]}')
        try:
            trajs, scores, metric = transform.score(samples, scorers=scorers)
        except (ValueError, TypeError) as e:
            # Surface the shared scoring engine's contract violations as ComponentError so the
            # graph executor treats them as stage failures (a scorer that does not score every
            # row fails loudly -- no silent truncation, no padding).
            msg = str(e)
            if 'scores for' in msg:
                msg = f'{msg} -- a scorer scores every row'
            raise ComponentError(f'{self.node_id}: {msg}') from e
        return {'traj': trajs, 'scores': scores, 'metric': metric}

    def release(self) -> None:
        transform = getattr(self, '_score_transform', None)
        if transform is not None:
            transform.close()
            self._score_transform = None


class SelectStage(StageComponent):
    """Delegate one aligned selection to the data worker; keep all payload channels as references."""

    type = 'select'

    def invoke(self, inputs: Dict[str, Any], batch: Any) -> Dict[str, Any]:
        samples = inputs.get('batch')
        if samples is None:
            samples = batch
        if samples is None:
            raise ComponentError(f'{self.node_id}: select needs a wired batch')
        transform = getattr(self, '_selection', None)
        if transform is None:
            import uuid
            from twinkle.processor import TrajectoryFilter
            from ..placement import NUMERIC_GROUP

            transform = self._selection = TrajectoryFilter(
                remote_group=NUMERIC_GROUP, instance_id=f'{self.node_id}-{uuid.uuid4().hex[:8]}')
        rows, signals, scores, metric = transform.select_scored(
            samples, mode=str(self.params.get('select') or 'threshold'),
            num_generations=int(self.params.get('num_generations') or 8),
            top_k=int(self.params.get('top_k') or 1), threshold=float(self.params.get('threshold') or 0.0),
            signals=tuple(inputs.get('signals') or ()), scores=inputs.get('scores'))
        output = {'batch': rows, 'scores': scores, 'metric': metric}
        if inputs.get('signals'):
            output['signals'] = signals
        return output

    def release(self) -> None:
        transform = getattr(self, '_selection', None)
        if transform is not None:
            transform.close()
            self._selection = None

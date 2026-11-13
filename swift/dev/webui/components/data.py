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
        from swift.dev.builders import build_prompt_dataloader
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
    """One wired scorer as a driver ORM callable: a Future resolving to ``[score]``.

    The driver's orm channel expects ``scorer([traj]) -> Future`` (it registers completion
    callbacks and reads ``future.result()[0]``); a webui scorer is a plain
    ``scorer(rows) -> list[float]`` callable, so the adapter resolves synchronously into a
    completed Future -- the driver's event thread never blocks on it and a raising scorer
    fails the run through the driver's own error path.
    """
    from concurrent.futures import Future

    def orm(trajs: List[Any]) -> Any:
        future: Future = Future()
        try:
            future.set_result([scorer(trajs)])
        except Exception as exc:  # noqa: BLE001 -- the driver's error path re-raises it
            future.set_exception(exc)
        return future

    return orm


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

        from swift.dev.builders.sampler import to_sampling_params
        from swift.dev.builders.streaming import build_rollout, build_streaming_driver
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
            rollout = self._owned_rollout = build_rollout(
                engines[0], template, rollout_config, sampling_params=asdict(to_sampling_params(gen_config)))
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
        orms = tuple(_orm_adapter(scorer) for _slot, scorer in scorers)
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
        # The single-return contract: one next() hands the batch as the driver delivers it --
        # the rows plus the light facts carrying the row-aligned score column. The webui
        # runs the driver IN-PROCESS, so its rows are plain dicts this component owns: the
        # admission scores are merged onto row copies HERE (the write-back post sits with
        # the executor holding the row bodies) and the downstream row-shaped stages -- the
        # score stage's no-rescore reuse, select's criterion, the advantage seam -- keep
        # reading the named columns off the rows.
        rows, facts = next(self._iterator)
        scores = facts.get('scores') or ()
        if scores:
            merged = [dict(row) if isinstance(row, dict) else row for row in rows]
            for row, score in zip(merged, scores):
                if not isinstance(row, dict):
                    continue
                rewards = dict(row.get('rewards') or {})
                rewards.update(score.get('rewards') or {})
                row['rewards'] = rewards
                if score.get('prm_segments') is not None:
                    row['prm_segments'] = score['prm_segments']
            rows = merged
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
        """Terminate the source: stop the driver, then the rollout this node built."""
        driver = self.obj
        if driver is None:
            return
        self.obj = None
        self._iterator = None
        for closeable in (driver, getattr(self, '_owned_rollout', None)):
            if closeable is None:
                continue
            try:
                closeable.close()
            except Exception:
                pass
        self._owned_rollout = None


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
        from twinkle.data_format import attach_user_data

        from swift.dev.utils.scoring import collect
        episode = inputs.get('episode')
        engine = inputs.get('engine')
        if (episode is None) == (engine is None):
            raise ComponentError(f'{self.node_id}: generate needs exactly one wired engine or episode')
        rows = batch if isinstance(batch, list) else (batch or {}).get('rows', [])
        if not rows:
            raise ComponentError(f'{self.node_id}: generate received an empty batch')
        # Sampling knobs are the REAL GenerationConfig fields (the typed
        # generation_config param), translated through the SAME builder the
        # recipes use; each episode carries them in user_data, which the
        # rollout reads over its own defaults. No config means the engine's
        # construction-time defaults stand.
        from dataclasses import asdict

        from swift.dev.builders.sampler import to_sampling_params
        from .resources import _configs, _fill
        gen_kv = dict(self.params.get('generation_config') or {})
        if gen_kv:
            params = asdict(to_sampling_params(_fill(_configs().GenerationConfig, gen_kv, self.node_id)))
            rows = [attach_user_data(row, sampling_params=params) for row in rows]
        num_generations = int(self.params.get('num_generations', '8') or 8)
        # One episode per (row, generation): a group is num_generations consecutive trajectories.
        trajs = [dict(row) for row in rows for _ in range(num_generations)]
        rollout = episode if episode is not None else self._rollout_for(engine)
        return {'traj': collect(rollout(trajs))}

    def _rollout_for(self, engine: Any) -> Any:
        """One resident episode engine per wired sampler."""
        rollouts = getattr(self, '_rollouts', None)
        if rollouts is None:
            rollouts = self._rollouts = {}
        rollout = rollouts.get(id(engine))
        if rollout is None:
            from twinkle_agentic.rollout import MultiTurnRollout
            rollout = MultiTurnRollout(engine, getattr(engine, 'template', None))
            rollouts[id(engine)] = rollout
        return rollout


def _slice_output(value: Any, keep: List[int]) -> Any:
    """Slice one MODEL_OUTPUT value's per-row channels by the selection mask.

    A whole ModelOutput's data channels (and its per-row identity lists) carry one entry
    per row: they are sliced by ``keep``. ``field_frames`` is per-FIELD metadata and
    scalars ride whole -- neither is row-indexed. A bare per-row list slices directly.
    """
    value = _deref(value)
    if isinstance(value, dict):
        sliced = {}
        for key, item in value.items():
            if key == 'field_frames':
                sliced[key] = item
            elif isinstance(item, list):
                sliced[key] = [item[i] for i in keep]
            else:
                sliced[key] = item
        return sliced
    if isinstance(value, list):
        return [value[i] for i in keep]
    return value


def _deref(value: Any) -> Any:
    """Resolve a maybe-lazy remote result to its value (the shared dual-mode shim)."""
    from swift.dev.utils.scoring import collect

    return collect(value)


class ScoreStage(StageComponent):
    """Actually call r(tau): every wired scorer contributes one named column.

    The scorers arrive on the many-port as ``(slot, scorer)`` pairs -- the slot
    names its column. Each scorer is called once over the SAME row protocol;
    the named columns are written onto copies of the trajectories as
    ``rewards.orm`` (a scorer that does not score every row fails loudly -- no
    silent truncation, no padding). Rows already carrying a slot's column
    from the driver's admission scoring are NOT rescored: one episode, one
    scorer, one score. The metric port carries this batch's per-column means.
    """

    type = 'score'

    def invoke(self, inputs: Dict[str, Any], batch: Any) -> Dict[str, Any]:
        scorers = inputs.get('scorers')
        if not scorers:
            raise ComponentError(f'{self.node_id}: score needs wired scorers (one per column)')
        # The sample list rides the wired traj port when the frame source is
        # a driver or generate (trajectory rows); the positional batch is the
        # loader shape. The port is single-entry: its value IS the sample list.
        port_samples = inputs.get('traj')
        samples = port_samples if isinstance(port_samples, list) else (
            batch if isinstance(batch, list) else [])
        # Every producer (generate, the streaming driver, the challenger lanes)
        # hands over trajectory dicts -- the scorers' native input, passed verbatim.
        rows = [sample for sample in samples if isinstance(sample, dict)]
        if len(rows) != len(samples):
            raise ComponentError(f'{self.node_id}: score takes trajectory dicts')
        columns: List[List[Any]] = []
        scorer_ids: List[str] = []
        new_rows = [dict(row) for row in rows]
        for slot, scorer in scorers:
            if not callable(scorer):
                raise ComponentError(f'{self.node_id}: scorer slot {slot!r} is not callable')
            name = str(slot)
            has_column = [name in ((row.get('rewards') or {}).get('orm') or {}) for row in rows]
            if all(has_column):
                # The admission binding already scored every row under this slot: reuse,
                # never rescore (overwriting would have to be an explicit request).
                column = [((row.get('rewards') or {}).get('orm') or {}).get(name) for row in rows]
            elif any(has_column):
                raise ComponentError(
                    f'{self.node_id}: scorer {name!r} has already scored SOME of the rows '
                    '(the admission binding); a partial column is inconsistent -- either '
                    'the driver scored the whole batch or none of it.')
            else:
                rewards = scorer(rows)
                column = list(rewards) if rewards is not None else []
                if len(column) != len(rows):
                    raise ComponentError(
                        f'{self.node_id}: scorer {name!r} returned {len(column)} scores '
                        f'for {len(rows)} rows -- a scorer scores every row')
            for row, value in zip(new_rows, column):
                rewards = dict(row.get('rewards') or {})
                orm = dict(rewards.get('orm') or {})
                orm[name] = value
                rewards['orm'] = orm
                row['rewards'] = rewards
            columns.append(column)
            scorer_ids.append(name)
        # The metric port's summary: one mean per named column; None / nan counted separately.
        values: Dict[str, Any] = {}
        invalid = 0
        for name, column in zip(scorer_ids, columns):
            finite = [float(v) for v in column if v is not None and v == v]
            invalid += len(column) - len(finite)
            values[name] = (sum(finite) / len(finite)) if finite else None
        return {'traj': new_rows,
                'metric': {'values': values, 'invalid_count': invalid}}


class SelectStage(StageComponent):
    """Keep rows / whole groups by the declared criterion; every aligned
    ModelOutput lane is co-selected with the same mask.

    The criterion is the trajectories' single named score column
    (``rewards.orm``); a multi-column reward is an ambiguous criterion --
    combine the columns first. The kept rows keep their rewards (nothing is
    stripped), and each wired signal lane's rows are sliced by the same mask.
    """

    type = 'select'

    def invoke(self, inputs: Dict[str, Any], batch: Any) -> Dict[str, Any]:
        port_batch = inputs.get('batch')
        samples = port_batch if isinstance(port_batch, list) else (
            batch if isinstance(batch, list) else [])
        mode = self.params.get('select', 'best_of_n')
        num_generations = int(self.params.get('num_generations', '8') or 8)
        if not samples:
            raise ComponentError(f'{self.node_id}: select received an empty batch')
        # The criterion column: every row carries the SAME named scorer set, and the
        # set has exactly one column -- a wider set is ambiguous.
        column_sets = {tuple(sorted((row.get('rewards') or {}).get('orm') or {})) for row in samples}
        if len(column_sets) != 1 or not next(iter(column_sets), ()):
            raise ComponentError(
                f'{self.node_id}: select needs the trajectories to carry one named score '
                'column (rewards.orm); score them first')
        names = next(iter(column_sets))
        if len(names) > 1:
            raise ComponentError(
                f'{self.node_id}: select received a multi-column reward ({len(names)} scorers); '
                'it selects on ONE criterion. Combine the columns into a single score first.')
        rewards = [((row.get('rewards') or {}).get('orm') or {}).get(names[0]) for row in samples]
        keep: List[int] = []
        if mode in ('best_of_n', 'top_k'):
            top_k = 1 if mode == 'best_of_n' else int(self.params.get('top_k', '1') or 1)
            for start in range(0, len(samples), num_generations):
                group = list(range(start, min(start + num_generations, len(samples))))
                ranked = sorted(group, key=lambda i: -(rewards[i] or 0.0))
                keep.extend(ranked[:top_k])
        elif mode == 'threshold':
            threshold = float(self.params.get('threshold', '0.0') or 0.0)
            keep = [i for i, value in enumerate(rewards)
                    if value is not None and value == value and value >= threshold]
        else:
            raise ComponentError(f'{self.node_id}: unknown select mode {mode!r}')
        kept = [samples[i] for i in keep]
        kept_signals = None
        signals = inputs.get('signals')
        if signals:
            kept_signals = [_slice_output(value, keep) for _slot, value in signals]
        out: Dict[str, Any] = {'batch': kept,
                               'metric': {'values': {'kept': len(keep), 'dropped': len(samples) - len(keep)},
                                          'invalid_count': 0}}
        if kept_signals is not None:
            out['signals'] = kept_signals
        return out

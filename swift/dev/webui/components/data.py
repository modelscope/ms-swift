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

    def take_batch(self) -> Any:
        return next(self._iterator)

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


class StreamingDriverComponent(IteratorComponent):
    """The async control plane as an on-demand batch source.

    Construction uses the existing assembly (build_driver) with the TOTAL
    budget; iteration starts once. ``take_batch`` returns the native
    List[RolloutSample] (as an infra reference when the source runs remotely);
    ``commit`` reports the REAL optimizer steps of the consumption.
    """

    type = 'streaming_driver'

    def start(self, inputs: Dict[str, Any]) -> None:
        """Build the driver if needed, then iter() exactly once."""
        if self.obj is None:
            self._build_driver(inputs)
        self._iterator = iter(self.obj)
        self._current: Optional[Any] = None

    def _build_driver(self, inputs: Dict[str, Any]) -> None:
        from swift.dev.builders.streaming import build_streaming_driver
        from .resources import _configs, _fill
        engines = slot_values(inputs.get('engine'))
        cluster = engines[0] if engines else None
        if cluster is None:
            raise ComponentError(
                f'{self.node_id}: streaming_driver needs a wired engine (the cluster)')
        # The prompt stream is the wired dataloader's resident loader object
        # (its batch port projects the loader itself): batching, seed and the
        # face choice live on THAT node, not here.
        prompt_loader = inputs.get('prompts')
        if prompt_loader is None:
            raise ComponentError(
                f'{self.node_id}: streaming_driver needs a wired prompt stream (a dataloader)')
        config_pkg = _configs()
        # The panel's scalar knobs are the REAL Config fields' convenience face
        # (dict params win; the scalars fill their Config's field when unset).
        # The click budget is NOT passed here: max_steps is the frame's total
        # budget owner; a click must never make the driver terminate.
        rl_kv = dict(self.params.get('rl_config') or {})
        rl_kv.setdefault('num_generations', self.params.get('num_generations', 8))
        rollout_kv = dict(self.params.get('rollout_config') or {})
        rollout_kv.setdefault('async_mode', self.params.get('async_mode', 'none'))
        rollout_kv.setdefault('max_staleness', self.params.get('max_staleness', 1))
        gen_kv = dict(self.params.get('generation_config') or {})
        # Scoring, regeneration and snapshots are the graph's own stages; the
        # builder assembles the algorithm-free control plane only.
        self.obj = build_streaming_driver(
            engine=cluster,
            prompt_dataloader=prompt_loader,
            rollout_config=_fill(config_pkg.RolloutConfig, rollout_kv, self.node_id),
            rl_config=_fill(config_pkg.RLConfig, rl_kv, self.node_id),
            generation_config=(
                _fill(config_pkg.GenerationConfig, gen_kv, self.node_id) if gen_kv else None),
            train_batch_size=int(self.params.get('train_batch_size', 8) or 8),
            gradient_accumulation_steps=int(
                self.params.get('gradient_accumulation_steps', 1) or 1),
            num_train_epochs=float(self.params.get('num_train_epochs', 1.0) or 1.0),
            max_steps=int(self.params.get('max_steps', 0) or 0))

    def take_batch(self) -> Any:
        batch = next(self._iterator)
        # The samples port projects the LAST delivered batch: downstream stages
        # wired to this port read batch data, not the control-plane object.
        self._current = batch
        return batch

    def port_value(self, key: str) -> Any:
        """The iterator's lanes: samples = the last delivered batch, source =
        this control handle (for ports that name the driver itself)."""
        if key == 'source':
            return self
        return getattr(self, '_current', None) or []

    def commit(self, steps: int) -> None:
        self.obj.commit(int(steps))

    def skip_batch(self) -> None:
        self.obj.commit(0)

    def request_publish(self, push: Any = None) -> None:
        """Register the real push through the existing public entry. The push
        runs in the driver's next pass -- never awaited before commit."""
        self.obj.request_publish(push)

    def release(self) -> None:
        """Terminate the source.

        The single-slot queue may already hold a delivered batch; the runner's
        final take has already consumed it. Draining happens through the public
        iterator: one more ``next`` (if a batch is queued) receives it, THEN
        abort joins without the queue-full deadlock. The private queue is never
        touched here.
        """
        driver = self.obj
        if driver is None:
            return
        try:
            if getattr(self, '_iterator', None) is not None:
                try:
                    next(self._iterator)  # receive a queued delivery, if any
                except StopIteration:
                    pass
        finally:
            try:
                driver.abort()
            except Exception:
                pass
            self.obj = None
            self._iterator = None


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
            dict(self.params.get('template_config') or {}), self.node_id)
        cfg = dict(self.params.get('challenger_config') or {})
        try:
            self.obj = AgenticChallenger(
                backend, api=inputs.get('api'),
                seed_fn=(lambda text=seed: text) if seed else None,
                envs=[env], solver_rollout=inputs.get('solver_rollout'),
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

    The engine face is whichever the wire carries: a rollout/cluster component
    already IS the episode engine (its ``rollout`` attribute); a bare sampler
    gets a cached MultiTurnRollout wrapper -- the same class the streaming
    driver uses, so the sample objects downstream (score/advantage/train)
    see one shape regardless of who generated them."""

    type = 'generate'

    def invoke(self, inputs: Dict[str, Any], batch: Any) -> Dict[str, Any]:
        engine = inputs.get('engine') or inputs.get('episode')
        if engine is None:
            raise ComponentError(f'{self.node_id}: generate needs a wired engine')
        rows = batch if isinstance(batch, list) else (batch or {}).get('rows', [])
        if not rows:
            raise ComponentError(f'{self.node_id}: generate received an empty batch')
        prompts = [row['messages'] for row in rows]
        extras = [{k: v for k, v in row.items() if k != 'messages'} for row in rows]
        # Sampling knobs are the REAL GenerationConfig fields (the typed
        # generation_config param), translated through the SAME builder the
        # recipes use; no config means the engine's construction-time
        # defaults stand (None is passed, not an empty override).
        from swift.dev.builders.sampler import to_sampling_params
        from .resources import _configs, _fill
        gen_kv = dict(self.params.get('generation_config') or {})
        gen_config = _fill(_configs().GenerationConfig, gen_kv, self.node_id) if gen_kv else None
        samples = self._rollout_for(engine).generate(
            prompts,
            num_samples=int(self.params.get('num_generations', '8') or 8),
            sampling_params=to_sampling_params(gen_config) if gen_config is not None else None,
            prompt_extras=extras or None)
        return {'samples': samples}

    def _rollout_for(self, engine: Any) -> Any:
        """One resident episode engine per wired engine object."""
        rollouts = getattr(self, '_rollouts', None)
        if rollouts is None:
            rollouts = self._rollouts = {}
        rollout = rollouts.get(id(engine))
        if rollout is None:
            native = getattr(engine, 'rollout', None)
            if native is not None:
                rollout = native  # a cluster face: the rollout it assembled
            elif callable(getattr(engine, 'generate', None)):
                rollout = engine  # already an episode engine (a rollout component)
            else:
                from twinkle_agentic.rollout import MultiTurnRollout
                rollout = MultiTurnRollout(engine, getattr(engine, 'template', None))
            rollouts[id(engine)] = rollout
        return rollout


class ScoreStage(StageComponent):
    """Actually call r(tau): every wired scorer contributes one column.

    The scorers arrive on the many-port as ``(slot, scorer)`` pairs -- the
    slot names its column. Each scorer is called once over the SAME row
    protocol; the reward port carries ``values[B, C]`` with ``scorer_ids``
    naming the columns in wire order. A scorer that does not score every row
    fails loudly -- no silent truncation, no padding."""

    type = 'score'

    def invoke(self, inputs: Dict[str, Any], batch: Any) -> Dict[str, Any]:
        scorers = inputs.get('scorers')
        if not scorers:
            raise ComponentError(f'{self.node_id}: score needs wired scorers (one per column)')
        # The sample list rides the wired samples port when the frame source is
        # a loader (dict rows) or when the producer is an explicit upstream; a
        # driver-sourced frame already carries the rollout samples in ``batch``.
        # The port is single-entry: its value IS the sample list.
        port_samples = inputs.get('samples')
        if port_samples is not None:
            samples = port_samples if isinstance(port_samples, list) else []
        else:
            samples = batch if isinstance(batch, list) else []
        # 行协议（ScoreWorker._score_rows 同一形态）：messages + flat extras +
        # response_token_ids；messages 缺失的样本用 prompt+decoded 合成——
        # 规则 / RM / judge 都读这一种行，绑层只做载荷转协议。一个原生
        # Trajectory dict（challenger 两路出流的载荷形态）已经是评分器的
        # 原生输入，逐字直通，不做任何转写。
        rows = []
        for sample in samples:
            if isinstance(sample, dict):
                rows.append(sample)
                continue
            row = {**(getattr(sample, 'extra', None) or {}),
                   'response_token_ids': list(getattr(sample, 'response_token_ids', None) or [])}
            messages = getattr(sample, 'messages', None)
            if messages is None:
                prompt = list(getattr(sample, 'prompt', None) or [])
                if not prompt or prompt[-1].get('role') != 'assistant':
                    prompt = prompt + [{'role': 'assistant', 'content': sample.decoded}]
                messages = prompt
            row['messages'] = list(messages)
            rows.append(row)
        columns: List[List[Any]] = []
        scorer_ids: List[str] = []
        for slot, scorer in scorers:
            if not callable(scorer):
                raise ComponentError(f'{self.node_id}: scorer slot {slot!r} is not callable')
            rewards = scorer(rows)
            column = list(rewards) if rewards is not None else []
            if len(column) != len(rows):
                raise ComponentError(
                    f'{self.node_id}: scorer {slot!r} returned {len(column)} scores '
                    f'for {len(rows)} rows -- a scorer scores every row')
            columns.append(column)
            scorer_ids.append(str(slot))
        # values[B, C]: one row per sample, one column per wired scorer.
        values = [[column[i] for column in columns] for i in range(len(rows))]
        return {'reward': {'values': values, 'scorer_ids': scorer_ids}}


class SelectStage(StageComponent):
    """Keep rows / whole groups / pairs by the declared criterion; every
    aligned signal is co-selected with the same mask."""

    type = 'select'

    def invoke(self, inputs: Dict[str, Any], batch: Any) -> Dict[str, Any]:
        samples = batch if isinstance(batch, list) else []
        reward = inputs.get('reward') or {}
        values = reward.get('values') if isinstance(reward, dict) else None
        mode = self.params.get('select', 'best_of_n')
        num_generations = int(self.params.get('num_generations', '8') or 8)
        keep: List[int] = []
        if values is not None and mode in ('best_of_n', 'top_k'):
            top_k = 1 if mode == 'best_of_n' else int(self.params.get('top_k', '1') or 1)
            for start in range(0, len(samples), num_generations):
                group = list(range(start, min(start + num_generations, len(samples))))
                ranked = sorted(group, key=lambda i: -(values[i][0] if values[i] else 0.0))
                keep.extend(ranked[:top_k])
        elif values is not None and mode == 'threshold':
            threshold = float(self.params.get('threshold', '0.0') or 0.0)
            keep = [i for i, row in enumerate(values)
                    if row and (row[0] if isinstance(row, list) else row) >= threshold]
        else:
            raise ComponentError(
                f'{self.node_id}: select needs a wired reward (the criterion scores)')
        kept = [samples[i] for i in keep]
        signals = slot_values(inputs.get('signals'))
        kept_signals = None
        if signals:
            kept_signals = [signals[i] for i in keep] if hasattr(signals, '__getitem__') else signals
        out = {'batch': kept}
        if kept_signals is not None:
            out['signals'] = kept_signals
        return out

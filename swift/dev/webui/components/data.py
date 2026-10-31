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
    """for batch in dataloader -- advanced on demand, cursor preserved."""

    type = 'dataloader'

    def start(self, inputs: Dict[str, Any]) -> None:
        from swift.dev.builders import build_prompt_dataloader
        dataset = inputs.get('data')
        if not isinstance(dataset, dict) or 'prompts' not in dataset:
            raise ComponentError(f'{self.node_id}: dataloader needs a wired dataset')
        self._dataset = dataset
        self.obj = build_prompt_dataloader(
            dataset['prompts'], dataset.get('extras') or [],
            seed=int(self.params.get('seed', '42') or 42),
            batch_size=int(self.params.get('batch_size', '8') or 8))
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
        from swift.dev.recipe.grpo.assembly import build_driver
        engines = slot_values(inputs.get('engine'))
        cluster = engines[0] if engines else None
        if cluster is None:
            raise ComponentError(
                f'{self.node_id}: streaming_driver needs a wired engine (the cluster)')
        dataset = inputs.get('prompts')
        if not isinstance(dataset, dict) or 'prompts' not in dataset:
            raise ComponentError(f'{self.node_id}: streaming_driver needs a wired dataset')
        # The click budget is NOT passed here: max_steps is the frame's total
        # budget owner; a click must never make the driver terminate.
        self.obj = build_driver(
            cluster=cluster, prompt_dataloader=self._loader_for(dataset),
            max_steps=int(self.params.get('total_steps', '0') or 0))

    def _loader_for(self, dataset: Dict[str, Any]) -> Any:
        from swift.dev.builders import build_prompt_dataloader
        return build_prompt_dataloader(
            dataset['prompts'], dataset.get('extras') or [],
            seed=int(self.params.get('seed', '42') or 42),
            batch_size=int(self.params.get('generation_batch_size', '8') or 8))

    def take_batch(self) -> Any:
        return next(self._iterator)

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
    """One blocking batch generation: rollout.generate over the prompt rows."""

    type = 'generate'

    def invoke(self, inputs: Dict[str, Any], batch: Any) -> Dict[str, Any]:
        engine = inputs.get('engine') or inputs.get('episode')
        if engine is None:
            raise ComponentError(f'{self.node_id}: generate needs a wired engine')
        rollout = getattr(engine, 'rollout', engine)
        rows = batch if isinstance(batch, list) else (batch or {}).get('rows', [])
        prompts = [row['messages'] for row in rows]
        extras = [{k: v for k, v in row.items() if k != 'messages'} for row in rows]
        samples = rollout.generate(
            prompts,
            num_samples=int(self.params.get('num_generations', '8') or 8),
            sampling_params={'temperature': float(self.params.get('temperature', '1.0') or 1.0),
                             'max_new_tokens': int(self.params.get('max_new_tokens', '512') or 512)},
            prompt_extras=extras or None)
        return {'samples': samples}


class ScoreStage(StageComponent):
    """Actually call r(tau): the scorer is a param -- the real reward family
    off the ``orms`` plugin registry, constructed once with typed kwargs."""

    type = 'score'

    def _scorer(self) -> Any:
        if getattr(self, '_scorer_obj', None) is None:
            from swift.dev.rewards import orms
            name = str(self.params.get('scorer') or 'math')
            cls = orms.get(name)
            if cls is None:
                raise ComponentError(
                    f'{self.node_id}: unknown scorer {name!r} (not in the orms registry)')
            kwargs = dict(self.params.get('scorer_kwargs') or {})
            try:
                self._scorer_obj = cls(args=None, **kwargs)
            except TypeError as exc:
                raise ComponentError(
                    f'{self.node_id}: scorer {name!r} rejects kwargs {sorted(kwargs)}: {exc}') from exc
        return self._scorer_obj

    def invoke(self, inputs: Dict[str, Any], batch: Any) -> Dict[str, Any]:
        scorer = self._scorer()
        samples = batch if isinstance(batch, list) else []
        completions = [getattr(s, 'decoded', '') or '' for s in samples]
        rewards = scorer(completions, messages=[getattr(s, 'messages', None) for s in samples])
        # values[B, 1]: one row per sample, the single declared scorer column.
        column = list(rewards) if rewards is not None else []
        values = [[row] for row in column] if column else []
        return {'reward': {'values': values, 'scorer_ids': [str(self.params.get('scorer') or 'math')]}}


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

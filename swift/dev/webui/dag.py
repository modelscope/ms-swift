# Copyright (c) ModelScope Contributors. All rights reserved.
"""The cyclic-DAG controller: click semantics over a resident graph.

One controller drives one graph revision on the session's single execution
thread. Every control-surface method (``pause`` / ``resume`` / ``stop`` /
``get_running_nodes`` / ``get_output``) is safe to call from the command
thread while a run is active, so preview and control never wait for
execution.

Execution units:

- ``run_component(target)`` -- one click. A loop-free target runs only what
  has not run yet (a re-click with every prerequisite done runs the target
  alone); a target inside a frame turns that frame one full circle -- the
  frame's later members, then its earlier members around the feedback edge --
  without rebuilding any resident resource. Cross-frame producers turn
  first: every consuming round reads a fresh production.
- ``run(target=None)`` -- the whole graph: an unceasing sequence of frame
  circles until the sources exhaust, pause or stop; a ``target`` degenerates
  to one click on that node.
- ``observe=True`` runs a click with training side effects suppressed and the
  batch left pending (the next real click reuses it).

Parallelism (requirement 7): every segment executes as ready sets -- all
nodes whose same-segment dependencies are complete are submitted together;
unrelated branches never serialise. Resource builds stay on the driving
thread (they mutate process-level build context); stage and task invocations
run in the pool, and a node reports 'completed' only when its reference
outputs are truly ready (LazyCollect handles are completion-probed).

Batch handover lives in ``SourceHandle``: one resident iterator per source;
take / commit-exactly-once / pending reuse / terminal exhaustion. Only real
optimizer completions count as steps; a click budget counts the same way.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import FIRST_COMPLETED, ALL_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any, Callable, Dict, List, Optional, Set

from .graph import FrameSpec, Graph

#: Effects suppressed by an observe click: training, publishing and persisting
#: side effects. ``code`` stays live -- a code component is a data-plane
#: transform the round's data flow depends on.
SUPPRESSED_EFFECTS = frozenset({'train', 'sync', 'save', 'serve', 'external'})

#: Concurrent stage invocations per segment. Remote execution makes each
#: invoke a thin submit; the bound only caps driver-side thread count.
WAVE_WORKERS = 16


class DAGError(RuntimeError):
    """A controller-contract violation (double commit, unstarted source, misuse)."""


class SourceExhausted(DAGError):
    """A take discovered the iterator is done: natural exhaustion, not a failure."""


def _collect_refs(value: Any, depth: int = 0) -> List[Any]:
    """The LazyCollect handles in a stage's outputs (bounded depth).

    Only the shapes bindings actually return -- port dicts and one nesting
    level of lists -- are scanned; anything deeper is not part of the wire
    contract."""
    refs: List[Any] = []
    if depth > 2:
        return refs
    if isinstance(value, dict):
        for item in value.values():
            refs.extend(_collect_refs(item, depth + 1))
    elif isinstance(value, (list, tuple)):
        for item in value:
            refs.extend(_collect_refs(item, depth + 1))
    elif callable(value) and getattr(value, '_is_lazy_collect', False):
        refs.append(value)
    return refs


class SourceHandle:
    """Batch handover around one resident iterator component.

    Invariants:

    - ``start`` runs once per source; a second call is a contract violation
      (StreamingDriver is single-use and a rebuilt iterator would lose the
      cursor).
    - At most one delivered batch is unconsumed at a time.
    - Each delivered batch is acknowledged exactly once: ``commit(steps)``
      after full consumption, ``commit(0)`` for an explicit skip.
    - ``exhausted`` is terminal: a terminated iterator is never reused.
    """

    def __init__(self, node_id: str, component: Any):
        self.node_id = node_id
        self.component = component
        self._iterator: Any = None
        self._current: Any = None
        self._acked = True
        self.exhausted = False
        self.last_steps: int = 0

    @property
    def started(self) -> bool:
        return self._iterator is not None

    @property
    def has_pending(self) -> bool:
        """A delivered batch awaiting consumption or acknowledgement."""
        return self._current is not None and not self._acked

    def start(self, inputs: Dict[str, Any]) -> None:
        if self.started:
            raise DAGError(f'source {self.node_id!r}: iterator already started')
        if self.exhausted:
            raise DAGError(f'source {self.node_id!r}: iterator exhausted; restart '
                           'the source explicitly, it never auto-rewinds')
        self.component.start(inputs)
        self._iterator = self.component

    def take(self) -> Any:
        """The current batch when one is pending (observe re-click reuse), else
        the next batch off the iterator."""
        if self.exhausted:
            raise DAGError(f'source {self.node_id!r}: exhausted')
        if self.has_pending:
            return self._current
        if not self.started:
            raise DAGError(f'source {self.node_id!r}: take before start')
        try:
            batch = self.component.take_batch()
        except StopIteration as exc:
            self.exhausted = True
            raise DAGError(f'source {self.node_id!r}: exhausted: {exc}') from None
        self._current = batch
        self._acked = False
        return batch

    def commit(self, steps: int) -> None:
        """Acknowledge the current batch (exactly once) with the OPTIMIZER steps
        the training actually completed on it."""
        if self._current is None or self._acked:
            raise DAGError(f'source {self.node_id!r}: commit without a pending batch')
        if steps < 0:
            raise DAGError(f'source {self.node_id!r}: negative commit steps {steps}')
        self.component.commit(int(steps))
        self.last_steps = int(steps)
        self._acked = True
        self._current = None

    def skip(self) -> None:
        """Explicitly drop the current batch: commit(0), cursor advanced, no step."""
        if not self.has_pending:
            raise DAGError(f'source {self.node_id!r}: skip without a pending batch')
        self.component.skip_batch()
        self.last_steps = 0
        self._acked = True
        self._current = None

    def release(self) -> None:
        """Terminate the source; idempotent."""
        if self.exhausted:
            return
        self.exhausted = True
        self._current = None
        self._acked = True
        release = getattr(self.component, 'release', None)
        if callable(release):
            release()


class DAGController:
    """Drives clicks and whole-graph runs over the session's resident bindings.

    The session supplies the component surface (``resolve``), the event
    channel (``event``) and the output recorder (``record_output``); the
    sources table is INJECTED and owned by the session so iterator cursors
    survive across calls. Nothing here imports session internals.
    """

    def __init__(self, graph: Graph, *, resolve: Callable[[str], Any],
                 event: Callable[..., None], record_output: Callable[[str, str, Any], None],
                 sources: Optional[Dict[str, SourceHandle]] = None):
        self.graph = graph
        self._resolve = resolve
        self._event = event
        self._record_output = record_output
        #: Iterator consumption progress is INJECTED by the session (the
        #: click-to-click cursor); SourceHandle state is per-source.
        self.sources: Dict[str, SourceHandle] = sources if sources is not None else {}
        #: node_id -> port values of the node's LAST completion. Residents and
        #: iterators hold projections; stages hold invoke outputs (references).
        self._outputs: Dict[str, Dict[str, Any]] = {}
        #: frame_id -> completed rounds; cross-frame edge id -> producer round
        #: the consumer last read. A consuming round turns a producer whose
        #: rounds have not advanced past the consumed marker.
        self._frame_rounds: Dict[str, int] = {}
        self._consumed: Dict[str, int] = {}
        self._frames: Dict[str, FrameSpec] = {frame.id: frame for frame in graph.frames}
        self._turning: Set[str] = set()
        self._lock = threading.Lock()  # guards _outputs and _running cross-thread
        self._running: Dict[str, None] = {}  # insertion-ordered running node set
        self._paused = threading.Event()  # set = pause requested
        self._stop = threading.Event()
        self._paused_emitted = False

    # -- control surface (any thread, immediate) -----------------------------

    def pause(self) -> Dict[str, Any]:
        """Pause at the next node boundary: running nodes complete, no further
        node starts. The receipt is immediate; a ``paused`` event lands once
        the running set drains."""
        self._paused.set()
        return {'ok': True, 'state': 'pausing', 'running': self.get_running_nodes()}

    def resume(self) -> Dict[str, Any]:
        if not self._paused.is_set():
            return {'ok': False, 'error': 'not paused'}
        self._paused.clear()
        self._paused_emitted = False
        self._event('resumed')
        return {'ok': True, 'state': 'running'}

    def stop(self) -> Dict[str, Any]:
        """Stop this call: no new node starts; submitted atomic work finishes
        and reports its real result. An incomplete round leaves its batch
        pending (the next call reuses it) and does not count as turned."""
        self._stop.set()
        self._paused.clear()  # a stop also releases a paused wait
        return {'ok': True, 'state': 'stopping', 'running': self.get_running_nodes()}

    def get_running_nodes(self) -> List[str]:
        with self._lock:
            return list(self._running)

    def get_output(self, node_id: str) -> Dict[str, Any]:
        """The node's recorded port outputs, references unopened. THIS is the
        pressure point the preview surface materialises through -- the only
        place output data is asked for at all."""
        with self._lock:
            if node_id not in self._outputs:
                raise DAGError(f'{node_id}: no recorded output; run the node first')
            return self._outputs[node_id]

    def drop_outputs(self, node_ids: List[str]) -> None:
        """Session invalidation hook: forget outputs of dropped residents."""
        with self._lock:
            for node_id in node_ids:
                self._outputs.pop(node_id, None)

    # -- execution entry points (the session's single execution thread) ------

    def run_component(self, target: Optional[str] = None, *, steps: Optional[int] = None,
                      observe: bool = False) -> Dict[str, Any]:
        """One click on ``target`` (default: the graph's last node).

        Loop-free target: run the not-yet-run upstream closure, then the
        target -- or the target alone on a re-click. Frame target: turn the
        frame one circle (``steps`` extends the click to that many optimizer
        steps; one circle is the default).
        """
        target = target or (self.graph.order[-1] if self.graph.order else None)
        if target is None:
            raise DAGError('empty graph has nothing to run')
        self.graph.node(target)  # unknown target fails loudly here
        self._stop.clear()
        self._paused.clear()
        self._paused_emitted = False
        frame = self.graph.training_frame(target)
        if frame is not None:
            receipt = self._run_frame_click(frame, steps=steps, observe=observe)
            receipt['target'] = target
            return receipt
        return self._run_segment_click(target, observe=observe)

    def run(self, target: Optional[str] = None, *, steps: Optional[int] = None) -> Dict[str, Any]:
        """The whole graph as an unceasing sequence of single-component runs.

        No ``target``: turn every frame circle after circle until the sources
        exhaust, pause or stop (loop-free tasks stay click-driven). A
        ``target`` degenerates to one click on that node.
        """
        if target:
            return self.run_component(target, steps=steps)
        self._stop.clear()
        self._paused.clear()
        self._paused_emitted = False
        if not self.graph.frames:
            # A pure DAG runs to completion once: one click on the sink.
            return self._run_segment_click(self.graph.order[-1])
        participation = set().union(*(
            self.graph.upstream_closure(node_id) for frame in self.graph.frames
            for node_id in frame.members))
        self._emit_plan(participation)
        rounds = 0
        completed = 0
        stalled: Dict[str, int] = {}
        while not self._stop.is_set():
            live = [frame for frame in self._frame_order()
                    if self._not_exhausted(frame) and stalled.get(frame.id, 0) < 2]
            if not live:
                break
            for frame in live:
                self._check_pause()
                if self._stop.is_set():
                    break
                receipt = self._turn_frame(frame)
                rounds += 1
                trained = receipt.get('optimizer_steps', 0)
                completed += trained
                stalled[frame.id] = stalled.get(frame.id, 0) + 1 if trained <= 0 else 0
                if receipt.get('stopped') or receipt.get('exhausted'):
                    stalled[frame.id] = 2  # terminal for this run
                    break
            if steps is not None and completed >= steps:
                break
        # A frame is terminal for this run when its own source exhausted or it
        # was dropped (starved producer or a no-progress stall).
        terminal = all(not self._not_exhausted(frame)
                       or stalled.get(frame.id, 0) >= 2 for frame in self.graph.frames)
        return {'mode': 'run', 'rounds': rounds, 'optimizer_steps': completed,
                'state': 'stopped' if self._stop.is_set() else 'completed',
                'exhausted': terminal}

    def _not_exhausted(self, frame: FrameSpec) -> bool:
        """Whether the frame's source may still deliver (peeks the handle; a
        source that never started is not exhausted)."""
        handle = self.sources.get(frame.sources[0])
        return handle is None or not handle.exhausted

    # -- frame rounds ---------------------------------------------------------

    def _run_frame_click(self, frame: FrameSpec, *, steps: Optional[int],
                         observe: bool) -> Dict[str, Any]:
        """A click on a frame member: the frame's circles (preparation and
        cross-frame freshening ride each turn)."""
        # Peek WITHOUT resolving: creating the handle here would build the
        # component before its upstream resources exist, and the resident's
        # fingerprint would then legitimately change at the next resolve.
        handle = self.sources.get(frame.sources[0])
        if handle is not None and handle.exhausted:
            raise DAGError(
                f'source {handle.node_id!r} is exhausted; a loop never auto-rewinds -- '
                'release and rebuild the source explicitly')
        rounds = 0
        completed = 0
        while not self._stop.is_set():
            self._check_pause()
            receipt = self._turn_frame(frame, observe=observe)
            rounds += 1
            completed += receipt.get('optimizer_steps', 0)
            if receipt.get('stopped') or receipt.get('exhausted'):
                break
            if steps is None:
                break  # one click turns exactly one circle (requirement 8)
            if completed >= steps:
                break
            if rounds > 1 and receipt.get('optimizer_steps', 0) <= 0:
                break  # no-progress guard: a circle that trains nothing ends the click
        handle = self.sources.get(frame.sources[0])
        return {'mode': 'component', 'rounds': rounds,
                'optimizer_steps': completed,
                'state': 'stopped' if self._stop.is_set() else 'completed',
                'exhausted': bool(handle is not None and handle.exhausted)}

    def _turn_frame(self, frame: FrameSpec, *, observe: bool = False) -> Dict[str, Any]:
        """One frame circle: prepare own prerequisites, freshen cross-frame
        producers, take a batch, run the members as ready sets, commit with
        the REAL optimizer steps."""
        source_id = frame.sources[0]
        # Peek without resolving (see _run_frame_click); the handle itself is
        # created only after preparation, when the component's upstream
        # residents exist and its fingerprint is stable.
        existing = self.sources.get(source_id)
        if existing is not None and existing.exhausted:
            return {'optimizer_steps': 0, 'exhausted': True}
        if frame.id in self._turning:
            raise DAGError(f'frame {frame.id!r}: re-entered while turning '
                           '(cross-frame dependency cycle between frames)')
        self._turning.add(frame.id)
        try:
            participation = set().union(*(self.graph.upstream_closure(node_id)
                                          for node_id in frame.members))
            self._emit_plan(participation)
            self._prepare_round(participation)
            handle = self._source(source_id)
            if self._freshen_producers(frame, observe=observe):
                # Every producer already advanced as far as it ever will: the
                # consuming loop is starved and stops WITHOUT re-reading stale data.
                return {'optimizer_steps': 0, 'exhausted': True}
            self._check_pause()
            if self._stop.is_set():
                return {'optimizer_steps': 0, 'stopped': True}
            try:
                batch = self._take_source(source_id)
            except SourceExhausted:
                # Natural exhaustion discovered mid-drive ends the turn gracefully.
                return {'optimizer_steps': 0, 'exhausted': True}
            self._refresh_iterators()
            members = set(frame.members)
            plan = [node_id for node_id in self.graph.order
                    if node_id in members and node_id != source_id]
            done = self._execute_wave(plan, batch, observe=observe)
            stopped = len(done) != len(plan)
            trained = self._trained_steps(done)
            if not observe and not stopped:
                handle.commit(trained)
                self._frame_rounds[frame.id] = self._frame_rounds.get(frame.id, 0) + 1
            self._event('frame_progress', frame=frame.id,
                        rounds=self._frame_rounds.get(frame.id, 0),
                        optimizer_steps=trained, pending=handle.has_pending,
                        stopped=stopped)
            return {'optimizer_steps': trained, 'stopped': stopped}
        finally:
            self._turning.discard(frame.id)

    def _freshen_producers(self, frame: FrameSpec, *, observe: bool) -> bool:
        """Cross-frame data edges are round boundaries: every consuming round
        reads a FRESH production -- the producing loop turns once more unless
        it already advanced past the round this loop last consumed. Returns
        True when a required producer is exhausted with nothing fresher left
        (the consuming loop is starved)."""
        starved = False
        members = set(frame.members)
        for edge in self.graph.cross_frame_edges():
            if edge.target not in members:
                continue
            producer_id = self.graph.frame_of(edge.source)
            producer = self._frames[producer_id]
            consumed = self._consumed.get(edge.id, 0)
            if self._frame_rounds.get(producer_id, 0) > consumed:
                # Fresher than what this loop last read (the producer advanced
                # on its own): consume it now and mark THAT round as consumed,
                # so the next round turns the producer again.
                self._consumed[edge.id] = self._frame_rounds[producer_id]
                continue
            handle = self.sources.get(producer.sources[0])  # peek, never resolve
            if handle is not None and handle.exhausted:
                starved = True
                continue
            self._turn_frame(producer, observe=observe)
            self._consumed[edge.id] = self._frame_rounds.get(producer_id, 0)
        return starved

    def _ensure_cross_available(self, closure: Set[str]) -> None:
        """Loop-free clicks only need cross-frame producers to have delivered
        once; later clicks read the current value (a loop-invariant)."""
        for edge in self.graph.cross_frame_edges():
            if edge.target not in closure:
                continue
            producer_id = self.graph.frame_of(edge.source)
            if self._frame_rounds.get(producer_id, 0) == 0:
                self._turn_frame(self._frames[producer_id])
                if self._frame_rounds.get(producer_id, 0) == 0:
                    raise DAGError(
                        f'edge {edge.id}: producer frame {producer_id!r} exhausted '
                        'before delivering anything the click can consume')

    def _frame_order(self) -> List[FrameSpec]:
        """Frames ordered producer-first over cross-frame edges, so a cascade
        during a consumer's circle never double-turns a producer."""
        producers: Dict[str, Set[str]] = {frame_id: set() for frame_id in self._frames}
        for edge in self.graph.cross_frame_edges():
            producer_id = self.graph.frame_of(edge.source)
            consumer_id = self.graph.frame_of(edge.target)
            if producer_id is not None and consumer_id is not None and producer_id != consumer_id:
                producers[consumer_id].add(producer_id)
        ordered: List[str] = []
        visiting: Set[str] = set()

        def visit(frame_id: str) -> None:
            if frame_id in ordered or frame_id in visiting:
                return
            visiting.add(frame_id)
            for dep in sorted(producers[frame_id]):
                visit(dep)
            visiting.discard(frame_id)
            ordered.append(frame_id)

        for frame_id in sorted(self._frames):
            visit(frame_id)
        return [self._frames[frame_id] for frame_id in ordered]

    # -- loop-free segment clicks ---------------------------------------------

    def _run_segment_click(self, target: str, *, observe: bool) -> Dict[str, Any]:
        """A click on a loop-free node: run the un-run upstream closure plus
        the target; with everything already done, the target alone."""
        closure = self.graph.upstream_closure(target)
        self._ensure_cross_available(closure)
        unrun = [node_id for node_id in self.graph.order
                 if node_id in closure and node_id not in self._outputs]
        if not unrun:
            unrun = [target]  # requirement 8: re-click re-runs the target alone
        elif target not in unrun:
            unrun.append(target)
        participation = closure | {target}
        self._emit_plan(participation)
        self._prepare_round(participation)
        sources = [node_id for node_id in self.graph.order if node_id in closure
                   and self.graph.descriptor(node_id).lifecycle == 'Iterator']
        plan = [node_id for node_id in unrun
                if self.graph.descriptor(node_id).lifecycle != 'Resource']
        batch = None
        needs_batch = (unrun != [target]
                       and any(self.graph.descriptor(node_id).lifecycle == 'Stage'
                               for node_id in plan))
        if needs_batch and sources:
            batch = self._take_source(sources[0])
            self._refresh_iterators()
        done = self._execute_wave(plan, batch, observe=observe)
        # A consumed batch is acknowledged with the real steps of this segment.
        if batch is not None and sources and not observe:
            handle = self._source(sources[0])
            if handle.has_pending:
                handle.commit(self._trained_steps(done))
        return {'mode': 'component', 'target': target, 'nodes': len(done),
                'optimizer_steps': self._trained_steps(done),
                'state': 'stopped' if self._stop.is_set() else 'completed'}

    # -- the ready-set executor ------------------------------------------------

    def _execute_wave(self, plan: List[str], batch: Any, *, observe: bool) -> Set[str]:
        """Run one segment's nodes as parallel ready sets.

        All nodes whose same-segment dependencies are complete submit
        together; completion of any node frees its dependants. Resources
        build inline on the driving thread; stages and tasks invoke in the
        pool. Pause lands on node boundaries; stop leaves the segment
        incomplete (the caller decides the receipt). A failed node cancels
        the un-started rest and re-raises.
        """
        members = set(plan)
        dependencies = {node_id: {edge.source for edge in self.graph.incoming(node_id)
                                  if edge.source in members
                                  and not self.graph.is_feedback(edge.id)}
                        for node_id in plan}
        pending = list(plan)
        done: Set[str] = set()
        futures: Dict[Future, str] = {}
        with ThreadPoolExecutor(max_workers=WAVE_WORKERS, thread_name_prefix='webui-dag') as pool:
            while pending or futures:
                if not self._paused.is_set() and not self._stop.is_set():
                    # Submit in rounds: an inline completion (resource build,
                    # observe suppression) can free further nodes without any
                    # pool wait in between.
                    while True:
                        ready = [nid for nid in pending if dependencies[nid] <= done]
                        if not ready:
                            break
                        for node_id in ready:
                            pending.remove(node_id)
                            if self._submit_node(node_id, batch, observe, futures, pool):
                                done.add(node_id)
                if futures:
                    finished, _unused = wait(
                        list(futures),
                        return_when=ALL_COMPLETED if self._paused.is_set() else FIRST_COMPLETED)
                    for future in finished:
                        node_id = futures.pop(future)
                        self._mark_running(node_id, running=False)
                        error = future.exception()
                        if error is not None:
                            for other in futures:
                                other.cancel()
                            raise error
                        with self._lock:
                            self._outputs[node_id] = future.result()
                        self._record_output(node_id, self.graph.descriptor(node_id).type,
                                            self._outputs[node_id])
                        done.add(node_id)
                    continue
                if self._stop.is_set():
                    return done
                if self._paused.is_set():
                    self._check_pause()  # running set has drained; block or return
                    if self._stop.is_set():
                        return done
                    continue
                if pending:
                    raise DAGError(f'segment stalled with unmet dependencies: {sorted(pending)}')
                break
        return done

    def _submit_node(self, node_id: str, batch: Any, observe: bool,
                     futures: Dict[Future, str], pool: ThreadPoolExecutor) -> bool:
        """Dispatch one node by lifecycle. Returns True when it completed
        inline (the driving thread); False when a pool future carries it."""
        descriptor = self.graph.descriptor(node_id)
        if observe and set(descriptor.effects) & SUPPRESSED_EFFECTS:
            # Suppressed on the driving thread: no inputs are assembled (an
            # effect node's outputs do not exist this round) and the node
            # completes without ever entering the pool.
            self._event('node', node_id=node_id, state='cached',
                        note='observe: side effect suppressed, batch stays pending')
            outputs = {'observe': True}
            with self._lock:
                self._outputs[node_id] = outputs
            self._record_output(node_id, descriptor.type, outputs)
            return True
        component = self._resolve(node_id)  # resident access stays on the driving thread
        inputs = self._inputs(node_id)
        lifecycle = descriptor.lifecycle
        if lifecycle == 'Resource':
            outputs = self._build_resource(node_id, component, inputs)
            with self._lock:
                self._outputs[node_id] = outputs
            self._record_output(node_id, descriptor.type, outputs)
            return True
        if lifecycle == 'Stage':
            self._mark_running(node_id)
            futures[pool.submit(self._invoke_stage, node_id, component, inputs,
                                batch)] = node_id
            return False
        # Task: one invocation per request, never per batch.
        self._mark_running(node_id)
        futures[pool.submit(self._invoke_task, node_id, component, inputs)] = node_id
        return False

    def _invoke_stage(self, node_id: str, component: Any,
                      inputs: Dict[str, Any], batch: Any) -> Dict[str, Any]:
        self._event('node', node_id=node_id, state='running')
        try:
            outputs = component.invoke(inputs, batch)
            self._await_stage_outputs(node_id, outputs)
        except Exception as exc:
            self._event('node', node_id=node_id, state='failed', error=str(exc))
            raise
        self._event('node', node_id=node_id, state='completed')
        return outputs

    def _invoke_task(self, node_id: str, component: Any,
                     inputs: Dict[str, Any]) -> Dict[str, Any]:
        self._event('node', node_id=node_id, state='running')
        try:
            outputs = component.run(inputs)
        except Exception as exc:
            self._event('node', node_id=node_id, state='failed', error=str(exc))
            raise
        self._event('node', node_id=node_id, state='completed')
        return outputs

    @staticmethod
    def _await_stage_outputs(node_id: str, outputs: Any) -> None:
        """Hold 'running' until the stage's reference outputs are truly done.

        Remote-execution stages return LazyCollect handles; a handle is
        completion-probed with the public ``ready()`` (a non-materialising
        ray.wait(timeout=0)) so 'completed' always means the computation
        finished -- never merely that the call was submitted. A FAILED task
        also reports ready: the exception surfaces at the consuming collect,
        exactly like the kernel's own handle semantics."""
        refs = _collect_refs(outputs)
        while refs and not all(ref.ready() for ref in refs):
            time.sleep(0.05)

    # -- inputs and projections ------------------------------------------------

    def _inputs(self, node_id: str) -> Dict[str, Any]:
        """Assemble a node's named inputs from recorded outputs.

        Single ports receive the producing port's value directly. Many ports
        receive the slot-bearing pairs ``[(slot, value), ...]`` in edge
        declaration order: the slot IS the consuming parameter name, never
        guessed. Scope ports without a wire are completed from the
        graph-validated unique upstream Resource -- the value shape is
        identical to a wired port, so bindings never learn which path
        delivered it."""
        inputs: Dict[str, Any] = {}
        for port_key, source in self.graph.inputs_of(node_id).items():
            if isinstance(source, list):  # many: EdgeSpec list in declaration order
                inputs[port_key] = [(edge.slot, self._output_value(edge.source, edge.source_port))
                                    for edge in source]
            else:
                inputs[port_key] = self._output_value(source.source, source.source_port)
            for edge in source if isinstance(source, list) else [source]:
                self._event('edge', edge_id=edge.id, state='used')
        for port_key, (source_id, source_port) in self.graph.scope_of(node_id).items():
            if port_key not in inputs:
                inputs[port_key] = self._output_value(source_id, source_port)
        return inputs

    def _output_value(self, node_id: str, port_key: str) -> Any:
        try:
            return self._outputs[node_id][port_key]
        except KeyError:
            raise DAGError(f'{node_id}.{port_key}: upstream output unavailable; '
                           'run the producing node first') from None

    def _projection(self, node_id: str, component: Any) -> Dict[str, Any]:
        """A resident node's port values: per-port projection when the
        component declares ``port_value`` (an iterator's lanes off the LAST
        delivered batch), the resident object otherwise."""
        out: Dict[str, Any] = {}
        project = getattr(component, 'port_value', None)
        for port in self.graph.descriptor(node_id).ports_out:
            out[port.key] = project(port.key) if callable(project) else component.obj
        return out

    def _refresh_iterators(self) -> None:
        """Re-project every started iterator: port projections are live views
        of the LAST delivered batch (the frame source's samples lane)."""
        for node_id in list(self._outputs):
            if self.graph.descriptor(node_id).lifecycle == 'Iterator':
                self._project_iterator(node_id)

    def _project_iterator(self, node_id: str) -> None:
        outputs = self._projection(node_id, self._resolve(node_id))
        with self._lock:
            self._outputs[node_id] = outputs
        self._record_output(node_id, self.graph.descriptor(node_id).type, outputs)

    # -- preparation -----------------------------------------------------------

    def _prepare_round(self, participation: Set[str]) -> None:
        """Build un-built resources in dependency order and start the
        loop-free iterators feeding this segment (started once, projected
        resident -- the consumer pulls from the resident object)."""
        self._prepare_resources(participation)
        for node_id in [nid for nid in self.graph.order if nid in participation
                        and self.graph.descriptor(nid).lifecycle == 'Iterator']:
            if node_id in self._outputs:
                continue
            handle = self._source(node_id)
            if not handle.started:
                if handle.exhausted:
                    continue
                self._start_source(node_id)
            self._project_iterator(node_id)

    def _prepare_resources(self, participation: Set[str]) -> None:
        """按已就绪依赖构建资源；CPU 构建与设备资源构建重叠。

        模型和采样器构建会修改进程级环境与加载上下文，仍在调用线程串行执行；
        不把两个设备构建放进线程以共享这些临时状态。每个并行任务只写自己的
        输出副本，资源依赖在上一轮完成后才可用。依赖 Stage 输出的资源留给
        正常波次执行。"""
        pending = [node_id for node_id in self.graph.order if node_id in participation
                   and node_id not in self._outputs
                   and self.graph.descriptor(node_id).lifecycle == 'Resource']
        while pending:
            ready = [node_id for node_id in pending
                     if self._resource_dependencies(node_id) <= set(self._outputs)]
            if not ready:
                return
            # 绑定与懒导入在调用线程完成；worker 不修改 resident 表或指纹。
            bindings = {node_id: self._resolve(node_id) for node_id in ready}
            self._prewarm_imports(bindings)
            lanes = self.graph.access_plan(ready)
            device_lanes = []
            cpu_lanes = []
            for lane in lanes:
                owners = set().union(*(self.graph.resource_owners(node_id)
                                       for node_id in lane))
                for node_id in lane:
                    owners.update(f'{self.graph.descriptor(source).owner_kind}:{source}'
                                  for source, _port in self.graph.scope_of(node_id).values())
                destination = device_lanes if any(
                    owner.startswith(('MODEL:', 'SAMPLER:')) for owner in owners) else cpu_lanes
                destination.append(lane)
            with ThreadPoolExecutor() as pool:
                futures = [pool.submit(self._build_resource_lane, lane, bindings)
                           for lane in cpu_lanes]
                for lane in device_lanes:
                    self._merge_resource_lane(self._build_resource_lane(lane, bindings))
                for future in futures:
                    self._merge_resource_lane(future.result())
            pending = [node_id for node_id in pending if node_id not in ready]

    def _resource_dependencies(self, node_id: str) -> Set[str]:
        dependencies = {edge.source for source in self.graph.inputs_of(node_id).values()
                        for edge in (source if isinstance(source, list) else [source])}
        dependencies.update(source for source, _port in self.graph.scope_of(node_id).values())
        return dependencies

    def _merge_resource_lane(self, lane_outputs: Dict[str, Dict[str, Any]]) -> None:
        for node_id, outputs in lane_outputs.items():
            with self._lock:
                self._outputs[node_id] = outputs
            self._record_output(node_id, self.graph.descriptor(node_id).type, outputs)

    def _build_resource_lane(self, lane: List[str],
                             bindings: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        return {node_id: self._build_resource(node_id, bindings[node_id])
                for node_id in lane}

    def _build_resource(self, node_id: str, component: Any) -> Dict[str, Any]:
        if getattr(component, 'obj', None) is None:
            self._event('node', node_id=node_id, state='running')
            try:
                component.build(self._inputs(node_id))
            except Exception as exc:
                self._event('node', node_id=node_id, state='failed', error=str(exc))
                try:
                    component.release()
                except Exception:
                    pass  # a failed close must not mask the build's root cause
                raise
            self._event('node', node_id=node_id, state='completed')
        else:
            self._event('node', node_id=node_id, state='cached')
        return self._projection(node_id, component)

    @staticmethod
    def _prewarm_imports(bindings: Dict[str, Any]) -> None:
        """Serially import each ready component's build-chain entry modules.

        Runs on the driving thread BEFORE the pool starts. Importing one
        module executes its whole dependency chain; done serially here, every
        worker's later ``from X import Y`` hits sys.modules instead of racing
        another thread's first load of X (a from-import against a
        half-initialised module raises ImportError)."""
        import importlib
        for component in bindings.values():
            for module_name in getattr(component, 'IMPORTS', ()):
                importlib.import_module(module_name)

    # -- sources ---------------------------------------------------------------

    def _source(self, node_id: str) -> SourceHandle:
        """The resident source handle for an Iterator node (created on first use)."""
        if node_id not in self.sources:
            self.sources[node_id] = SourceHandle(node_id, self._resolve(node_id))
        return self.sources[node_id]

    def _start_source(self, node_id: str) -> None:
        handle = self._source(node_id)
        if handle.started:
            return
        self._event('node', node_id=node_id, state='running')
        try:
            handle.start(self._inputs(node_id))
        except Exception as exc:
            self._event('node', node_id=node_id, state='failed', error=str(exc))
            raise
        self._event('node', node_id=node_id, state='completed')

    def _take_source(self, node_id: str) -> Any:
        handle = self._source(node_id)
        reused = handle.has_pending  # must be read BEFORE take()
        self._event('node', node_id=node_id, state='cached' if reused else 'running')
        try:
            batch = handle.take()
        except Exception as exc:
            if handle.exhausted:
                raise SourceExhausted(f'source {node_id!r}: exhausted') from None
            self._event('node', node_id=node_id, state='failed', error=str(exc))
            raise
        if not reused:
            self._event('node', node_id=node_id, state='completed')
        self._event('source_batch', node_id=node_id, reused=reused)
        return batch

    # -- bookkeeping -----------------------------------------------------------

    def _trained_steps(self, node_ids: Set[str]) -> int:
        """Optimizer steps the given nodes' receipts actually completed."""
        total = 0
        for node_id in node_ids:
            node_out = self._outputs.get(node_id)
            if isinstance(node_out, dict) and isinstance(node_out.get('optimizer_steps'), int):
                total += node_out['optimizer_steps']
        return total

    def _emit_plan(self, participation: Set[str]) -> None:
        """本次参与范围不等于节点正在执行；下游无需先转动才能显示上游出线。"""
        edge_ids = []
        for node_id in self.graph.order:
            if node_id not in participation:
                continue
            for source in self.graph.inputs_of(node_id).values():
                for edge in source if isinstance(source, list) else [source]:
                    if edge.source in participation:
                        edge_ids.append(edge.id)
        self._event('run_plan', nodes=[node_id for node_id in self.graph.order
                                       if node_id in participation], edges=edge_ids)

    def _mark_running(self, node_id: str, running: bool = True) -> None:
        with self._lock:
            if running:
                self._running[node_id] = None
            else:
                self._running.pop(node_id, None)

    def _check_pause(self) -> None:
        """Block on a node boundary while paused (the running set has
        drained); stop releases the wait. Emits ``paused`` exactly once."""
        if not self._paused.is_set():
            return
        if not self._paused_emitted:
            self._event('paused', running=[])
            self._paused_emitted = True
        while self._paused.is_set() and not self._stop.is_set():
            time.sleep(0.05)

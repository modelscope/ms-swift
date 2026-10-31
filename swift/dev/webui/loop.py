# Copyright (c) ModelScope Contributors. All rights reserved.
"""The on-demand execution loop: click budgets, batch handover, training state.

Responsibilities (and nothing else):

- ``StepState``: the training-frame cadence -- global/micro/optimizer step
  counters, the click budget, the pending mini/micro cursor and frozen
  anchors. Only a real optimizer completion advances ``global_step``.
- ``SourceHandle``: the batch-handover contract around one resident iterator.
  The iterator is created once; every later call only takes the next batch.
  The CURRENT delivered batch and its acknowledgement state live here: a debug
  re-click reuses this batch, a fully consumed batch is acknowledged exactly
  once, an explicit skip acknowledges with 0 steps (cursor advances, no step).
- ``LoopRunner``: drives one graph call (``to_target`` / ``debug`` / ``train``)
  over the session's resident components. Click budgets belong to the runner,
  never to a driver's terminating ``max_steps``; a click never aborts the
  iterator, never triggers its ``finally`` cleanup.

The runner asks the session for resident bindings (``resolve``), emits events
through it (``event``) and records outputs (``record_output``) -- interfaces,
not imports, so the loop layer stays free of session internals.
"""
from __future__ import annotations

import time

from typing import Any, Callable, Dict, List, Optional

from .graph import FrameSpec, Graph


class LoopError(RuntimeError):
    """A loop-contract violation (double commit, unstarted source, budget misuse)."""


def _int_param(params: Dict[str, Any], key: str, default: int) -> int:
    """One integer node parameter (a typed v3 value or a legacy string)."""
    raw = params.get(key, default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        raise LoopError(f'parameter {key!r} must be an integer, got {raw!r}') from None


def _float_param(params: Dict[str, Any], key: str, default: float) -> float:
    """One float node parameter (a typed v3 value or a legacy string)."""
    raw = params.get(key, default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        raise LoopError(f'parameter {key!r} must be a number, got {raw!r}') from None


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
            raise LoopError(f'source {self.node_id!r}: iterator already started')
        if self.exhausted:
            raise LoopError(f'source {self.node_id!r}: iterator exhausted; restart '
                            'the source explicitly, it never auto-rewinds')
        self.component.start(inputs)
        self._iterator = self.component

    def take(self) -> Any:
        """The current batch when one is pending (debug re-click reuse), else
        the next batch off the iterator."""
        if self.exhausted:
            raise LoopError(f'source {self.node_id!r}: exhausted')
        if self.has_pending:
            return self._current
        if not self.started:
            raise LoopError(f'source {self.node_id!r}: take before start')
        try:
            batch = self.component.take_batch()
        except StopIteration as exc:
            self.exhausted = True
            raise LoopError(f'source {self.node_id!r}: exhausted: {exc}') from None
        self._current = batch
        self._acked = False
        return batch

    def commit(self, steps: int) -> None:
        """Acknowledge the current batch (exactly once) with the OPTIMIZER steps
        the training actually completed on it."""
        if self._current is None or self._acked:
            raise LoopError(f'source {self.node_id!r}: commit without a pending batch')
        if steps < 0:
            raise LoopError(f'source {self.node_id!r}: negative commit steps {steps}')
        self.component.commit(int(steps))
        self.last_steps = int(steps)
        self._acked = True
        self._current = None

    def skip(self) -> None:
        """Explicitly drop the current batch: commit(0), cursor advanced, no step."""
        if not self.has_pending:
            raise LoopError(f'source {self.node_id!r}: skip without a pending batch')
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


class StepState:
    """The training-frame cadence shared by every recipe family.

    The TOTAL horizon (LR schedule) is construction-time; the CLICK budget is
    per run-call and never reconfigures anything. ``global_step`` counts real
    optimizer completions only; ``micro_step`` counts forward_backward calls.
    """

    def __init__(self, *, gradient_accumulation_steps: int, max_grad_norm: float,
                 max_steps: int, click_steps: Optional[int] = None):
        self.ga = int(gradient_accumulation_steps)
        self.max_grad_norm = float(max_grad_norm)
        self.max_steps = int(max_steps)  #: the frame's total budget owner
        self.click_steps = click_steps  #: this click's追加 budget; None = unbounded
        self.global_step = 0
        self.micro_step = 0
        #: pending plan when a stop landed mid-batch: (mini index, micro offset)
        self.pending_cursor: Optional[Dict[str, int]] = None

    def reached_max(self) -> bool:
        if self.max_steps and self.global_step >= self.max_steps:
            return True
        return bool(self.click_steps) and self.global_step >= int(self.click_steps)

    def is_grad_sync_boundary(self) -> bool:
        return self.micro_step % self.ga == 0

    def record_optimizer_step(self) -> None:
        self.global_step += 1


class LoopRunner:
    """Drives one graph call over resident components.

    The session supplies the component surface; this class owns the call shape:
    upstream closure execution, per-batch stage invocation in topological
    order, the click-budget loop, the batch handover and the publish/commit
    pairing.
    """

    def __init__(self, graph: Graph, *, resolve: Callable[[str], Any],
                 event: Callable[..., None], record_output: Callable[[str, str, Any], None]):
        self.graph = graph
        self._resolve = resolve
        self._event = event
        self._record_output = record_output
        self.sources: Dict[str, SourceHandle] = {}

    # -- component surface ------------------------------------------------

    def source(self, node_id: str) -> SourceHandle:
        """The resident source handle for an Iterator node (created on first use)."""
        if node_id not in self.sources:
            self.sources[node_id] = SourceHandle(node_id, self._resolve(node_id))
        return self.sources[node_id]

    def _invoke_stage(self, node_id: str, inputs: Dict[str, Any], batch: Any) -> Dict[str, Any]:
        component = self._resolve(node_id)
        self._event('node', node_id=node_id, state='running')
        try:
            outputs = component.invoke(inputs, batch)
            self._await_stage_outputs(node_id, outputs)
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
        exactly like the kernel's own handle semantics. Local mode has no
        handles and pays nothing."""
        refs = _collect_refs(outputs)
        while refs and not all(ref.ready() for ref in refs):
            time.sleep(0.05)

    # -- inputs ------------------------------------------------------------

    def _inputs(self, node_id: str, outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Assemble a node's named inputs from upstream outputs.

        Single ports receive the producing port's value directly. Many ports
        receive the slot-bearing pairs ``[(slot, value), ...]`` in edge
        declaration order: the slot IS the consuming parameter name (a loss
        kwarg, a named lane), never guessed from shapes or completion order.
        Scope ports without a wire are completed from the graph-validated
        unique upstream Resource (connectivity shares variables) -- the value
        shape is identical to a wired port, so bindings never learn which
        path delivered it. Heavy values pass through as references either way."""
        inputs: Dict[str, Any] = {}
        for port_key, source in self.graph.inputs_of(node_id).items():
            if isinstance(source, list):  # many: EdgeSpec list in declaration order
                inputs[port_key] = [
                    (edge.slot, outputs[edge.source][edge.source_port]) for edge in source]
            else:
                inputs[port_key] = outputs[source.source][source.source_port]
        for port_key, (source_id, source_port) in self.graph.scope_of(node_id).items():
            if port_key not in inputs and source_id in outputs:
                inputs[port_key] = outputs[source_id][source_port]
        return inputs

    # -- execution ---------------------------------------------------------

    def run(self, target: Optional[str], mode: str = 'to_target',
            steps: Optional[int] = None) -> Dict[str, Any]:
        """One graph call.

        - ``to_target``: execute the target's same-round upstream closure once;
          no training-frame effects, no loop.
        - ``debug``: one batch flows to the target; train steps do NOT train
          (their frame effects are suppressed and the batch stays pending).
        - ``train``: execute the target's training frame: per iteration, take a
          batch off the frame's source, run the members in order, register the
          publish, commit with the REAL optimizer steps, until the click budget
          or the frame budget is reached.
        """
        target = target or (self.graph.order[-1] if self.graph.order else None)
        if target is None:
            raise LoopError('empty graph has nothing to run')
        closure = self.graph.upstream_closure(target)
        frame = self.graph.training_frame(target)

        if mode == 'train':
            if frame is None:
                raise LoopError(
                    f'target {target!r} is not a training-frame member; use to_target or debug')
            return self._run_frame(frame, target, steps)

        outputs: Dict[str, Dict[str, Any]] = {}
        batch = None
        source_ids = [nid for nid in closure
                      if self.graph.descriptor(nid).lifecycle == 'Iterator']
        for source_id in source_ids:
            handle = self.source(source_id)
            if not handle.started:
                self._prepare_upstream(source_id, outputs)
                self._event('node', node_id=source_id, state='running')
                handle.start(self._inputs(source_id, outputs))
                self._event('node', node_id=source_id, state='completed')
        if source_ids:
            batch = self.source(source_ids[0]).take()
            self._event('source_batch', node_id=source_ids[0],
                        reused=self.source(source_ids[0]).has_pending is False)
        for node_id in [nid for nid in self.graph.order if nid in closure]:
            descriptor = self.graph.descriptor(node_id)
            self._run_node(node_id, descriptor, outputs, batch, debug=(mode == 'debug'))
        if mode != 'debug' and source_ids:
            # to_target consumed the batch: acknowledge with the real steps (0
            # when the closure trained nothing).
            handle = self.source(source_ids[0])
            if handle.has_pending:
                handle.commit(self._trained_steps(outputs))
        return {'mode': mode, 'target': target, 'outputs': self._bounded_outputs(outputs)}

    def _prepare_upstream(self, node_id: str, outputs: Dict[str, Dict[str, Any]]) -> None:
        """Build the source's resident upstream before the iterator starts.

        A source's inputs wire Resource nodes (the engine, the prompts); those
        residents must EXIST for the iterator's own construction, so every
        Resource in the source's upstream closure runs here, in graph order
        (upstream first, so chained resources resolve). Re-running a node
        whose object is already resident is a no-op."""
        closure = self.graph.upstream_closure(node_id)
        for member in [nid for nid in self.graph.order if nid in closure]:
            descriptor = self.graph.descriptor(member)
            if descriptor.lifecycle == 'Resource':
                self._run_node(member, descriptor, outputs, None)

    def _resident_ports(self, node_id: str, component: Any, descriptor) -> Dict[str, Any]:
        """A resident node's port values: per-port projection when the component
        declares ``port_value`` (an iterator's lanes off the LAST delivered
        batch), the resident object itself otherwise (a resource IS its ports)."""
        out: Dict[str, Any] = {'__resident__': node_id}
        project = getattr(component, 'port_value', None)
        for port in descriptor.ports_out:
            out[port.key] = project(port.key) if callable(project) else component.obj
        return out

    def _run_node(self, node_id: str, descriptor, outputs: Dict[str, Dict[str, Any]],
                  batch: Any, debug: bool = False) -> None:
        lifecycle = descriptor.lifecycle
        if lifecycle == 'Resource':
            component = self._resolve(node_id)
            if getattr(component, 'obj', None) is None:
                self._event('node', node_id=node_id, state='running')
                inputs = self._inputs(node_id, outputs)
                component.build(inputs)
                self._event('node', node_id=node_id, state='completed')
            outputs[node_id] = {'__resident__': node_id}
            for port in descriptor.ports_out:
                outputs[node_id][port.key] = component.obj
            return
        if lifecycle == 'Iterator':
            component = self._resolve(node_id)
            outputs[node_id] = self._resident_ports(node_id, component, descriptor)
            return
        if lifecycle == 'Task':
            component = self._resolve(node_id)
            self._event('node', node_id=node_id, state='running')
            try:
                outputs[node_id] = component.run(self._inputs(node_id, outputs))
            except Exception as exc:
                self._event('node', node_id=node_id, state='failed', error=str(exc))
                raise
            self._event('node', node_id=node_id, state='completed')
            self._record_output(node_id, descriptor.type, outputs[node_id])
            return
        # Stage: invoke per batch. In debug mode training-effect stages run in
        # dry form (the binding decides what that means; the batch stays pending).
        component = self._resolve(node_id)
        inputs = self._inputs(node_id, outputs)
        if debug and 'train' in descriptor.effects:
            self._event('node', node_id=node_id, state='cached',
                        note='debug: training effect suppressed, batch stays pending')
            outputs[node_id] = {'debug': True}
            return
        outputs[node_id] = self._invoke_stage(node_id, inputs, batch)
        self._record_output(node_id, descriptor.type, outputs[node_id])

    def _trained_steps(self, outputs: Dict[str, Dict[str, Any]]) -> int:
        """Optimizer steps the closure's train steps actually completed."""
        total = 0
        for node_id, node_out in outputs.items():
            if isinstance(node_out, dict) and isinstance(node_out.get('optimizer_steps'), int):
                total += node_out['optimizer_steps']
        return total

    def _run_frame(self, frame: FrameSpec, target: str, steps: Optional[int]) -> Dict[str, Any]:
        """The training loop for one click: batches until the click budget ends.

        The frame's budget owner contributes max_steps / GA / clip; the click
        budget only bounds THIS call. Each iteration: take a batch, run the
        members in order, register the publish intent, commit the source with
        the real optimizer steps. Stopping between clicks keeps everything
        resident; nothing here aborts a source.
        """
        owner = self.graph.node(frame.budget_owner or target)
        owner_params = owner.params
        step_state = StepState(
            gradient_accumulation_steps=_int_param(owner_params, 'gradient_accumulation_steps', 1),
            max_grad_norm=_float_param(owner_params, 'max_grad_norm', 1.0),
            max_steps=_int_param(owner_params, 'max_steps', 0),
            click_steps=steps)
        source_ids = [nid for nid in frame.members
                      if self.graph.descriptor(nid).lifecycle == 'Iterator']
        if not source_ids:
            raise LoopError(f'frame {frame.name!r} has no source iterator')
        source = self.source(source_ids[0])
        if not source.started:
            prep: Dict[str, Dict[str, Any]] = {}
            self._prepare_upstream(source.node_id, prep)
            self._event('node', node_id=source.node_id, state='running')
            source.start(self._inputs(source.node_id, prep))
            self._event('node', node_id=source.node_id, state='completed')
        iterations = 0
        try:
            while not step_state.reached_max() and not source.exhausted:
                batch = source.take()
                iterations += 1
                outputs: Dict[str, Dict[str, Any]] = {}
                for node_id in [nid for nid in self.graph.order if nid in set(frame.members)]:
                    descriptor = self.graph.descriptor(node_id)
                    if descriptor.lifecycle == 'Iterator':
                        component = self._resolve(node_id)
                        outputs[node_id] = self._resident_ports(node_id, component, descriptor)
                        continue
                    self._run_node(node_id, descriptor, outputs, batch, debug=False)
                    if node_id == target and descriptor.lifecycle != 'Stage':
                        break
                trained = self._trained_steps(outputs)
                # publish registration precedes the commit; the push itself runs in
                # the driver's next pass. Never wait for the push before committing.
                source.commit(trained if trained >= 0 else 0)
                if trained <= 0 and iterations > 1:
                    break  # no progress guard: a frame that trains nothing stops
        finally:
            self._event('frame_progress', iterations=iterations,
                        global_step=step_state.global_step,
                        pending=source.has_pending)
        return {'mode': 'train', 'target': target, 'iterations': iterations,
                'outputs': {}}

    def _bounded_outputs(self, outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Outputs for the caller: residents collapse to their node id (the
        session holds the real objects); everything else passes as declared."""
        bounded: Dict[str, Dict[str, Any]] = {}
        for node_id, node_out in outputs.items():
            if isinstance(node_out, dict):
                bounded[node_id] = {k: v for k, v in node_out.items() if k != '__resident__'}
            else:
                bounded[node_id] = node_out
        return bounded

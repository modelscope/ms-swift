# Copyright (c) ModelScope Contributors. All rights reserved.
"""The resident graph session: one workflow key, one long-lived ray process.

Command protocol (the vite dev server drives this over stdin/stdout):

- stdin: one JSON object per line -- ``run`` / ``pause`` / ``resume`` /
  ``stop`` / ``release`` / ``preview`` / ``update`` / ``reset``, each
  carrying a ``request_id`` (duplicate ids are acknowledged as duplicates,
  not re-executed).
- stdout: every line is ``[workflow] {json}`` -- session lifecycle, node
  running/completed/failed/cached, edge usage, paused/resumed markers and
  run results. Logs go to stderr; the stdout channel stays parseable.

The session ALWAYS boots ray (there is no local mode): ``_bootstrap`` plans
the graph's DeviceGroups from the declaration's resource section and
declares them once at ``twinkle.initialize``.

Execution runs on ONE background thread (the DAGController drives parallel
ready sets there); every command replies immediately -- ``run`` returns
``accepted`` and the result rides the ``run_end`` event, while ``pause`` /
``resume`` / ``stop`` / ``preview`` never wait for execution.

State machine: ``idle / running / pausing / paused / stopping / stopped /
failed / reset_required / releasing / released``. ``stopped`` only means
THIS call stopped issuing new node invocations -- resident components may
still be finishing submitted work and the source may hold a bounded
prefetch; ``release`` is the only path that terminates sources and closes
resources. A browser disconnect never releases; the frontend closes its
session explicitly.
"""
from __future__ import annotations

import json
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from .component import COMPONENT_CLASSES, ComponentError
from .dag import DAGController, DAGError
from .graph import Graph, GraphValidationError, NodeSpec
from .placement import NUMERIC_GROUP, plan_device_groups

#: Fixed stdout prefix; the frontend splits on it and never parses logs.
PREFIX = '[workflow] '

STATES = ('idle', 'running', 'pausing', 'paused', 'stopping', 'stopped', 'failed',
          'reset_required', 'releasing', 'released')

#: Preview page bounds: one lane page, bounded for the stdout channel.
PREVIEW_LIMIT_MAX = 1000


class GraphSession:
    """The resident execution surface for one workflow declaration."""

    def __init__(self, declaration: Dict[str, Any], *, output_dir: str = 'output'):
        self._graph = Graph.from_declaration(declaration)
        self._declaration = self._graph.to_dict()
        self._output_dir = output_dir
        self._device_plan = plan_device_groups(self._graph)
        self._residents: Dict[str, Any] = {}
        self._outputs: Dict[str, Dict[str, Any]] = {}
        self._fingerprints: Dict[str, str] = {}
        self._has_training_state = False
        self.state = 'idle'
        self.epoch = 0
        self.revision = 0
        self._run_lock = threading.Lock()
        self._active = False
        #: One execution thread: the controller drives waves there while the
        #: command loop keeps answering (pause/preview arrive mid-run).
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='webui-run')
        self._controller: Optional[DAGController] = None
        #: Iterator consumption progress, resident across runs (the
        #: click-to-click cursor). SourceHandle state is per-source.
        self._sources: Dict[str, Any] = {}
        self._seq = 0
        self._event_lock = threading.Lock()
        self._request_id = ''
        self._seen_requests: set = set()

    # -- events -----------------------------------------------------------

    def emit(self, kind: str, **payload: Any) -> None:
        """One stdout event line; seq is monotonic per session."""
        # 并行构建的事件必须整行写出，序号与写入顺序保持一致。
        with self._event_lock:
            self._seq += 1
            row = {'event': kind, 'epoch': self.epoch, 'revision': self.revision,
                   'seq': self._seq, 'request_id': self._request_id,
                   'state': self.state, **payload}
            try:
                sys.stdout.write(PREFIX + json.dumps(row, ensure_ascii=False, default=str) + '\n')
                sys.stdout.flush()
            except Exception:
                pass  # a closed pipe must not take the session down mid-run

    def event(self, kind: str, **payload: Any) -> None:
        """The controller's event channel; pause markers move the state."""
        if kind == 'paused':
            self.state = 'paused'
        elif kind == 'resumed':
            self.state = 'running'
        self.emit(kind, **payload)

    # -- component surface --------------------------------------------------

    def resolve(self, node_id: str) -> Any:
        """The resident binding for a node: built once, keyed by node id.

        Fingerprints guard the once-ness: a changed construction param set
        invalidates the resident (and, when training state exists, demands an
        explicit reset instead of silently rebuilding under it). GPU
        components receive their planned DeviceGroup name here -- the one
        place residents are born.
        """
        node = self._graph.node(node_id)
        if node_id in self._residents:
            fingerprint = self._fingerprint(node)
            if self._fingerprints.get(node_id) != fingerprint:
                if self._has_training_state and self._affects_training(node):
                    self.state = 'reset_required'
                    raise ComponentError(
                        f'{node_id}: construction changed with training state present -- '
                        'reset_required; confirm the reset to rebuild')
                self._invalidate(node_id)
            else:
                return self._residents[node_id]
        component = self._build_binding(node)
        self._residents[node_id] = component
        self._fingerprints[node_id] = self._fingerprint(node)
        return component

    def _build_binding(self, node: NodeSpec) -> Any:
        # 懒导入 components 包：其 import 副作用把全部内置绑定注册进
        # COMPONENT_CLASSES（链条触及 twinkle，不能在模块顶层拉起）。
        from . import components  # noqa: F401 -- registration side effect
        cls = COMPONENT_CLASSES.get(node.type)
        if cls is None:
            raise ComponentError(f'{node.id}: no component binding for type {node.type!r}')
        component = cls(node.id, {k: v for k, v in node.params.items()})
        component.remote_group = self._device_plan.remote_groups.get(node.id, '')
        return component

    def _fingerprint(self, node: NodeSpec) -> str:
        """Construction fingerprint: the node's own params + its upstream
        resources' fingerprints (a changed upstream IS a changed resident).

        Upstream means wired inputs AND scope-resolved sources: a variable
        the node consumes through connectivity (never through a wire) still
        invalidates the resident when its defining node changes."""
        upstream_ids = []
        for source in self._graph.inputs_of(node.id).values():
            for edge in (source if isinstance(source, list) else [source]):
                upstream_ids.append(edge.source)
        upstream_ids.extend(source_id for source_id, _port in self._graph.scope_of(node.id).values())
        return json.dumps({'params': node.params,
                           'upstream': [self._fingerprints.get(src, '') for src in upstream_ids]},
                          sort_keys=True, default=str)

    def _affects_training(self, node: NodeSpec) -> bool:
        return node.type in ('model', 'train_step')

    def _invalidate(self, node_id: str) -> None:
        """Drop a resident and everyone downstream of it."""
        drop = {node_id}
        changed = True
        while changed:
            changed = False
            for e in self._graph.edges:
                if e.source in drop and e.target not in drop:
                    drop.add(e.target)
                    changed = True
        for nid in drop:
            resident = self._residents.pop(nid, None)
            if resident is not None:
                release = getattr(resident, 'release', None)
                if callable(release):
                    try:
                        release()
                    except Exception:
                        pass
            handle = self._sources.pop(nid, None)
            if handle is not None:
                try:
                    handle.release()
                except Exception:
                    pass
            self._fingerprints.pop(nid, None)
            self._outputs.pop(nid, None)
        if self._controller is not None:
            self._controller.drop_outputs(sorted(drop))

    def record_output(self, node_id: str, node_type: str, output: Any) -> None:
        """Keep one invocation's output for the preview surface, as-is.

        Remote-execution stages hand back references: they are recorded
        UNOPENED (a reference costs the session nothing), and only the
        preview command -- the user explicitly asking to see the value --
        materialises and bounds them."""
        self._outputs[node_id] = {'node': node_id, 'type': node_type, 'output': output}

    # -- controller ----------------------------------------------------------

    def controller(self) -> DAGController:
        """The graph's controller (rebuilt per revision; cursors persist in
        ``self._sources`` across rebuilds)."""
        if self._controller is None:
            self._controller = DAGController(
                self._graph, resolve=self.resolve, event=self.event,
                record_output=self.record_output, sources=self._sources)
        return self._controller

    # -- commands -----------------------------------------------------------

    def command(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch one stdin command; the caller writes the reply."""
        request_id = str(cmd.get('request_id', ''))
        if request_id and request_id in self._seen_requests:
            return {'ok': True, 'duplicate': True, 'request_id': request_id}
        op = str(cmd.get('op', cmd.get('command', '')))
        if op == 'run':
            return self._run(cmd, request_id)
        if op == 'pause':
            return self._pause()
        if op == 'resume':
            return self._resume()
        if op == 'stop':
            return self._stop()
        if op == 'release':
            return self._release()
        if op == 'preview':
            return self._preview(cmd)
        if op == 'update':
            return self._update(cmd)
        if op == 'reset':
            return self.reset()
        return {'ok': False, 'error': f'unknown command {op!r}'}

    def _run(self, cmd: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Submit one run to the execution thread; the reply is immediate and
        the result rides the ``run_end`` event."""
        with self._run_lock:
            if self._active:
                return {'ok': False, 'error': 'busy: a run is already active',
                        'state': self.state}
            if self.state in ('releasing', 'released'):
                return {'ok': False, 'error': f'session is {self.state}', 'state': self.state}
            self._active = True
            self._request_id = request_id
        if request_id:
            self._seen_requests.add(request_id)
        mode = str(cmd.get('mode', 'component'))
        if mode not in ('component', 'full'):
            with self._run_lock:
                self._active = False
                self._request_id = ''
            return {'ok': False, 'error': f"mode must be 'component' or 'full', got {mode!r}"}
        target = cmd.get('target') or None
        steps = cmd.get('steps')
        observe = bool(cmd.get('observe'))
        self.state = 'running'
        self.emit('run_begin', request_id=request_id, target=target or '', mode=mode,
                  observe=observe)
        self._executor.submit(self._drive, target, mode, steps, observe, request_id)
        return {'ok': True, 'accepted': True, 'state': self.state, 'request_id': request_id}

    def _drive(self, target: Optional[str], mode: str, steps: Optional[Any],
               observe: bool, request_id: str) -> None:
        """The execution thread body: drive one controller call to its end."""
        try:
            controller = self.controller()
            if mode == 'full':
                result = controller.run(target, steps=_optional_int(steps, 'steps'))
            else:
                result = controller.run_component(target, steps=_optional_int(steps, 'steps'),
                                                  observe=observe)
            self._has_training_state = True
            self.state = 'stopped' if result.get('state') == 'stopped' else 'idle'
            self.emit('run_end', request_id=request_id, state=self.state, result=_bound(result))
        except (DAGError, ComponentError, GraphValidationError) as exc:
            self.state = 'failed'
            self.emit('run_end', request_id=request_id, state='failed', error=str(exc))
        except Exception as exc:  # unexpected failures surface with the trace
            self.state = 'failed'
            trace = traceback.format_exc()
            print(trace, file=sys.stderr)
            self.emit('run_end', request_id=request_id, state='failed', error=str(exc))
        finally:
            with self._run_lock:
                self._active = False
                self._request_id = ''

    def _pause(self) -> Dict[str, Any]:
        """Pause at the next node boundary: running nodes complete, no
        further node starts. The ``paused`` event lands once the running set
        drains."""
        if not self._active or self._controller is None:
            return {'ok': False, 'error': 'no active run to pause', 'state': self.state}
        reply = self._controller.pause()
        if reply['ok']:
            self.state = 'pausing'
            self.emit('pause', state=self.state)
        return reply

    def _resume(self) -> Dict[str, Any]:
        if self._controller is None:
            return {'ok': False, 'error': 'no controller', 'state': self.state}
        reply = self._controller.resume()
        if reply['ok']:
            self.state = 'running'
            self.emit('resume', state=self.state)
        return reply

    def _stop(self) -> Dict[str, Any]:
        """Stop this call: no new node invocations start; submitted atomic
        training/sync work may finish and report its real result."""
        if self._controller is not None:
            self._controller.stop()
        with self._run_lock:
            self.state = 'stopping' if self._active else 'stopped'
        self.emit('stop', state=self.state)
        return {'ok': True, 'state': self.state}

    def _release(self) -> Dict[str, Any]:
        """Terminate sources and close resources. Idempotent; the only path
        that ends resident components."""
        self.state = 'releasing'
        self.emit('release_begin')
        for handle in list(self._sources.values()):
            try:
                handle.release()
            except Exception:
                pass
        self._sources.clear()
        for node_id in list(self._residents):
            resident = self._residents.pop(node_id)
            release = getattr(resident, 'release', None)
            if callable(release):
                try:
                    release()
                except Exception:
                    pass
        self._residents.clear()
        self._fingerprints.clear()
        self._outputs.clear()
        self._controller = None
        self.state = 'released'
        self.emit('release_end')
        return {'ok': True, 'state': 'released'}

    def _preview(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        """Bounded, PAGINATED output preview -- the one place references are
        materialised (the user explicitly asked to see the value).

        Without a node: the recorded nodes and their lanes. With a node but
        no lane: every lane's bounded summary. With a lane: that lane's page
        (``offset``/``limit`` over its rows; a non-list lane returns whole).
        """
        node_id = str(cmd.get('node_id', ''))
        if not node_id:
            return {'ok': True, 'nodes': {
                nid: {'type': row['type'], 'lanes': sorted(row['output'])}
                for nid, row in self._outputs.items()}}
        if node_id not in self._outputs:
            return {'ok': False, 'error': f'{node_id}: no recorded output; run the node first'}
        row = self._outputs[node_id]
        lane = str(cmd.get('lane', ''))
        if not lane:
            return {'ok': True, 'node': node_id, 'type': row['type'],
                    'lanes': {key: _lane_summary(_deref(value))
                              for key, value in row['output'].items()}}
        if lane not in row['output']:
            return {'ok': False,
                    'error': f'{node_id}.{lane}: 未声明输出通道；可用 {sorted(row["output"])}'}
        offset = _bounded_int(cmd.get('offset'), 'offset', 0, minimum=0)
        limit = _bounded_int(cmd.get('limit'), 'limit', 20, minimum=1, maximum=PREVIEW_LIMIT_MAX)
        value = _deref(row['output'][lane])
        if isinstance(value, (list, tuple)):
            rows = [_bound(_deref(item)) for item in value[offset:offset + limit]]
            return {'ok': True, 'node': node_id, 'type': row['type'], 'lane': lane,
                    'offset': offset, 'limit': limit, 'total': len(value), 'rows': rows}
        return {'ok': True, 'node': node_id, 'type': row['type'], 'lane': lane,
                'value': _bound(value)}

    def _update(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        """A new declaration revision (edit-then-continue): revalidate; only
        the changed residents invalidate, training state demands a reset.

        The device plan is immutable -- groups were declared at
        ``twinkle.initialize`` -- so any placement change is rejected with a
        rebuild instruction instead of silently running on the old groups.
        """
        declaration = cmd.get('declaration') or cmd.get('graph')
        if not declaration:
            return {'ok': False, 'error': 'update needs a declaration'}
        try:
            graph = Graph.from_declaration(declaration)
            plan = plan_device_groups(graph)
        except GraphValidationError as exc:
            return {'ok': False, 'error': str(exc)}
        if plan.signature() != self._device_plan.signature():
            return {'ok': False, 'state': 'rebuild_required',
                    'error': '资源分配（资源页选择或组件占卡）已变更：DeviceGroup 在 '
                             'twinkle.initialize 时一次性声明，不可运行期改组 -- 请重建会话'}
        self._graph = graph
        self._declaration = graph.to_dict()
        self._device_plan = plan
        self.revision += 1
        for node_id in list(self._residents):
            node = self._graph.node(node_id) if node_id in {n.id for n in graph.nodes} else None
            if node is None:
                self._invalidate(node_id)
                continue
            if self._fingerprints.get(node_id) != self._fingerprint(node):
                if self._has_training_state and self._affects_training(node):
                    self.state = 'reset_required'
                    self.emit('update', state='reset_required', node=node_id)
                    continue
                self._invalidate(node_id)
        for node_id in list(self._sources):
            if node_id not in {n.id for n in graph.nodes}:
                handle = self._sources.pop(node_id)
                try:
                    handle.release()
                except Exception:
                    pass
        # A revision rebuilds the controller: the new graph's first click
        # recomputes what still counts as un-run; residents and cursors stay.
        self._controller = None
        self.emit('update', revision=self.revision)
        return {'ok': True, 'revision': self.revision, 'state': self.state}

    def reset(self) -> Dict[str, Any]:
        """Explicit reset: drop training state and rebuild everything."""
        self._has_training_state = False
        for handle in list(self._sources.values()):
            try:
                handle.release()
            except Exception:
                pass
        self._sources.clear()
        for node_id in list(self._residents):
            self._invalidate(node_id)
        self._controller = None
        self.state = 'idle'
        self.emit('reset')
        return {'ok': True, 'state': 'idle'}


def _deref(value: Any) -> Any:
    """Resolve a LazyCollect handle (inlined to keep this module import-light)."""
    if callable(value) and getattr(value, '_is_lazy_collect', False):
        return value()
    return value


def _lane_summary(value: Any) -> Any:
    """One lane's overview: a list's head plus its total, else the value."""
    if isinstance(value, (list, tuple)):
        return {'total': len(value), 'head': [_bound(item) for item in value[:2]]}
    return _bound(value)


def _bound(value: Any, depth: int = 0) -> Any:
    """Bound a value for stdout: dicts/lists keep structure (bounded depth);
    everything else becomes its string form. Heavy payloads never reach here
    -- components record bounded summaries only. Depth 6: a lane page wraps
    rows in a structural layer over node/port lanes, so a scalar riding a
    preview page sits several layers deep."""
    if depth > 6:
        return '...'
    if isinstance(value, dict):
        return {str(k): _bound(v, depth + 1) for k, v in list(value.items())[:32]}
    if isinstance(value, (list, tuple)):
        return [_bound(v, depth + 1) for v in list(value)[:16]]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)[:256]


def _bounded_int(value: Any, name: str, default: int, *, minimum: int = 0,
                 maximum: Optional[int] = None) -> int:
    try:
        result = int(value if value is not None else default)
    except (TypeError, ValueError):
        raise GraphValidationError(f'{name}: 需要整数，得到 {value!r}') from None
    if result < minimum or maximum is not None and result > maximum:
        top = f'..{maximum}' if maximum is not None else ''
        raise GraphValidationError(f'{name}: 需要在 {minimum}{top} 范围内，得到 {result}')
    return result


def _optional_int(value: Any, name: str) -> Optional[int]:
    if value is None:
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise GraphValidationError(f'{name}: 需要整数，得到 {value!r}') from None
    return result if result >= 0 else None


def _bootstrap(declaration: Dict[str, Any]) -> None:
    """Bring up the ALWAYS-ray runtime and declare the graph's device groups.

    Local: the resource page's cards become the process's visible GPUs (ray
    maps its slots onto them). Cluster: attach to the given address; groups
    are numbered over the announced GPU count. The numeric worker's CPU
    group rides along -- every group must exist here because twinkle only
    declares groups at init time.
    """
    import os

    graph = Graph.from_declaration(declaration)
    plan = plan_device_groups(graph)
    import twinkle
    from twinkle import DeviceGroup, Platform

    if graph.resource.mode == 'cluster':
        import ray
        ray.init(address=graph.resource.cluster_address, ignore_reinit_error=True,
                 namespace='webui_session')
    else:
        # Ray's local slots map onto exactly the selected physical cards.
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(card) for card in graph.resource.cards)
    device_groups = [
        DeviceGroup(name=group.name, ranks=list(group.ranks),
                    device_type=Platform.get_platform().__name__,
                    gpus_per_worker=group.gpus_per_worker, shared=False)
        for group in plan.groups]
    device_groups.append(DeviceGroup(name=NUMERIC_GROUP, ranks=1, device_type='CPU'))
    twinkle.initialize(
        mode='ray',
        nproc_per_node=plan.nproc_per_node,
        groups=device_groups,
        name='webui_session',
    )
    # The CPU group's ranks stay the integer process count (twinkle's
    # DeviceGroup contract); the GPU groups carry explicit rank lists.
    print(f'[workflow-bootstrap] device groups: '
          f'{[(group.name, group.ranks, group.gpus_per_worker) for group in device_groups]}',
          file=sys.stderr)


def main(argv: Optional[List[str]] = None) -> int:
    """The resident session process entry.

    Reads JSON commands from stdin (one per line), writes ``[workflow]``-prefixed
    JSON events and command replies to stdout, logs to stderr. The process ends
    at EOF (the frontend closed the session) with a final released state.

    Usage: ``python -m swift.dev.webui.session graph.json``.
    """
    declaration: Optional[Dict[str, Any]] = None
    output_dir = 'output'
    argv = list(sys.argv[1:] if argv is None else argv)
    args = [arg for arg in argv if not arg.startswith('--')]
    if args:
        with open(args[0], encoding='utf-8') as f:
            declaration = json.load(f)
    if not declaration:
        print('usage: python -m swift.dev.webui.session graph.json',
              file=sys.stderr)
        return 2
    _bootstrap(declaration)
    session = GraphSession(declaration, output_dir=output_dir)
    session.emit('session', state='idle', nodes=len(session._graph.nodes))
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
        except ValueError:
            reply = {'event': 'reply', 'ok': False, 'error': 'malformed command line'}
        else:
            op = str(cmd.get('op', cmd.get('command', '')))
            reply = session.command(cmd)
            reply = {'event': 'reply', 'op': op,
                     'request_id': str(cmd.get('request_id', '')), **reply}
        try:
            sys.stdout.write(PREFIX + json.dumps(reply, ensure_ascii=False, default=str) + '\n')
            sys.stdout.flush()
        except Exception:
            break
        if session.state == 'released':
            break
    if session.state != 'released':
        session._release()
    session._executor.shutdown(wait=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

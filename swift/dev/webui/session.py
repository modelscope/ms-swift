# Copyright (c) ModelScope Contributors. All rights reserved.
"""The resident graph session: one workflow key, one long-lived process.

Command protocol (the vite dev server drives this over stdin/stdout):

- stdin: one JSON object per line -- ``run`` / ``stop`` / ``release`` /
  ``preview`` commands, each carrying a ``request_id`` (duplicate ids are
  acknowledged as duplicates, not re-executed).
- stdout: every line is ``[workflow] {json}`` -- session lifecycle, node
  begin/completed/failed/cached, edge usage and run results. Logs go to
  stderr; the stdout channel stays parseable.

State machine: ``idle / running / stopping / stopped / failed /
reset_required / releasing / released``. ``stopped`` only means THIS graph
call stopped issuing new calls -- resident components may still be finishing
submitted work and the source may hold a bounded prefetch; ``release`` is the
only path that terminates sources and closes resources. A browser disconnect
never releases; the frontend closes its session explicitly.
"""
from __future__ import annotations

import json
import sys
import threading
import traceback
from typing import Any, Dict, List, Optional

from .component import COMPONENT_CLASSES, ComponentError, ResourceComponent, TaskComponent
from .graph import Graph, GraphValidationError, NodeSpec
from .loop import LoopError, LoopRunner, SourceHandle

#: Fixed stdout prefix; the frontend splits on it and never parses logs.
PREFIX = '[workflow] '

STATES = ('idle', 'running', 'stopping', 'stopped', 'failed',
          'reset_required', 'releasing', 'released')


class GraphSession:
    """The resident execution surface for one workflow declaration."""

    def __init__(self, declaration: Dict[str, Any], *, output_dir: str = 'output'):
        self._graph = Graph.from_declaration(declaration)
        self._declaration = self._graph.to_dict()
        self._output_dir = output_dir
        self._residents: Dict[str, Any] = {}
        self._outputs: Dict[str, Dict[str, Any]] = {}
        self._fingerprints: Dict[str, str] = {}
        self._has_training_state = False
        self.state = 'idle'
        self.epoch = 0
        self.revision = 0
        self._run_lock = threading.Lock()
        self._active = False
        self._stop_requested = False
        self._runner: Optional[LoopRunner] = None
        #: Iterator consumption progress, resident across runs (the click-to-
        #: click cursor). SourceHandle state is per-source, never per-call.
        self._sources: Dict[str, SourceHandle] = {}
        self._seq = 0
        self._seen_requests: set = set()

    # -- events -----------------------------------------------------------

    def emit(self, kind: str, **payload: Any) -> None:
        """One stdout event line; seq is monotonic per session."""
        self._seq += 1
        row = {'event': kind, 'epoch': self.epoch, 'revision': self.revision,
               'seq': self._seq, 'state': self.state, **payload}
        try:
            sys.stdout.write(PREFIX + json.dumps(row, ensure_ascii=False, default=str) + '\n')
            sys.stdout.flush()
        except Exception:
            pass  # a closed pipe must not take the session down mid-run

    # -- component surface --------------------------------------------------

    def resolve(self, node_id: str) -> Any:
        """The resident binding for a node: built once, keyed by node id.

        Fingerprints guard the once-ness: a changed construction param set
        invalidates the resident (and, when training state exists, demands an
        explicit reset instead of silently rebuilding under it).
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
        return cls(node.id, {k: v for k, v in node.params.items()})

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

    def record_output(self, node_id: str, node_type: str, output: Any) -> None:
        """Keep one invocation's output for the preview surface, as-is.

        Remote-execution stages hand back references: they are recorded
        UNOPENED (a reference costs the session nothing), and only the
        preview command -- the user explicitly asking to see the value --
        materialises and bounds them."""
        self._outputs[node_id] = {'node': node_id, 'type': node_type, 'output': output}

    def event(self, kind: str, **payload: Any) -> None:
        self.emit(kind, **payload)

    # -- commands -----------------------------------------------------------

    def command(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch one stdin command; the caller writes the reply."""
        request_id = str(cmd.get('request_id', ''))
        if request_id and request_id in self._seen_requests:
            return {'ok': True, 'duplicate': True, 'request_id': request_id}
        op = str(cmd.get('op', cmd.get('command', '')))
        if op == 'run':
            return self._run(cmd, request_id)
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
        with self._run_lock:
            if self._active:
                return {'ok': False, 'error': 'busy: a run is already active',
                        'state': self.state}
            if self.state in ('releasing', 'released'):
                return {'ok': False, 'error': f'session is {self.state}', 'state': self.state}
            self._active = True
            self._stop_requested = False
        if request_id:
            self._seen_requests.add(request_id)
        self.state = 'running'
        self.emit('run_begin', request_id=request_id,
                  target=str(cmd.get('target') or ''), mode=str(cmd.get('mode', 'to_target')))
        try:
            runner = LoopRunner(
                self._graph, resolve=self.resolve, event=self.event,
                record_output=self.record_output, sources=self._sources)
            self._runner = runner
            result = runner.run(
                target=cmd.get('target') or None,
                mode=str(cmd.get('mode', 'to_target')),
                steps=cmd.get('steps'))
            self._has_training_state = True
            self.state = 'idle' if not self._stop_requested else 'stopped'
            self.emit('run_end', request_id=request_id, state=self.state,
                      result=_bound(result))
            return {'ok': True, 'state': self.state, 'result': _bound(result)}
        except (LoopError, ComponentError, GraphValidationError) as exc:
            self.state = 'failed'
            self.emit('run_end', request_id=request_id, state='failed', error=str(exc))
            return {'ok': False, 'error': str(exc), 'state': 'failed'}
        except Exception as exc:  # unexpected failures surface with the trace
            self.state = 'failed'
            trace = traceback.format_exc()
            print(trace, file=sys.stderr)
            self.emit('run_end', request_id=request_id, state='failed', error=str(exc))
            return {'ok': False, 'error': str(exc), 'state': 'failed'}
        finally:
            with self._run_lock:
                self._active = False

    def _stop(self) -> Dict[str, Any]:
        """Stop THIS graph call: no new node invocations start; submitted
        atomic training/sync work may finish and report its real result."""
        with self._run_lock:
            self._stop_requested = True
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
        self.state = 'released'
        self.emit('release_end')
        return {'ok': True, 'state': 'released'}

    def _preview(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        """Bounded output preview: the recorded outputs plus the session state.

        THIS is the one place the session materialises a reference -- the user
        explicitly asked to see the value. References are dereferenced, then
        bounded the same way every event payload is."""
        node_id = str(cmd.get('node_id', ''))
        if node_id and node_id in self._outputs:
            return {'ok': True, 'output': _materialize(self._outputs[node_id])}
        return {'ok': True, 'outputs': {
            nid: _materialize(row) for nid, row in self._outputs.items()}}

    def _update(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        """A new declaration revision (edit-then-continue): revalidate; only
        the changed residents invalidate, training state demands a reset."""
        declaration = cmd.get('declaration') or cmd.get('graph')
        if not declaration:
            return {'ok': False, 'error': 'update needs a declaration'}
        try:
            graph = Graph.from_declaration(declaration)
        except GraphValidationError as exc:
            return {'ok': False, 'error': str(exc)}
        self._graph = graph
        self._declaration = graph.to_dict()
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
        self.state = 'idle'
        self.emit('reset')
        return {'ok': True, 'state': 'idle'}


def _deref(value: Any) -> Any:
    """Resolve a LazyCollect handle (inlined to keep this module import-light)."""
    if callable(value) and getattr(value, '_is_lazy_collect', False):
        return value()
    return value


def _materialize(row: Dict[str, Any]) -> Dict[str, Any]:
    """One recorded output row, dereferenced and bounded for the preview reply."""
    def walk(value: Any, depth: int = 0) -> Any:
        if depth > 4:
            return '...'
        if isinstance(value, dict):
            return {str(k): walk(v, depth + 1) for k, v in list(value.items())[:32]}
        if isinstance(value, (list, tuple)):
            return [walk(v, depth + 1) for v in list(value)[:16]]
        return _bound(_deref(value))

    return {'node': row['node'], 'type': row['type'], 'output': walk(row['output'])}


def _bound(value: Any, depth: int = 0) -> Any:
    """Bound a value for stdout: dicts/lists keep structure (bounded depth);
    everything else becomes its string form. Heavy payloads never reach here
    -- components record bounded summaries only."""
    if depth > 4:
        return '...'
    if isinstance(value, dict):
        return {str(k): _bound(v, depth + 1) for k, v in list(value.items())[:32]}
    if isinstance(value, (list, tuple)):
        return [_bound(v, depth + 1) for v in list(value)[:16]]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)[:256]


def _bootstrap(mode: str) -> None:
    """Bring up the twinkle infra for this session process.

    ``local`` (the default, and the currently runnable surface): nothing to
    do -- a component without a ``remote_group`` constructs in-process.
    ``ray``: the session joins/starts the cluster and declares the
    ``webui_numeric`` CPU group the numeric worker's ``remote_group`` names;
    without the declared group, dispatch would fail on the first
    remote-execution stage. GPU groups for model/sampler residents are the
    resource-plan side of the ray form and are NOT invented here."""
    if mode != 'ray':
        return
    import twinkle
    from twinkle import DeviceGroup
    twinkle.initialize(
        mode='ray',
        groups=[DeviceGroup(name='webui_numeric', ranks=1, device_type='CPU')],
        name='webui_session',
    )


def main(argv: Optional[List[str]] = None) -> int:
    """The resident session process entry.

    Reads JSON commands from stdin (one per line), writes ``[workflow]``-prefixed
    JSON events and command replies to stdout, logs to stderr. The process ends
    at EOF (the frontend closed the session) with a final released state.

    Usage: ``python -m swift.dev.webui.session graph.json [--mode local|ray]``.
    """
    declaration: Optional[Dict[str, Any]] = None
    output_dir = 'output'
    mode = 'local'
    argv = list(sys.argv[1:] if argv is None else argv)
    args = [arg for arg in argv if not arg.startswith('--')]
    for i, arg in enumerate(argv):
        if arg.startswith('--mode='):
            mode = arg.partition('=')[2]
        elif arg == '--mode' and i + 1 < len(argv):
            mode = argv[i + 1]
    if mode not in ('local', 'ray'):
        print(f'unknown mode {mode!r}: expected local or ray', file=sys.stderr)
        return 2
    if args:
        with open(args[0], encoding='utf-8') as f:
            declaration = json.load(f)
    if not declaration:
        print('usage: python -m swift.dev.webui.session graph.json [--mode local|ray]',
              file=sys.stderr)
        return 2
    _bootstrap(mode)
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
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

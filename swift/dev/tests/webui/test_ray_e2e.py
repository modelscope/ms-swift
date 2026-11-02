# Copyright (c) ModelScope Contributors. All rights reserved.
"""L2: the drag-orchestration surface over a REAL ray cluster.

One circle of GRPO over a tiny model -- dataloader feeds the streaming driver,
a real vllm engine generates, a code scorer scores, the advantage runs on the
numeric worker, train_step runs a REAL forward_backward, weight_sync pushes
the weights back over the native bucketed transfer. The session is the
production GraphSession; the assertions anchor to the drag requirements
(run receipts, cursor continuation, preview materialisation, event lighting)
and to independent oracles (the placement plan's explicit group table, the
run receipt's own counters, parseable loss values).

Case map (see .temp/v5_design/ut_plan.md section 20i):

- S20-65 always-ray runtime plane  -> test_grpo_circle_real_ray (stage 1)
- S20-66 card-occupancy plane      -> test_two_card_model_group
- S20-67 full GRPO circle          -> test_grpo_circle_real_ray (stage 2)
- S20-68 re-click cursor continue  -> test_grpo_circle_real_ray (stage 5)
- S20-69 preview materialisation   -> test_grpo_circle_real_ray (stage 3)
- S20-70 parallel runtime plane    -> test_grpo_circle_real_ray (stage 4)
- S20-71 real pause / resume       -> test_grpo_circle_real_ray (stage 6)
- S20-72 megatron tier             -> test_megatron_tier_circle (engine-gated)
- S20-73 sglang parameter surface  -> test_sglang_sampler_surface (engine-gated)
- S20-74 algorithm A3 circle       -> test_a3_rejection_ema_circle
- S20-77 event-lighting flow       -> test_grpo_circle_real_ray (stage 4)
- S20-78 replay equivalence        -> test_grpo_circle_real_ray (stage 7)
- UI protocol form (the exact stdin/stdout channel the vite middleware
  drives) -> test_session_subprocess_protocol
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from swift.dev.tests.tiny import TinyData, TinyModel

_REPO = Path(__file__).resolve().parents[4]
_TWINKLE_SRC = _REPO / 'twinkle' / 'src'

#: A scorer whose rewards VARY inside one generation group: with a constant
#: scorer GRPO's group-relative advantage is identically zero and the loss
#: degenerates -- the assertion "loss is finite and non-zero" needs variance.
SCORER_CODE = 'def score(rows):\n    return [0.0 if i % 2 else 1.0 for i in range(len(rows))]\n'

#: vllm on a shared box: a small slice of one card is enough for the tiny
#: model, and the engine refuses to start when its desired share exceeds the
#: free memory, so the slice stays explicit.
ENGINE_ARGS = {'gpu_memory_utilization': 0.3, 'max_model_len': 512}

_TIMEOUT = 1800.0


def _tiny_tokenizer_dir() -> Optional[str]:
    """The modelscope cache's tokenizer snapshot, when this box already has it.

    Building the tiny model must not depend on the hub being reachable: the
    revision check modelscope makes even on a cache hit fails loudly during a
    hub outage. A cached snapshot short-circuits that (tiny.tokenizer_dir
    passes local directories through); ``None`` keeps the normal download.
    """
    cache = os.environ.get('MODELSCOPE_CACHE',
                            os.path.expanduser('~/.cache/modelscope/hub'))
    snapshot = Path(cache) / 'models' / 'Qwen--Qwen2.5-0.5B-Instruct' / 'snapshots' / 'master'
    if (snapshot / 'tokenizer_config.json').exists():
        return str(snapshot)
    return None


_TINY_TOKENIZER = _tiny_tokenizer_dir()


def _node(node_id: str, type_key: str, params: Optional[dict] = None) -> Dict[str, Any]:
    return {'id': node_id, 'type': type_key, 'params': dict(params or {})}


def _edge(eid: str, source: str, source_port: str, target: str, target_port: str,
          slot: str = '') -> Dict[str, Any]:
    value = {'id': eid, 'source': source, 'source_port': source_port,
             'target': target, 'target_port': target_port}
    if slot:
        value['slot'] = slot
    return value


def grpo_declaration(model: str, data: str, cards: Tuple[int, ...] = (0, 1)) -> Dict[str, Any]:
    """The GRPO circle the probe validated end to end (tiny model + vllm)."""
    return {
        'version': 4, 'workflow_id': 'grpo-e2e',
        'resource': {'mode': 'local', 'cards': list(cards)},
        'nodes': [
            _node('ds', 'dataset', {'dataset': [data]}),
            _node('loader', 'dataloader'),
            _node('policy', 'model', {'model': model, 'gpu_count': 1,
                                      'model_config': {'model_type': 'qwen2'}}),
            _node('smp', 'sampler', {'model': model, 'engine': 'vllm', 'gpu_count': 1,
                                     'model_config': {'model_type': 'qwen2'},
                                     'engine_args': ENGINE_ARGS}),
            _node('loss_n', 'loss', {'kind': 'grpo'}),
            _node('scorer', 'scorer', {'code': SCORER_CODE}),
            _node('driver', 'streaming_driver', {'tokenizer': model,
                                                 'model_config': {'model_type': 'qwen2'}}),
            _node('score', 'score'),
            _node('adv', 'advantage'),
            _node('train', 'train_step', {'tokenizer': model,
                                          'model_config': {'model_type': 'qwen2'}}),
            _node('sync', 'weight_sync'),
        ],
        'edges': [
            _edge('e2', 'ds', 'data', 'loader', 'data'),
            _edge('e1', 'loader', 'batch', 'driver', 'prompts'),
            _edge('e3', 'smp', 'sampler', 'driver', 'engine', 'smp'),
            _edge('e_s', 'scorer', 'scorer', 'score', 'scorers', 'scorer'),
            _edge('e4', 'driver', 'samples', 'score', 'samples'),
            _edge('e5', 'driver', 'samples', 'adv', 'traj'),
            _edge('e6', 'score', 'reward', 'adv', 'reward'),
            _edge('e7', 'adv', 'adv', 'train', 'arguments', 'advantages'),
            _edge('e7c', 'adv', 'old_logps', 'train', 'arguments', 'old_logps'),
            _edge('e7b', 'driver', 'samples', 'train', 'batch'),
            _edge('e8', 'loss_n', 'loss', 'train', 'loss'),
            _edge('e9', 'policy', 'model', 'train', 'model'),
            _edge('e10', 'train', 'then', 'sync', 'after'),
            _edge('e11', 'smp', 'sampler', 'sync', 'targets', 'smp'),
            _edge('f1', 'sync', 'then', 'driver', 'after'),
        ],
        'frames': [
            {'id': 'loop', 'name': 'grpo', 'members': ['driver', 'score', 'adv', 'train', 'sync'],
             'sources': ['driver'], 'consumers': ['score', 'adv', 'train', 'sync'],
             'budget_owner': 'train', 'feedback': ['f1']},
        ],
    }


class _Recorder:
    """Timestamped event capture: parallelism assertions need wall-clock."""

    def __init__(self) -> None:
        self.rows: List[Tuple[float, str, Dict[str, Any]]] = []

    def __call__(self, kind: str, **payload: Any) -> None:
        self.rows.append((time.monotonic(), kind, payload))

    def of(self, kind: str) -> List[Tuple[float, Dict[str, Any]]]:
        return [(ts, payload) for ts, k, payload in self.rows if k == kind]

    def clear(self) -> None:
        self.rows.clear()


def _wait_run_end(recorder: _Recorder, timeout: float = _TIMEOUT) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rows = recorder.of('run_end')
        if rows:
            return rows[-1][1]
        time.sleep(1.0)
    raise AssertionError(f'no run_end within {timeout}s; events: '
                         f'{[(k, p) for _t, k, p in recorder.rows[-20:]]}')


def _span(recorder: _Recorder, node_id: str) -> Optional[Tuple[float, float]]:
    """[begin, end] of one node's first running->completed pair."""
    began = ended = None
    for _ts, payload in recorder.of('node'):
        if payload.get('node_id') != node_id:
            continue
        if payload.get('state') == 'running' and began is None:
            began = _ts
        if payload.get('state') == 'completed':
            ended = _ts
            break
    return (began, ended) if began is not None and ended is not None else None


def _completed_order(recorder: _Recorder, node_ids: List[str]) -> List[str]:
    """Node ids in first-completed order, restricted to ``node_ids``."""
    seen = []
    for _ts, payload in recorder.of('node'):
        if payload.get('state') == 'completed' and payload.get('node_id') in node_ids:
            if payload['node_id'] not in seen:
                seen.append(payload['node_id'])
    return seen


@pytest.mark.slow
@pytest.mark.accel(2)
@pytest.mark.requires_vllm
def test_grpo_circle_real_ray(tmp_path: Any) -> None:
    """The whole always-ray surface in one flow (one bootstrap amortised):
    the circle, the preview channels, the re-click cursor, pause/resume and
    the full-mode replay -- each stage's failure mode has its own assertion.
    """
    from swift.dev.webui.numeric import get_numeric_worker
    from swift.dev.webui.placement import plan_device_groups
    from swift.dev.webui.graph import Graph
    from swift.dev.webui.session import GraphSession, _bootstrap
    import twinkle.infra as infra

    model = TinyModel.build(str(tmp_path / "tiny_model"), tokenizer_id=_TINY_TOKENIZER)
    data = TinyData.prompt_only(str(tmp_path / 'prompts.jsonl'), n=8)
    declaration = grpo_declaration(model, data)

    # Stage 1 (S20-65): the runtime plane IS ray, with the planned groups.
    plan = plan_device_groups(Graph.from_declaration(declaration))
    group_table = {group.name: group for group in plan.groups}
    assert group_table['policy'].ranks == (0,)
    assert group_table['smp'].ranks == (1,)
    _bootstrap(declaration)
    assert infra._mode == 'ray'
    session = GraphSession(declaration, output_dir=str(tmp_path / 'output'))
    recorder = _Recorder()
    session.emit = recorder  # type: ignore[assignment]

    # Stage 2 (S20-67): one full circle with a REAL optimizer step.
    reply = session.command({'op': 'run', 'request_id': 'r1', 'mode': 'component',
                             'target': 'train'})
    assert reply['ok'] and reply['accepted'], reply
    end = _wait_run_end(recorder)
    result = end['result']
    assert result['state'] == 'completed', end
    assert result['optimizer_steps'] >= 1, result
    for node_id in ('ds', 'loader', 'policy', 'smp', 'loss_n', 'scorer', 'driver',
                    'score', 'adv', 'train', 'sync'):
        span = _span(recorder, node_id)
        assert span is not None, f'{node_id} never completed'
    # The numeric engine is a ray actor pool, not a local object: the driver
    # handle carries the actor set of the CPU group the bootstrap declared.
    numeric = get_numeric_worker()
    assert getattr(numeric, '_actors', None), 'numeric worker is not a ray actor'

    # Stage 3 (S20-69): preview materialises the recorded lanes (the one
    # deref point), and a bogus lane fails loudly naming the real ones.
    #
    # The CHAIN evidence, not the loss VALUE: on-policy GRPO's reported loss
    # is mathematically ~zero at every step -- the sampler and the trainer
    # share weights (ratio~1) against group-centered advantages, so only the
    # numerical residue shows up in the 5-decimal metric string. What a
    # broken main link actually zeroes out is one of: the rewards (scorer),
    # the advantages (grouping), the sampler logps (the rollout's encoding --
    # a template lost over the ray boundary lands here) or the optimizer
    # steps. Each has its own assertion below.
    preview = session.command({'op': 'preview', 'node_id': 'train'})
    assert preview['ok'], preview
    lanes = set(preview['lanes'])
    assert {'step', 'metric', 'model'} <= lanes, preview
    metric = session.command({'op': 'preview', 'node_id': 'train', 'lane': 'metric'})
    assert metric['ok'], metric
    loss_raw = metric['value']['values']['loss']
    loss = float(loss_raw)  # twinkle metrics format as fixed-precision strings
    # On-policy GRPO's reported loss is the numerical residue of
    # ratio~1 against group-centered advantages -- TINY, and its sign is
    # arbitrary (a real residue can be -1e-05). The contract is FINITE and of
    # sane magnitude; the "training really ran" evidence is the chain below
    # plus a log with zero unretrieved worker errors.
    assert math.isfinite(loss) and abs(loss) < 1.0, loss_raw
    reward_lane = session.command({'op': 'preview', 'node_id': 'score', 'lane': 'reward'})
    reward_values = [row[0] for row in reward_lane['value']['values']]
    assert len(reward_values) == 8 and len(set(reward_values)) > 1, reward_lane
    adv_lane = session.command({'op': 'preview', 'node_id': 'adv', 'lane': 'adv'})
    # The adv lane is the BARE per-sample value sequence (the loss's kwarg
    # shape) -- a dict wrapper here would make the loss iterate its keys.
    # Like every list lane it pages: total is the row count.
    assert adv_lane['ok'] and adv_lane['total'] == 8, adv_lane
    assert any(abs(float(v)) > 1e-6 for v in adv_lane['rows']), adv_lane
    old_lane = session.command({'op': 'preview', 'node_id': 'adv', 'lane': 'old_logps'})
    # A list lane pages: total is the row count, rows the materialised page.
    assert old_lane['ok'] and old_lane['total'] == 8, old_lane
    assert all(len(row) > 0 for row in old_lane['rows']), old_lane
    step = session.command({'op': 'preview', 'node_id': 'train', 'lane': 'step'})
    assert step['value']['status'] == 'completed', step
    assert step['value']['optimizer_steps'] == result['optimizer_steps'], step
    bogus = session.command({'op': 'preview', 'node_id': 'train', 'lane': 'no_such_lane'})
    assert not bogus['ok'] and 'no_such_lane' in bogus['error'], bogus
    unrun = session.command({'op': 'preview', 'node_id': 'never_ran'})
    assert not unrun['ok'] and 'never_ran' in unrun['error'], unrun

    # Stage 4 (S20-70 / S20-77): the two GPU resources built OVERLAPPING in
    # wall-clock (serialised device builds would place the engine's begin
    # after the model's completion), and the CPU-lane dataset build overlaps
    # the engine's build window. And every edge fires 'used' only after its
    # producer node completed -- the lighting order the frontend consumes.
    policy_span = _span(recorder, 'policy')
    smp_span = _span(recorder, 'smp')
    ds_span = _span(recorder, 'ds')
    assert policy_span and smp_span and ds_span
    assert policy_span[0] < smp_span[1] and smp_span[0] < policy_span[1], \
        f'the model and the sampler built serially: {policy_span} {smp_span}'
    assert ds_span[0] < smp_span[1], 'the dataset build missed the engine build window'
    completed_at: Dict[str, float] = {}
    for ts, payload in recorder.of('node'):
        if payload.get('state') == 'completed':
            completed_at.setdefault(payload['node_id'], ts)
    edge_targets = {'e2': 'ds', 'e1': 'loader', 'e3': 'smp', 'e_s': 'scorer',
                    'e4': 'driver', 'e5': 'driver', 'e6': 'score', 'e7': 'adv',
                    'e7b': 'driver', 'e7c': 'adv', 'e8': 'loss_n', 'e9': 'policy',
                    'e10': 'train', 'e11': 'smp', 'f1': 'sync'}
    used = {payload['edge_id']: ts for ts, payload in recorder.of('edge')
            if payload.get('state') == 'used'}
    assert set(used) == set(edge_targets), sorted(set(edge_targets) - set(used))
    for edge_id, producer in edge_targets.items():
        assert used[edge_id] >= completed_at[producer], \
            f'edge {edge_id} lit before its producer {producer} completed'

    # Stage 5 (S20-68): the second click resumes from the cursor -- the
    # resources stay resident (no rebuild events) and the micro-step counter
    # accumulates instead of restarting.
    micro_before = step['value']['micro_step']
    recorder.clear()
    reply = session.command({'op': 'run', 'request_id': 'r2', 'mode': 'component',
                             'target': 'train'})
    assert reply['ok'], reply
    end = _wait_run_end(recorder)
    assert end['result']['state'] == 'completed', end
    rebuilt = [payload['node_id'] for _ts, payload in recorder.of('node')
               if payload.get('state') == 'running'
               and payload.get('node_id') in ('ds', 'policy', 'smp', 'loss_n', 'scorer')]
    assert rebuilt == [], f'resources were rebuilt on re-click: {rebuilt}'
    step2 = session.command({'op': 'preview', 'node_id': 'train', 'lane': 'step'})
    assert step2['value']['micro_step'] == micro_before + micro_before, step2

    # Stage 6 (S20-71): pause lands at a node boundary and resume finishes
    # the circle -- the run is neither killed nor rewound.
    recorder.clear()
    reply = session.command({'op': 'run', 'request_id': 'r3', 'mode': 'component',
                             'target': 'train'})
    assert reply['ok'], reply
    pause_reply = session.command({'op': 'pause'})
    assert pause_reply['ok'], pause_reply
    deadline = time.monotonic() + 300.0
    while time.monotonic() < deadline and not recorder.of('paused'):
        time.sleep(0.5)
    assert recorder.of('paused'), (
        'no paused event at the node boundary; pause receipt '
        f'{pause_reply}; recent events: '
        f'{[(k, p) for _t, k, p in recorder.rows[-30:]]}')
    reply = session.command({'op': 'resume'})
    assert reply['ok'], reply
    end = _wait_run_end(recorder)
    assert end['result']['state'] == 'completed', end

    # Stage 7 (S20-78): a full-mode run of the same graph walks the SAME
    # frame-member state machine as the component-mode circle.
    frame_members = ['driver', 'score', 'adv', 'train', 'sync']
    component_order = _completed_order(recorder, frame_members)
    recorder.clear()
    reply = session.command({'op': 'run', 'request_id': 'r4', 'mode': 'full'})
    assert reply['ok'], reply
    end = _wait_run_end(recorder)
    assert end['result']['state'] == 'completed', end
    full_order = _completed_order(recorder, frame_members)
    assert full_order == component_order, (component_order, full_order)

    session._release()


class _SessionProcess:
    """A ``python -m swift.dev.webui.session`` child, driven exactly the way
    the vite middleware drives it: the declaration as argv, JSON commands on
    stdin, ``[workflow]`` JSON lines on stdout."""

    def __init__(self, graph_path: Path) -> None:
        self._proc = subprocess.Popen(
            [sys.executable, '-m', 'swift.dev.webui.session', str(graph_path)],
            cwd=str(_REPO), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
            env={**os.environ, 'PYTHONPATH': f'{_TWINKLE_SRC}{os.pathsep}{_REPO}'})
        self._lines: List[str] = []
        # Continuously drain the child's stderr: an unread PIPE fills (64 KiB)
        # and BLOCKS the child mid-log -- a model actor stalled writing NCCL
        # debug output misses its NCCL rendezvous and the peer's connect is
        # refused. The tail doubles as the failure diagnostics below.
        self._stderr_tail: List[str] = []
        threading.Thread(target=self._pump_stderr, daemon=True).start()

    def _pump_stderr(self) -> None:
        try:
            for line in self._proc.stderr:  # type: ignore[union-attr]
                self._stderr_tail.append(line)
                if len(self._stderr_tail) > 1000:
                    del self._stderr_tail[:500]
        except Exception:
            pass  # the child closing its stderr ends the pump

    def _drain_stderr(self) -> str:
        """The tail of what the child wrote to stderr (crash traces, NCCL)."""
        return ''.join(self._stderr_tail[-80:])

    def command(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        assert self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(cmd) + '\n')
        self._proc.stdin.flush()
        # The reply is the next [workflow] line carrying this request_id.
        request_id = str(cmd.get('request_id', ''))
        deadline = time.monotonic() + _TIMEOUT
        while time.monotonic() < deadline:
            line = self._proc.stdout.readline()  # type: ignore[union-attr]
            if not line:
                raise AssertionError('session process closed stdout early; stderr: '
                                     f'{self._drain_stderr()[-2000:]}')
            if line.startswith('[workflow] '):
                self._lines.append(line)
                try:
                    row = json.loads(line[len('[workflow] '):])
                except ValueError:
                    continue
                if row.get('event') == 'reply' and row.get('request_id') == request_id:
                    return row
        raise AssertionError(f'no reply for request {request_id}')

    def wait_event(self, kind: str, timeout: float = _TIMEOUT) -> Dict[str, Any]:
        """Block until one event of ``kind`` arrives on the protocol stream."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = self._proc.stdout.readline()  # type: ignore[union-attr]
            if not line:
                raise AssertionError(f'session process closed stdout before {kind}; '
                                     f'stderr: {self._drain_stderr()[-2000:]}')
            if line.startswith('[workflow] '):
                self._lines.append(line)
                try:
                    row = json.loads(line[len('[workflow] '):])
                except ValueError:
                    continue
                if row.get('event') == kind:
                    return row
        raise AssertionError(f'no {kind} event within {timeout}s')

    def close(self) -> List[str]:
        if self._proc.poll() is None and self._proc.stdin is not None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
        try:
            self._proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=60)
        return self._lines


@pytest.mark.slow
@pytest.mark.accel(2)
@pytest.mark.requires_vllm
def test_session_subprocess_protocol(tmp_path: Any) -> None:
    """The exact channel the frontend drives: a resident session subprocess
    with JSON commands in and ``[workflow]`` events out, run to a completed
    GRPO circle with a preview round-trip."""
    model = TinyModel.build(str(tmp_path / "tiny_model"), tokenizer_id=_TINY_TOKENIZER)
    data = TinyData.prompt_only(str(tmp_path / 'prompts.jsonl'), n=8)
    graph_path = tmp_path / 'graph.json'
    graph_path.write_text(json.dumps(grpo_declaration(model, data)), encoding='utf-8')

    proc = _SessionProcess(graph_path)
    try:
        session_row = proc.wait_event('session', timeout=300.0)
        assert session_row['state'] == 'idle', session_row
        reply = proc.command({'op': 'run', 'request_id': 'r1', 'mode': 'component',
                              'target': 'train'})
        assert reply['ok'] and reply['accepted'], reply
        end = proc.wait_event('run_end')
        assert end.get('result', {}).get('state') == 'completed', end.get('error') or end
        preview = proc.command({'op': 'preview', 'node_id': 'train', 'lane': 'metric'})
        assert preview['ok'], preview
        loss = float(preview['value']['values']['loss'])
        # The on-policy GRPO residue is tiny and its sign is arbitrary.
        assert math.isfinite(loss) and abs(loss) < 1.0, preview
    finally:
        lines = proc.close()
    # Every stdout line the frontend would parse is a [workflow] JSON row.
    assert all(line.startswith('[workflow] ') for line in lines), \
        'non-protocol lines leaked onto the session stdout'


@pytest.mark.slow
@pytest.mark.accel(2)
def test_two_card_model_group(tmp_path: Any) -> None:
    """S20-66: a model occupying two cards becomes a two-rank DeviceGroup
    with two GPU actors -- the resource page's occupancy is real."""
    from swift.dev.webui.graph import Graph
    from swift.dev.webui.placement import plan_device_groups

    declaration = {
        'version': 4, 'workflow_id': 'two-card',
        'resource': {'mode': 'local', 'cards': [0, 1]},
        'nodes': [
            _node('policy', 'model', {'model': 'Qwen/Qwen2.5-0.5B', 'gpu_count': 2,
                                      'parallel_spec': 'dp2'}),
        ],
        'edges': [],
        'frames': [],
    }
    plan = plan_device_groups(Graph.from_declaration(declaration))
    group = {group.name: group for group in plan.groups}['policy']
    assert group.ranks == (0, 1), group
    assert plan.cards['policy'] == (0, 1), plan.cards

    # The real bootstrap prints the declared groups; the run builds the model
    # on both ranks. A tiny local directory keeps the download out of the
    # test; the identity fields pin the ambiguous model_type.
    model = TinyModel.build(str(tmp_path / "tiny_model"), tokenizer_id=_TINY_TOKENIZER)
    declaration['nodes'][0]['params']['model'] = model
    declaration['nodes'][0]['params']['model_config'] = {'model_type': 'qwen2'}
    graph_path = tmp_path / 'graph.json'
    graph_path.write_text(json.dumps(declaration), encoding='utf-8')

    proc = _SessionProcess(graph_path)
    try:
        proc.wait_event('session', timeout=300.0)
        reply = proc.command({'op': 'run', 'request_id': 'r1', 'mode': 'component',
                              'target': 'policy'})
        assert reply['ok'] and reply['accepted'], reply
        end = proc.wait_event('run_end')
        assert end.get('result', {}).get('state') == 'completed', end.get('error') or end
    finally:
        proc.close()


@pytest.mark.slow
@pytest.mark.accel(3)
@pytest.mark.requires_vllm
def test_a3_rejection_ema_circle(tmp_path: Any) -> None:
    """S20-74: the rejection-sampling + EMA-anchor algorithm assembled purely
    from the stock components -- generate / select / weight_average /
    checkpoint -- runs a real circle (composition, not a wired demo)."""
    model = TinyModel.build(str(tmp_path / "tiny_model"), tokenizer_id=_TINY_TOKENIZER)
    data = TinyData.prompt_only(str(tmp_path / 'prompts.jsonl'), n=8)
    declaration = {
        'version': 4, 'workflow_id': 'a3-rejection-ema',
        # Three GPU consumers (sampler + policy + anchor), so three cards.
        'resource': {'mode': 'local', 'cards': [0, 1, 2]},
        'nodes': [
            _node('data', 'dataset', {'dataset': [data]}),
            _node('loader', 'dataloader', {'batch_size': 4}),
            _node('sampler', 'sampler', {'model': model, 'engine': 'vllm', 'gpu_count': 1,
                                         'model_config': {'model_type': 'qwen2'},
                                         'engine_args': ENGINE_ARGS}),
            _node('policy', 'model', {'model': model, 'train_type': 'lora', 'gpu_count': 1,
                                      'model_config': {'model_type': 'qwen2'}}),
            _node('anchor', 'model', {'model': model, 'freeze': True, 'gpu_count': 1,
                                      'model_config': {'model_type': 'qwen2'}}),
            _node('gen', 'generate', {'num_generations': 4}),
            _node('scor', 'scorer', {'code': SCORER_CODE}),
            _node('score', 'score'),
            _node('sel', 'select', {'select': 'best_of_n', 'top_k': 1}),
            _node('loss', 'loss', {'kind': 'cross_entropy'}),
            _node('train', 'train_step', {'tokenizer': model, 'lr': 1e-4,
                                          'model_config': {'model_type': 'qwen2'}}),
            _node('wavg', 'weight_average', {'mode': 'ema', 'alpha': 0.9}),
            _node('sync', 'weight_sync'),
            _node('ckpt', 'checkpoint', {'profile': 'model'}),
        ],
        'edges': [
            _edge('g1', 'data', 'data', 'loader', 'data'),
            _edge('g2', 'loader', 'batch', 'gen', 'prompts'),
            _edge('g3', 'sampler', 'sampler', 'gen', 'engine'),
            _edge('g4', 'scor', 'scorer', 'score', 'scorers', 'scorer'),
            _edge('g5', 'gen', 'samples', 'score', 'samples'),
            _edge('g6', 'score', 'reward', 'sel', 'reward'),
            _edge('g7', 'gen', 'samples', 'sel', 'batch'),
            _edge('g8', 'sel', 'batch', 'train', 'batch'),
            _edge('g9', 'loss', 'loss', 'train', 'loss'),
            _edge('g10', 'policy', 'model', 'train', 'model'),
            _edge('g11', 'train', 'step', 'wavg', 'step'),
            _edge('g12', 'train', 'model', 'wavg', 'source'),
            _edge('g13', 'anchor', 'model', 'wavg', 'target'),
            _edge('g14', 'train', 'then', 'sync', 'after'),
            _edge('g15', 'sync', 'then', 'loader', 'after'),
            _edge('g16', 'sampler', 'sampler', 'sync', 'targets', 'tgt'),
            _edge('g17', 'train', 'step', 'ckpt', 'step'),
        ],
        'frames': [
            {'id': 'loop', 'name': 'a3', 'members': ['loader', 'gen', 'score', 'sel', 'train',
                                                     'wavg', 'sync'],
             'sources': ['loader'], 'consumers': ['gen', 'score', 'sel', 'train', 'wavg', 'sync'],
             'budget_owner': 'train', 'feedback': ['g15']},
        ],
    }
    graph_path = tmp_path / 'graph.json'
    graph_path.write_text(json.dumps(declaration), encoding='utf-8')

    proc = _SessionProcess(graph_path)
    try:
        proc.wait_event('session', timeout=300.0)
        reply = proc.command({'op': 'run', 'request_id': 'r1', 'mode': 'component',
                              'target': 'train'})
        assert reply['ok'] and reply['accepted'], reply
        end = proc.wait_event('run_end')
        assert end.get('result', {}).get('state') == 'completed', end.get('error') or end
    finally:
        proc.close()


@pytest.mark.slow
@pytest.mark.accel(1)
@pytest.mark.requires_megatron
def test_megatron_tier_circle(tmp_path: Any) -> None:
    """S20-72: backend is a parameter -- the same train_step / weight_sync
    circle with backend=megatron builds and steps (skipped where the
    megatron bridge is absent)."""
    model = TinyModel.build(str(tmp_path / "tiny_model"), tokenizer_id=_TINY_TOKENIZER)
    data = TinyData.prompt_only(str(tmp_path / 'prompts.jsonl'), n=8)
    declaration = grpo_declaration(model, data)
    declaration['nodes'][2]['params']['backend'] = 'megatron'
    graph_path = tmp_path / 'graph.json'
    graph_path.write_text(json.dumps(declaration), encoding='utf-8')

    proc = _SessionProcess(graph_path)
    try:
        proc.wait_event('session', timeout=300.0)
        reply = proc.command({'op': 'run', 'request_id': 'r1', 'mode': 'component',
                              'target': 'policy'})
        assert reply['ok'] and reply['accepted'], reply
        end = proc.wait_event('run_end')
        assert end.get('result', {}).get('state') == 'completed', end.get('error') or end
    finally:
        proc.close()


@pytest.mark.slow
@pytest.mark.accel(1)
@pytest.mark.requires_sglang
def test_sglang_sampler_surface(tmp_path: Any) -> None:
    """S20-73: engine is a parameter -- the sampler node with engine=sglang
    constructs the same disaggregated shape (its own DeviceGroup, mesh from
    the card spec; engine-specific knobs ride engine_args)."""
    model = TinyModel.build(str(tmp_path / "tiny_model"), tokenizer_id=_TINY_TOKENIZER)
    declaration = {
        'version': 4, 'workflow_id': 'sglang-surface',
        'resource': {'mode': 'local', 'cards': [0]},
        'nodes': [
            _node('smp', 'sampler', {'model': model, 'engine': 'sglang', 'gpu_count': 1,
                                     'model_config': {'model_type': 'qwen2'}}),
        ],
        'edges': [],
        'frames': [],
    }
    graph_path = tmp_path / 'graph.json'
    graph_path.write_text(json.dumps(declaration), encoding='utf-8')

    proc = _SessionProcess(graph_path)
    try:
        proc.wait_event('session', timeout=300.0)
        reply = proc.command({'op': 'run', 'request_id': 'r1', 'mode': 'component',
                              'target': 'smp'})
        assert reply['ok'] and reply['accepted'], reply
        end = proc.wait_event('run_end')
        assert end.get('result', {}).get('state') == 'completed', end.get('error') or end
    finally:
        proc.close()

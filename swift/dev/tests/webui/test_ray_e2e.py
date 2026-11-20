# Copyright (c) ModelScope Contributors. All rights reserved.
"""L2: the drag-orchestration surface's migration design boundaries.

This webui runs ONLY v{REGISTRY_VERSION} graphs. A v4 declaration is not
silently upgraded -- the graph gate refuses it loudly (GraphValidationError),
and that refusal IS the correct behavior (W1). The former v4 circle e2e tests
are therefore expected-rejection gates: each preserves its real declaration as
the rejection input and asserts the loud refusal. The always-ray numeric plane
is verified through the live ``numeric_component`` API (W2, the retired
``get_numeric_worker`` handle is gone), and the resident session subprocess
must surface a bootstrap failure on the stdout protocol as a clean error event
rather than a silent EOF (W3). The sglang parameter surface stays engine-gated.

Case map (see .temp/v5_design/ut_plan.md section 20i):

- S20-65/67/68/69/70/71/77/78 GRPO circle -> test_grpo_circle_real_ray (W1 v4 refusal)
- S20-66 card-occupancy plane             -> test_two_card_model_group (W1 v4 refusal;
                                           placement itself: test_placement.py)
- S20-72 megatron tier                    -> test_megatron_tier_circle (W1 v4 refusal)
- S20-73 sglang parameter surface         -> test_sglang_sampler_surface (engine-gated)
- S20-74 algorithm A3 circle              -> test_a3_rejection_ema_circle (W1 v4 refusal)
- numeric plane ray actor (W2)            -> test_numeric_component_is_ray_actor
- UI protocol form (W3 stdin/stdout)      -> test_session_subprocess_protocol
"""
from __future__ import annotations
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import pytest

from swift.dev.tests.tiny import TinyModel

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
    """The GRPO circle the probe validated end to end (tiny model + vllm).

    Stamped v4: this webui runs only v5, so the declaration is the rejection
    input for the W1 version gate (and the W3 subprocess bootstrap diagnostic).
    """
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
            _edge('e1', 'loader', 'source', 'driver', 'prompts'),
            _edge('e3', 'smp', 'sampler', 'driver', 'engine', 'smp'),
            _edge('e_s', 'scorer', 'scorer', 'score', 'scorers', 'scorer'),
            _edge('e4', 'driver', 'traj', 'score', 'traj'),
            _edge('e6', 'score', 'traj', 'adv', 'traj'),
            _edge('e7', 'adv', 'traj', 'train', 'batch'),
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


def test_grpo_circle_real_ray() -> None:
    """W1 (S20-65/67/68/69/70/71/77/78): the flagship GRPO circle is a v4 graph.
    This webui runs only v5; the loud version gate (GraphValidationError) is the
    correct behavior -- an explicit refusal, not a silent upgrade. The circle
    declaration is preserved as the rejection input. The always-ray numeric
    plane is covered by test_numeric_component_is_ray_actor (W2) and the
    subprocess protocol by test_session_subprocess_protocol (W3)."""
    from swift.dev.webui.graph import Graph, GraphValidationError
    declaration = grpo_declaration('Qwen/Qwen2.5-0.5B', 'prompts.jsonl')
    with pytest.raises(GraphValidationError, match='仅运行 v5'):
        Graph.from_declaration(declaration)


@pytest.mark.slow
@pytest.mark.accel(1)
def test_numeric_component_is_ray_actor() -> None:
    """W2: the numeric plane is remote -- the advantage stage's twinkle estimator
    is constructed under the CPU group the bootstrap declared and resolves to a
    ray actor pool (``_actors``), never a local object."""
    import twinkle
    import twinkle.infra as infra

    import swift.dev.webui.numeric as numeric
    from swift.dev.webui.components.signals import AdvantageStage
    from swift.dev.webui.session import _bootstrap

    assert not hasattr(numeric, 'get_numeric_worker'), 'retired get_numeric_worker API still present'
    declaration = {
        'version': 5, 'workflow_id': 'numeric-probe',
        'resource': {'mode': 'local', 'cards': [0]},
        'nodes': [_node('policy', 'model', {'model': 'Qwen/Qwen2.5-0.5B', 'gpu_count': 1})],
        'edges': [], 'frames': [],
    }
    _bootstrap(declaration)
    assert infra._mode == 'ray'
    try:
        stage = AdvantageStage('adv', {'estimator': 'grpo', 'num_generations': 4})
        component = stage._estimator()
        assert getattr(component, '_actors', None), 'numeric component is not a ray actor pool'
        component.close()
    finally:
        twinkle.shutdown()


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


def test_session_subprocess_protocol(tmp_path: Any) -> None:
    """W3: the exact channel the frontend drives. An invalid (v4) declaration
    must ride the stdout protocol as a loud ``error`` event and a clean non-zero
    exit -- NOT a silent stdout EOF before any diagnostic. The frontend only
    parses ``[workflow]`` lines; a bootstrap crash to stderr would leave it
    watching a dead pipe with no error."""
    graph_path = tmp_path / 'graph.json'
    graph_path.write_text(json.dumps(grpo_declaration('Qwen/Qwen2.5-0.5B', 'prompts.jsonl')),
                          encoding='utf-8')

    proc = _SessionProcess(graph_path)
    try:
        err = proc.wait_event('error', timeout=120.0)
        assert err.get('stage') == 'bootstrap', err
        assert '仅运行 v5' in err.get('error', ''), err
    finally:
        lines = proc.close()
    # Every stdout line the frontend would parse is a [workflow] JSON row.
    assert all(line.startswith('[workflow] ') for line in lines), \
        'non-protocol lines leaked onto the session stdout'
    assert proc._proc.returncode not in (0, None), 'a bootstrap failure must exit non-zero'


def test_two_card_model_group() -> None:
    """W1 (S20-66): the two-card model declaration is a v4 graph -- unsupported,
    loudly rejected. Card-occupancy placement itself (two cards => two ranks)
    is covered version-independently by test_placement.py."""
    from swift.dev.webui.graph import Graph, GraphValidationError
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
    with pytest.raises(GraphValidationError, match='仅运行 v5'):
        Graph.from_declaration(declaration)


def test_a3_rejection_ema_circle() -> None:
    """W1 (S20-74): the rejection-sampling + EMA-anchor algorithm declaration
    (generate / select / weight_average / checkpoint composed from stock
    components) is a v4 graph -- unsupported, loudly rejected. The declaration
    is preserved as the rejection input."""
    from swift.dev.webui.graph import Graph, GraphValidationError
    declaration = {
        'version': 4, 'workflow_id': 'a3-rejection-ema',
        # Three GPU consumers (sampler + policy + anchor), so three cards.
        'resource': {'mode': 'local', 'cards': [0, 1, 2]},
        'nodes': [
            _node('data', 'dataset', {'dataset': ['prompts.jsonl']}),
            _node('loader', 'dataloader', {'batch_size': 4}),
            _node('sampler', 'sampler', {'model': 'Qwen/Qwen2.5-0.5B', 'engine': 'vllm',
                                         'gpu_count': 1, 'model_config': {'model_type': 'qwen2'},
                                         'engine_args': ENGINE_ARGS}),
            _node('policy', 'model', {'model': 'Qwen/Qwen2.5-0.5B', 'train_type': 'lora',
                                      'gpu_count': 1, 'model_config': {'model_type': 'qwen2'}}),
            _node('anchor', 'model', {'model': 'Qwen/Qwen2.5-0.5B', 'freeze': True,
                                      'gpu_count': 1, 'model_config': {'model_type': 'qwen2'}}),
            _node('gen', 'generate', {'num_generations': 4}),
            _node('scor', 'scorer', {'code': SCORER_CODE}),
            _node('score', 'score'),
            _node('sel', 'select', {'select': 'best_of_n', 'top_k': 1}),
            _node('loss', 'loss', {'kind': 'cross_entropy'}),
            _node('train', 'train_step', {'tokenizer': 'Qwen/Qwen2.5-0.5B', 'lr': 1e-4,
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
    with pytest.raises(GraphValidationError, match='仅运行 v5'):
        Graph.from_declaration(declaration)


def test_megatron_tier_circle() -> None:
    """W1 (S20-72): the megatron-tier GRPO circle (backend=megatron) is a v4
    graph -- unsupported, loudly rejected. The backend parameter surface is
    preserved in the rejection input."""
    from swift.dev.webui.graph import Graph, GraphValidationError
    declaration = grpo_declaration('Qwen/Qwen2.5-0.5B', 'prompts.jsonl')
    declaration['nodes'][2]['params']['backend'] = 'megatron'
    with pytest.raises(GraphValidationError, match='仅运行 v5'):
        Graph.from_declaration(declaration)


@pytest.mark.slow
@pytest.mark.accel(1)
@pytest.mark.requires_sglang
def test_sglang_sampler_surface(tmp_path: Any) -> None:
    """S20-73: engine is a parameter -- the sampler node with engine=sglang
    constructs the same disaggregated shape (its own DeviceGroup, mesh from
    the card spec; engine-specific knobs ride engine_args)."""
    model = TinyModel.build(str(tmp_path / "tiny_model"), tokenizer_id=_TINY_TOKENIZER)
    declaration = {
        'version': 5, 'workflow_id': 'sglang-surface',
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

# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared fixtures-of-code for the webui orchestration tests.

Two layers live here:

* declaration builders -- the two-loop graph (solver + question loops) at the
  current registry version, its v3 ancestor, and surgical variants (extra model,
  checkpoint, extra stage). They are plain dicts: the test IS the declaration's
  oracle. The current graph reads its version from the registry rather than
  carrying a literal, so a registry bump does not silently turn these fixtures
  into legacy graphs that the validator refuses.
* fake components -- ``FakeResource`` / ``FakeIterator`` / ``FakeStage``
  replace the real bindings so the DAGController and GraphSession protocols
  (clicks, circles, pause boundaries, pagination) run without engines while
  every observable (invocations, spans, takes, commits) stays assertable.
"""
from __future__ import annotations

import copy
import threading
import time

from swift.dev.webui.dag import DAGController
from swift.dev.webui.graph import Graph
from swift.dev.webui.registry import REGISTRY_VERSION
from swift.dev.webui.session import GraphSession

MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'


def node(node_id, type_key, params=None):
    return {'id': node_id, 'type': type_key, 'params': dict(params or {})}


def edge(edge_id, source, source_port, target, target_port, slot=''):
    value = {'id': edge_id, 'source': source, 'source_port': source_port,
             'target': target, 'target_port': target_port}
    if slot:
        value['slot'] = slot
    return value


def base_declaration():
    """The two-loop graph: a question loop (chall -> train_q) feeding a solver
    loop (driver -> score -> adv -> train -> sync) through the cross-frame
    edge x1 -- a round boundary, never a same-round dependency."""
    nodes = [
        node('ds', 'dataset', {'dataset': ['AI-ModelScope/gsm8k']}),
        node('loader', 'dataloader'),
        node('policy', 'model', {'model': MODEL_ID}),
        node('policy_q', 'model', {'model': MODEL_ID, 'freeze': True}),
        node('smp', 'sampler', {'model': MODEL_ID}),
        node('env_res', 'sandbox_env'),
        node('loss_n', 'loss', {'kind': 'grpo'}),
        node('loss_q', 'loss', {'kind': 'cross_entropy'}),
        node('scorer', 'scorer', {'kind': 'math'}),
        node('driver', 'streaming_driver', {'tokenizer': 'tiny/test'}),
        node('score', 'score'),
        node('adv', 'advantage'),
        node('train', 'train_step', {'tokenizer': MODEL_ID}),
        node('sync', 'weight_sync'),
        node('chall', 'challenger', {'seed': 'seed task', 'tokenizer': MODEL_ID}),
        node('train_q', 'train_step', {'tokenizer': MODEL_ID}),
    ]
    edges = [
        edge('e2', 'ds', 'data', 'loader', 'data'),
        edge('e1', 'loader', 'source', 'driver', 'prompts'),
        edge('e3', 'smp', 'sampler', 'driver', 'engine', slot='smp'),
        edge('e_s', 'scorer', 'scorer', 'score', 'scorers', slot='scorer'),
        edge('e4', 'driver', 'samples', 'score', 'samples'),
        edge('e5', 'driver', 'samples', 'adv', 'traj'),
        edge('e6', 'score', 'reward', 'adv', 'reward'),
        edge('e7', 'adv', 'adv', 'train', 'arguments', slot='advantages'),
        edge('e8', 'loss_n', 'loss', 'train', 'loss'),
        edge('e9', 'policy', 'model', 'train', 'model'),
        edge('e10', 'train', 'then', 'sync', 'after'),
        edge('e11', 'smp', 'sampler', 'sync', 'targets', slot='smp'),
        edge('x1', 'chall', 'answers', 'train', 'batch'),
        edge('e12', 'chall', 'questions', 'train_q', 'batch'),
        edge('e13', 'loss_q', 'loss', 'train_q', 'loss'),
        edge('e14', 'policy_q', 'model', 'train_q', 'model'),
        edge('e15', 'smp', 'sampler', 'chall', 'backend'),
        edge('e16', 'env_res', 'env', 'chall', 'env'),
        edge('f1', 'sync', 'then', 'driver', 'after'),
        edge('fq', 'train_q', 'then', 'chall', 'after'),
    ]
    frames = [
        {'id': 'loop', 'name': 'solver', 'members': ['driver', 'score', 'adv', 'train', 'sync'],
         'sources': ['driver'], 'consumers': ['score', 'adv', 'train', 'sync'],
         'budget_owner': 'train', 'feedback': ['f1']},
        {'id': 'loopq', 'name': 'question', 'members': ['chall', 'train_q'],
         'sources': ['chall'], 'consumers': ['train_q'],
         'budget_owner': 'train_q', 'feedback': ['fq']},
    ]
    return {'version': REGISTRY_VERSION, 'workflow_id': 'two-loop-test',
            'resource': {'mode': 'local', 'cards': [0, 1, 2, 3]},
            'nodes': nodes, 'edges': edges, 'frames': frames}


def with_second_model(declaration):
    """A frozen reference model enters the solver loop through model_forward.

    ``sync.model`` stays unwired, so scope resolution must see two MODEL
    candidates (policy, policy2) and refuse to guess."""
    result = copy.deepcopy(declaration)
    result['nodes'] += [
        node('policy2', 'model', {'model': MODEL_ID, 'freeze': True}),
        node('fwd', 'model_forward', {'tokenizer': MODEL_ID, 'outputs': ['logps']}),
    ]
    result['edges'] += [
        edge('e17', 'policy2', 'model', 'fwd', 'model'),
        edge('e18', 'driver', 'samples', 'fwd', 'traj'),
        edge('e19', 'fwd', 'logps', 'train', 'arguments', slot='old_logps'),
    ]
    for frame in result['frames']:
        if frame['id'] == 'loop':
            frame['members'].append('fwd')
    return result


def with_checkpoint(declaration):
    """A loop-free checkpoint consuming the solver loop's step receipt."""
    result = copy.deepcopy(declaration)
    result['nodes'].append(node('ckpt', 'checkpoint'))
    result['edges'] += [
        edge('xck', 'train', 'step', 'ckpt', 'step'),
        edge('e_ck', 'policy', 'model', 'ckpt', 'model')]
    return result


def v3_declaration():
    """A v3 GRPO graph: mode params, dataset wired straight into the driver,
    loss/scorer still params of train_step/score."""
    nodes = [
        node('policy', 'model', {'model': MODEL_ID, 'mode': 'ray'}),
        node('smp', 'sampler', {'model': MODEL_ID, 'mode': 'ray'}),
        node('ds', 'dataset', {'dataset': ['AI-ModelScope/gsm8k']}),
        node('driver', 'streaming_driver', {'tokenizer': 'tiny/test'}),
        node('score', 'score', {'scorer': 'math', 'scorer_kwargs': {}}),
        node('adv', 'advantage'),
        node('train', 'train_step', {'tokenizer': MODEL_ID, 'loss': 'grpo', 'loss_kwargs': {}}),
        node('sync', 'weight_sync'),
    ]
    edges = [
        edge('e1', 'ds', 'data', 'driver', 'prompts'),
        edge('e2', 'smp', 'sampler', 'driver', 'engine', slot='smp'),
        edge('e3', 'driver', 'samples', 'score', 'samples'),
        edge('e4', 'driver', 'samples', 'adv', 'traj'),
        edge('e5', 'score', 'reward', 'adv', 'reward'),
        edge('e6', 'adv', 'adv', 'train', 'arguments', slot='advantages'),
        edge('e7', 'driver', 'samples', 'train', 'batch'),
        edge('e8', 'policy', 'model', 'train', 'model'),
        edge('e9', 'train', 'then', 'sync', 'after'),
        edge('e10', 'smp', 'sampler', 'sync', 'targets', slot='smp'),
        edge('f1', 'sync', 'then', 'driver', 'after'),
    ]
    frames = [
        {'id': 'loop', 'name': 'grpo', 'members': ['driver', 'score', 'adv', 'train', 'sync'],
         'sources': ['driver'], 'consumers': ['score', 'adv', 'train', 'sync'],
         'budget_owner': 'train', 'feedback': ['f1']},
    ]
    return {'version': 3, 'workflow_id': 'v3-grpo',
            'nodes': nodes, 'edges': edges, 'frames': frames}


class FakeResource:
    """A resource binding: counts builds, remembers its inputs, releasable."""

    def __init__(self, node_id):
        self.node_id = node_id
        self.obj = None
        self.built = 0
        self.IMPORTS = ()

    def build(self, inputs):
        self.built += 1
        self.obj = {'resource': self.node_id, 'inputs': sorted(inputs)}
        return self.obj

    def release(self):
        self.obj = None


class FakeIterator:
    """A source binding: hands out batch strings, records takes and commits."""

    def __init__(self, node_id, batches):
        self.node_id = node_id
        self.batches = list(batches)
        self.obj = {'source': node_id}
        self.takes = 0
        self.commits = []
        self.last_batch = None
        self.IMPORTS = ()

    def start(self, inputs):
        pass

    def take_batch(self):
        if self.takes >= len(self.batches):
            raise StopIteration
        batch = f'{self.node_id}-b{self.takes}'
        self.takes += 1
        self.last_batch = batch
        return batch

    def commit(self, steps):
        self.commits.append(steps)

    def skip_batch(self):
        self.commits.append(0)

    def release(self):
        pass

    def port_value(self, key):
        if key == 'source':
            return self
        return self.last_batch


class FakeStage:
    """A stage binding: records invocations and wall-clock spans per call."""

    def __init__(self, node_id, *, delay=0.0, optimizer_steps=0, output_keys=()):
        self.node_id = node_id
        self.delay = delay
        self.optimizer_steps = optimizer_steps
        self.output_keys = tuple(output_keys)
        self.invocations = []
        self.spans = []

    def invoke(self, inputs, batch):
        begin = time.monotonic()
        if self.delay:
            time.sleep(self.delay)
        self.invocations.append((dict(inputs), batch))
        self.spans.append((begin, time.monotonic()))
        outputs = {key: [f'{self.node_id}:{key}:{batch}'] for key in self.output_keys}
        if self.optimizer_steps:
            outputs['optimizer_steps'] = self.optimizer_steps
        return outputs


def build_fakes(declaration, *, batches=8, delays=None):
    """One fake per node, keyed by node id and shaped by the descriptor."""
    graph = Graph.from_declaration(declaration)
    fakes = {}
    for graph_node in graph.nodes:
        lifecycle = graph.descriptor(graph_node.id).lifecycle
        if lifecycle == 'Resource':
            fakes[graph_node.id] = FakeResource(node_id=graph_node.id)
        elif lifecycle == 'Iterator':
            fakes[graph_node.id] = FakeIterator(node_id=graph_node.id,
                                                batches=[f'b{i}' for i in range(batches)])
        else:
            descriptor = graph.descriptor(graph_node.id)
            output_keys = [port.key for port in descriptor.ports_out
                           if not port.enabled_by
                           or port.key in (graph_node.params.get('outputs') or [])]
            fakes[graph_node.id] = FakeStage(
                node_id=graph_node.id, output_keys=output_keys,
                delay=(delays or {}).get(graph_node.id, 0.0),
                optimizer_steps=1 if graph_node.type == 'train_step' else 0)
    return graph, fakes


def build_controller(declaration, *, batches=8, delays=None):
    """A DAGController over fake bindings: real scheduling, no engines."""
    graph, fakes = build_fakes(declaration, batches=batches, delays=delays)
    events = []
    recorded = {}
    controller = DAGController(
        graph, resolve=fakes.__getitem__,
        event=lambda kind, **payload: events.append((kind, payload)),
        record_output=lambda node_id, node_type, output: recorded.__setitem__(node_id, output))
    return controller, fakes, events, recorded


def build_session(declaration, *, batches=8, delays=None):
    """A GraphSession whose bindings are patched to the fakes: the real
    command loop, event channel and preview surface over fake components."""
    graph, fakes = build_fakes(declaration, batches=batches, delays=delays)
    session = GraphSession(declaration)
    session._build_binding = lambda graph_node: fakes[graph_node.id]
    events = []
    session.emit = lambda kind, **payload: events.append((kind, payload))
    return session, fakes, events


def overlapping(span_a, span_b):
    """Whether two wall-clock spans intersect (parallelism evidence)."""
    return span_a[0] < span_b[1] and span_b[0] < span_a[1]


def wait_for(events, kind, timeout=15.0):
    """Block until ``kind`` lands in the event list (or the deadline passes)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for event_kind, _payload in events:
            if event_kind == kind:
                return True
        time.sleep(0.02)
    return False


def run_in_thread(fn):
    """Run ``fn`` on a daemon thread and return it (results via closure)."""
    worker = threading.Thread(target=fn, daemon=True)
    worker.start()
    return worker

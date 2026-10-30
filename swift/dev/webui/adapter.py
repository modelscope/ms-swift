# Copyright (c) ModelScope Contributors. All rights reserved.
"""Graph compiler: turn a canvas declaration into a construction plan.

The canvas graph (nodes + edges) is compiled here -- parse, wiring validation,
topological order -- and the output is a flat construction plan: one step per
component, each step carrying the factory path and the upstream wiring as plain
data. Nothing is imported or executed at compile time; realizing the plan against
real builders is a separate concern.

This is the Python half of the canvas flow controller (the frontend owns editing
and serialisation to this declaration format).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .registry import REGISTRY

@dataclass(frozen=True)
class NodeSpec:
    """A declared canvas node: type key plus parameter overrides."""
    id: str
    type: str
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class EdgeSpec:
    """A declared canvas edge: from node/port to node/port."""
    source: str
    source_port: str
    target: str
    target_port: str


@dataclass(frozen=True)
class CompileStep:
    """One construction step: what to build and which upstream outputs to feed.

    ``inputs`` maps each input port key of the component to the node id whose
    output feeds it; the executor resolves ids to objects in the same order.
    """
    node_id: str
    type: str
    factory: str
    inputs: dict
    params: dict


class GraphValidationError(ValueError):
    """Raised when the declaration cannot be compiled; message names the seam."""


class GraphCompiler:
    """Validates and linearises a canvas declaration into a construction plan."""

    def __init__(self, nodes: list[NodeSpec], edges: list[EdgeSpec]):
        self._nodes = {n.id: n for n in nodes}
        self._edges = list(edges)

    def compile(self) -> list[CompileStep]:
        """Validate, order topologically, then emit one step per node."""
        self._validate_ports()
        order = self._topo_order()
        return [self._step(self._nodes[nid]) for nid in order]

    # -- validation -----------------------------------------------------

    def _validate_ports(self) -> None:
        """Every edge must land on a declared port of a declared node, with a
        matching channel kind -- a bad wiring fails here, not at runtime.

        A non-collection port takes ONE in-edge: two edges into the same port
        would silently overwrite each other in the plan, so the duplicate is
        rejected here (collection ports declared ``many=True`` stack freely).

        A required input port (not ``many``, not ``optional``) must carry at
        least one in-edge -- a dangling trainer.data would otherwise default
        the dataset and quietly train on the wrong rows."""
        fan_in: dict = {}
        for e in self._edges:
            for node_id, port_key, direction in (
                    (e.source, e.source_port, 'out'), (e.target, e.target_port, 'in')):
                node = self._nodes.get(node_id)
                if node is None:
                    raise GraphValidationError(f'edge references unknown node {node_id!r}')
                descriptor = REGISTRY.get(node.type)
                if descriptor is None:
                    raise GraphValidationError(
                        f'node {node_id!r} has unregistered type {node.type!r}')
                try:
                    descriptor.port(port_key)
                except KeyError:
                    raise GraphValidationError(
                        f'{node.type}.{port_key}: no such {direction} port') from None
            src = self._port_kind(e.source, e.source_port, 'out')
            dst = self._port_kind(e.target, e.target_port, 'in')
            if src != dst:
                raise GraphValidationError(
                    f'{e.source}.{e.source_port} ({src}) cannot feed '
                    f'{e.target}.{e.target_port} ({dst})')
            port = REGISTRY[self._nodes[e.target].type].port(e.target_port)
            if not port.many:
                key = (e.target, e.target_port)
                if key in fan_in:
                    raise GraphValidationError(
                        f'{e.target}.{e.target_port} takes one in-edge, got several '
                        f'(from {fan_in[key]!r} and {e.source!r})')
                fan_in[key] = e.source
        wired = {(e.target, e.target_port) for e in self._edges}
        for node in self._nodes.values():
            descriptor = REGISTRY[node.type]
            for port in descriptor.ports_in:
                if port.many or port.optional:
                    continue
                if (node.id, port.key) not in wired:
                    raise GraphValidationError(
                        f'{node.id}.{port.key} ({port.kind}) is required but unwired -- '
                        f'{node.type} cannot run without it')

    def _port_kind(self, node_id: str, port_key: str, direction: str) -> str:
        node = self._nodes[node_id]
        descriptor = REGISTRY[node.type]
        port = descriptor.port(port_key)
        return port.kind

    # -- ordering -------------------------------------------------------

    def _topo_order(self) -> list[str]:
        """Kahn ordering over data edges; declaration order breaks ties.

        A straight pipeline compiles to its declaration order unchanged; only
        fan-in graphs pay for the sort.
        """
        pending = {nid: set() for nid in self._nodes}
        followers: dict[str, list[str]] = {nid: [] for nid in self._nodes}
        # CTRL edges carry pure ordering and may close the training loop:
        # a cycle made only of CTRL wires is legal and excluded from the sort.
        data_edges = [
            e for e in self._edges
            if self._port_kind(e.source, e.source_port, 'out') != 'CTRL']
        for e in data_edges:
            # 同一对节点可能有多条边（如优势的 batch 与 adv 双投训练步），
            # 依赖只记一次；判定以「target 的依赖集尚未收录该 source」为准。
            if e.source not in pending[e.target]:
                pending[e.target].add(e.source)
                followers[e.source].append(e.target)
        declared = list(self._nodes)
        ready = [nid for nid in declared if not pending[nid]]
        order: list[str] = []
        while ready:
            nid = ready.pop(0)
            order.append(nid)
            for nxt in followers[nid]:
                pending[nxt].discard(nid)
                if not pending[nxt]:
                    ready.append(nxt)
        if len(order) != len(declared):
            stuck = sorted(set(declared) - set(order))
            raise GraphValidationError(f'graph has a cycle through: {", ".join(stuck)}')
        return order

    # -- plan emission ---------------------------------------------------

    def _step(self, node: NodeSpec) -> CompileStep:
        descriptor = REGISTRY[node.type]
        inputs: dict = {}
        for e in self._edges:
            if e.target != node.id:
                continue
            if REGISTRY[node.type].port(e.target_port).many:
                # A symmetric collection port (e.g. several teachers): keep every
                # source in wiring order -- collapsing to one would drop members.
                inputs.setdefault(e.target_port, []).append(e.source)
            else:
                inputs[e.target_port] = e.source
        return CompileStep(
            node_id=node.id,
            type=node.type,
            factory=self._factory_for(node),
            inputs=inputs,
            params=dict(node.params),
        )

    def _factory_for(self, node: NodeSpec) -> str:
        """Deployment mode is a parameter tier: server mode swaps the factory to
        the descriptor's declared client construction point, leaving ports and
        wiring untouched. Components without a client_factory (the trainable
        policy, offline trainers) always construct on the training side."""
        descriptor = REGISTRY[node.type]
        if node.params.get('deploy_mode') == 'server' and descriptor.client_factory:
            return descriptor.client_factory
        return descriptor.factory


def compile_graph(nodes: list[NodeSpec], edges: list[EdgeSpec]) -> list[CompileStep]:
    """Convenience entry: validate and linearise in one call."""
    return GraphCompiler(nodes, edges).compile()

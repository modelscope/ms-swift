# Copyright (c) ModelScope Contributors. All rights reserved.
"""v3 声明、端口校验与同轮依赖规划；不加载模型或执行用户代码。"""
from __future__ import annotations

import copy
import json
import pprint
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .descriptor import ComponentDescriptor, PortSpec
from .registry import REGISTRY, REGISTRY_VERSION
from .schema import CHANNEL_SCHEMAS


class GraphValidationError(ValueError):
    pass


@dataclass(frozen=True)
class NodeSpec:
    id: str
    type: str
    params: Dict[str, Any] = field(default_factory=dict)
    code: str = ''


@dataclass(frozen=True)
class EdgeSpec:
    id: str
    source: str
    source_port: str
    target: str
    target_port: str
    slot: str = ''


@dataclass(frozen=True)
class FrameSpec:
    members: Tuple[str, ...]
    feedback: Tuple[str, ...] = ()
    budget_owner: str = ''
    name: str = ''
    sources: Tuple[str, ...] = ()
    consumers: Tuple[str, ...] = ()
    id: str = ''


class Graph:
    def __init__(self, nodes: List[NodeSpec], edges: List[EdgeSpec],
                 frames: Optional[List[FrameSpec]] = None, *, workflow_id: str = ''):
        self.workflow_id = workflow_id or uuid.uuid4().hex
        self.nodes = copy.deepcopy(list(nodes))
        self.edges = list(edges)
        self.frames = list(frames or [])
        if not self.nodes:
            raise GraphValidationError('nodes: 图不能为空')
        if len(self.nodes) > 4096 or len(self.edges) > 32768:
            raise GraphValidationError('图超过声明规模限制（4096 节点/32768 边）')
        self._nodes = {}
        self._descriptors = {}
        self._edges_by_id = {}
        self._output_cache = {}
        for node in self.nodes:
            self._identity(node.id, 'node.id')
            if node.id in self._nodes:
                raise GraphValidationError(f'重复 node.id {node.id!r}')
            self._nodes[node.id] = node
        for edge in self.edges:
            self._identity(edge.id, 'edge.id')
            if edge.id in self._edges_by_id:
                raise GraphValidationError(f'重复 edge.id {edge.id!r}')
            self._edges_by_id[edge.id] = edge
        self._feedback_edge_ids = {edge_id for frame in self.frames for edge_id in frame.feedback}
        #: (node_id, port_key) -> (source_id, source_port)：作用域解析结果。
        #: 一条边把源节点带入目标节点的作用域（连通即共享），沿非反馈边传递；
        #: 只有 Resource 节点的输出口是作用域变量，Stage 的输出走显式连线。
        self._scope_resolution: Dict[Tuple[str, str], Tuple[str, str]] = {}
        self._validate_nodes()
        self._validate_edges()
        self._validate_inputs()
        self._validate_frames()
        self._ordered = self._order()
        self._validate_feedback()

    @staticmethod
    def _identity(value: Any, location: str) -> str:
        if type(value) is not str or not value or not value.strip():
            raise GraphValidationError(f'{location}: 需要非空字符串身份')
        return value

    @staticmethod
    def _mapping(value: Any, location: str, allowed: Set[str]) -> dict:
        if type(value) is not dict:
            raise GraphValidationError(f'{location}: 需要 object')
        unknown = set(value) - allowed
        if unknown:
            raise GraphValidationError(f'{location}: 未声明字段 {sorted(unknown)}')
        return value

    @staticmethod
    def _list(value: Any, location: str) -> list:
        if type(value) is not list:
            raise GraphValidationError(f'{location}: 需要 list')
        return value

    @classmethod
    def from_declaration(cls, raw: Dict[str, Any]) -> 'Graph':
        raw = cls._mapping(raw, 'graph', {'version', 'workflow_id', 'nodes', 'edges', 'frames'})
        if type(raw.get('version')) is not int or raw['version'] != REGISTRY_VERSION:
            raise GraphValidationError('version: 仅运行 v3；旧图必须显式升级并确认，原声明不会修改')
        workflow_id = cls._identity(raw.get('workflow_id'), 'workflow_id')
        nodes = []
        for index, value in enumerate(cls._list(raw.get('nodes'), 'nodes')):
            value = cls._mapping(value, f'nodes[{index}]', {'id', 'type', 'params', 'code'})
            params = value.get('params', {})
            if type(params) is not dict or any(type(key) is not str for key in params):
                raise GraphValidationError(f'nodes[{index}].params: 需要字符串键的 object')
            code = value.get('code', '')
            if type(code) is not str:
                raise GraphValidationError(f'nodes[{index}].code: 需要字符串，代码空白原样保留')
            nodes.append(NodeSpec(cls._identity(value.get('id'), f'nodes[{index}].id'),
                                  cls._identity(value.get('type'), f'nodes[{index}].type'), params, code))
        edges = []
        for index, value in enumerate(cls._list(raw.get('edges', []), 'edges')):
            value = cls._mapping(value, f'edges[{index}]',
                                 {'id', 'source', 'source_port', 'target', 'target_port', 'slot'})
            fields = [cls._identity(value.get(key), f'edges[{index}].{key}')
                      for key in ('id', 'source', 'source_port', 'target', 'target_port')]
            slot = value.get('slot', '')
            if type(slot) is not str:
                raise GraphValidationError(f'edges[{index}].slot: 需要字符串')
            edges.append(EdgeSpec(*fields, slot))
        frames = []
        for index, value in enumerate(cls._list(raw.get('frames', []), 'frames')):
            value = cls._mapping(value, f'frames[{index}]',
                                 {'id', 'name', 'members', 'feedback', 'budget_owner', 'sources', 'consumers'})
            collections = {}
            for key in ('members', 'feedback', 'sources', 'consumers'):
                collections[key] = tuple(cls._identity(item, f'frames[{index}].{key}')
                                         for item in cls._list(value.get(key, []), f'frames[{index}].{key}'))
            name = value.get('name', '')
            owner = value.get('budget_owner', '')
            if type(name) is not str or type(owner) is not str:
                raise GraphValidationError(f'frames[{index}]: name/budget_owner 需要字符串')
            frames.append(FrameSpec(**collections, budget_owner=owner, name=name,
                                    id=cls._identity(value.get('id'), f'frames[{index}].id')))
        return cls(nodes, edges, frames, workflow_id=workflow_id)

    def node(self, node_id: str) -> NodeSpec:
        try:
            return self._nodes[node_id]
        except KeyError:
            raise GraphValidationError(f'未知节点 {node_id!r}') from None

    def descriptor(self, node_id: str) -> ComponentDescriptor:
        self.node(node_id)
        return self._descriptors[node_id]

    def incoming(self, node_id: str) -> List[EdgeSpec]:
        return [edge for edge in self.edges if edge.target == node_id]

    def outgoing(self, node_id: str) -> List[EdgeSpec]:
        return [edge for edge in self.edges if edge.source == node_id]

    def inputs_of(self, node_id: str, *, feedback: bool = False) -> Dict[str, Any]:
        """保留完整边身份，many 顺序等于边的声明顺序。"""
        result = {}
        for edge in self.incoming(node_id):
            if self._is_feedback(edge) and not feedback:
                continue
            port = self.descriptor(node_id).port(edge.target_port, 'in')
            if port.many:
                result.setdefault(port.key, []).append(edge)
            else:
                if port.key in result:
                    raise GraphValidationError(f'{node_id}.{port.key}: 单输入不能连接多条边')
                result[port.key] = edge
        return result

    def output_ports(self, node_id: str, trail: Tuple[str, ...] = ()) -> Tuple[PortSpec, ...]:
        if node_id in self._output_cache:
            return self._output_cache[node_id]
        if node_id in trail or len(trail) > 128:
            raise GraphValidationError(f'{node_id}: 动态输出推导存在环或超过深度限制')
        node = self.node(node_id)
        descriptor = self.descriptor(node_id)
        result = []
        for port in descriptor.output_ports(node.params):
            if port.profile_param:
                result.append(replace(port, profiles=(node.params[port.profile_param],)))
            elif port.mirror_input:
                incoming = [edge for edge in self.incoming(node_id) if edge.target_port == port.mirror_input]
                if not incoming:
                    continue
                sources = [(edge, self.output_port(edge.source, edge.source_port, (*trail, node_id)))
                           for edge in incoming]
                # 数值组合输出只有一个；筛选则逐具名输入保留独立输出。
                if port.key == 'out':
                    kinds = {(source.kind, source.profiles) for _edge, source in sources}
                    if len(kinds) != 1:
                        raise GraphValidationError(f'{node_id}.{port.key}: 组合输入 kind/profile 必须一致')
                    source = sources[0][1]
                    result.append(replace(port, kind=source.kind, profiles=source.profiles))
                else:
                    many = descriptor.port(port.mirror_input, 'in').many
                    for edge, source in sources:
                        key = f'{port.key}:{edge.slot}' if many else port.key
                        result.append(replace(port, key=key, kind=source.kind, profiles=source.profiles))
            else:
                result.append(port)
        self._output_cache[node_id] = tuple(result)
        return tuple(result)

    def output_port(self, node_id: str, key: str, trail: Tuple[str, ...] = ()) -> PortSpec:
        for port in self.output_ports(node_id, trail):
            if port.key == key:
                return port
        raise GraphValidationError(f'{node_id}.{key}: 输出不存在或未在 outputs 中启用')

    def _validate_nodes(self) -> None:
        updated = []
        for node in self.nodes:
            descriptor = REGISTRY.get(node.type)
            if descriptor is None:
                raise GraphValidationError(f'{node.id}: 未注册组件 {node.type!r}')
            try:
                params = descriptor.validate_params(node.params)
            except (ValueError, TypeError) as exc:
                raise GraphValidationError(f'{node.id}.{exc}') from None
            if node.code and node.type != 'code_component':
                raise GraphValidationError(f'{node.id}: 代码只能放在显式自定义组件')
            if node.type == 'code_component':
                if not node.code:
                    raise GraphValidationError(f'{node.id}.code: 自定义组件必须提供代码')
                try:
                    inputs = tuple(PortSpec(**port) for port in self._custom_ports(params['ports_in']))
                    outputs = tuple(PortSpec(**port) for port in self._custom_ports(params['ports_out']))
                    descriptor = replace(descriptor, lifecycle=params['lifecycle'], ports_in=inputs,
                                         ports_out=outputs, effects=tuple(dict.fromkeys(('code', *params['effects']))))
                except (TypeError, ValueError) as exc:
                    raise GraphValidationError(f'{node.id}: 自定义端口声明错误：{exc}') from None
            for port in descriptor.ports_in + descriptor.ports_out:
                if port.kind not in CHANNEL_SCHEMAS:
                    raise GraphValidationError(f'{node.id}.{port.key}: 未声明通道 {port.kind}')
                for profile in port.profiles:
                    try:
                        CHANNEL_SCHEMAS[port.kind].profile(profile)
                    except KeyError as exc:
                        raise GraphValidationError(f'{node.id}.{port.key}: {exc}') from None
            node = replace(node, params=params)
            self._nodes[node.id] = node
            self._descriptors[node.id] = descriptor
            updated.append(node)
        self.nodes = updated

    @staticmethod
    def _custom_ports(ports: list) -> list:
        result = []
        for value in ports:
            value = dict(value)
            for key in ('profiles', 'accepted_kinds'):
                items = value.get(key, [])
                if type(items) is not list or any(type(item) is not str for item in items):
                    raise ValueError(f'{key}: 需要字符串 list')
                value[key] = tuple(items)
            result.append(value)
        return result

    def _validate_edges(self) -> None:
        slots = set()
        singles = set()
        for edge in self.edges:
            self.node(edge.source)
            self.node(edge.target)
            try:
                target = self.descriptor(edge.target).port(edge.target_port, 'in')
            except KeyError:
                raise GraphValidationError(f'edge {edge.id}: {edge.target}.{edge.target_port} 不是输入口') from None
            source = self.output_port(edge.source, edge.source_port)
            if not target.accepts(source.profiles, source.kind):
                raise GraphValidationError(f'edge {edge.id}: {source.kind}{source.profiles} 不满足 '
                                           f'{edge.target}.{target.key} {target.kind}{target.profiles}')
            if target.many:
                self._identity(edge.slot, f'edge {edge.id}.slot')
                key = (edge.target, edge.target_port, edge.slot)
                if key in slots:
                    raise GraphValidationError(f'edge {edge.id}: 重复 slot {edge.slot!r}')
                slots.add(key)
            else:
                key = (edge.target, edge.target_port)
                if key in singles:
                    raise GraphValidationError(f'{edge.target}.{edge.target_port}: 单输入连接了多条边')
                if edge.slot:
                    raise GraphValidationError(f'edge {edge.id}: 单输入不能声明 slot')
                singles.add(key)
            if source.control != target.control and (source.kind == 'SOURCE' or target.kind == 'SOURCE'):
                raise GraphValidationError(f'edge {edge.id}: source 控制句柄不能作为数据批次消费')

    def _validate_inputs(self) -> None:
        wired = {(edge.target, edge.target_port) for edge in self.edges}
        for node in self.nodes:
            descriptor = self.descriptor(node.id)
            keys = {key for target, key in wired if target == node.id}
            for port in descriptor.ports_in:
                if port.key in keys:
                    continue
                if port.scope:
                    # 连通即共享：未接线的作用域口从上游 Resource 解析唯一同 kind 变量。
                    # 本节点已显式绑定的来源不重复参与其它口的解析（绑过的不算候选）。
                    bound = {edge.source for edge in self.incoming(node.id)}
                    candidates = sorted(
                        item for item in self._scope_sources(node.id).get(port.kind, ())
                        if item[0] not in bound)
                    if len(candidates) == 1:
                        self._scope_resolution[(node.id, port.key)] = candidates[0]
                        continue
                    if len(candidates) > 1:
                        raise GraphValidationError(
                            f'{node.id}.{port.key}: 作用域内多个 {port.kind} 候选 '
                            f'{[item[0] for item in candidates]}，需显式连线消歧')
                    if not port.optional:
                        raise GraphValidationError(
                            f'{node.id}.{port.key}: 必需输入未接线且作用域内无 {port.kind} 资源')
                    continue
                if not port.optional and port.key not in keys:
                    raise GraphValidationError(f'{node.id}.{port.key}: 必需输入未接线')
            for group in descriptor.exclusive_inputs:
                if sum(key in keys for key in group) != 1:
                    raise GraphValidationError(f'{node.id}: {group} 必须恰接入一种来源')
            for param, value, ports in descriptor.required_when:
                if node.params[param] == value and not set(ports) <= keys:
                    raise GraphValidationError(f'{node.id}: {param}={value!r} 时必需输入 {ports}')
            if node.type == 'model' and 'artifact' not in keys and not node.params['model'].strip():
                raise GraphValidationError(f'{node.id}.model: 模型来源或 artifact 必须明确')
            if node.type == 'sampler' and not node.params['model'].strip():
                raise GraphValidationError(f'{node.id}.model: 采样器模型来源必须明确')
            if node.type == 'streaming_driver' and (('engine' in keys) == ('episode' in keys)):
                raise GraphValidationError(f'{node.id}: engine 与 episode 必须恰接入一种来源')
            if node.type == 'advantage' and node.params['estimator'] == 'gae' and node.params['profile'] != 'token':
                raise GraphValidationError(f'{node.id}.profile: GAE 必须输出 token 优势')
            if node.type == 'reward_shape' and node.params['beta'] and node.params['kl'] == 'none':
                raise GraphValidationError(f'{node.id}.kl: 非零 beta 必须选择两路 KL 估计')
            if node.type == 'signal_combine' and node.params['mode'] == 'probability_mixture':
                if self.output_port(node.id, 'out').profiles != ('logps',):
                    raise GraphValidationError(f'{node.id}: 概率混合只能消费 logps')

    def _scope_sources(self, node_id: str) -> Dict[str, List[Tuple[str, str]]]:
        """沿非反馈边可达的 Resource 节点导出变量（kind -> [(节点, 输出口)]）。

        传递规则与 ``upstream_closure`` 同一边集：数据边与前向 CTRL 边共享变量，
        feedback 回边是同轮末尾回到轮初，不向轮初注入变量；SOURCE 控制句柄边
        只传句柄不传变量。Stage 的输出口不是作用域变量——需要 Stage 产物里的
        身份（如 train_step 的训练后模型引用）必须显式连线。"""
        result: Dict[str, List[Tuple[str, str]]] = {}
        for member in self.upstream_closure(node_id):
            if member == node_id:
                continue
            descriptor = self.descriptor(member)
            if descriptor.lifecycle != 'Resource':
                continue
            for port in descriptor.ports_out:
                if not port.control:
                    result.setdefault(port.kind, []).append((member, port.key))
        return result

    def scope_of(self, node_id: str) -> Dict[str, Tuple[str, str]]:
        """该节点经作用域解析的输入口：port_key -> (源节点, 源输出口)。"""
        self.node(node_id)
        return {port_key: source for (nid, port_key), source in self._scope_resolution.items()
                if nid == node_id}

    def _validate_frames(self) -> None:
        frame_ids = set()
        members_seen = set()
        feedback_seen = set()
        for frame in self.frames:
            self._identity(frame.id, 'frame.id')
            if frame.id in frame_ids:
                raise GraphValidationError(f'重复 frame.id {frame.id}')
            frame_ids.add(frame.id)
            members = set(frame.members)
            if not members or len(members) != len(frame.members) or members & members_seen:
                raise GraphValidationError(f'frame {frame.id}: members 不能为空、重复或跨 frame 共享')
            members_seen.update(members)
            for node_id in members:
                if self.descriptor(node_id).lifecycle == 'Resource':
                    raise GraphValidationError(f'frame {frame.id}: Resource {node_id} 不能进入循环')
                if 'external' in self.descriptor(node_id).effects:
                    raise GraphValidationError(f'frame {frame.id}: 外部部署/发布必须单独授权运行')
            sources = {node_id for node_id in members if self.descriptor(node_id).lifecycle == 'Iterator'}
            if not sources or set(frame.sources) != sources or len(set(frame.sources)) != len(frame.sources):
                raise GraphValidationError(f'frame {frame.id}: sources 必须明确列出全部 Iterator 成员')
            if not frame.consumers or len(set(frame.consumers)) != len(frame.consumers) or not set(frame.consumers) <= members - sources:
                raise GraphValidationError(f'frame {frame.id}: consumers 必须明确列出唯一非 source 消费终点')
            if frame.budget_owner not in frame.consumers or 'train' not in self.descriptor(frame.budget_owner).effects:
                raise GraphValidationError(f'frame {frame.id}: budget_owner 必须是消费者中的真实训练节点 ID')
            for edge_id in frame.feedback:
                edge = self._edges_by_id.get(edge_id)
                if edge is None or edge_id in feedback_seen:
                    raise GraphValidationError(f'frame {frame.id}: feedback 不存在或重复 {edge_id}')
                feedback_seen.add(edge_id)
                if edge.source not in members or edge.target not in sources:
                    raise GraphValidationError(f'frame {frame.id}: feedback 必须从本框成员回到本框 source')
                if self.output_port(edge.source, edge.source_port).kind != 'CTRL':
                    raise GraphValidationError(f'frame {frame.id}: feedback 只能是 CTRL，不能是数据环')
            # 声明的消费者必须覆盖本框全部副作用，否则无法确认消费结束。
            effects = {node_id for node_id in members if self.descriptor(node_id).effects}
            if not effects <= set(frame.consumers):
                raise GraphValidationError(f'frame {frame.id}: consumers 遗漏副作用节点 {sorted(effects - set(frame.consumers))}')

    def _validate_feedback(self) -> None:
        for edge_id in self._feedback_edge_ids:
            edge = self._edges_by_id[edge_id]
            if edge.target not in self.upstream_closure(edge.source):
                raise GraphValidationError(f'edge {edge.id}: feedback 不是实际同轮路径的回边')

    def _is_feedback(self, edge: EdgeSpec) -> bool:
        return edge.id in self._feedback_edge_ids

    def _data_edges(self) -> List[EdgeSpec]:
        """同轮依赖包括正向 CTRL；反馈和控制句柄不触发数据提取。"""
        return [edge for edge in self.edges if not self._is_feedback(edge)
                and not self.descriptor(edge.target).port(edge.target_port, 'in').control]

    def _order(self) -> List[str]:
        dependencies = {node_id: set() for node_id in self._nodes}
        followers = {node_id: [] for node_id in self._nodes}
        # 控制句柄仍依赖 source 的构造，但不等待 source 取批。
        for edge in self.edges:
            if not self._is_feedback(edge) and edge.source not in dependencies[edge.target]:
                dependencies[edge.target].add(edge.source)
                followers[edge.source].append(edge.target)
        ready = [node_id for node_id in self._nodes if not dependencies[node_id]]
        result = []
        while ready:
            node_id = ready.pop(0)
            result.append(node_id)
            for target in followers[node_id]:
                dependencies[target].discard(node_id)
                if not dependencies[target]:
                    ready.append(target)
        if len(result) != len(self._nodes):
            raise GraphValidationError(f'非法同轮环：{sorted(set(self._nodes) - set(result))}')
        return result

    @property
    def order(self) -> List[str]:
        return list(self._ordered)

    def upstream_closure(self, target: str) -> Set[str]:
        self.node(target)
        reverse = {}
        for edge in self._data_edges():
            reverse.setdefault(edge.target, []).append(edge.source)
        result = {target}
        pending = [target]
        while pending:
            for source in reverse.get(pending.pop(), ()):
                if source not in result:
                    result.add(source)
                    pending.append(source)
        return result

    def training_frame(self, target: str) -> Optional[FrameSpec]:
        return next((frame for frame in self.frames if target in frame.members), None)

    def resource_owners(self, node_id: str) -> Set[str]:
        result = set()
        descriptor = self.descriptor(node_id)
        if descriptor.owner_kind:
            result.add(f'{descriptor.owner_kind}:{node_id}')
        for edge in self.incoming(node_id):
            if self._is_feedback(edge):
                continue
            source = self.descriptor(edge.source)
            if source.owner_kind:
                result.add(f'{source.owner_kind}:{edge.source}')
        return result

    def resource_owner(self, node_id: str) -> Optional[str]:
        owners = sorted(self.resource_owners(node_id))
        return owners[0] if owners else None

    def access_plan(self, node_ids: Iterable[str]) -> List[List[str]]:
        selected = set(node_ids)
        lanes = []
        for node_id in self.order:
            if node_id not in selected:
                continue
            owners = self.resource_owners(node_id)
            touching = [index for index, (_lane, resources) in enumerate(lanes) if owners & resources]
            if not touching:
                lanes.append(([node_id], owners))
            else:
                first = touching[0]
                lanes[first][0].append(node_id)
                lanes[first][1].update(owners)
                for index in reversed(touching[1:]):
                    lane, resources = lanes.pop(index)
                    lanes[first][0].extend(lane)
                    lanes[first][1].update(resources)
        return [[node_id for node_id in self.order if node_id in lane] for lane, _owners in lanes]

    @classmethod
    def from_yaml(cls, text: str) -> 'Graph':
        import yaml
        return cls.from_declaration(yaml.safe_load(text))

    def to_yaml(self) -> str:
        import yaml
        return yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False)

    def to_python(self) -> str:
        declaration = pprint.pformat(self.to_dict(), sort_dicts=False, width=100)
        return ('from swift.dev.webui.graph import Graph\n'
                'from swift.dev.webui.session import GraphSession\n\n'
                f'declaration = {declaration}\n'
                'graph = Graph.from_declaration(declaration)\n'
                'session = GraphSession(graph.to_dict())\n')

    def to_dict(self, *, redact: bool = True) -> Dict[str, Any]:
        nodes = []
        for node in self.nodes:
            params = copy.deepcopy(node.params)
            if redact:
                for param in self.descriptor(node.id).params:
                    if param.secret:
                        params.pop(param.name, None)
            nodes.append({'id': node.id, 'type': node.type, 'params': params,
                          **({'code': node.code} if node.code else {})})
        return {'version': REGISTRY_VERSION, 'workflow_id': self.workflow_id, 'nodes': nodes,
                'edges': [{'id': edge.id, 'source': edge.source, 'source_port': edge.source_port,
                           'target': edge.target, 'target_port': edge.target_port,
                           **({'slot': edge.slot} if edge.slot else {})} for edge in self.edges],
                'frames': [{'id': frame.id, 'name': frame.name, 'members': list(frame.members),
                            'feedback': list(frame.feedback), 'budget_owner': frame.budget_owner,
                            'sources': list(frame.sources), 'consumers': list(frame.consumers)} for frame in self.frames]}


class GraphUpgrade:
    """旧图只在副本中展开；待选择项和校验错误未清除时不允许确认。"""

    RENAMES = {'forward_signal': 'model_forward', 'signal_fusion': 'signal_combine',
               'reference_update': 'weight_average'}
    USE_ARGUMENTS = {'training_old': 'old_logps', 'ref': 'ref_logps',
                     'teacher': 'teacher_logps', 'rollout': 'rollout_logps'}

    def __init__(self, declaration: dict):
        self.original = copy.deepcopy(declaration)
        self.candidate = copy.deepcopy(declaration)
        self.changes = []
        self.choices = []
        self._uses = {}

    def _issue(self, location: str, reason: str, options: Tuple[str, ...] = ()) -> None:
        self.choices.append({'location': location, 'reason': reason, 'options': list(options)})

    def _nodes(self) -> dict:
        return {node['id']: node for node in self.candidate['nodes']}

    def _incoming(self, node_id: str, port: str = '') -> list:
        return [edge for edge in self.candidate['edges'] if edge['target'] == node_id
                and (not port or edge['target_port'] == port)]

    def _add_edge(self, source: str, source_port: str, target: str, target_port: str, slot: str = '') -> None:
        edge_id = f'upgrade:{source}:{source_port}:{target}:{target_port}:{slot}'
        existing = {edge['id'] for edge in self.candidate['edges']}
        while edge_id in existing:
            edge_id += ':1'
        self.candidate['edges'].append({'id': edge_id, 'source': source, 'source_port': source_port,
                                        'target': target, 'target_port': target_port,
                                        **({'slot': slot} if slot else {})})

    def _new_node(self, node_id: str, type_key: str, params: dict) -> str:
        while node_id in self._nodes():
            node_id += ':1'
        self.candidate['nodes'].append({'id': node_id, 'type': type_key, 'params': params})
        self.changes.append({'node': node_id, 'action': 'add', 'type': type_key})
        return node_id

    def _remove_node(self, node_id: str) -> None:
        self.candidate['nodes'] = [node for node in self.candidate['nodes'] if node['id'] != node_id]
        self.candidate['edges'] = [edge for edge in self.candidate['edges']
                                   if edge['source'] != node_id and edge['target'] != node_id]
        self.changes.append({'node': node_id, 'action': 'remove'})

    def preview(self, selections: Optional[dict] = None) -> dict:
        # 反复预览不继续改上次副本，也不迁移运行状态。
        self.candidate = copy.deepcopy(self.original)
        self.changes = []
        self.choices = []
        self._uses = {}
        version = self.original.get('version', 2)
        if type(version) is not int or version not in (2, 3):
            self._issue('version', '未知版本，保留原声明并禁止运行')
            return self._result()
        if version == 3:
            return self._result()
        if type(self.original.get('nodes')) is not list or type(self.original.get('edges', [])) is not list:
            self._issue('graph', '旧图节点或边结构畸形，不能静默过滤')
            return self._result()
        self.candidate['version'] = 3
        self.candidate.setdefault('workflow_id', uuid.uuid4().hex)
        self.candidate.setdefault('edges', [])
        self.candidate.setdefault('frames', [])
        selections = selections or {}
        try:
            self._expand_views(selections)
            self._expand_trainers()
            self._expand_bindings()
            self._migrate_nodes(selections)
            self._migrate_edges()
            self._migrate_frames(selections)
        except (KeyError, TypeError, ValueError) as exc:
            self._issue('graph', f'旧声明无法确定迁移：{exc}')
        return self._result()

    def _expand_views(self, selections: dict) -> None:
        for node in list(self.candidate['nodes']):
            if node['type'] != 'model_view':
                continue
            inputs = self._incoming(node['id'], 'model')
            if len(inputs) != 1:
                self._issue(node['id'], '模型视图没有唯一来源')
                continue
            source = inputs[0]['source']
            view = node.get('params', {}).get('view', 'adapter')
            if view == 'frozen':
                selection = selections.get(f'{node["id"]}.view')
                if selection not in ('independent_frozen', 'shared_base'):
                    self._issue(f'{node["id"]}.view', '冻结视图不能猜测为独立权重或共享 base',
                                ('independent_frozen', 'shared_base'))
                    continue
                if selection == 'independent_frozen':
                    original = self._nodes()[source]
                    if original['type'] != 'model':
                        self._issue(node['id'], '独立冻结模型缺少完整模型构造来源')
                        continue
                    params = copy.deepcopy(original.get('params', {}))
                    params['freeze'] = True
                    source = self._new_node(f'{node["id"]}:frozen', 'model', params)
                    view = 'adapter'
                else:
                    view = 'base'
            for edge in self.candidate['edges']:
                if edge['source'] == node['id']:
                    edge['source'] = source
                    edge['source_port'] = 'base' if view == 'base' else 'model'
            self._remove_node(node['id'])

    def _expand_trainers(self) -> None:
        for trainer in list(self.candidate['nodes']):
            if trainer['type'] != 'trainer':
                continue
            incoming = {edge['target_port']: edge for edge in self._incoming(trainer['id'])}
            if 'model' not in incoming:
                self._issue(trainer['id'], '训练器缺少模型来源')
                continue
            model_id = incoming['model']['source']
            model_params = self._nodes()[model_id].get('params', {})
            # loss 与 optimizer 已是 train_step 的参数：接线或模型参数携带的取值
            # 折叠进目标训练步，不再物化成独立节点。
            loss_name = model_params.get('loss')
            loss_edge = incoming.get('loss')
            if loss_edge is not None:
                loss_name = self._nodes()[loss_edge['source']].get('params', {}).get('loss', loss_name)
            optim_name = model_params.get('optim')
            spec_edge = incoming.get('spec')
            if spec_edge is not None:
                optim_name = self._nodes()[spec_edge['source']].get('params', {}).get('optim', optim_name)
            for edge in list(self.candidate['edges']):
                if edge['source'] != trainer['id']:
                    continue
                target = self._nodes()[edge['target']]
                if edge['source_port'] == 'model':
                    edge['source'] = model_id
                    edge['source_port'] = 'model'
                elif target['type'] == 'train_step' and edge['target_port'] == 'trainer':
                    edge.update(source=model_id, source_port='model', target_port='model')
                    params = target.setdefault('params', {})
                    if loss_name:
                        params['loss'] = loss_name
                    if optim_name:
                        params['optim'] = optim_name
                    for port_key, node_type in (('loss', 'loss'), ('spec', 'optimizer')):
                        wired = incoming.get(port_key)
                        if wired is None:
                            continue
                        source_params = self._nodes()[wired['source']].get('params', {})
                        for param_key in ('lr', 'lr_scheduler_type', 'warmup_ratio',
                                          'weight_decay', 'num_training_steps'):
                            if param_key in source_params:
                                params[param_key] = source_params[param_key]
                        if node_type == 'loss' and source_params.get('kwargs'):
                            params['loss_kwargs'] = source_params['kwargs']
                        if node_type == 'optimizer' and source_params.get('optimizer_config'):
                            params['optimizer_config'] = source_params['optimizer_config']
                else:
                    self._issue(edge['id'], '训练器输出用途不明确，请改为模型或训练步骤来源')
            self._remove_node(trainer['id'])

    def _expand_bindings(self) -> None:
        for binding in list(self.candidate['nodes']):
            if binding['type'] != 'bind_signals':
                continue
            incoming = self._incoming(binding['id'])
            consumers = [edge for edge in self.candidate['edges'] if edge['source'] == binding['id']]
            features = [edge for edge in incoming if edge['target_port'] == 'features']
            if len(features) != 1:
                self._issue(binding['id'], '轨迹到训练输入需要明确编码模板与 profile，不能隐式适配')
                continue
            for consumer in consumers:
                target = consumer['target']
                if self._nodes()[target]['type'] != 'train_step':
                    self._issue(consumer['id'], '信号绑定输出不是训练输入')
                    continue
                consumer.update(source=features[0]['source'], source_port=features[0]['source_port'], target_port='batch')
                for edge in incoming:
                    if edge['target_port'] in ('signals', 'adv'):
                        slot = edge.get('slot') or ('advantages' if edge['target_port'] == 'adv' else '')
                        if not slot:
                            source = self._nodes()[edge['source']]
                            use = source.get('params', {}).get('use')
                            slot = self.USE_ARGUMENTS.get(use, '')
                        if not slot:
                            self._issue(edge['id'], 'loss argument 缺少确定用途/slot')
                            continue
                        self._add_edge(edge['source'], edge['source_port'], target, 'arguments', slot)
            self._remove_node(binding['id'])

    def _migrate_nodes(self, selections: dict) -> None:
        for node in self.candidate['nodes']:
            old_type = node['type']
            node['type'] = self.RENAMES.get(old_type, old_type)
            node.pop('status', None)
            node.pop('result', None)
            params = node.setdefault('params', {})
            if old_type != node['type']:
                self.changes.append({'node': node['id'], 'action': 'rename', 'from': old_type, 'to': node['type']})
            if old_type == 'model':
                params.pop('loss', None)
                params.pop('optim', None)
                for key in ('max_length', 'truncation_strategy', 'loss_scale', 'system'):
                    if key in params:
                        self._issue(f'{node["id"]}.{key}', '模型中的编码配置必须迁入消费组件的 tokenizer 参数，不能丢弃')
            if old_type == 'template':
                self._issue(node['id'], 'template 已除名：tokenizer 来源迁入消费组件（train_step/model_forward/sampler 等）的参数')
            if old_type == 'forward_signal':
                self._uses[node['id']] = params.pop('use', None)
                params.pop('compute_loss', None)
                fields = [edge['source_port'] for edge in self.candidate['edges'] if edge['source'] == node['id']]
                if any(key not in ('logps', 'values', 'logits', 'topk', 'embeddings') for key in fields):
                    self._issue(node['id'], '旧前向输出缺少明确数值 profile')
                params['outputs'] = list(dict.fromkeys(fields or ['logps']))
            if node['type'] in ('forward_signal', 'train_step') and not params.get('tokenizer'):
                self._issue(f'{node["id"]}.tokenizer', '编码已并入组件内部，必须填 tokenizer 来源')
            if old_type == 'train_step':
                combined = params.pop('mini_micro_batch', None)
                if combined is not None:
                    parts = combined.split('/') if type(combined) is str else []
                    if len(parts) == 2 and all(part.strip().isdigit() for part in parts):
                        params['micro_batch_size'] = int(parts[0].strip())
                        params['gradient_accumulation_steps'] = int(parts[1].strip())
                    else:
                        self._issue(f'{node["id"]}.mini_micro_batch', '批划分歧义，需要分别设置整数参数')
                for key in ('save_steps', 'publish_cadence', 'max_steps'):
                    if key in params:
                        self._issue(f'{node["id"]}.{key}', '节奏/预算需迁到 checkpoint、weight_sync 或运行命令')
            descriptor = REGISTRY.get(node['type'])
            if descriptor is None:
                self._issue(node['id'], f'未知插件/旧类型 {node["type"]}，需要真实契约')
                continue
            declared = {param.name: param for param in descriptor.params}
            for key, value in list(params.items()):
                if key not in declared:
                    self._issue(f'{node["id"]}.{key}', '旧参数没有确定的新构造字段，不静默删除')
                    continue
                param = declared[key]
                if type(value) is str and param.type in ('bool', 'int', 'float'):
                    try:
                        parsed = json.loads(value)
                        params[key] = param.validate(parsed)
                        self.changes.append({'node': node['id'], 'param': key, 'action': 'explicit_type_upgrade'})
                    except (ValueError, TypeError):
                        self._issue(f'{node["id"]}.{key}', f'不能明确转换为 {param.type}')

    def _migrate_edges(self) -> None:
        for edge in self.candidate['edges']:
            source = self._nodes()[edge['source']]
            target = self._nodes()[edge['target']]
            if source['type'] == 'model' and edge['source_port'] == 'base':
                if target['type'] == 'model_forward' and edge['target_port'] == 'model':
                    target['params']['disable_lora'] = True
                    edge['source_port'] = 'model'
                else:
                    self._issue(edge['id'], 'base 必须是显式前向调用，不是另一个模型对象')
            descriptor = REGISTRY.get(target['type'])
            if descriptor:
                try:
                    port = descriptor.port(edge['target_port'], 'in')
                    if port.many and not edge.get('slot'):
                        edge['slot'] = edge['source']
                        self.changes.append({'edge': edge['id'], 'action': 'name_slot', 'slot': edge['slot']})
                except KeyError:
                    self._issue(edge['id'], '端口已移除或 profile 不明确，需要重新接线')

    def _migrate_frames(self, selections: dict) -> None:
        known = set(self._nodes())
        for index, frame in enumerate(self.candidate['frames']):
            frame.pop('status', None)
            frame['members'] = [node_id for node_id in frame.get('members', []) if node_id in known]
            frame.setdefault('id', f'upgrade:frame:{index}')
            for key in ('sources', 'consumers', 'budget_owner', 'feedback'):
                selection = selections.get(f'{frame["id"]}.{key}')
                if selection is not None:
                    frame[key] = selection
                elif key not in frame or not frame[key]:
                    self._issue(f'{frame["id"]}.{key}', '必须明确选择真实 source、消费终点、预算 owner 和反馈边')

    def _result(self) -> dict:
        validation_error = ''
        if not self.choices:
            try:
                Graph.from_declaration(self.candidate)
            except GraphValidationError as exc:
                validation_error = str(exc)
        return {'original': copy.deepcopy(self.original), 'candidate': copy.deepcopy(self.candidate),
                'changes': copy.deepcopy(self.changes), 'choices': copy.deepcopy(self.choices),
                'validation_error': validation_error, 'can_confirm': not self.choices and not validation_error}

    @staticmethod
    def confirm(preview: dict) -> Graph:
        if not preview.get('can_confirm') or preview.get('choices') or preview.get('validation_error'):
            raise GraphValidationError('升级仍有待选择项或校验错误，不能确认运行')
        return Graph.from_declaration(preview['candidate'])


def parse_declaration(raw: Dict[str, Any]) -> Tuple[List[NodeSpec], List[EdgeSpec], List[FrameSpec]]:
    graph = Graph.from_declaration(raw)
    return graph.nodes, graph.edges, graph.frames

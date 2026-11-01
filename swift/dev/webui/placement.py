# Copyright (c) ModelScope Contributors. All rights reserved.
"""Device-group planning: the graph's GPU components onto the resource cards.

Pure planning, no twinkle import: the bootstrap converts the plan into real
DeviceGroups and declares them once at ``twinkle.initialize`` -- groups exist
only at init time, so the whole graph's demand is planned before the first
component builds.

Placement rules (the disaggregated form the resource page implies):

- Every GPU component (any descriptor declaring ``gpu_count``) owns one
  DISJOINT card segment, assigned in graph order; colocate has no knob to
  reject because it has no knob at all.
- ``gpu_ids`` wins over ``gpu_count``; a count that disagrees with the list
  (anything besides the default 1 or the list's length) is a construction
  error.
- A model spends one rank per card -- its ``parallel_spec`` shapes the mesh,
  never the group. A sampler's engine shards ``tp`` GPUs per rank, so its
  segment becomes ``cards / tp`` ranks of ``tp`` GPUs each.
- Demand beyond the allocated cards, two components naming the same card, or
  two node ids sanitising to one group name all fail loudly with both names.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Tuple

from .graph import Graph, GraphValidationError

#: The CPU group the numeric worker's ``remote_group`` names.
NUMERIC_GROUP = 'webui_numeric'

_GROUP_NAME = re.compile(r'[^0-9a-zA-Z_]')
_TP_TOKEN = re.compile(r'([a-z]+)(\d*)')


@dataclass(frozen=True)
class GroupSpec:
    """One planned device group: name, rank positions, GPUs per rank."""
    name: str
    ranks: Tuple[int, ...]
    gpus_per_worker: int = 1


@dataclass(frozen=True)
class DevicePlan:
    """The graph's whole placement: groups plus the per-node group names."""
    groups: Tuple[GroupSpec, ...]
    remote_groups: Dict[str, str]
    cards: Dict[str, Tuple[int, ...]]
    nproc_per_node: int

    def signature(self) -> str:
        """A stable fingerprint: any change here needs a session rebuild."""
        import json
        return json.dumps({
            'nproc': self.nproc_per_node,
            'groups': [[group.name, list(group.ranks), group.gpus_per_worker]
                       for group in self.groups]}, sort_keys=True)


def group_name(node_id: str) -> str:
    """The DeviceGroup name for a node: its id made group-safe."""
    return _GROUP_NAME.sub('_', node_id)


#: The native DeviceMesh grammar's separators (``device_mesh._SPEC_SEPARATORS``).
_SPEC_SEPARATORS = '_-,. \t'


def sampler_tp(spec: str, cards: int) -> int:
    """The engine's tensor-parallel width off a ``parallel_spec`` string.

    Reads the ``tp`` token of the native DeviceMesh grammar (``dp2tp2``-style
    concatenation); the mesh itself is built by the component through
    ``DeviceMesh.from_spec`` -- this only sizes the group's ranks. The
    default is the single-dim form: one engine spanning every card.
    """
    text = str(spec or '').strip().lower()
    while text:
        if text[0] in _SPEC_SEPARATORS:
            text = text[1:]
            continue
        match = _TP_TOKEN.match(text)
        if match is None or not match.group(1):
            raise GraphValidationError(f'parallel_spec: 无法解析 {spec!r}')
        key, digits = match.group(1), match.group(2)
        if key == 'tp':
            if not digits or int(digits) < 1:
                raise GraphValidationError(f'parallel_spec: tp 需要正整数，得到 {spec!r}')
            tp = int(digits)
            if cards % tp:
                raise GraphValidationError(
                    f'parallel_spec: tp={tp} 不整除占卡数 {cards}，无法分片引擎')
            return tp
        text = text[match.end():]
    return cards


def plan_device_groups(graph: Graph) -> DevicePlan:
    """Plan every GPU component's disjoint card segment, in graph order."""
    slot_count, slot_of_card = _slots(graph)
    gpu_nodes = [node for node in graph.nodes if _declares_gpu(graph, node.id)]
    remote_groups: Dict[str, str] = {}
    cards: Dict[str, Tuple[int, ...]] = {}
    names: Dict[str, str] = {}
    owner: Dict[int, str] = {}
    free = list(range(slot_count))
    for node in gpu_nodes:
        name = group_name(node.id)
        if name in names:
            raise GraphValidationError(
                f'节点 {node.id!r} 与 {names[name]!r} 的组名同为 {name!r}；重命名其中一个')
        ids = list(node.params.get('gpu_ids') or [])
        count = int(node.params.get('gpu_count') or 1)
        if ids:
            if count != 1 and count != len(ids):
                raise GraphValidationError(
                    f'{node.id}: gpu_ids 与 gpu_count 同时设置是构造错误；'
                    '卡列表优先，请只保留其一')
            segment = []
            for card in ids:
                if card not in slot_of_card:
                    raise GraphValidationError(
                        f'{node.id}.gpu_ids: 卡 {card} 不在资源页分配的卡内 '
                        f'{sorted(slot_of_card)}')
                slot = slot_of_card[card]
                if slot in owner:
                    raise GraphValidationError(
                        f'卡 {card}（槽位 {slot}）同时被 {owner[slot]!r} 与 {node.id!r} 声明')
                owner[slot] = node.id
                segment.append(slot)
                free.remove(slot)
        else:
            if count < 1:
                raise GraphValidationError(f'{node.id}.gpu_count: 需要 >= 1 的占卡数')
            if count > len(free):
                raise GraphValidationError(
                    f'{node.id}: 需要再占 {count} 张卡，但资源页仅剩 {len(free)} 张空闲'
                    f'（共 {slot_count} 张）；请扩大分配或调低占卡数')
            segment = free[:count]
            free = free[count:]
            owner.update({slot: node.id for slot in segment})
        names[name] = node.id
        remote_groups[node.id] = name
        cards[node.id] = tuple(segment)
    groups = []
    for node in gpu_nodes:
        segment = cards[node.id]
        if graph.descriptor(node.id).type == 'sampler':
            tp = sampler_tp(str(node.params.get('parallel_spec') or ''), len(segment))
            width, rank_count = tp, len(segment) // tp
        else:
            width, rank_count = 1, len(segment)
        groups.append(GroupSpec(group_name(node.id), tuple(segment), width))
    return DevicePlan(tuple(groups), remote_groups, cards, slot_count)


def _slots(graph: Graph) -> Tuple[int, Dict[int, int]]:
    """The session's card slots: local cards are physical ids mapped to slot
    positions; cluster cards are already positions over the announced count."""
    resource = graph.resource
    if resource.mode == 'cluster':
        if resource.cluster_gpus < 1:
            raise GraphValidationError('resource.cluster_gpus: 集群卡数必须是正整数')
        return resource.cluster_gpus, {slot: slot for slot in range(resource.cluster_gpus)}
    if not resource.cards:
        raise GraphValidationError(
            'resource: 本机模式未选择任何卡 -- 在资源页选择卡；空分配意味着没有 GPU')
    ordered = sorted(resource.cards)
    return len(ordered), {card: slot for slot, card in enumerate(ordered)}


def _declares_gpu(graph: Graph, node_id: str) -> bool:
    return any(param.name == 'gpu_count' for param in graph.descriptor(node_id).params)

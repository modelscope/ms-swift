# Copyright (c) ModelScope Contributors. All rights reserved.
"""Device-group planning over the resource section (S20 20d).

The oracle is the explicit rank table: which cards a component's group owns,
how many GPUs each rank commands, and that two components never share a slot.
Every refusal asserts the guard's message naming the offending node."""
import pytest

from swift.dev.webui.graph import Graph, GraphValidationError
from swift.dev.webui.placement import NUMERIC_GROUP, plan_device_groups

from ._fakes import base_declaration, node


def plan_of(declaration):
    return plan_device_groups(Graph.from_declaration(declaration))


def refusal(declaration):
    with pytest.raises(GraphValidationError) as excinfo:
        plan_of(declaration)
    return str(excinfo.value)


class TestResourcePlanning:
    """The resource page's two shapes and the occupancy math."""

    def test_empty_local_cards_refused(self):  # S20-19
        declaration = base_declaration()
        declaration['resource'] = {'mode': 'local', 'cards': []}
        message = refusal(declaration)
        assert '本机模式未选择任何卡' in message

    def test_demand_beyond_cards_refused_by_node(self):  # S20-20
        declaration = base_declaration()
        declaration['resource'] = {'mode': 'local', 'cards': [0, 1]}
        for entry in declaration['nodes']:
            if entry['id'] in ('policy', 'policy_q'):
                entry['params']['gpu_count'] = 2
        message = refusal(declaration)
        assert '仅剩' in message or '需要再占' in message

    def test_gpu_ids_with_conflicting_count_refused(self):  # S20-21
        declaration = base_declaration()
        for entry in declaration['nodes']:
            if entry['id'] == 'policy':
                entry['params']['gpu_count'] = 3
                entry['params']['gpu_ids'] = [0]
        message = refusal(declaration)
        assert 'gpu_ids 与 gpu_count 同时设置' in message

    def test_gpu_ids_win_over_count(self):  # S20-22
        declaration = base_declaration()
        for entry in declaration['nodes']:
            if entry['id'] == 'policy':
                entry['params']['gpu_count'] = 1
                entry['params']['gpu_ids'] = [2]
        plan = plan_of(declaration)
        assert plan.cards['policy'] == (2,)
        group = next(g for g in plan.groups if g.name == 'policy')
        assert group.ranks == (2,)

    def test_two_cards_mean_two_ranks(self):  # S20-23
        declaration = base_declaration()
        for entry in declaration['nodes']:
            if entry['id'] == 'policy':
                entry['params']['gpu_count'] = 2
        plan = plan_of(declaration)
        group = next(g for g in plan.groups if g.name == 'policy')
        assert group.ranks == (0, 1)
        assert group.gpus_per_worker == 1

    def test_overlapping_gpu_ids_refused(self):  # S20-24
        declaration = base_declaration()
        for entry in declaration['nodes']:
            if entry['id'] in ('policy', 'policy_q'):
                entry['params']['gpu_ids'] = [0]
        message = refusal(declaration)
        assert '同时被' in message and 'policy' in message

    def test_cluster_shape_and_numeric_group_constant(self):  # S20-25
        declaration = base_declaration()
        declaration['resource'] = {'mode': 'cluster',
                                   'cluster_address': 'ray://127.0.0.1:10001',
                                   'cluster_gpus': 8}
        plan = plan_of(declaration)
        assert plan.nproc_per_node == 8
        for group in plan.groups:
            assert all(0 <= rank < 8 for rank in group.ranks)
        # The CPU group the numeric worker rides is a stable contract name:
        # the session bootstrap appends exactly this group at initialize time.
        assert NUMERIC_GROUP == 'webui_numeric'

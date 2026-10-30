# Copyright (c) ModelScope Contributors. All rights reserved.
import importlib
import pytest
import ray
from types import SimpleNamespace
from unittest.mock import Mock

from swift.ray.megatron.resource_pool import ResourcePool, ResourcePoolManager


@pytest.fixture
def scheduler(monkeypatch):
    nodes = [{
        'NodeID': name,
        'NodeManagerAddress': address,
        'Labels': {
            'ray.io/node-id': name
        }
    } for name, address in [('b', '10.0.0.2'), ('a', '10.0.0.1'), ('c', '10.0.0.3')]]
    groups = []
    ready = Mock()
    remove = Mock()

    def create(bundles, **kwargs):
        index = len(groups)
        pg = SimpleNamespace(id=index, bundles=bundles, node=nodes[index]['NodeID'])
        pg.ready = Mock(return_value=f'ready-{index}')
        groups.append(pg)
        return pg

    placement = Mock(side_effect=create)
    table = Mock(side_effect=lambda pg_id: {'bundles_to_node_id': {0: groups[pg_id].node}})

    def discover(pool):
        # Each count must still describe its own PG after rank-order sorting.
        assert [len(pg.bundles) for pg in pool.pgs] == pool.process_on_nodes
        pool.bundle_infos = [(pg.node, str(i)) for pg in pool.pgs for i in range(len(pg.bundles))]
        pool.node_ips = [
            next(node['NodeManagerAddress'] for node in nodes if node['NodeID'] == pg.node) for pg in pool.pgs
        ]

    monkeypatch.setattr(ray, 'nodes', lambda: nodes)
    monkeypatch.setattr(ray, 'get', ready)
    monkeypatch.setattr(ray.util, 'remove_placement_group', remove)
    monkeypatch.setattr(ray._private.state.state, 'placement_group_table', table)
    monkeypatch.setattr(importlib.import_module('ray.util.placement_group'), 'placement_group', placement)
    monkeypatch.setattr(ResourcePool, '_discover_bundle_infos', discover)
    return SimpleNamespace(nodes=nodes, groups=groups, placement=placement, table=table, ready=ready, remove=remove)


@pytest.mark.parametrize('device_name,resource', [('GPU', 'GPU'), ('cuda', 'GPU'), ('NPU', 'NPU'), ('npu', 'NPU')])
@pytest.mark.parametrize('colocate_count', [0, 3])
def test_pool_excludes_previous_nodes_and_preserves_reservations(scheduler, device_name, resource, colocate_count):
    pool = ResourcePool([2, 2, 2], max_colocate_count=colocate_count)
    pool.create(device_name)

    submissions = scheduler.placement.call_args_list
    expected_bundle = {resource: 1, 'CPU': max(colocate_count, 1)}
    for invocation in submissions:
        assert invocation.args == ([expected_bundle] * 2, )
        assert invocation.kwargs['strategy'] == 'STRICT_PACK'
    assert 'bundle_label_selector' not in submissions[0].kwargs
    assert submissions[1].kwargs['bundle_label_selector'] == [{'ray.io/node-id': '!in(b)'}] * 2
    assert submissions[2].kwargs['bundle_label_selector'] == [{'ray.io/node-id': '!in(b,a)'}] * 2
    assert [pg.node for pg in pool.pgs] == ['a', 'b', 'c']
    assert pool.world_size == 6 and pool.num_nodes == 3
    assert len(pool.bundle_infos) == 6


@pytest.mark.parametrize('counts', [[1, 3], [3, 1], [2, 2], [1]])
def test_uneven_counts_follow_their_sorted_groups(scheduler, counts):
    original = counts.copy()
    pool = ResourcePool(counts)
    pool.create()
    assert [len(pg.bundles) for pg in scheduler.groups] == sorted(original, reverse=True)
    assert pool.process_on_nodes == [len(pg.bundles) for pg in pool.pgs]
    assert pool.world_size == sum(original)
    assert len(pool.bundle_infos) == sum(original)
    assert counts == original


@pytest.mark.parametrize('stage', ['submit', 'ready', 'metadata', 'discover', 'interrupt'])
def test_failed_creation_releases_groups_and_restores_counts(scheduler, monkeypatch, stage):
    failure = KeyboardInterrupt() if stage == 'interrupt' else RuntimeError('injected pool failure')
    pool = ResourcePool([1, 3])
    original = pool.process_on_nodes
    if stage == 'submit':
        create = scheduler.placement.side_effect

        def fail_second(*args, **kwargs):
            if scheduler.groups:
                raise failure
            return create(*args, **kwargs)

        scheduler.placement.side_effect = fail_second
    elif stage in ('ready', 'interrupt'):
        scheduler.ready.side_effect = [None, failure]
    elif stage == 'metadata':
        table = scheduler.table.side_effect

        def fail_metadata(pg_id):
            if pg_id != 0:
                raise failure
            return table(pg_id)

        scheduler.table.side_effect = fail_metadata
    else:

        def fail_discovery(self):
            self.bundle_infos = [('partial', '0')]
            self.node_ips = ['partial']
            raise failure

        monkeypatch.setattr(ResourcePool, '_discover_bundle_infos', fail_discovery)

    with pytest.raises(type(failure)) as raised:
        pool.create()
    assert raised.value is failure
    assert {args.args[0].id for args in scheduler.remove.call_args_list} == {pg.id for pg in scheduler.groups}
    assert scheduler.remove.call_count == len(scheduler.groups)
    assert pool.pgs == [] and pool.bundle_infos == [] and pool.node_ips == []
    assert pool.process_on_nodes is original


@pytest.mark.parametrize('ray_version', ['2.47.0', '2.53.0'])
def test_unsupported_ray_fails_before_reserving_multi_node_pool(scheduler, monkeypatch, ray_version):
    monkeypatch.setattr(ray, '__version__', ray_version)
    with pytest.raises(RuntimeError, match='Ray >= 2.54.0'):
        ResourcePool([1, 1]).create()
    scheduler.placement.assert_not_called()
    scheduler.remove.assert_not_called()


def test_single_node_pool_does_not_require_label_support(scheduler, monkeypatch):
    monkeypatch.setattr(ray, '__version__', '2.47.0')
    pool = ResourcePool([1])
    pool.create()
    assert len(pool.pgs) == 1
    assert 'bundle_label_selector' not in scheduler.placement.call_args.kwargs


def test_shared_pools_are_created_once_and_separate_pools_can_use_other_nodes(scheduler):
    shared = ResourcePool([4], max_colocate_count=2)
    separate = ResourcePool([4])
    manager = ResourcePoolManager({'train': shared, 'rollout': shared, 'teacher': separate})
    manager.create_all()
    assert len(scheduler.groups) == 2
    assert [pg.node for pg in shared.pgs] == ['b']
    assert [pg.node for pg in separate.pgs] == ['a']
    assert all('bundle_label_selector' not in submission.kwargs for submission in scheduler.placement.call_args_list)
    manager.destroy_all()
    assert scheduler.remove.call_count == 2
    assert not shared.bundle_infos and not separate.bundle_infos


def test_manager_failure_also_releases_previously_created_pools(scheduler):
    first = ResourcePool([1])
    second = ResourcePool([1])
    manager = ResourcePoolManager({'train': first, 'rollout': first, 'teacher': second})
    failure = RuntimeError('second pool failed')
    scheduler.ready.side_effect = [None, failure]
    with pytest.raises(RuntimeError) as raised:
        manager.create_all()
    assert raised.value is failure
    assert {args.args[0].id for args in scheduler.remove.call_args_list} == {0, 1}
    assert scheduler.remove.call_count == 2
    assert not first.pgs and not second.pgs
    assert not first.bundle_infos and not second.bundle_infos


def test_equal_ips_have_stable_node_id_order(scheduler):
    for node in scheduler.nodes:
        node['NodeManagerAddress'] = '127.0.0.1'
    pool = ResourcePool([1, 1])
    pool.create()
    assert [pg.node for pg in pool.pgs] == ['a', 'b']

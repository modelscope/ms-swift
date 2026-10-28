# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest
import ray
from unittest.mock import Mock

from swift.ray.megatron.checkpoint_engine.manager import CheckpointEngineManager
from swift.ray.megatron.rollout.replica import RolloutReplica


@pytest.fixture
def rollout_manager(monkeypatch):
    servers = [Mock() for _ in range(4)]
    replicas = [RolloutReplica.__new__(RolloutReplica) for _ in range(2)]
    for replica, shard in zip(replicas, [servers[:2], servers[2:]]):
        replica._servers = shard
    manager = CheckpointEngineManager([], replicas, train_group=Mock())
    wait = Mock()
    monkeypatch.setattr(ray, 'get', wait)
    return manager, servers, wait


@pytest.mark.parametrize('sleep_level,is_colocated,expected_level', [(1, False, 1), (2, False, 2), (2, True, 1)])
def test_rollout_transitions_submit_before_waiting(rollout_manager, sleep_level, is_colocated, expected_level):
    manager, servers, wait = rollout_manager
    manager = CheckpointEngineManager([],
                                      manager._rollout_replicas,
                                      train_group=Mock(),
                                      sleep_level=sleep_level,
                                      is_colocated=is_colocated)
    operation = 'sleep'
    before = set()

    def complete(refs):
        methods = [getattr(server, operation).remote for server in servers]
        assert all(method.called for method in methods)
        assert refs == [method.return_value for method in methods]
        assert manager._sleeping_tags == before

    wait.side_effect = complete
    manager.wake_up_rollout()
    wait.assert_not_called()
    manager.sleep_rollout()
    for server in servers:
        server.sleep.remote.assert_called_once_with(expected_level)
    assert manager._sleeping_tags == {'weights', 'kv_cache'}
    manager.sleep_rollout()
    assert wait.call_count == 1

    operation = 'wake_up'
    before = {'weights', 'kv_cache'}
    tags = ['weights']
    manager.wake_up_rollout(tags=tags)
    for server in servers:
        server.wake_up.remote.assert_called_once_with(tags=['weights'])
    assert tags == ['weights']
    assert manager._sleeping_tags == {'kv_cache'}

    before = {'kv_cache'}
    manager.wake_up_rollout()
    for server in servers:
        server.wake_up.remote.assert_called_with(tags=None)
    assert manager._sleeping_tags == set()
    manager.wake_up_rollout()
    assert wait.call_count == 3

    operation = 'sleep'
    before = set()
    manager.sleep_rollout()
    assert wait.call_count == 4
    assert manager._sleeping_tags == {'weights', 'kv_cache'}


@pytest.mark.parametrize('operation,tags', [('sleep', None), ('wake_up', None), ('wake_up', ['weights'])])
@pytest.mark.parametrize('failure_at', ['submit', 'wait'])
def test_failed_rollout_transition_preserves_state_and_error(rollout_manager, operation, tags, failure_at):
    manager, servers, wait = rollout_manager
    before = set() if operation == 'sleep' else {'weights', 'kv_cache'}
    manager._sleeping_tags = before.copy()
    failure = RuntimeError('rollout transition failed')
    if failure_at == 'submit':
        getattr(servers[-1], operation).remote.side_effect = failure
    else:
        wait.side_effect = failure

    kwargs = {} if operation == 'sleep' else {'tags': tags}
    with pytest.raises(RuntimeError) as raised:
        getattr(manager, operation + '_rollout')(**kwargs)
    assert raised.value is failure
    assert manager._sleeping_tags == before

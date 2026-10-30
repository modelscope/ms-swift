# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest
from unittest.mock import Mock, call

from swift.ray.megatron.checkpoint_engine.manager import CheckpointEngineManager


@pytest.mark.parametrize('is_colocated', [False, True])
@pytest.mark.parametrize('base_sync_done,merge_and_sync', [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize('export_fails', [False, True])
def test_naive_sync_exports_resident_weights(is_colocated, base_sync_done, merge_and_sync, export_fails):
    group = Mock()
    manager = CheckpointEngineManager([], [], train_group=group, is_colocated=is_colocated)
    manager.base_sync_done = base_sync_done
    failure = RuntimeError('weight export failed')
    if export_fails:
        group.update_weights.side_effect = failure

    if export_fails:
        with pytest.raises(RuntimeError) as raised:
            manager._sync_weights_naive(merge_and_sync)
        assert raised.value is failure
        assert manager.base_sync_done is base_sync_done
    else:
        manager._sync_weights_naive(merge_and_sync)
        assert manager.base_sync_done is True

    adapter_only = base_sync_done and not merge_and_sync
    expected = [call.update_weights(adapter_only=adapter_only)]
    if merge_and_sync:
        expected = [call.merge_lora(), *expected, call.unmerge_lora()]
    assert group.mock_calls == expected


@pytest.mark.parametrize('base_sync_done', [False, True])
def test_naive_sync_unmerge_failure_preserves_state(base_sync_done):
    group = Mock()
    manager = CheckpointEngineManager([], [], train_group=group, is_colocated=True)
    manager.base_sync_done = base_sync_done
    failure = RuntimeError('unmerge failed')
    group.unmerge_lora.side_effect = failure

    with pytest.raises(RuntimeError) as raised:
        manager._sync_weights_naive(merge_and_sync=True)
    assert raised.value is failure
    assert manager.base_sync_done is base_sync_done
    assert group.mock_calls == [call.merge_lora(), call.update_weights(adapter_only=False), call.unmerge_lora()]

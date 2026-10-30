# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest
import ray
import torch
from unittest.mock import AsyncMock, Mock

from swift.ray.megatron.rollout import weight_transfer
from swift.ray.megatron.rollout.adapter import RolloutAdapter


@pytest.fixture
def allocator(monkeypatch):
    import zmq

    import swift.utils

    state = {'config': '', 'expandable': False, 'threshold': 0.0}
    setter = Mock()

    def set_settings(config):
        # PyTorch resets the threshold on each parse, but retains omitted bools.
        assert not config.startswith(',')
        state.update(config=config, threshold=0.0)
        for option in config.split(','):
            if option.startswith('expandable_segments:'):
                state['expandable'] = option.endswith('True')
            elif option.startswith('garbage_collection_threshold:'):
                state['threshold'] = float(option.split(':')[1])

    setter.side_effect = set_settings
    monkeypatch.setattr(torch._C, '_accelerator_setAllocatorSettings', setter, raising=False)
    monkeypatch.setattr(torch.cuda.memory, '_set_allocator_settings', setter)
    snapshot = Mock(side_effect=lambda: {
        'allocator_settings': {
            'PYTORCH_CUDA_ALLOC_CONF': state['config'],
            'expandable_segments': state['expandable']
        }
    })
    monkeypatch.setattr(torch.cuda.memory, '_snapshot', snapshot)
    monkeypatch.setattr(torch.cuda.memory, 'get_allocator_backend', lambda: 'native')
    monkeypatch.setattr(weight_transfer, 'get_torch_device', lambda: torch.cuda, raising=False)
    monkeypatch.setattr(weight_transfer, 'get_current_device', lambda: 'cpu')
    monkeypatch.setattr(swift.utils, 'get_current_device', lambda: 'cpu')
    monkeypatch.setattr(swift.utils, 'gc_collect', Mock())
    monkeypatch.setattr(swift.utils, 'ipc_collect', Mock())
    monkeypatch.setattr(zmq.Context, 'instance', Mock(return_value=Mock()))
    return state, setter, snapshot


@pytest.mark.parametrize('expandable,explicit_flag', [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize('failure_at', [None, 'allocate', 'export'])
def test_sender_allocates_exportable_buffer_and_restores_settings(allocator, monkeypatch, tmp_path, expandable,
                                                                  explicit_flag, failure_at):
    from torch.multiprocessing import reductions

    state, setter, _ = allocator
    config = 'garbage_collection_threshold:0.8'
    if explicit_flag:
        config += f',expandable_segments:{expandable}'
    state.update(config=config, expandable=expandable, threshold=0.8)
    before = state.copy()
    failure = RuntimeError('injected IPC failure')
    empty = torch.empty
    allocations = []

    def allocate(*args, **kwargs):
        allocations.append(state.copy())
        assert state['expandable'] is False
        assert state['threshold'] == 0.8
        if failure_at == 'allocate':
            raise failure
        return empty(*args, **kwargs)

    def export(buffer):
        assert state == before
        if failure_at == 'export':
            raise failure
        return ('handshake', buffer)

    monkeypatch.setattr(torch, 'empty', allocate)
    monkeypatch.setattr(reductions, 'reduce_tensor', export)
    sender = weight_transfer.BucketedWeightSender(f'ipc://{tmp_path / "weights.sock"}', bucket_size_mb=1)
    try:
        if failure_at:
            with pytest.raises(RuntimeError) as raised:
                sender._init_socket_and_buffer()
            assert raised.value is failure
        else:
            sender._init_socket_and_buffer()
            assert sender._pending_handshake[1] is sender.buffer
        assert state == before
        assert len(allocations) == 1
        if not expandable:
            setter.assert_not_called()
    finally:
        sender.cleanup()


@pytest.mark.parametrize('use_shm', [False, True])
def test_sender_bypasses_allocator_for_reused_or_shared_buffer(allocator, monkeypatch, tmp_path, use_shm):
    from multiprocessing import shared_memory
    from torch.multiprocessing import reductions

    _, setter, snapshot = allocator
    buffer = torch.zeros(1 << 20, dtype=torch.uint8)
    monkeypatch.setattr(reductions, 'reduce_tensor', lambda tensor: ('handshake', tensor))
    monkeypatch.setattr(shared_memory, 'SharedMemory', Mock(return_value=Mock(buf=bytearray(1 << 20))))
    sender = weight_transfer.BucketedWeightSender(
        f'ipc://{tmp_path / "weights.sock"}', bucket_size_mb=1, use_shm=use_shm, external_buffer=buffer)
    try:
        sender._init_socket_and_buffer()
        assert sender._pending_handshake is not None
        if not use_shm:
            assert sender.buffer is buffer
        setter.assert_not_called()
        snapshot.assert_not_called()
    finally:
        sender.cleanup()


@pytest.mark.parametrize('sync_fails', [False, True])
@pytest.mark.parametrize('use_shm,is_primary', [(False, True), (True, True), (False, False)])
def test_adapter_restores_allocator_before_export_and_reuses_buffer(allocator, monkeypatch, sync_fails, use_shm,
                                                                    is_primary):
    state, setter, snapshot = allocator
    state.update(config='expandable_segments:True,garbage_collection_threshold:0.8', expandable=True, threshold=0.8)
    before = state.copy()
    failure = RuntimeError('sync failed')
    empty = torch.empty
    allocations = []

    def allocate(*args, **kwargs):
        allocations.append(state.copy())
        assert state['expandable'] is False
        return empty(*args, **kwargs)

    def weights():
        # Bridge export is lazy and must run after the allocation scope exits.
        assert state == before
        if sync_fails:
            raise failure
        yield 'weight', torch.ones(4)

    sender = Mock()
    sender.__aenter__ = AsyncMock(return_value=sender)
    sender.__aexit__ = AsyncMock(return_value=False)
    sender.handshake = AsyncMock()
    sender.send_weights = AsyncMock(side_effect=lambda items: list(items))
    factory = Mock(return_value=sender)
    monkeypatch.setattr(weight_transfer, 'BucketedWeightSender', factory)
    monkeypatch.setattr(torch, 'empty', allocate)
    monkeypatch.setattr(ray, 'get', Mock())
    adapter = RolloutAdapter(bucket_size_mb=1)
    adapter.use_shm = use_shm
    adapter.is_primary = is_primary
    adapter._server_handle = Mock()

    for _ in range(2):
        if sync_fails:
            with pytest.raises(RuntimeError) as raised:
                adapter.update_weights(weights())
            assert raised.value is failure
        else:
            adapter.update_weights(weights())
        assert state == before
    if is_primary and not use_shm:
        assert len(allocations) == 1
        assert factory.call_args.kwargs['external_buffer'] is adapter._ipc_buffer
    else:
        assert not allocations
        setter.assert_not_called()
        snapshot.assert_not_called()


@pytest.mark.parametrize('cuda_backend', [False, True])
def test_other_allocator_backends_are_unchanged(allocator, monkeypatch, tmp_path, cuda_backend):
    from torch.multiprocessing import reductions

    _, setter, snapshot = allocator
    monkeypatch.setattr(weight_transfer, 'get_torch_device', lambda: torch.cuda if cuda_backend else torch.cpu)
    monkeypatch.setattr(torch.cuda.memory, 'get_allocator_backend', lambda: 'cudaMallocAsync')
    monkeypatch.setattr(reductions, 'reduce_tensor', lambda tensor: ('handshake', tensor))
    sender = weight_transfer.BucketedWeightSender(f'ipc://{tmp_path / "weights.sock"}', bucket_size_mb=1)
    try:
        sender._init_socket_and_buffer()
        setter.assert_not_called()
        snapshot.assert_not_called()
    finally:
        sender.cleanup()


@pytest.mark.parametrize('allocation_fails', [False, True])
@pytest.mark.parametrize('config', [
    '', 'garbage_collection_threshold:0.8', 'garbage_collection_threshold:0.8,', 'garbage_collection_threshold:0.8, \n'
])
def test_older_pytorch_setter_restores_settings(allocator, monkeypatch, allocation_fails, config):
    state, _, _ = allocator
    state.update(config=config, expandable=True, threshold=0.8 if config else 0.0)
    before = state.copy()
    monkeypatch.delattr(torch._C, '_accelerator_setAllocatorSettings')
    failure = RuntimeError('allocation failed')

    def allocate():
        with weight_transfer._regular_cuda_ipc():
            assert state['expandable'] is False
            assert state['threshold'] == before['threshold']
            if allocation_fails:
                raise failure

    if allocation_fails:
        with pytest.raises(RuntimeError) as raised:
            allocate()
        assert raised.value is failure
    else:
        allocate()
    assert state == before

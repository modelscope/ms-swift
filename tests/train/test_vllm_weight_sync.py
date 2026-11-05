import torch
from contextlib import contextmanager
from unittest.mock import patch

from swift.rlhf_trainers.vllm_client import _broadcast_tensors_for_vllm_weight_sync


class _FakeTensor:

    def __init__(self, device):
        self.device = device
        self.to_calls = []

    def to(self, *, device, non_blocking):
        self.to_calls.append((device, non_blocking))
        return _FakeTensor(device)


class _FakeCommunicator:

    def __init__(self, device=torch.device('cuda:1')):
        self.device = device
        self.rank = 2
        self.events = None

    def broadcast(self, tensor, src, stream):
        self.events.append(('broadcast', tensor.device, src, stream.device))


class _FakeStream:

    def __init__(self, device):
        self.device = device


class _FakeDeviceModule:

    def __init__(self, events):
        self.events = events
        # ThreadPoolExecutor workers can start with a different current device.
        self.current_device = torch.device('cuda:0')

    @contextmanager
    def device(self, device):
        previous_device = self.current_device
        self.current_device = device
        self.events.append(('enter_device', device))
        try:
            yield
        finally:
            self.events.append(('exit_device', device))
            self.current_device = previous_device

    def current_stream(self):
        self.events.append(('current_stream', self.current_device))
        return _FakeStream(self.current_device)


def _run_broadcast(tensors, communicator):
    events = []
    device_module = _FakeDeviceModule(events)
    communicator.events = events

    def fake_synchronize(device):
        events.append(('synchronize', device))

    with patch(
            'swift.rlhf_trainers.vllm_client.get_torch_device', return_value=device_module), patch(
                'swift.rlhf_trainers.vllm_client.synchronize', side_effect=fake_synchronize), patch(
                    'swift.rlhf_trainers.utils.get_torch_device', return_value=device_module), patch(
                        'swift.rlhf_trainers.utils.is_torch_npu_available', return_value=False):
        _broadcast_tensors_for_vllm_weight_sync(communicator, tensors)

    return events


def test_weight_sync_same_device_has_no_copy():
    communicator = _FakeCommunicator()
    tensor = _FakeTensor(torch.device('cuda:1'))

    events = _run_broadcast([tensor], communicator)

    assert tensor.to_calls == []
    assert ('current_stream', torch.device('cuda:1')) in events
    assert ('broadcast', torch.device('cuda:1'), communicator.rank, torch.device('cuda:1')) in events
    assert events.count(('synchronize', torch.device('cuda:1'))) == 2


def test_weight_sync_aligns_mismatched_tensor_to_communicator_device():
    communicator = _FakeCommunicator()
    tensor = _FakeTensor(torch.device('cuda:0'))

    events = _run_broadcast([tensor], communicator)

    assert tensor.to_calls == [(torch.device('cuda:1'), False)]
    assert ('synchronize', torch.device('cuda:0')) in events
    assert ('current_stream', torch.device('cuda:1')) in events
    assert ('broadcast', torch.device('cuda:1'), communicator.rank, torch.device('cuda:1')) in events
    assert events[-2:] == [('synchronize', torch.device('cuda:1')), ('exit_device', torch.device('cuda:1'))]


def test_weight_sync_copies_cpu_tensor_without_accelerator_synchronize_on_cpu():
    communicator = _FakeCommunicator()
    tensor = _FakeTensor(torch.device('cpu'))

    events = _run_broadcast([tensor], communicator)

    assert tensor.to_calls == [(torch.device('cuda:1'), False)]
    assert ('synchronize', torch.device('cpu')) not in events
    assert ('current_stream', torch.device('cuda:1')) in events
    assert ('broadcast', torch.device('cuda:1'), communicator.rank, torch.device('cuda:1')) in events


def test_weight_sync_synchronizes_each_source_device_once():
    communicator = _FakeCommunicator()
    tensors = [
        _FakeTensor(torch.device('cuda:0')),
        _FakeTensor(torch.device('cuda:0')),
        _FakeTensor(torch.device('cuda:1')),
    ]

    events = _run_broadcast(tensors, communicator)

    first_broadcast = next(i for i, event in enumerate(events) if event[0] == 'broadcast')
    assert events[:2] == [('synchronize', torch.device('cuda:0')), ('synchronize', torch.device('cuda:1'))]
    assert all(event[1] == torch.device('cuda:1') and event[3] == torch.device('cuda:1')
               for event in events[first_broadcast:] if event[0] == 'broadcast')

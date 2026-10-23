# Copyright (c) ModelScope Contributors. All rights reserved.
import importlib.util
import json
import os
import pytest
import sys
import torch
import types
from pathlib import Path
from transformers import TrainerCallback
from types import SimpleNamespace
from unittest.mock import Mock


@pytest.fixture
def plugin(monkeypatch):
    # Match the two callback constructors without requiring Megatron/CANN hardware.
    class SwiftCallback(TrainerCallback):

        def __init__(self, args, trainer):
            self.args, self.trainer = args, trainer

    class MegatronCallback:

        def __init__(self, trainer):
            self.trainer, self.args, self.state = trainer, trainer.args, trainer.state

    modules = {
        'swift': types.ModuleType('swift'),
        'swift.callbacks': types.ModuleType('swift.callbacks'),
        'swift.megatron': types.ModuleType('swift.megatron'),
        'swift.megatron.callbacks': types.ModuleType('swift.megatron.callbacks'),
    }
    modules['swift.callbacks'].TrainerCallback = SwiftCallback
    modules['swift.callbacks'].callbacks_map = {}
    modules['swift.megatron.callbacks'].MegatronCallback = MegatronCallback
    modules['swift.megatron.callbacks'].megatron_callbacks_map = {}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    for name in list(os.environ):
        if name.startswith('SWIFT_PROFILER_'):
            monkeypatch.delenv(name)
    spec = importlib.util.spec_from_file_location('_swift_profiler_test_plugin',
                                                  Path(__file__).resolve().parents[2] / 'examples/custom/profiler.py')
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake_api(plugin, monkeypatch):
    api = SimpleNamespace(
        ProfilerActivity=SimpleNamespace(CPU='cpu', CUDA='cuda', NPU='npu'),
        ProfilerLevel=SimpleNamespace(Level0=0, Level1=1, Level2=2),
        _ExperimentalConfig=Mock(),
        profile=Mock(),
        schedule=Mock(),
        tensorboard_trace_handler=Mock())
    monkeypatch.setattr(plugin, '_resolve_device', lambda requested, args: (requested, api))
    return api


def session(plugin, tmp_path, **kwargs):
    args = SimpleNamespace(output_dir=str(tmp_path), process_index=0)
    config = plugin.ProfilerConfig(**kwargs)
    return plugin.ProfilerSession(args, 'test', config)


@pytest.mark.parametrize('device', ['cuda', 'npu'])
def test_exact_target_and_no_restart(plugin, fake_api, tmp_path, device):
    capture = session(plugin, tmp_path, start_step=1000, device=device)
    for completed in range(999):
        capture.begin_step(completed)
        capture.end_step()
    fake_api.profile.assert_not_called()
    capture.begin_step(999)
    fake_api.profile.return_value.start.assert_called_once()
    options = fake_api.profile.call_args.kwargs
    assert options['activities'] == ['cpu', device]
    assert ('experimental_config' in options) == (device == 'npu')
    if device == 'npu':
        fake_api._ExperimentalConfig.assert_called_once_with(profiler_level=1)
    capture.end_step()
    for completed in range(1000, 1004):
        capture.begin_step(completed)
        capture.end_step()
    fake_api.profile.return_value.step.assert_called_once()
    fake_api.profile.return_value.stop.assert_called_once()
    assert capture.finished
    metadata = json.loads((capture.trace_dir / 'config.json').read_text())
    assert metadata['first_active_step'] == 1000


def test_warmup_and_multi_step_window(plugin, fake_api, tmp_path):
    capture = session(plugin, tmp_path, start_step=1000, warmup_steps=1, active_steps=3)
    capture.begin_step(997)
    fake_api.profile.assert_not_called()
    for completed in range(998, 1002):
        capture.begin_step(completed)
        assert capture.prof is not None
        capture.end_step()
    fake_api.schedule.assert_called_once_with(wait=0, warmup=1, active=3, repeat=1, skip_first=0)
    assert fake_api.profile.return_value.step.call_count == 4
    fake_api.profile.return_value.stop.assert_called_once()


def test_resume_missed_window_and_rank_filter(plugin, fake_api, tmp_path):
    resumed = session(plugin, tmp_path, start_step=1000, warmup_steps=1)
    resumed.begin_step(999)
    assert resumed.finished
    unselected = session(plugin, tmp_path, start_step=1, ranks='1,8')
    unselected.begin_step(0)
    unselected.train_end()
    fake_api.profile.assert_not_called()
    assert not list(tmp_path.iterdir())


def test_early_end_and_exception_cleanup(plugin, fake_api, tmp_path):
    capture = session(plugin, tmp_path, start_step=1, active_steps=3)
    capture.begin_step(0)
    capture.end_step()
    capture.train_end()
    capture.close()
    fake_api.profile.return_value.stop.assert_called_once()
    fake_api.profile.return_value.stop.reset_mock()
    failed = session(plugin, tmp_path, start_step=1)
    failed.begin_step(0)
    fake_api.profile.return_value.step.side_effect = RuntimeError('device error')
    with pytest.raises(RuntimeError, match='device error'):
        failed.end_step()
    fake_api.profile.return_value.stop.assert_called_once()
    assert failed.prof is None


@pytest.mark.parametrize('backend', ['transformers', 'megatron'])
def test_callback_interfaces(plugin, fake_api, monkeypatch, tmp_path, backend):
    monkeypatch.setenv('SWIFT_PROFILER_START_STEP', '2')
    args = SimpleNamespace(output_dir=str(tmp_path), process_index=0)
    state = SimpleNamespace(global_step=0, iteration=0)
    trainer = SimpleNamespace(args=args, state=state)
    if backend == 'transformers':
        callback = sys.modules['swift.callbacks'].callbacks_map['profiler'](args, trainer)
    else:
        callback = sys.modules['swift.megatron.callbacks'].megatron_callbacks_map['profiler'](trainer)

    def event(name):
        if backend == 'transformers':
            getattr(callback, name)(args, state, None)
        else:
            getattr(callback, name)()

    for completed in range(4):
        state.global_step = state.iteration = completed
        event('on_step_begin')
        if backend == 'transformers':
            # Gradient accumulation substeps must not advance the profiler.
            callback.on_substep_end(args, state, None)
            callback.on_substep_end(args, state, None)
        state.global_step = state.iteration = completed + 1
        event('on_step_end')
    event('on_train_end')
    fake_api.profile.return_value.start.assert_called_once()
    fake_api.profile.return_value.step.assert_called_once()
    fake_api.profile.return_value.stop.assert_called_once()


def test_standard_backend_does_not_import_megatron(plugin, monkeypatch):
    monkeypatch.delitem(sys.modules, 'swift.megatron')
    monkeypatch.delitem(sys.modules, 'swift.megatron.callbacks')
    plugin.register_callbacks()
    assert 'swift.megatron' not in sys.modules


def test_cuda_does_not_import_torch_npu(plugin, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    importer = Mock(side_effect=AssertionError('CUDA must not import torch_npu'))
    monkeypatch.setattr(plugin.importlib, 'import_module', importer)
    device, api = plugin._resolve_device('auto', SimpleNamespace(device=torch.device('cuda:0')))
    assert device == 'cuda' and api is torch.profiler
    importer.assert_not_called()


def test_npu_auto_detection_and_global_rank(plugin, monkeypatch):
    npu = SimpleNamespace(is_available=lambda: True)
    monkeypatch.setattr(torch, 'npu', npu, raising=False)
    api = SimpleNamespace()
    monkeypatch.setattr(plugin.importlib, 'import_module', lambda name: SimpleNamespace(profiler=api))
    device, resolved_api = plugin._resolve_device('auto', SimpleNamespace())
    assert device == 'npu' and resolved_api is api
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 9)
    assert plugin._global_rank(SimpleNamespace(process_index=0, rank=0)) == 9


@pytest.mark.parametrize('kwargs', [
    dict(start_step=0),
    dict(active_steps=0),
    dict(warmup_steps=-1),
    dict(start_step=1, warmup_steps=1),
    dict(ranks='0,-1'),
    dict(ranks=''),
    dict(device='gpu'),
    dict(npu_level='Level3')
])
def test_invalid_configuration(plugin, kwargs):
    with pytest.raises(ValueError):
        plugin.ProfilerConfig(**kwargs)


@pytest.mark.parametrize('name,minimum', [('START_STEP', 1), ('ACTIVE_STEPS', 1), ('WARMUP_STEPS', 0)])
@pytest.mark.parametrize('value', ['abc', '1.5', '1.0', '', 'true'])
def test_integer_env_errors_identify_name_value_and_range(plugin, monkeypatch, name, minimum, value):
    monkeypatch.setenv(f'SWIFT_PROFILER_{name}', value)
    with pytest.raises(ValueError) as exc:
        plugin.ProfilerConfig.from_env()
    assert str(exc.value) == (f'SWIFT_PROFILER_{name} must be an integer >= {minimum}; got {value!r}')
    assert exc.value.__suppress_context__


@pytest.mark.parametrize('name,value,minimum', [('START_STEP', '0', 1), ('ACTIVE_STEPS', '-1', 1),
                                                ('WARMUP_STEPS', '-1', 0)])
def test_integer_env_range_errors(plugin, monkeypatch, name, value, minimum):
    monkeypatch.setenv(f'SWIFT_PROFILER_{name}', value)
    with pytest.raises(ValueError) as exc:
        plugin.ProfilerConfig.from_env()
    assert str(exc.value) == (f'SWIFT_PROFILER_{name} must be an integer >= {minimum}; got {int(value)!r}')


def test_warmup_constraint_identifies_both_settings(plugin, monkeypatch):
    monkeypatch.setenv('SWIFT_PROFILER_START_STEP', '10')
    monkeypatch.setenv('SWIFT_PROFILER_WARMUP_STEPS', '10')
    with pytest.raises(ValueError) as exc:
        plugin.ProfilerConfig.from_env()
    assert str(exc.value) == ('SWIFT_PROFILER_WARMUP_STEPS (10) must be smaller than SWIFT_PROFILER_START_STEP (10)')


def test_integer_env_defaults_and_valid_values(plugin, monkeypatch):
    assert plugin.ProfilerConfig.from_env() == plugin.ProfilerConfig()
    monkeypatch.setenv('SWIFT_PROFILER_START_STEP', ' 20 ')
    monkeypatch.setenv('SWIFT_PROFILER_WARMUP_STEPS', '0')
    monkeypatch.setenv('SWIFT_PROFILER_ACTIVE_STEPS', '3')
    cfg = plugin.ProfilerConfig.from_env()
    assert (cfg.start_step, cfg.warmup_steps, cfg.active_steps) == (20, 0, 3)


def test_real_cpu_trace_has_only_requested_active_steps(plugin, tmp_path):
    capture = session(
        plugin,
        tmp_path,
        device='cpu',
        start_step=3,
        warmup_steps=1,
        active_steps=2,
        record_shapes=False,
        profile_memory=False)
    try:
        for completed in range(6):
            capture.begin_step(completed)
            with torch.profiler.record_function(f'train_step_{completed + 1}'):
                torch.ones(8, 8) @ torch.ones(8, 8)
            capture.end_step()
    finally:
        capture.close()
    traces = list(capture.trace_dir.glob('*.pt.trace.json'))
    assert len(traces) == 1
    events = json.loads(traces[0].read_text(encoding='utf-8'))['traceEvents']
    names = {event['name'] for event in events}
    assert {'train_step_3', 'train_step_4', 'aten::mm'} <= names
    assert not {'train_step_1', 'train_step_2', 'train_step_5', 'train_step_6'} & names

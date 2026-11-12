# Copyright (c) ModelScope Contributors. All rights reserved.
import importlib.util
import json
import os
import pytest
import subprocess
import sys
import torch
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


@pytest.fixture
def profiler(monkeypatch):
    from swift.utils import profiler

    for name in list(os.environ):
        if name.startswith('SWIFT_PROFILER_'):
            monkeypatch.delenv(name)
    return profiler


@pytest.fixture
def megatron_callbacks(monkeypatch):
    # Load the real callback package and registry without requiring Megatron/CANN.
    # Only the backend package initializer and its distributed reduction helper are stubbed.
    import swift

    backend = types.ModuleType('swift.megatron')
    utils = types.ModuleType('swift.megatron.utils')
    utils.reduce_max_stat_across_model_parallel_group = Mock()
    monkeypatch.setitem(sys.modules, 'swift.megatron', backend)
    monkeypatch.setitem(sys.modules, 'swift.megatron.utils', utils)
    name = '_swift_test_megatron_callbacks'
    directory = Path(swift.__file__).resolve().parent / 'megatron' / 'callbacks'
    spec = importlib.util.spec_from_file_location(
        name, directory / '__init__.py', submodule_search_locations=[str(directory)])
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    try:
        spec.loader.exec_module(module)
        yield module.megatron_callbacks_map
    finally:
        for loaded in list(sys.modules):
            if loaded.startswith(name + '.'):
                sys.modules.pop(loaded)


@pytest.fixture
def fake_api(profiler, monkeypatch):
    api = SimpleNamespace(
        ProfilerActivity=SimpleNamespace(CPU='cpu', CUDA='cuda', NPU='npu'),
        ProfilerLevel=SimpleNamespace(Level0=0, Level1=1, Level2=2),
        _ExperimentalConfig=Mock(),
        profile=Mock(),
        schedule=Mock(),
        tensorboard_trace_handler=Mock())
    monkeypatch.setattr(profiler, '_resolve_device', lambda requested, args: (requested, api))
    return api


def session(profiler, tmp_path, **kwargs):
    args = SimpleNamespace(output_dir=str(tmp_path), process_index=0)
    config = profiler.ProfilerConfig(**kwargs)
    return profiler.ProfilerSession(args, 'test', config)


@pytest.mark.parametrize('device', ['cuda', 'npu'])
def test_exact_target_and_no_restart(profiler, fake_api, tmp_path, device):
    capture = session(profiler, tmp_path, start_step=1000, device=device)
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
        fake_api.tensorboard_trace_handler.assert_called_once_with(
            str(capture.trace_dir), async_mode=False, analyse_flag=True)
    else:
        fake_api.tensorboard_trace_handler.assert_called_once_with(str(capture.trace_dir))
    capture.end_step()
    for completed in range(1000, 1004):
        capture.begin_step(completed)
        capture.end_step()
    fake_api.profile.return_value.step.assert_called_once()
    fake_api.profile.return_value.stop.assert_called_once()
    assert capture.finished
    metadata = json.loads((capture.trace_dir / 'config.json').read_text())
    assert metadata['first_active_step'] == 1000


def test_warmup_and_multi_step_window(profiler, fake_api, tmp_path):
    capture = session(profiler, tmp_path, start_step=1000, warmup_steps=1, active_steps=3)
    capture.begin_step(997)
    fake_api.profile.assert_not_called()
    for completed in range(998, 1002):
        capture.begin_step(completed)
        assert capture.prof is not None
        capture.end_step()
    fake_api.schedule.assert_called_once_with(wait=0, warmup=1, active=3, repeat=1, skip_first=0)
    assert fake_api.profile.return_value.step.call_count == 4
    fake_api.profile.return_value.stop.assert_called_once()


def test_resume_missed_window_and_rank_filter(profiler, fake_api, tmp_path):
    resumed = session(profiler, tmp_path, start_step=1000, warmup_steps=1)
    resumed.begin_step(999)
    assert resumed.finished
    unselected = session(profiler, tmp_path, start_step=1, ranks='1,8')
    unselected.begin_step(0)
    unselected.train_end()
    fake_api.profile.assert_not_called()
    assert not list(tmp_path.iterdir())


def test_early_end_and_exception_cleanup(profiler, fake_api, tmp_path):
    capture = session(profiler, tmp_path, start_step=1, active_steps=3)
    capture.begin_step(0)
    capture.end_step()
    capture.train_end()
    capture.close()
    fake_api.profile.return_value.stop.assert_called_once()
    fake_api.profile.return_value.stop.reset_mock()
    failed = session(profiler, tmp_path, start_step=1)
    failed.begin_step(0)
    fake_api.profile.return_value.step.side_effect = RuntimeError('device error')
    with pytest.raises(RuntimeError, match='device error'):
        failed.end_step()
    fake_api.profile.return_value.stop.assert_called_once()
    assert failed.prof is None


@pytest.mark.parametrize('backend', ['transformers', 'megatron'])
def test_callback_interfaces(profiler, fake_api, monkeypatch, tmp_path, backend, request):
    monkeypatch.setenv('SWIFT_PROFILER_START_STEP', '2')
    args = SimpleNamespace(output_dir=str(tmp_path), process_index=0)
    state = SimpleNamespace(global_step=0, iteration=0)
    trainer = SimpleNamespace(args=args, state=state)
    if backend == 'transformers':
        from swift.callbacks import callbacks_map
        callback = callbacks_map['profiler'](args, trainer)
    else:
        callbacks_map = request.getfixturevalue('megatron_callbacks')
        callback = callbacks_map['profiler'](trainer)

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


def test_builtin_registration_is_inactive_and_does_not_import_optional_backends(tmp_path):
    # A fresh process catches import-time registration/configuration regressions.
    script = """
import sys
import torch
from unittest.mock import patch
with patch.object(torch.profiler, 'profile', side_effect=AssertionError('profiler started during import')):
    from swift.callbacks import callbacks_map
    from swift.callbacks.profiler import SwiftProfilerCallback
    assert callbacks_map['profiler'] is SwiftProfilerCallback
assert not any(name in sys.modules for name in ('swift.megatron', 'megatron', 'mindspeed', 'torch_npu'))
"""
    env = dict(os.environ, SWIFT_PROFILER_START_STEP='invalid-unless-enabled')
    result = subprocess.run([sys.executable, '-c', script],
                            cwd=tmp_path,
                            env=env,
                            capture_output=True,
                            text=True,
                            timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_cuda_does_not_import_torch_npu(profiler, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    importer = Mock(side_effect=AssertionError('CUDA must not import torch_npu'))
    monkeypatch.setattr(profiler.importlib, 'import_module', importer)
    device, api = profiler._resolve_device('auto', SimpleNamespace(device=torch.device('cuda:0')))
    assert device == 'cuda' and api is torch.profiler
    importer.assert_not_called()


@pytest.mark.parametrize('async_mode', [False, True])
@pytest.mark.parametrize('analyse_flag', [False, True])
def test_npu_trace_analysis_modes(profiler, fake_api, monkeypatch, tmp_path, async_mode, analyse_flag):
    monkeypatch.setenv('SWIFT_PROFILER_DEVICE', 'npu')
    monkeypatch.setenv('SWIFT_PROFILER_START_STEP', '1')
    monkeypatch.setenv('SWIFT_PROFILER_NPU_ASYNC_MODE', str(async_mode))
    monkeypatch.setenv('SWIFT_PROFILER_NPU_ANALYSE_FLAG', str(analyse_flag))
    args = SimpleNamespace(output_dir=str(tmp_path), process_index=0)
    capture = profiler.ProfilerSession(args, 'test')
    try:
        capture.begin_step(0)
        fake_api.tensorboard_trace_handler.assert_called_once_with(
            str(capture.trace_dir), async_mode=async_mode, analyse_flag=analyse_flag)
        options = fake_api.profile.call_args.kwargs
        assert options['on_trace_ready'] is fake_api.tensorboard_trace_handler.return_value
        assert 'async_mode' not in options and 'analyse_flag' not in options
        metadata = json.loads((capture.trace_dir / 'config.json').read_text())
        assert metadata['config']['npu_async_mode'] is async_mode
        assert metadata['config']['npu_analyse_flag'] is analyse_flag
        capture.end_step()
        fake_api.profile.return_value.stop.assert_called_once()
    finally:
        capture.close()


@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('async_mode', [False, True])
@pytest.mark.parametrize('analyse_flag', [False, True])
def test_npu_analysis_settings_do_not_affect_other_devices(profiler, monkeypatch, tmp_path, device, async_mode,
                                                           analyse_flag):
    monkeypatch.setenv('SWIFT_PROFILER_START_STEP', '1')
    monkeypatch.setenv('SWIFT_PROFILER_NPU_ASYNC_MODE', str(async_mode))
    monkeypatch.setenv('SWIFT_PROFILER_NPU_ANALYSE_FLAG', str(analyse_flag))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    importer = Mock(side_effect=AssertionError('CUDA/CPU must not import torch_npu'))
    monkeypatch.setattr(profiler.importlib, 'import_module', importer)
    # Use the real handler signature so NPU-only kwargs would raise on CUDA/CPU.
    trace_handler = Mock(wraps=torch.profiler.tensorboard_trace_handler)
    profile = Mock()
    schedule = Mock()
    monkeypatch.setattr(torch.profiler, 'tensorboard_trace_handler', trace_handler)
    monkeypatch.setattr(torch.profiler, 'profile', profile)
    monkeypatch.setattr(torch.profiler, 'schedule', schedule)
    args = SimpleNamespace(output_dir=str(tmp_path), process_index=0, device=torch.device(device))
    capture = profiler.ProfilerSession(args, 'test')
    try:
        capture.begin_step(0)
        trace_handler.assert_called_once_with(str(capture.trace_dir))
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device == 'cuda':
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        profile.assert_called_once_with(
            activities=activities,
            schedule=schedule.return_value,
            on_trace_ready=profile.call_args.kwargs['on_trace_ready'],
            record_shapes=True,
            profile_memory=True,
            with_stack=False)
        assert callable(profile.call_args.kwargs['on_trace_ready'])
        schedule.assert_called_once_with(wait=0, warmup=0, active=1, repeat=1, skip_first=0)
        capture.end_step()
        profile.return_value.stop.assert_called_once()
        importer.assert_not_called()
    finally:
        capture.close()


@pytest.mark.parametrize('backend', ['transformers', 'megatron'])
def test_callbacks_forward_npu_analysis_settings(profiler, fake_api, monkeypatch, tmp_path, backend, request):
    monkeypatch.setenv('SWIFT_PROFILER_DEVICE', 'npu')
    monkeypatch.setenv('SWIFT_PROFILER_START_STEP', '1')
    monkeypatch.setenv('SWIFT_PROFILER_NPU_ASYNC_MODE', '1')
    monkeypatch.setenv('SWIFT_PROFILER_NPU_ANALYSE_FLAG', '0')
    args = SimpleNamespace(output_dir=str(tmp_path), process_index=0)
    state = SimpleNamespace(global_step=0, iteration=0)
    trainer = SimpleNamespace(args=args, state=state)
    if backend == 'transformers':
        from swift.callbacks import callbacks_map
        callback = callbacks_map['profiler'](args, trainer)
        callback.on_step_begin(args, state, None)
        callback.on_step_end(args, state, None)
        callback.on_train_end(args, state, None)
    else:
        callback = request.getfixturevalue('megatron_callbacks')['profiler'](trainer)
        callback.on_step_begin()
        callback.on_step_end()
        callback.on_train_end()
    fake_api.tensorboard_trace_handler.assert_called_once_with(
        str(callback.session.trace_dir), async_mode=True, analyse_flag=False)
    fake_api.profile.return_value.stop.assert_called_once()


@pytest.mark.parametrize('name', ['NPU_ASYNC_MODE', 'NPU_ANALYSE_FLAG'])
@pytest.mark.parametrize('value,expected', [('true', True), ('FALSE', False), ('1', True), ('0', False)])
def test_npu_boolean_env_values(profiler, monkeypatch, name, value, expected):
    monkeypatch.setenv(f'SWIFT_PROFILER_{name}', value)
    assert getattr(profiler.ProfilerConfig.from_env(), name.lower()) is expected


@pytest.mark.parametrize('name', ['NPU_ASYNC_MODE', 'NPU_ANALYSE_FLAG'])
def test_invalid_npu_boolean_env_values(profiler, monkeypatch, name):
    monkeypatch.setenv(f'SWIFT_PROFILER_{name}', 'yes')
    with pytest.raises(ValueError, match=f'SWIFT_PROFILER_{name} must be true/false or 1/0'):
        profiler.ProfilerConfig.from_env()


def test_npu_auto_detection_and_global_rank(profiler, monkeypatch):
    npu = SimpleNamespace(is_available=lambda: True)
    monkeypatch.setattr(torch, 'npu', npu, raising=False)
    api = SimpleNamespace()
    monkeypatch.setattr(profiler.importlib, 'import_module', lambda name: SimpleNamespace(profiler=api))
    device, resolved_api = profiler._resolve_device('auto', SimpleNamespace())
    assert device == 'npu' and resolved_api is api
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 9)
    assert profiler._global_rank(SimpleNamespace(process_index=0, rank=0)) == 9


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
def test_invalid_configuration(profiler, kwargs):
    with pytest.raises(ValueError):
        profiler.ProfilerConfig(**kwargs)


@pytest.mark.parametrize('name,minimum', [('START_STEP', 1), ('ACTIVE_STEPS', 1), ('WARMUP_STEPS', 0)])
@pytest.mark.parametrize('value', ['abc', '1.5', '1.0', '', 'true'])
def test_integer_env_errors_identify_name_value_and_range(profiler, monkeypatch, name, minimum, value):
    monkeypatch.setenv(f'SWIFT_PROFILER_{name}', value)
    with pytest.raises(ValueError) as exc:
        profiler.ProfilerConfig.from_env()
    assert str(exc.value) == (f'SWIFT_PROFILER_{name} must be an integer >= {minimum}; got {value!r}')
    assert exc.value.__suppress_context__


@pytest.mark.parametrize('name,value,minimum', [('START_STEP', '0', 1), ('ACTIVE_STEPS', '-1', 1),
                                                ('WARMUP_STEPS', '-1', 0)])
def test_integer_env_range_errors(profiler, monkeypatch, name, value, minimum):
    monkeypatch.setenv(f'SWIFT_PROFILER_{name}', value)
    with pytest.raises(ValueError) as exc:
        profiler.ProfilerConfig.from_env()
    assert str(exc.value) == (f'SWIFT_PROFILER_{name} must be an integer >= {minimum}; got {int(value)!r}')


def test_warmup_constraint_identifies_both_settings(profiler, monkeypatch):
    monkeypatch.setenv('SWIFT_PROFILER_START_STEP', '10')
    monkeypatch.setenv('SWIFT_PROFILER_WARMUP_STEPS', '10')
    with pytest.raises(ValueError) as exc:
        profiler.ProfilerConfig.from_env()
    assert str(exc.value) == ('SWIFT_PROFILER_WARMUP_STEPS (10) must be smaller than SWIFT_PROFILER_START_STEP (10)')


def test_integer_env_defaults_and_valid_values(profiler, monkeypatch):
    assert profiler.ProfilerConfig.from_env() == profiler.ProfilerConfig()
    monkeypatch.setenv('SWIFT_PROFILER_START_STEP', ' 20 ')
    monkeypatch.setenv('SWIFT_PROFILER_WARMUP_STEPS', '0')
    monkeypatch.setenv('SWIFT_PROFILER_ACTIVE_STEPS', '3')
    cfg = profiler.ProfilerConfig.from_env()
    assert (cfg.start_step, cfg.warmup_steps, cfg.active_steps) == (20, 0, 3)


def test_real_cpu_trace_has_only_requested_active_steps(profiler, tmp_path):
    capture = session(
        profiler,
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


def test_transformers_training_uses_builtin_profiler(profiler, monkeypatch, tmp_path):
    from transformers import Trainer

    from swift.trainers import TrainingArguments
    from swift.trainers.mixin import SwiftMixin

    class TinyModel(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 2)

        def forward(self, input_ids, labels=None):
            logits = self.linear(input_ids.float())
            loss = torch.nn.functional.cross_entropy(logits, labels) if labels is not None else None
            return {'loss': loss, 'logits': logits}

    for name, value in {
            'DEVICE': 'cpu',
            'START_STEP': '2',
            'ACTIVE_STEPS': '1',
            'RECORD_SHAPES': 'false',
            'PROFILE_MEMORY': 'false'
    }.items():
        monkeypatch.setenv(f'SWIFT_PROFILER_{name}', value)
    args = TrainingArguments(
        output_dir=str(tmp_path),
        max_steps=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        gradient_checkpointing=False,
        use_cpu=True,
        report_to=[],
        save_strategy='no',
        disable_tqdm=True)
    args.callbacks = ['profiler']
    dataset = [{'input_ids': torch.ones(4), 'labels': torch.tensor(0)} for _ in range(6)]
    trainer = Trainer(model=TinyModel(), args=args, train_dataset=dataset)
    callbacks = SwiftMixin._get_callbacks(trainer, args)
    for callback in callbacks:
        trainer.add_callback(callback)
    trainer.train()
    capture = callbacks[0].session
    assert trainer.state.global_step == 3
    assert capture.finished and capture.steps == 1
    assert len(list(capture.trace_dir.glob('*.pt.trace.json'))) == 1
    metadata = json.loads((capture.trace_dir / 'config.json').read_text())
    assert metadata['backend'] == 'transformers'
    assert metadata['first_active_step'] == metadata['last_active_step'] == 2

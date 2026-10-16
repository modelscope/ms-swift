# Copyright (c) ModelScope Contributors. All rights reserved.
"""ms-swift external profiler plugin for Transformers and Megatron.

Load with --external_plugins examples/custom/profiler.py --callbacks profiler.
See docs/source_en/Instruction/Profiling.md for configuration and examples.
"""

import atexit
import importlib
import json
import os
import socket
import sys
import time
import torch
from dataclasses import asdict, dataclass
from pathlib import Path


def _env(name, default):
    return os.environ.get(f'SWIFT_PROFILER_{name}', default)


def _int_env(name, default, minimum):
    value = _env(name, str(default))
    try:
        return int(value)
    except ValueError:
        raise ValueError(f'SWIFT_PROFILER_{name} must be an integer >= {minimum}; got {value!r}') from None


def _bool_env(name, default):
    value = str(_env(name, str(default))).lower()
    if value not in {'true', 'false', '1', '0'}:
        raise ValueError(f'SWIFT_PROFILER_{name} must be true/false or 1/0')
    return value in {'true', '1'}


@dataclass(frozen=True)
class ProfilerConfig:
    # One-based optimizer step: step 1000 begins with 999 completed updates.
    start_step: int = 1000
    active_steps: int = 1
    warmup_steps: int = 0
    device: str = 'auto'
    output_dir: str = ''
    ranks: str = 'all'
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = False
    npu_level: str = 'Level1'

    def __post_init__(self):
        for name, value, minimum in (
            ('START_STEP', self.start_step, 1),
            ('ACTIVE_STEPS', self.active_steps, 1),
            ('WARMUP_STEPS', self.warmup_steps, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f'SWIFT_PROFILER_{name} must be an integer >= {minimum}; got {value!r}')
        if self.warmup_steps >= self.start_step:
            raise ValueError(f'SWIFT_PROFILER_WARMUP_STEPS ({self.warmup_steps}) must be smaller than '
                             f'SWIFT_PROFILER_START_STEP ({self.start_step})')
        if self.device not in {'auto', 'cuda', 'npu', 'cpu'}:
            raise ValueError('device must be auto, cuda, npu or cpu')
        if self.npu_level not in {'Level0', 'Level1', 'Level2'}:
            raise ValueError('npu_level must be Level0, Level1 or Level2')
        if self.ranks != 'all':
            try:
                ranks = [int(rank.strip()) for rank in self.ranks.split(',')]
            except ValueError as exc:
                raise ValueError('ranks must be all or comma-separated nonnegative global ranks') from exc
            if not ranks or min(ranks) < 0:
                raise ValueError('ranks must be all or comma-separated nonnegative global ranks')

    @classmethod
    def from_env(cls):
        return cls(
            start_step=_int_env('START_STEP', 1000, 1),
            active_steps=_int_env('ACTIVE_STEPS', 1, 1),
            warmup_steps=_int_env('WARMUP_STEPS', 0, 0),
            device=_env('DEVICE', 'auto').lower(),
            output_dir=_env('OUTPUT_DIR', ''),
            ranks=_env('RANKS', 'all').lower().strip(),
            record_shapes=_bool_env('RECORD_SHAPES', True),
            profile_memory=_bool_env('PROFILE_MEMORY', True),
            with_stack=_bool_env('WITH_STACK', False),
            npu_level=_env('NPU_LEVEL', 'Level1'),
        )


def _global_rank(args):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    for name in ('process_index', 'rank'):
        value = getattr(args, name, None)
        if value is not None and int(value) >= 0:
            return int(value)
    return max(0, int(os.environ.get('RANK', '0')))


def _device_type(device):
    return str(getattr(device, 'type', device)).split(':')[0]


def _resolve_device(requested, args):
    device = requested
    if device == 'auto':
        # TrainingArguments.device is authoritative, including CPU training on a GPU host.
        hint = _device_type(getattr(args, 'device', None))
        if hint in {'cuda', 'npu', 'cpu'}:
            device = hint
        elif hasattr(torch, 'npu') and torch.npu.is_available():
            device = 'npu'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            try:
                importlib.import_module('torch_npu')
            except ModuleNotFoundError as exc:
                if exc.name != 'torch_npu':
                    raise
            if hasattr(torch, 'npu') and torch.npu.is_available():
                device = 'npu'
            else:
                raise RuntimeError('No CUDA/NPU device found; use SWIFT_PROFILER_DEVICE=cpu for CPU tracing')
    if device == 'npu':
        torch_npu = importlib.import_module('torch_npu')
        if not torch.npu.is_available():
            raise RuntimeError('NPU profiler requested but torch.npu.is_available() is False')
        return device, torch_npu.profiler
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA profiler requested but torch.cuda.is_available() is False')
    return device, torch.profiler


class ProfilerSession:
    """One collection window per process, independent of trainer callback signatures."""

    def __init__(self, args, backend, config=None):
        self.args = args
        self.backend = backend
        self.config = config if config is not None else ProfilerConfig.from_env()
        self.rank = _global_rank(args)
        self.prof = None
        self.finished = False
        self.steps = 0
        self.trace_dir = None
        self.selected = (
            self.config.ranks == 'all' or self.rank in {int(rank)
                                                        for rank in self.config.ranks.split(',')})
        self._log(f'created: backend={backend}, start_step={self.config.start_step}, '
                  f'warmup={self.config.warmup_steps}, active={self.config.active_steps}, selected={self.selected}')

    def _log(self, message):
        print(f'[SwiftProfiler] rank={self.rank} {message}', flush=True)

    def begin_step(self, completed_steps):
        if not self.selected or self.finished or self.prof is not None:
            return
        trigger = self.config.start_step - self.config.warmup_steps - 1
        if completed_steps < trigger:
            return
        if completed_steps > trigger:
            # Do not silently move the requested window when resuming a checkpoint.
            self.finished = True
            self._log(f'skipped: requested window begins after {trigger} completed steps; '
                      f'already completed={completed_steps}. Set a future START_STEP.')
            return
        self._start()

    def _start(self):
        cfg = self.config
        device, api = _resolve_device(cfg.device, self.args)
        root = cfg.output_dir or str(Path(self.args.output_dir) / 'profiling')
        run_name = f'{socket.gethostname()}_{os.getpid()}_{time.time_ns()}'
        self.trace_dir = Path(root) / f'profile_step_{cfg.start_step}' / f'rank_{self.rank}' / run_name
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        metadata = dict(
            config=asdict(cfg),
            backend=self.backend,
            device=device,
            rank=self.rank,
            torch_version=str(torch.__version__),
            first_active_step=cfg.start_step,
            last_active_step=cfg.start_step + cfg.active_steps - 1)
        (self.trace_dir / 'config.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        activities = [api.ProfilerActivity.CPU]
        if device != 'cpu':
            activities.append(getattr(api.ProfilerActivity, 'NPU' if device == 'npu' else 'CUDA'))
        extra = {}
        if device == 'npu':
            extra['experimental_config'] = api._ExperimentalConfig(
                profiler_level=getattr(api.ProfilerLevel, cfg.npu_level))
        self.prof = api.profile(
            activities=activities,
            schedule=api.schedule(wait=0, warmup=cfg.warmup_steps, active=cfg.active_steps, repeat=1, skip_first=0),
            on_trace_ready=api.tensorboard_trace_handler(str(self.trace_dir)),
            record_shapes=cfg.record_shapes,
            profile_memory=cfg.profile_memory,
            with_stack=cfg.with_stack,
            **extra,
        )
        # Fallback for Python exception exits; normal cleanup happens in callbacks.
        atexit.register(self._exit_cleanup)
        try:
            self.prof.start()
        except BaseException:
            self.close()
            raise
        self._log(f'started: device={device}, output={self.trace_dir}')

    def end_step(self):
        if self.prof is None:
            return
        try:
            self.prof.step()
            self.steps += 1
        except BaseException:
            self.close()
            raise
        if self.steps >= self.config.warmup_steps + self.config.active_steps:
            self.close()

    def close(self):
        if self.prof is None:
            return
        prof, self.prof = self.prof, None
        self.finished = True
        atexit.unregister(self._exit_cleanup)
        try:
            prof.stop()
        finally:
            self._log(f'stopped: completed_window_steps={self.steps}, output={self.trace_dir}')

    def train_end(self):
        if self.selected and not self.finished and self.prof is None:
            self._log('training ended before requested collection window')
        self.close()

    def _exit_cleanup(self):
        try:
            self.close()
        except Exception as exc:
            self._log(f'cleanup failed: {exc}')


def register_callbacks():
    from swift.callbacks import TrainerCallback, callbacks_map

    class SwiftProfilerCallback(TrainerCallback):

        def __init__(self, args, trainer):
            super().__init__(args, trainer)
            self.session = ProfilerSession(args, 'transformers')

        def on_step_begin(self, args, state, control, **kwargs):
            self.session.begin_step(state.global_step)

        def on_step_end(self, args, state, control, **kwargs):
            self.session.end_step()

        def on_train_end(self, args, state, control, **kwargs):
            self.session.train_end()

    callbacks_map['profiler'] = SwiftProfilerCallback

    # Importing swift.megatron initializes Megatron/MindSpeed. Only access it when
    # the Megatron entry point has already loaded it, so normal swift needs neither.
    if 'swift.megatron' in sys.modules:
        from swift.megatron.callbacks import MegatronCallback, megatron_callbacks_map

        class MegatronProfilerCallback(MegatronCallback):

            def __init__(self, trainer):
                super().__init__(trainer)
                self.session = ProfilerSession(self.args, 'megatron')

            def on_step_begin(self):
                self.session.begin_step(self.trainer.state.iteration)

            def on_step_end(self):
                self.session.end_step()

            def on_train_end(self):
                self.session.train_end()

        megatron_callbacks_map['profiler'] = MegatronProfilerCallback


register_callbacks()

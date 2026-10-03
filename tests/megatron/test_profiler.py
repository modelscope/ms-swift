import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

_IMPORT_ERROR = ''
try:
    import torch
    from swift.megatron.callbacks.profiler import ProfilerCallback
except ImportError as e:
    torch = None
    ProfilerCallback = None
    _IMPORT_ERROR = str(e)


class _FakeProfiler:

    def __init__(self):
        self.steps = 0
        self.started = False
        self.stopped = False
        self.execution_trace_observer = None

    def start(self):
        self.started = True

    def step(self):
        self.steps += 1

    def stop(self):
        self.stopped = True


@unittest.skipUnless(ProfilerCallback is not None, f'Megatron dependencies not available: {_IMPORT_ERROR}')
class TestProfilerCallback(unittest.TestCase):

    def _trainer(self, output_dir, **kwargs):
        defaults = dict(
            profile=True,
            profile_step_start=1,
            profile_step_end=2,
            use_pytorch_profiler=True,
            profile_ranks=[],
            profile_output_dir=output_dir,
            pytorch_profiler_collect_shapes=False,
            pytorch_profiler_collect_callstack=False,
            pytorch_profiler_collect_chakra=False,
            record_shapes=False,
            nvtx_ranges=False,
        )
        defaults.update(kwargs)
        args = types.SimpleNamespace(**defaults)
        return types.SimpleNamespace(args=args, state=types.SimpleNamespace(iteration=1))

    def test_pytorch_profiler_follows_training_events(self):
        with tempfile.TemporaryDirectory() as output_dir:
            trainer = self._trainer(output_dir)
            fake_profiler = _FakeProfiler()
            with patch('torch.profiler.profile', return_value=fake_profiler), patch(
                    'torch.cuda.is_available', return_value=False):
                callback = ProfilerCallback(trainer)
                callback.on_train_begin()
                callback.on_step_begin()
                self.assertTrue(fake_profiler.started)
                self.assertEqual(fake_profiler.steps, 1)
                trainer.state.iteration = 2
                callback.on_step_end()
                self.assertTrue(fake_profiler.stopped)
                callback.on_train_end()

    def test_profile_rank_filter_skips_non_selected_rank(self):
        with tempfile.TemporaryDirectory() as output_dir:
            trainer = self._trainer(output_dir, profile_ranks=[1])
            with patch('torch.profiler.profile') as profiler_factory, patch.object(
                    torch.distributed, 'is_initialized', return_value=False):
                callback = ProfilerCallback(trainer)
                callback.on_train_begin()
                callback.on_step_begin()
                self.assertIsNone(callback.prof)
                profiler_factory.assert_not_called()

    def test_chakra_directory_is_created_before_registration(self):
        with tempfile.TemporaryDirectory() as output_dir:
            trainer = self._trainer(output_dir, pytorch_profiler_collect_chakra=True)
            fake_profiler = _FakeProfiler()
            observer = types.SimpleNamespace(register_callback=lambda path: self.assertTrue(Path(path).parent.exists()))
            with patch('torch.profiler.profile', return_value=fake_profiler), patch(
                    'torch.profiler.ExecutionTraceObserver', return_value=observer), patch(
                    'torch.cuda.is_available', return_value=False):
                callback = ProfilerCallback(trainer)
                callback.on_train_begin()


if __name__ == '__main__':
    unittest.main()

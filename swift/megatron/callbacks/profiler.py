# Copyright (c) ModelScope Contributors. All rights reserved.
import os
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Optional

import torch

from swift.utils import get_logger
from .base import MegatronCallback

logger = get_logger()


class ProfilerCallback(MegatronCallback):
    """Profile Megatron training steps with PyTorch profiler or Nsys.

    The step schedule follows Megatron-LM: PyTorch profiler steps are advanced at
    the beginning of each training iteration, while Nsys is started at
    ``profile_step_start`` and stopped at ``profile_step_end``.
    """

    def __init__(self, trainer):
        super().__init__(trainer)
        self.prof: Optional[torch.profiler.profile] = None
        self.nsys_nvtx_context: Optional[AbstractContextManager] = None
        self._profile_rank = self._is_profile_rank()
        self._stopped = False

    def _is_profile_rank(self) -> bool:
        profile_ranks = self.args.profile_ranks
        if not profile_ranks:
            return True
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return 0 in profile_ranks
        return torch.distributed.get_rank() in profile_ranks

    def _rank(self) -> int:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        return 0

    def _profile_dir(self) -> Path:
        output_dir = self.args.profile_output_dir or os.path.join(self.args.output_dir, 'torch_profile')
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _trace_handler(self, prof):
        trace_path = self._profile_dir() / f'rank-{self._rank()}.json.gz'
        prof.export_chrome_trace(str(trace_path))
        logger.info(f'PyTorch profiler trace written to {trace_path}')

    def on_train_begin(self):
        if not self.args.profile or not self._profile_rank:
            return
        if not self.args.use_pytorch_profiler:
            return

        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self.prof = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=max(self.args.profile_step_start - 1, 0),
                warmup=1 if self.args.profile_step_start > 0 else 0,
                active=self.args.profile_step_end - self.args.profile_step_start,
                repeat=1,
            ),
            on_trace_ready=self._trace_handler,
            record_shapes=self.args.pytorch_profiler_collect_shapes,
            with_stack=self.args.pytorch_profiler_collect_callstack,
            execution_trace_observer=self._create_execution_trace_observer(),
        )
        self.prof.start()

    def _create_execution_trace_observer(self):
        if not self.args.pytorch_profiler_collect_chakra:
            return None
        observer = torch.profiler.ExecutionTraceObserver()
        chakra_dir = self._profile_dir().parent / 'chakra'
        chakra_dir.mkdir(parents=True, exist_ok=True)
        observer.register_callback(str(chakra_dir / f'rank-{self._rank()}.json.gz'))
        return observer

    def _start_nsys(self):
        if not torch.cuda.is_available():
            raise RuntimeError('Nsys profiling requires a CUDA runtime.')
        if self.args.nvtx_ranges:
            self._configure_nvtx(True)
        torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStart())
        self.nsys_nvtx_context = torch.autograd.profiler.emit_nvtx(record_shapes=self.args.record_shapes)
        self.nsys_nvtx_context.__enter__()

    @staticmethod
    def _configure_nvtx(enabled: bool):
        try:
            from megatron.core.utils import configure_nvtx_profiling
        except ImportError:
            logger.warning('Megatron Core does not provide configure_nvtx_profiling; skipping NVTX ranges.')
            return
        configure_nvtx_profiling(enabled)

    def _stop(self):
        if self._stopped:
            return
        self._stopped = True
        if self.prof is not None:
            self.prof.stop()
            observer = getattr(self.prof, 'execution_trace_observer', None)
            if observer is not None:
                observer.unregister_callback()
            self.prof = None
        if self.nsys_nvtx_context is not None:
            if self.args.nvtx_ranges:
                self._configure_nvtx(False)
            self.nsys_nvtx_context.__exit__(None, None, None)
            self.nsys_nvtx_context = None
            torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStop())

    def on_step_begin(self):
        if not self.args.profile or not self._profile_rank:
            return
        if self.args.use_pytorch_profiler:
            if self.prof is not None:
                self.prof.step()
        elif self.state.iteration == self.args.profile_step_start:
            self._start_nsys()

    def on_step_end(self):
        if not self.args.profile or not self._profile_rank:
            return
        if self.state.iteration == self.args.profile_step_end:
            self._stop()

    def on_train_end(self):
        if self.args.profile and self._profile_rank:
            self._stop()

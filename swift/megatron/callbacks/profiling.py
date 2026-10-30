# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import torch
from megatron.core.utils import configure_nvtx_profiling
from pathlib import Path

from swift.utils import get_logger
from .base import MegatronCallback

logger = get_logger()


def start_memory_history_recording(args) -> None:
    """Enable the CUDA caching allocator trace so memory snapshots contain history.

    Mirrors Megatron-LM's ``start_memory_history_recording``
    (megatron/training/utils/utils.py). ``torch.cuda.memory._dump_snapshot()`` only
    includes allocation/free events and Python stack context after
    ``_record_memory_history()`` has been enabled. Must be invoked before model
    construction so every tensor allocation is captured. Guarded by
    ``profile_ranks`` so only ranks that will dump a snapshot pay the overhead.
    """
    if not args.record_memory_history:
        return
    if len(args.profile_ranks) != 0 and torch.distributed.get_rank() not in args.profile_ranks:
        return

    torch.cuda.memory._record_memory_history(
        True,
        # Retain up to 100k alloc/free events.
        trace_alloc_max_entries=100_000,
        # Record the Python stack at each event - lets memory_viz show call sites.
        trace_alloc_record_context=True,
    )

    def _oom_observer(device, alloc, device_alloc, device_free):
        # Dump a snapshot on OOM so we can inspect what was live at the failure.
        rank = torch.distributed.get_rank()
        base, ext = os.path.splitext(args.memory_snapshot_path)
        filename = f'{base}_oom_rank_{rank}{ext}'
        torch.cuda.memory._dump_snapshot(filename)
        logger.info(f'[OOM] rank {rank} saved memory snapshot to {filename}')

    # CUDA-only API; absent on non-CUDA (e.g. NPU) builds.
    if hasattr(torch._C, '_cuda_attach_out_of_memory_observer'):
        torch._C._cuda_attach_out_of_memory_observer(_oom_observer)
    logger.info(f'Memory history recording enabled (rank {torch.distributed.get_rank()}); '
                f"snapshots will be written to '{args.memory_snapshot_path}'.")


class ProfilingCallback(MegatronCallback):

    def __init__(self, trainer):
        super().__init__(trainer)

        self.use_pytorch_profiler = self.args.use_pytorch_profiler
        self.use_nsys_profiler = self.args.use_nsys_profiler
        self.profile = self.use_pytorch_profiler or self.use_nsys_profiler
        self.profile_step_start = self.args.profile_step_start
        self.profile_step_end = self.args.profile_step_end
        self.pytorch_profiler_collect_shapes = self.args.pytorch_profiler_collect_shapes
        self.pytorch_profiler_collect_callstack = self.args.pytorch_profiler_collect_callstack
        self.pytorch_profiler_collect_chakra = self.args.pytorch_profiler_collect_chakra
        self.profile_ranks = self.args.profile_ranks
        self.record_memory_history = self.args.record_memory_history
        self.memory_snapshot_path = self.args.memory_snapshot_path
        self.record_shapes = self.args.record_shapes
        self.nvtx_ranges = self.args.nvtx_ranges
        self.base_dir = Path(self.args.output_dir)

        # dynamic profiling objects
        self.prof = None
        self.nsys_nvtx_context = None  # reference to context for nsys profiling, so it can be cleaned up

    def on_train_begin(self):
        if (self.profile and (len(self.profile_ranks) == 0 or torch.distributed.get_rank() in self.profile_ranks)
                and self.use_pytorch_profiler):
            if self.pytorch_profiler_collect_chakra:
                et_dir = Path(f"{self.base_dir}/chakra")
                et_dir.mkdir(parents=True, exist_ok=True)
                et = torch.profiler.ExecutionTraceObserver().register_callback(
                    f"{et_dir}/rank-{torch.distributed.get_rank()}.json.gz")
            else:
                et = None

            def trace_handler(p):
                profile_dir = Path(f"{self.base_dir}/torch_profile")
                profile_dir.mkdir(parents=True, exist_ok=True)
                p.export_chrome_trace(f"{profile_dir}/rank-{torch.distributed.get_rank()}.json.gz")

            self.prof = torch.profiler.profile(
                schedule=torch.profiler.schedule(
                    wait=max(self.profile_step_start - 1, 0),
                    warmup=1 if self.profile_step_start > 0 else 0,
                    active=self.profile_step_end - self.profile_step_start,
                    repeat=1,
                ),
                on_trace_ready=trace_handler,
                record_shapes=self.pytorch_profiler_collect_shapes,
                with_stack=self.pytorch_profiler_collect_callstack,
                execution_trace_observer=et,
            )
            self.prof.start()

    def on_step_begin(self):
        if (self.profile and (len(self.profile_ranks) == 0 or torch.distributed.get_rank() in self.profile_ranks)):
            # Enable NVTX range when profiling starts and nvtx_ranges is set.
            if self.state.iteration == self.profile_step_start and self.nvtx_ranges:
                configure_nvtx_profiling(True)
            if self.use_pytorch_profiler:
                self.prof.step()
            elif self.state.iteration == self.profile_step_start:
                torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStart())
                self.nsys_nvtx_context = torch.autograd.profiler.emit_nvtx(record_shapes=self.record_shapes)
                self.nsys_nvtx_context.__enter__()

    def on_step_end(self):
        if (self.profile and self.state.iteration == self.profile_step_end
                and (len(self.profile_ranks) == 0 or torch.distributed.get_rank() in self.profile_ranks)):
            # Disable NVTX range when profiling ends.
            if self.nvtx_ranges:
                configure_nvtx_profiling(False)
            if self.use_pytorch_profiler:
                assert self.prof is not None
                self.prof.stop()
                if self.prof.execution_trace_observer is not None:
                    self.prof.execution_trace_observer.unregister_callback()
            else:
                torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStop())
                if self.nsys_nvtx_context is not None:
                    self.nsys_nvtx_context.__exit__(None, None, None)
            # Profiling is a one-shot window; short-circuit all subsequent
            # on_step_begin/on_step_end calls (e.g. prof.step() after stop()).
            self.profile = False

    def on_log(self, logs):
        # Dump a memory snapshot at each log interval (mirrors Megatron-LM's
        # log-interval behavior in megatron/training/training.py). The rank
        # suffix prevents multiple ranks from overwriting the same file.
        if self.record_memory_history and (len(self.profile_ranks) == 0
                                           or torch.distributed.get_rank() in self.profile_ranks):
            rank = torch.distributed.get_rank()
            base, ext = os.path.splitext(self.memory_snapshot_path)
            torch.cuda.memory._dump_snapshot(f'{base}_{rank}{ext}')

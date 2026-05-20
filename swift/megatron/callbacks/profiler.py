# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import torch
import torch.profiler
from swift.megatron.callbacks.base import MegatronCallback


class NsysCallback(MegatronCallback):
    """Profile steps [nsys_profile_start, nsys_profile_end] via cudaProfilerStart/Stop.

    Requires nsys launched with --start-later --capture-range=cudaProfilerApi.
    profile_rank: list of global ranks to profile; None = all ranks.
    """

    def __init__(self, trainer):
        super().__init__(trainer)
        self.start_step = getattr(self.args, 'nsys_profile_start', -1)
        self.end_step = getattr(self.args, 'nsys_profile_end', -1)
        self._local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self._global_rank = int(os.environ.get('RANK', 0))
        self._profile_ranks = getattr(self.args, 'profile_rank', None)
        self._profiling = False

    def _should_profile(self):
        return self._profile_ranks is None or self._global_rank in self._profile_ranks

    def on_step_begin(self):
        if not self._should_profile():
            return
        step = self.state.iteration + 1
        if step == self.start_step and not self._profiling:
            print(f'[nsys] cudaProfilerStart at step {step} (local_rank={self._local_rank})',
                  flush=True)
            torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStart())
            self._profiling = True

    def on_step_end(self):
        if not self._should_profile():
            return
        step = self.state.iteration
        if self._profiling and step >= self.end_step:
            print(f'[nsys] cudaProfilerStop after step {step} (local_rank={self._local_rank})',
                  flush=True)
            torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStop())
            self._profiling = False


class TorchProfilerCallback(MegatronCallback):
    """Profile steps [nsys_profile_start, nsys_profile_end] via torch.profiler.

    profile_rank: list of global ranks to profile; None = all ranks.
    Saves TensorBoard traces to {tensorboard_dir}/torch_profiler/rank{R}_node{N}/.
    Step numbers are 1-based.
    """

    def __init__(self, trainer):
        super().__init__(trainer)
        self.start_step = getattr(self.args, 'nsys_profile_start', 5)
        self.end_step = getattr(self.args, 'nsys_profile_end', 5)
        self._local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self._global_rank = int(os.environ.get('RANK', 0))
        self._node_rank = int(os.environ.get('NODE_RANK', 0))
        self._profile_ranks = getattr(self.args, 'profile_rank', None)
        self._prof = None
        self._trace_dir = None

    def _should_profile(self):
        return self._profile_ranks is None or self._global_rank in self._profile_ranks

    def on_train_begin(self):
        if not self._should_profile():
            return
        wait = max(0, self.start_step - 1)
        active = max(1, self.end_step - self.start_step + 1)
        base_dir = self.args.output_dir
        trace_dir = os.path.join(
            base_dir, 'torch_profiler',
            f'rank{self._local_rank}_node{self._node_rank}')
        os.makedirs(trace_dir, exist_ok=True)
        self._prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=wait, warmup=0, active=active, repeat=1),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
        )
        self._trace_dir = trace_dir
        self._prof.__enter__()
        print(f'[torch_profiler] started rank={self._local_rank} node={self._node_rank} '
              f'wait={wait} active={active} trace_dir={trace_dir}', flush=True)

    def on_step_end(self):
        if self._prof is not None:
            self._prof.step()

    def on_train_end(self):
        if self._prof is not None:
            self._prof.__exit__(None, None, None)
            chrome_path = os.path.join(self._trace_dir, 'chrome_trace.json')
            self._prof.export_chrome_trace(chrome_path)
            print(f'[torch_profiler] trace saved rank={self._local_rank} node={self._node_rank} '
                  f'chrome={chrome_path}', flush=True)
            self._prof = None
            self._trace_dir = None

import functools
import os
import torch
from typing import Callable, Optional

from .config import ProfilerConfig, TorchProfilerToolConfig


class DistProfiler:

    def __init__(self,
                 global_config=None,
                 rank: int = None,
                 config: Optional[ProfilerConfig] = None,
                 tool_config: Optional[object] = None,
                 **kwargs):
        # Default config
        if rank is None:
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
            else:
                rank = int(os.environ.get('RANK', 0))
                print(f"Warning: torch.distributed is not initialized, using RANK env var for rank: {rank}")
        if global_config is not None:
            config = ProfilerConfig(
                tool=global_config.profiler_tool,
                enable=global_config.enable_profiler,
                all_ranks=global_config.profiler_all_ranks,
                ranks=global_config.profiler_ranks,
                save_path=global_config.profiler_save_path,
                tool_config=tool_config or TorchProfilerToolConfig(
                    contents=global_config.profiler_contents, discrete=global_config.profiler_discrete),
            )
        elif not config:
            config = ProfilerConfig(ranks=[], enable=False, tool_config=None)

        if tool_config is None:
            tool_config = config.tool_config

        self.config = config
        self.tool_config = tool_config

        self._impl = None
        self._tool = getattr(config, 'tool', None)
        self._enable = config.enable
        self._this_step = False

        # Normalize rank selection
        self._this_rank = False
        if config.all_ranks:
            self._this_rank = True
        elif config.ranks:
            self._this_rank = rank in config.ranks
        else:
            # default rank 0 if enabled but ranks unspecified
            self._this_rank = (rank == 0) if self._enable else False

        self._discrete = getattr(tool_config, 'discrete', False) if tool_config else False

        if self._tool == 'torch':
            from .torch_profile import Profiler as _Torch

            self._impl = _Torch(rank=rank, config=config, tool_config=tool_config)
        else:
            # Fallback to a no-op impl
            self._impl = _NoOpProfiler()

    def check_enable(self):
        return self._enable

    def check_this_rank(self):
        return self._this_rank

    def check_this_step(self):
        return self._this_step

    def is_discrete_mode(self):
        return self._discrete

    def start(self, **kwargs):
        if self.check_enable() and self.check_this_rank():
            self._this_step = True
            return getattr(self._impl, 'start', lambda **_: None)(**kwargs)

    def stop(self):
        if self.check_enable() and self.check_this_rank():
            self._this_step = False
            return getattr(self._impl, 'stop', lambda: None)()

    @classmethod
    def annotate(
        cls,
        message: Optional[str] = None,
        color: Optional[str] = None,
        domain: Optional[str] = None,
        category: Optional[str] = None,
        **kwargs_outer,
    ) -> Callable:

        def decorator(func):

            @functools.wraps(func)
            def wrapper(self_instance, *args, **kwargs_inner):
                profiler = getattr(self_instance, 'profiler', None)

                if (not profiler or not profiler.check_enable() or not profiler.check_this_step()
                        or not profiler.check_this_rank()):
                    return func(self_instance, *args, **kwargs_inner)

                impl = profiler._impl
                if hasattr(impl, 'annotate'):
                    try:
                        actual_decorator = impl.annotate(
                            message=message, color=color, domain=domain, category=category, **kwargs_outer)

                        return actual_decorator(func)(self_instance, *args, **kwargs_inner)
                    except Exception:
                        return func(self_instance, *args, **kwargs_inner)
                return func(self_instance, *args, **kwargs_inner)

            return wrapper

        return decorator


class DistProfilerExtension:

    def __init__(self, profiler: DistProfiler):
        self.profiler = profiler

    def start_profile(self, **kwargs) -> None:
        """Start profiling for the current rank in the current training step."""
        self.profiler.start(**kwargs)

    def stop_profile(self) -> None:
        """Stop profiling for the current rank in the current training step."""
        self.profiler.stop()


class _NoOpProfiler:

    def start(self, **kwargs):
        return

    def stop(self):
        return

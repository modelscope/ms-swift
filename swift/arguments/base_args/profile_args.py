# Copyright (c) ModelScope Contributors. All rights reserved.
from dataclasses import dataclass, field
from typing import List, Optional

from swift.utils import get_logger

logger = get_logger()


@dataclass
class ProfilerArguments:

    enable_profiler: bool = False
    profiler_save_path: Optional[str] = None
    profiler_all_ranks: bool = False
    profiler_ranks: List[int] = field(default_factory=list)
    profiler_contents: List[str] = field(default_factory=list)  # e.g., "cpu", "cuda", "stack", "memory"."shape"
    profiler_discrete: bool = False
    profiler_tool: Optional[str] = 'torch'
    profiler_steps: Optional[List[int]] = field(default_factory=list)  # Steps to profile

    def __post_init__(self):
        assert not self.profiler_discrete, \
            'Profiler discrete mode is not supported yet, please set profiler_discrete to false'
        if self.enable_profiler and 'profiler' not in self.callbacks:
            self.callbacks.append('profiler')
        if 'profiler' in self.callbacks and not self.enable_profiler:
            self.enable_profiler = True
        if self.enable_profiler:
            assert self.profiler_save_path is not None, \
                'Profiler save path must be specified when profiler is enabled.'
            assert self.profiler_contents, \
                'Profiler contents must be specified when profiler is enabled.'
            assert self.profiler_steps, \
                'Profiler steps must be specified when profiler is enabled.'
            assert self.profiler_ranks != [] or self.profiler_all_ranks, \
                'Either profiler_ranks must be specified or profiler_all_ranks must be set to True.'
        if self.enable_profiler:
            assert 'profiler' in self.callbacks, \
                'Profiler callback must be included in callbacks when profiler is enabled.'
        if 'profiler' in self.callbacks:
            assert self.enable_profiler, \
                'Profiler callback is included in callbacks but profiler is not enabled.'

    def get_profiler_kwargs(self):
        return {
            'enable_profiler': self.enable_profiler,
            'profiler_save_path': self.profiler_save_path,
            'profiler_all_ranks': self.profiler_all_ranks,
            'profiler_ranks': self.profiler_ranks,
            'profiler_contents': self.profiler_contents,
            'profiler_discrete': self.profiler_discrete,
            'profiler_tool': self.profiler_tool,
            'profiler_steps': self.profiler_steps,
        }

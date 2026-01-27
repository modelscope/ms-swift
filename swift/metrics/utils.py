# Copyright (c) ModelScope Contributors. All rights reserved.
import time
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.distributed as dist

from swift.utils import get_current_device, get_logger

logger = get_logger()


class Metric(ABC):

    def __init__(self):
        self._default = {}
        self._default_factory = {}

    def add_state(self, name: str, default=None, default_factory=None) -> None:
        if not hasattr(self, '_default'):
            raise AttributeError('Please call super().__init__() first.')
        if default is None:
            self._default_factory[name] = default_factory
            assert name not in self._default, f'self._default: {self._default}'
            default = default_factory()
        else:
            self._default[name] = default
            assert name not in self._default_factory, f'self._default_factory: {self._default_factory}'
        setattr(self, name, default)

    def reset(self):
        for k, v in self._default.items():
            setattr(self, k, v)
        for k, v in self._default_factory.items():
            setattr(self, k, v())

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abstractmethod
    def compute(self):
        pass


class InferStats(Metric):

    def __init__(self):
        super().__init__()
        self.add_state('start_runtime', default_factory=lambda: time.perf_counter())
        self.add_state('num_prompt_tokens', default_factory=dict)
        self.add_state('num_generated_tokens', default_factory=dict)

    def update(self, output):
        id_ = output.id
        self.num_prompt_tokens[id_] = output.usage.prompt_tokens
        self.num_generated_tokens[id_] = output.usage.completion_tokens

    def compute(self):
        runtime = time.perf_counter() - self.start_runtime
        num_samples = len(self.num_generated_tokens)
        num_generated_tokens = sum(self.num_generated_tokens.values())
        return {
            'num_prompt_tokens': sum(self.num_prompt_tokens.values()),
            'num_generated_tokens': num_generated_tokens,
            'num_samples': num_samples,
            'runtime': runtime,
            'samples/s': num_samples / runtime,
            'tokens/s': num_generated_tokens / runtime,
        }


class MeanMetric(Metric):

    def __init__(self, nan_value=0, device=None, group=None):
        super().__init__()
        self.nan_value = nan_value
        self.add_state('state', default=0.)
        self.add_state('count', default=0)
        if device is None:
            device = get_current_device()
        self.device = device
        self.group = group

    def update(self, state: torch.Tensor):
        if isinstance(state, (torch.Tensor, np.ndarray)):
            if state.ndim == 0:
                count = 1
                state = state.item()
            else:
                count = state.shape[0]
                state = state.sum().item()
        elif isinstance(state, (list, tuple)):
            count = len(state)
            state = sum(state)
        else:
            count = 1

        self.state += state
        self.count += count

    def compute(self):
        if dist.is_initialized():
            tensor = torch.tensor([self.state, self.count], device=self.device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.group)
            self.state, self.count = tensor[0].item(), int(tensor[1].item())
        if self.count == 0:
            value = self.nan_value
        else:
            value = self.state / self.count
        return {
            'value': value,
        }

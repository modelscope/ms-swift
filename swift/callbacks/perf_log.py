# Copyright (c) ModelScope Contributors. All rights reserved.
import time
from typing import TYPE_CHECKING

import torch
from transformers import TrainerControl, TrainerState

from swift.utils import empty_cache, get_current_device, get_device_count, get_env_args, get_logger
from .base import TrainerCallback

if TYPE_CHECKING:
    from swift.trainers import TrainingArguments, Trainer

logger = get_logger()

device_flops_map = {
    'GB200': 2.5e15,
    'B200': 2.25e15,
    'MI300X': 1336e12,
    'H100': 312e12,
    'H800': 312e12,
    'H200': 989e12,
    'A100': 312e12,
    'A800': 312e12,
    'L40S': 362.05e12,
    'L40': 181.05e12,
    'A40': 149.7e12,
    'L20': 119.5e12,
    'H20': 148e12,
    '910B': 354e12,
    'Ascend910': 354e12,
    'RTX 3070 Ti': 21.75e12
}


class PerfMetricsLogCallback(TrainerCallback):
    """An callback for perf metrics (MFU etc) log implementation"""

    def __init__(self, args: 'TrainingArguments', trainer: 'Trainer'):
        super().__init__(args, trainer)
        self.device_tflops = None
        self.elapsed = 0.0
        self.step_start_time = None

    def on_init_end(self, args: 'TrainingArguments', state: TrainerState, control: TrainerControl, **kwargs):

        # Top priority. Specify by ENV
        tflops = get_env_args('DEVICE_TFLOPS', int, None)
        device_count = max(get_device_count(), 1)
        if tflops is not None:
            logger.info(f"Specify theoretical max TFLOPS through ENV 'DEVICE_TFLOPS'. [{tflops} TFLOPS]")
        else:
            # Run a estimating test.
            dtype = kwargs.get('model').dtype
            device = torch.device(get_current_device())
            logger.info(f'Estimating device TFLOPS baseline. Device: [{device}] dtype: [{dtype}]')
            tflops = self._estimate_device_tflops_by_dtype(device, dtype)
            logger.info(f'Estimate test finished. [{tflops} TFLOPS] Device count: [{device_count}]')
        # TODO Collect comprehensive TFLOPS data. Then provide a fallback strategy based on lookup tables.

        self.device_tflops = tflops * device_count

    def on_step_begin(self, args: 'TrainingArguments', state: TrainerState, control: TrainerControl, **kwargs):
        self.step_start_time = time.time()

    def on_step_end(self, args: 'TrainingArguments', state: TrainerState, control: TrainerControl, **kwargs):
        self.elapsed += time.time() - self.step_start_time

    def on_log(self, args: 'TrainingArguments', state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        total_flos = getattr(state, 'total_flos', 0)
        actual_flops = total_flos / self.elapsed
        theoretical_max_flops = self.device_tflops * 1e12
        mfu = actual_flops / theoretical_max_flops
        logger.debug(f'Total_flos[{total_flos}] elapsed_time[{self.elapsed}]sec Average MFU[{mfu}]')
        logs['MFU'] = round(mfu, 6)

    @staticmethod
    def _estimate_device_tflops_by_dtype(device: torch.device, dtype: torch.dtype, repeats: int = 60, dim: int = 8192):

        def device_synchronize(sync_device):
            if backend == 'cuda':
                torch.cuda.synchronize(sync_device)
            elif backend == 'npu':
                torch.npu.synchronize(sync_device)
            elif backend == 'cpu':
                torch.cpu.synchronize(sync_device)

        # Set matrix dimension
        shape = (dim, dim)
        backend = device.type
        if backend == 'npu':
            import torch_npu

        # Initialize matrices
        a = torch.randn(*shape, device=device, dtype=dtype)
        b = torch.randn(*shape, device=device, dtype=dtype)

        # Warm-up
        for _ in range(5):
            c = torch.matmul(a, b)
        device_synchronize(device)

        # Run benchmark test
        start = time.time()
        for _ in range(repeats):
            c = torch.matmul(a, b)
        device_synchronize(device)
        end = time.time()
        total_time = end - start
        avg_time = total_time / repeats

        # Adjust repeat count and retest if test duration is too short
        if total_time < 3:
            repeats = int(6 / avg_time)
            start = time.time()
            for _ in range(repeats):
                c = torch.matmul(a, b)
            device_synchronize(device)
            end = time.time()
            total_time = end - start
            avg_time = total_time / repeats

        del a, b, c
        empty_cache()

        tflops = (2 * dim**3 / avg_time) / 1e12
        logger.info(f'[Device {device}] Total time: {total_time:.4f}s, dtype: {dtype}, Perf: {tflops:.4f} TFLOPS')
        return tflops

    @staticmethod
    def _retrieve_flops_from_map(device):
        """Retrieve theoretical FLOPS from Map.    """

        device_name = device.get_device_name()
        flops = None
        for name, value in device_flops_map.items():
            if name in device_name:
                flops = value
                break

        return flops

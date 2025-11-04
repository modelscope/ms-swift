# Copyright (c) Alibaba, Inc. and its affiliates.
import time

import numpy as np
import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from swift.utils import get_logger, get_current_device, get_device_count

logger = get_logger()


class EarlyStopCallback(TrainerCallback):
    """An early stop implementation"""

    def __init__(self, total_interval=3):
        self.best_metric = None
        self.interval = 0
        self.total_interval = total_interval

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        operator = np.greater if args.greater_is_better else np.less
        if self.best_metric is None or operator(state.best_metric, self.best_metric):
            self.best_metric = state.best_metric
            self.interval = 0
        else:
            self.interval += 1

        if self.interval >= self.total_interval:
            logger.info(f'Training stop because of eval metric is stable at step {state.global_step}')
            control.should_training_stop = True


class PerfMetricsLogCallback(TrainerCallback):
    """A callback for perf metrics (MFU etc) log implementation"""

    def __init__(self):
        self.start_time = None
        self.device_tflops = None
        self.elapsed = 0.0
        self.step_start_time = None

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        from swift.utils import get_env_args

        # Top priority. Specify by ENV
        tflops = get_env_args('DEVICE_TFLOPS', int, None)
        device_count = max(get_device_count(), 1)
        if tflops is not None:
            logger.info(f"Specify theoretical max TFLOPS through ENV 'DEVICE_TFLOPS'. [{tflops} TFLOPS]")
        else:
            # Run a estimating test.
            dtype = kwargs.get("model").dtype
            device = torch.device(get_current_device())
            logger.info(f"Estimating device TFLOPS baseline. Device: [{device}] dtype: [{dtype}]")
            tflops = self._estimate_device_tflops_by_dtype(device, dtype)
            logger.info(f"Estimate test finished. [{tflops} TFLOPS] Device count: [{device_count}]")

        self.device_tflops = tflops * device_count

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.step_start_time = time.time()

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.elapsed += time.time() - self.step_start_time

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.start_time = time.time()

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        total_flos = getattr(state, 'total_flos', 0)
        actual_flops = total_flos / self.elapsed
        theoretical_max_flops = self.device_tflops * 1e12
        mfu = actual_flops / theoretical_max_flops
        logger.debug(f"Total_flos[{total_flos}] elapsed_time[{self.elapsed}]sec Average MFU[{mfu}]")
        logs['MFU'] = round(mfu, 6)

    @staticmethod
    def _estimate_device_tflops_by_dtype(device: torch.device, dtype: torch.dtype, repeats: int = 60,
                                         dim: int = 8192):
        # 默认矩阵规模
        shape = (dim, dim)
        backend = device.type
        if backend == "npu":
            import torch_npu

        # 创建矩阵
        a = torch.randn(*shape, device=device, dtype=dtype)
        b = torch.randn(*shape, device=device, dtype=dtype)

        # 预热
        for _ in range(5):
            c = torch.matmul(a, b)
        if backend == 'cuda':
            torch.cuda.synchronize(device)
        elif backend == 'npu':
            torch.npu.synchronize(device)
        elif backend == 'cpu':
            torch.cpu.synchronize(device)

        # 进行测试
        start = time.time()
        for _ in range(repeats):
            c = torch.matmul(a, b)
        if backend == 'cuda':
            torch.cuda.synchronize(device)
        elif backend == 'npu':
            torch.npu.synchronize(device)
        elif backend == 'cpu':
            torch.cpu.synchronize(device)
        end = time.time()
        total_time = end - start
        avg_time = total_time / repeats

        # 若测试时间过短，调整循环次数并重新测试
        if total_time < 3:
            repeats = int(6 / avg_time)
            start = time.time()
            for _ in range(repeats):
                c = torch.matmul(a, b)
            if backend == 'cuda':
                torch.cuda.synchronize(device)
            elif backend == 'npu':
                torch.npu.synchronize(device)
            elif backend == 'cpu':
                torch.cpu.synchronize(device)
            end = time.time()
            total_time = end - start
            avg_time = total_time / repeats

        del a, b, c
        if backend == 'cuda':
            torch.cuda.empty_cache()
        elif backend == 'npu':
            torch.npu.empty_cache()

        tflops = (2 * dim ** 3 / avg_time) / 1e12
        print(
            f"[设备 {device}] 测试总耗时：{total_time:.4f}s，平均耗时: {avg_time:.4f} s，dtype：{dtype}，性能: {tflops:.4f} TFLOPS")

        return tflops

    @staticmethod
    def _retrieve_flops_from_map(device):
        """Retrieve theoretical FLOPS from Map.    """

        device_name = device.get_device_name()
        flops = None
        for name, value in device_flops_map:
            if name in device_name:
                flops = value
                break

        return flops


device_flops_map = {
    "GB200": 2.5e15,
    "B200": 2.25e15,
    "MI300X": 1336e12,
    "H100": 312e12,
    "H800": 312e12,
    "H200": 989e12,
    "A100": 312e12,
    "A800": 312e12,
    "L40S": 362.05e12,
    "L40": 181.05e12,
    "A40": 149.7e12,
    "L20": 119.5e12,
    "H20": 148e12,
    "910B": 354e12,
    "Ascend910": 354e12,
    "RTX 3070 Ti": 21.75e12
}

extra_callbacks = [PerfMetricsLogCallback()]
# This example shows a simple example of EarlyStop Callback, uncomment this to use
# extra_callbacks = [EarlyStopCallback()]

# Copyright (c) ModelScope Contributors. All rights reserved.
from .default_flow import DefaultFlowCallback
from .profiler import NsysCallback, TorchProfilerCallback
from .print import PrintCallback
from .swanlab import SwanlabCallback
from .tensorboard import TensorboardCallback
from .wandb import WandbCallback

megatron_callbacks_map = {
    'print': PrintCallback,
    'default_flow': DefaultFlowCallback,
    'nsys': NsysCallback,
    'torch_profiler': TorchProfilerCallback,
    'swanlab': SwanlabCallback,
    'wandb': WandbCallback,
    'tensorboard': TensorboardCallback,
}

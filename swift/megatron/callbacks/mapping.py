# Copyright (c) ModelScope Contributors. All rights reserved.
from .default_flow import DefaultFlowCallback
from .print import PrintCallback
from .profiling import ProfilingCallback
from .swanlab import SwanlabCallback
from .tensorboard import TensorboardCallback
from .wandb import WandbCallback

megatron_callbacks_map = {
    'print': PrintCallback,
    'default_flow': DefaultFlowCallback,
    'swanlab': SwanlabCallback,
    'wandb': WandbCallback,
    'tensorboard': TensorboardCallback,
    'profiling': ProfilingCallback,
}

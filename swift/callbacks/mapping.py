# Copyright (c) ModelScope Contributors. All rights reserved.
from swift.activation_cpu_offload import ActivationCpuOffloadCallBack
from .adalora import AdaloraCallback
from .early_stop import EarlyStopCallback
from .lisa import LISACallback
from .perf_log import PerfMetricsLogCallback

callbacks_map = {
    'activation_cpu_offload': ActivationCpuOffloadCallBack,
    'adalora': AdaloraCallback,
    'early_stop': EarlyStopCallback,
    'lisa': LISACallback,
    'perf_log': PerfMetricsLogCallback
}

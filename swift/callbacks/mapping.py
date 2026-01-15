# Copyright (c) ModelScope Contributors. All rights reserved.
from .adalora import AdaloraCallback
from .early_stop import EarlyStopCallback
from .lisa import DynamicLayerActivationCallback
from .perf_log import PerfMetricsLogCallback

callbacks_map = {
    'adalora': AdaloraCallback,
    'early_stop': EarlyStopCallback,
    'lisa': DynamicLayerActivationCallback,
    'perf_log': PerfMetricsLogCallback
}

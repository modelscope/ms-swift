# Copyright (c) ModelScope Contributors. All rights reserved.
from .adalora import AdaloraCallback
from .deepspeed_elastic import DeepspeedElasticCallback, GracefulExitCallback
from .early_stop import EarlyStopCallback
from .lisa import LISACallback
from .perf_log import PerfMetricsLogCallback

callbacks_map = {
    'adalora': AdaloraCallback,
    'deepspeed_elastic': DeepspeedElasticCallback,
    'early_stop': EarlyStopCallback,
    'graceful_exit': GracefulExitCallback,
    'lisa': LISACallback,
    'perf_log': PerfMetricsLogCallback
}

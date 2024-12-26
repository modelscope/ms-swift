# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .callback import extra_callbacks
    from .loss import LOSS_MAPPING, get_loss_func
    from .loss_scale import loss_scale_map
    from .metric import InferStats, MeanMetric, Metric, compute_acc, get_metric
    from .optimizer import optimizers_map
    from .tools import get_tools_prompt, get_tools_keyword
    from .tuner import Tuner, extra_tuners

else:
    _extra_objects = {k: v for k, v in globals().items() if not k.startswith('_')}
    _import_structure = {
        'callback': ['extra_callbacks'],
        'loss': ['LOSS_MAPPING', 'get_loss_func'],
        'loss_scale': ['loss_scale_map'],
        'metric': ['InferStats', 'MeanMetric', 'Metric', 'compute_acc', 'get_metric'],
        'optimizer': ['optimizers_map'],
        'tools': ['get_tools_prompt', 'get_tools_keyword'],
        'tuner': ['Tuner', 'extra_tuners'],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )

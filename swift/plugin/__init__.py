# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .callback import extra_callbacks
    from .loss import loss_mapping, get_loss_func
    from .loss_scale import loss_scale_map
    from .metric import InferStats, MeanMetric, Metric, compute_acc, get_metric, compute_rouge_bleu
    from .optimizer import optimizers_map
    from .agent_template import agent_templates
    from .tuner import Tuner, extra_tuners, PeftTuner
    from .prm import prms, PRM
    from .orm import orms, ORM
    from .multi_turn import multi_turns
    from .rm_plugin import rm_plugins
    from .env import envs, Env
    from .context_manager import context_managers, ContextManager

else:
    _import_structure = {
        'callback': ['extra_callbacks'],
        'loss': ['loss_mapping', 'get_loss_func'],
        'loss_scale': ['loss_scale_map'],
        'metric': ['InferStats', 'MeanMetric', 'Metric', 'compute_acc', 'get_metric', 'compute_rouge_bleu'],
        'optimizer': ['optimizers_map'],
        'agent_template': ['agent_templates'],
        'tuner': ['Tuner', 'extra_tuners', 'PeftTuner'],
        'prm': ['prms', 'PRM'],
        'orm': ['orms', 'ORM'],
        'multi_turn': ['multi_turns'],
        'rm_plugin': ['rm_plugins'],
        'env': ['envs', 'Env'],
        'context_manager': ['context_managers', 'ContextManager'],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )

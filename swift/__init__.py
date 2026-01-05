# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from .utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .version import __version__, __release_datetime__
    from .tuners import Swift
    from .trainers import TrainingArguments, Seq2SeqTrainingArguments, Trainer, Seq2SeqTrainer
    from .arguments import (PretrainArguments, SftArguments, RLHFArguments, ExportArguments, InferArguments,
                            AppArguments, EvalArguments, SamplingArguments, RolloutArguments, DeployArguments)
    from .pipelines import (sft_main, pretrain_main, infer_main, rlhf_main, export_main, app_main, eval_main,
                            sampling_main, rollout_main, deploy_main, merge_lora)
    from .model import get_model
    from .template import get_template
    from .dataset import load_dataset
    from .utils import get_logger
else:
    _import_structure = {
        'version': ['__release_datetime__', '__version__'],
        'tuners': ['Swift'],
        'trainers': ['TrainingArguments', 'Seq2SeqTrainingArguments', 'Trainer', 'Seq2SeqTrainer'],
        'arguments': [
            'PretrainArguments', 'SftArguments', 'RLHFArguments', 'ExportArguments', 'InferArguments', 'AppArguments',
            'EvalArguments', 'SamplingArguments', 'RolloutArguments', 'DeployArguments'
        ],
        'pipelines': [
            'sft_main', 'pretrain_main', 'infer_main', 'rlhf_main', 'export_main', 'app_main', 'eval_main',
            'sampling_main', 'rollout_main', 'deploy_main', 'merge_lora'
        ],
        'model': ['get_model'],
        'template': ['get_template'],
        'dataset': ['load_dataset'],
        'utils': ['get_logger'],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )

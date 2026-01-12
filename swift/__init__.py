# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from .utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .version import __version__, __release_datetime__
    from .tuners import Swift
    from .infer_engine import (TransformersEngine, VllmEngine, SglangEngine, LmdeployEngine, InferRequest,
                               RequestConfig, AdapterRequest)
    from .trainers import TrainingArguments, Seq2SeqTrainingArguments, Trainer, Seq2SeqTrainer
    from .arguments import (PretrainArguments, SftArguments, RLHFArguments, ExportArguments, InferArguments,
                            AppArguments, EvalArguments, SamplingArguments, RolloutArguments, DeployArguments)
    from .pipelines import (sft_main, pretrain_main, infer_main, rlhf_main, export_main, app_main, eval_main,
                            sampling_main, rollout_main, deploy_main, merge_lora, run_deploy)
    from .model import get_model_processor, get_processor
    from .template import get_template
    from .dataset import load_dataset, EncodePreprocessor
    from .utils import get_logger, safe_snapshot_download
else:
    _import_structure = {
        'version': ['__release_datetime__', '__version__'],
        'tuners': ['Swift'],
        'infer_engine': [
            'TransformersEngine', 'VllmEngine', 'SglangEngine', 'LmdeployEngine', 'InferRequest', 'RequestConfig',
            'AdapterRequest'
        ],
        'trainers': ['TrainingArguments', 'Seq2SeqTrainingArguments', 'Trainer', 'Seq2SeqTrainer'],
        'arguments': [
            'PretrainArguments', 'SftArguments', 'RLHFArguments', 'ExportArguments', 'InferArguments', 'AppArguments',
            'EvalArguments', 'SamplingArguments', 'RolloutArguments', 'DeployArguments'
        ],
        'pipelines': [
            'sft_main', 'pretrain_main', 'infer_main', 'rlhf_main', 'export_main', 'app_main', 'eval_main',
            'sampling_main', 'rollout_main', 'deploy_main', 'merge_lora', 'run_deploy'
        ],
        'model': ['get_model_processor', 'get_processor'],
        'template': ['get_template'],
        'dataset': ['load_dataset', 'EncodePreprocessor'],
        'utils': ['get_logger', 'safe_snapshot_download'],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )

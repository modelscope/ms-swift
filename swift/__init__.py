# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING

from .utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .version import __version__, __release_datetime__
    from .tuners import Swift
    from .tuner_plugin import Tuner, PeftTuner, tuners_map
    from .infer_engine import (TransformersEngine, VllmEngine, SglangEngine, LmdeployEngine, InferRequest,
                               RequestConfig, AdapterRequest, InferEngine, InferClient, GRPOVllmEngine)
    from .trainers import TrainingArguments, Seq2SeqTrainingArguments, Trainer, Seq2SeqTrainer
    from .arguments import (PretrainArguments, SftArguments, RLHFArguments, ExportArguments, InferArguments,
                            AppArguments, EvalArguments, SamplingArguments, RolloutArguments, DeployArguments,
                            BaseArguments)
    from .pipelines import (sft_main, pretrain_main, infer_main, rlhf_main, export_main, app_main, eval_main,
                            sampling_main, rollout_main, deploy_main, merge_lora, run_deploy)
    from .model import get_model_processor, get_processor
    from .template import get_template
    from .dataset import load_dataset, EncodePreprocessor
    from .utils import get_logger, safe_snapshot_download
    from .agent_template import agent_template_map, BaseAgentTemplate
    from .loss import loss_map, BaseLoss
    from .metrics import eval_metrics_map, InferStats, MeanMetric
    from .optimizers import optimizers_map, OptimizerCallback
    from .callbacks import callbacks_map, TrainerCallback
    from .loss_scale import loss_scale_map, LossScale, get_loss_scale, ALL_BASE_STRATEGY, ConfigLossScale
else:
    _import_structure = {
        'version': ['__release_datetime__', '__version__'],
        'tuners': ['Swift'],
        'tuner_plugin': ['Tuner', 'PeftTuner', 'tuners_map'],
        'infer_engine': [
            'TransformersEngine', 'VllmEngine', 'SglangEngine', 'LmdeployEngine', 'InferRequest', 'RequestConfig',
            'AdapterRequest', 'InferEngine', 'InferClient', 'GRPOVllmEngine'
        ],
        'trainers': ['TrainingArguments', 'Seq2SeqTrainingArguments', 'Trainer', 'Seq2SeqTrainer'],
        'arguments': [
            'PretrainArguments', 'SftArguments', 'RLHFArguments', 'ExportArguments', 'InferArguments', 'AppArguments',
            'EvalArguments', 'SamplingArguments', 'RolloutArguments', 'DeployArguments', 'BaseArguments'
        ],
        'pipelines': [
            'sft_main', 'pretrain_main', 'infer_main', 'rlhf_main', 'export_main', 'app_main', 'eval_main',
            'sampling_main', 'rollout_main', 'deploy_main', 'merge_lora', 'run_deploy'
        ],
        'model': ['get_model_processor', 'get_processor'],
        'template': ['get_template'],
        'dataset': ['load_dataset', 'EncodePreprocessor'],
        'utils': ['get_logger', 'safe_snapshot_download'],
        'agent_template': ['agent_template_map', 'BaseAgentTemplate'],
        'loss': ['loss_map', 'BaseLoss'],
        'metrics': ['eval_metrics_map', 'InferStats', 'MeanMetric'],
        'optimizers': ['optimizers_map', 'OptimizerCallback'],
        'callbacks': ['callbacks_map', 'TrainerCallback'],
        'loss_scale': ['loss_scale_map', 'LossScale', 'get_loss_scale', 'ALL_BASE_STRATEGY', 'ConfigLossScale'],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )

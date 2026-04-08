# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING

from .utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .agent_template import BaseAgentTemplate, agent_template_map
    from .arguments import (AppArguments, BaseArguments, DeployArguments, EvalArguments, ExportArguments,
                            InferArguments, PretrainArguments, RLHFArguments, RolloutArguments, SamplingArguments,
                            SftArguments)
    from .callbacks import TrainerCallback, callbacks_map
    from .dataset import EncodePreprocessor, load_dataset
    from .infer_engine import (AdapterRequest, GRPOVllmEngine, InferClient, InferEngine, InferRequest, LmdeployEngine,
                               RequestConfig, SglangEngine, TransformersEngine, VllmEngine)
    from .loss import BaseLoss, loss_map
    from .loss_scale import ALL_BASE_STRATEGY, ConfigLossScale, LossScale, get_loss_scale, loss_scale_map
    from .metrics import InferStats, MeanMetric, eval_metrics_map
    from .model import get_model_processor, get_processor
    from .optimizers import OptimizerCallback, optimizers_map
    from .pipelines import (app_main, deploy_main, eval_main, export_main, infer_main, merge_lora, pretrain_main,
                            rlhf_main, rollout_main, run_deploy, sampling_main, sft_main)
    from .template import get_template
    from .trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments, Trainer, TrainingArguments
    from .tuner_plugin import PeftTuner, Tuner, tuners_map
    from .tuners import Swift
    from .utils import get_logger, safe_snapshot_download
    from .version import __release_datetime__, __version__
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

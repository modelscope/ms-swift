# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .arguments import (Seq2SeqTrainingArguments, TrainingArguments, DPOConfig, CPOConfig, KTOConfig, ORPOConfig,
                            PPOConfig)
    from .rlhf_trainer import CPOTrainer, DPOTrainer, KTOTrainer, ORPOTrainer, RewardTrainer, PPOTrainer
    from .trainer_factory import TrainerFactory
    from .trainers import Seq2SeqTrainer, Trainer
    from .mixin import SwiftMixin, RLHFTrainerMixin
    from .push_to_ms import PushToMsHubMixin
    from .loss import LOSS_MAPPING, LossName, register_loss_func, get_loss_func
    from .utils import (EvaluationStrategy, FSDPOption, HPSearchBackend, HubStrategy, IntervalStrategy, SchedulerType,
                        ShardedDDPOption, TrainerCallback)
else:
    _import_structure = {
        'arguments': [
            'Seq2SeqTrainingArguments', 'TrainingArguments', 'DPOConfig', 'CPOConfig', 'KTOConfig', 'ORPOConfig',
            'RewardConfig', 'PPOConfig'
        ],
        'rlhf_trainer': ['CPOTrainer', 'DPOTrainer', 'KTOTrainer', 'ORPOTrainer', 'RewardTrainer', 'PPOTrainer'],
        'trainer_factory': ['TrainerFactory'],
        'trainers': ['Seq2SeqTrainer', 'Trainer'],
        'mixin': ['SwiftMixin', 'RLHFTrainerMixin'],
        'push_to_ms': ['PushToMsHubMixin'],
        'loss': ['LOSS_MAPPING', 'LossName', 'register_loss_func', 'get_loss_func'],
        'utils': [
            'EvaluationStrategy', 'FSDPOption', 'HPSearchBackend', 'HubStrategy', 'IntervalStrategy', 'SchedulerType',
            'ShardedDDPOption', 'TrainerCallback'
        ]
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )

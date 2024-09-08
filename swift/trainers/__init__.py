# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .arguments import (Seq2SeqTrainingArguments, TrainingArguments, DPOConfig, CPOConfig, KTOConfig, ORPOConfig)
    from .dpo_trainer import DPOTrainer
    from .orpo_trainer import ORPOTrainer
    from .trainer_factory import TrainerFactory
    from .trainers import Seq2SeqTrainer, Trainer
    from .loss import LOSS_MAPPING, LossName, register_loss_func, get_loss_func
    from .utils import (EvaluationStrategy, FSDPOption, HPSearchBackend, HubStrategy, IntervalStrategy, SchedulerType,
                        ShardedDDPOption, TrainerCallback)
else:
    _import_structure = {
        'arguments':
        ['Seq2SeqTrainingArguments', 'TrainingArguments', 'DPOConfig', 'CPOConfig', 'KTOConfig', 'ORPOConfig'],
        'dpo_trainer': ['DPOTrainer'],
        'orpo_trainer': ['ORPOTrainer'],
        'trainer_factory': ['TrainerFactory'],
        'trainers': ['Seq2SeqTrainer', 'Trainer'],
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

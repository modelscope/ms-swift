# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .arguments import Seq2SeqTrainingArguments, TrainingArguments
    from .dpo_trainers import DPOTrainer
    from .orpo_trainers import ORPOTrainer
    from .simpo_trainers import SimPOTrainer
    from .rlhf_trainers import TrainerFactory
    from .trainers import Seq2SeqTrainer, Trainer
    from .utils import EvaluationStrategy, FSDPOption, HPSearchBackend, HubStrategy, \
        IntervalStrategy, SchedulerType, ShardedDDPOption, TrainerCallback
else:
    _import_structure = {
        'arguments': ['Seq2SeqTrainingArguments', 'TrainingArguments'],
        'dpo_trainers': ['DPOTrainer'],
        'orpo_trainers': ['ORPOTrainer'],
        'simpo_trainers': ['SimPOTrainer'],
        'rlhf_trainers': ['TrainerFactory'],
        'trainers': ['Seq2SeqTrainer', 'Trainer'],
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

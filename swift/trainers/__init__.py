# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .arguments import Seq2SeqTrainingArguments, TrainingArguments
    from .dpo_trainer import DPOTrainer
    from .orpo_trainer import ORPOTrainer
    from .simpo_trainer import SimPOTrainer
    from .rlhf_trainers import RLHFTrainerFactory
    from .trainers import Seq2SeqTrainer, Trainer
    from .utils import EvaluationStrategy, FSDPOption, HPSearchBackend, HubStrategy, \
        IntervalStrategy, SchedulerType, ShardedDDPOption, TrainerCallback,\
        build_tokenized_answer, concat_template
else:
    _import_structure = {
        'arguments': ['Seq2SeqTrainingArguments', 'TrainingArguments'],
        'dpo_trainer': ['DPOTrainer'],
        'orpo_trainer': ['ORPOTrainer'],
        'simpo_trainer': ['SimPOTrainer'],
        'rlhf_trainers': ['RLHFTrainerFactory'],
        'trainers': ['Seq2SeqTrainer', 'Trainer'],
        'utils': [
            'EvaluationStrategy', 'FSDPOption', 'HPSearchBackend', 'HubStrategy', 'IntervalStrategy', 'SchedulerType',
            'ShardedDDPOption', 'TrainerCallback', 'build_tokenized_answer', 'concat_template'
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

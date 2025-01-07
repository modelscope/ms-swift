# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import (EvaluationStrategy, FSDPOption, HPSearchBackend, HubStrategy, IntervalStrategy,
                                        SchedulerType)

from swift.utils.import_utils import _LazyModule
from . import callback

try:
    # https://github.com/huggingface/transformers/pull/25702
    from transformers.trainer_utils import ShardedDDPOption
except ImportError:
    ShardedDDPOption = None

if TYPE_CHECKING:
    from .arguments import Seq2SeqTrainingArguments, TrainingArguments
    from .rlhf_trainer import (CPOTrainer, DPOTrainer, KTOTrainer, ORPOTrainer, RLHFTrainerMixin, PPOTrainer,
                               RewardTrainer)
    from .rlhf_arguments import DPOConfig, CPOConfig, KTOConfig, ORPOConfig, PPOConfig, RewardConfig
    from .trainer_factory import TrainerFactory
    from .trainers import Seq2SeqTrainer, Trainer
    from .mixin import SwiftMixin

else:
    _extra_objects = {k: v for k, v in globals().items() if not k.startswith('_')}
    _import_structure = {
        'arguments': ['Seq2SeqTrainingArguments', 'TrainingArguments'],
        'rlhf_arguments': ['DPOConfig', 'CPOConfig', 'KTOConfig', 'ORPOConfig', 'PPOConfig', 'RewardConfig'],
        'rlhf_trainer':
        ['CPOTrainer', 'DPOTrainer', 'KTOTrainer', 'ORPOTrainer', 'RLHFTrainerMixin', 'PPOTrainer', 'RewardTrainer'],
        'trainer_factory': ['TrainerFactory'],
        'trainers': ['Seq2SeqTrainer', 'Trainer'],
        'mixin': ['SwiftMixin'],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects=_extra_objects,
    )

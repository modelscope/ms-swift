# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .arguments import TrainArgumentsMixin, Seq2SeqTrainingArguments, TrainingArguments
    from .embedding import EmbeddingTrainer
    from .mixin import DataLoaderMixin, SwiftMixin
    from .reranker import RerankerTrainer
    from .seq2seq_trainer import Seq2SeqTrainer
    from .trainer import Trainer
    from .trainer_factory import TrainerFactory
    from .utils import (disable_gradient_checkpointing, dynamic_gradient_checkpointing, per_token_loss_func,
                        calculate_max_steps)
else:
    _import_structure = {
        'arguments': ['TrainArgumentsMixin', 'Seq2SeqTrainingArguments', 'TrainingArguments'],
        'embedding': ['EmbeddingTrainer'],
        'mixin': ['DataLoaderMixin', 'SwiftMixin'],
        'reranker': ['RerankerTrainer'],
        'seq2seq_trainer': ['Seq2SeqTrainer'],
        'trainer': ['Trainer'],
        'trainer_factory': ['TrainerFactory'],
        'utils': [
            'disable_gradient_checkpointing', 'dynamic_gradient_checkpointing', 'per_token_loss_func',
            'calculate_max_steps'
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

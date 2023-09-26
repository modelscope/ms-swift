# Copyright (c) Alibaba, Inc. and its affiliates.

from dataclasses import dataclass, field

from transformers.training_args import TrainingArguments as HfTrainingArguments
from transformers.training_args_seq2seq import \
    Seq2SeqTrainingArguments as HfSeq2SeqTrainingArguments


@dataclass
class SwiftArgumentsMixin:
    # ckpt only save model
    only_save_model: bool = False
    train_sampler_random: bool = True
    push_hub_strategy: str = field(
        default='push_best',
        metadata={
            'choices':
            {'push_best', 'push_last', 'end', 'checkpoint', 'all_checkpoints'}
        })


@dataclass
class TrainingArguments(SwiftArgumentsMixin, HfTrainingArguments):
    pass


@dataclass
class Seq2SeqTrainingArguments(SwiftArgumentsMixin,
                               HfSeq2SeqTrainingArguments):
    pass

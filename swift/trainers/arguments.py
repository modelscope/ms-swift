# Copyright (c) Alibaba, Inc. and its affiliates.

from dataclasses import dataclass, field
from typing import List, Optional

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
            {'end', 'push_best', 'push_last', 'checkpoint', 'all_checkpoints'}
        })
    acc_strategy: str = field(
        default='token', metadata={'choices': ['token', 'sentence']})
    additional_saved_files: Optional[List[str]] = None

    def __post_init__(self):
        if self.additional_saved_files is None:
            self.additional_saved_files = []
        super().__post_init__()


@dataclass
class TrainingArguments(SwiftArgumentsMixin, HfTrainingArguments):
    pass


@dataclass
class Seq2SeqTrainingArguments(SwiftArgumentsMixin,
                               HfSeq2SeqTrainingArguments):
    pass

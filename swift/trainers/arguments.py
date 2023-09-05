# Copyright (c) Alibaba, Inc. and its affiliates.

from dataclasses import dataclass

from transformers.training_args import TrainingArguments as HfTrainingArguments
from transformers.training_args_seq2seq import \
    Seq2SeqTrainingArguments as HfSeq2SeqTrainingArguments


@dataclass
class SwiftArgumentsMixin:
    # ckpt only save model
    only_save_model: bool = False


@dataclass
class TrainingArguments(SwiftArgumentsMixin, HfTrainingArguments):
    pass


@dataclass
class Seq2SeqTrainingArguments(SwiftArgumentsMixin,
                               HfSeq2SeqTrainingArguments):
    pass

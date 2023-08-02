# Copyright (c) Alibaba, Inc. and its affiliates.

from transformers import Seq2SeqTrainer as HfSeq2SeqTrainer
from transformers import Trainer as HfTrainer

from .mixin import PushToMsHubMixin, SwiftMixin


class Trainer(PushToMsHubMixin, SwiftMixin, HfTrainer):
    pass


class Seq2SeqTrainer(PushToMsHubMixin, SwiftMixin, HfSeq2SeqTrainer):
    pass

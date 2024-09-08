# Copyright (c) Alibaba, Inc. and its affiliates.
from transformers import Seq2SeqTrainer as HfSeq2SeqTrainer
from transformers import Trainer as HfTrainer

from .mixin import Seq2SeqTrainerMixin, SwiftMixin
from .push_to_ms import PushToMsHubMixin


class Trainer(PushToMsHubMixin, SwiftMixin, HfTrainer):
    pass


class Seq2SeqTrainer(Seq2SeqTrainerMixin, PushToMsHubMixin, SwiftMixin, HfSeq2SeqTrainer):
    pass

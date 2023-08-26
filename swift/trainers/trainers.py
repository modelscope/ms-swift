# Copyright (c) Alibaba, Inc. and its affiliates.

from transformers import Seq2SeqTrainer as HfSeq2SeqTrainer
from transformers import Trainer as HfTrainer
from transformers import trainer

from .mixin import PushToMsHubMixin, SwiftMixin
from .trainer_patch import DefaultFlowCallbackNew, ProgressCallbackNew


class Trainer(PushToMsHubMixin, SwiftMixin, HfTrainer):
    pass


class Seq2SeqTrainer(PushToMsHubMixin, SwiftMixin, HfSeq2SeqTrainer):
    pass


# monkey patch
trainer.DEFAULT_PROGRESS_CALLBACK = ProgressCallbackNew
trainer.DEFAULT_CALLBACKS = [DefaultFlowCallbackNew]

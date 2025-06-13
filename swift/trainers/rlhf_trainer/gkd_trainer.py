# Copyright (c) Alibaba, Inc. and its affiliates.
from collections import defaultdict
from contextlib import nullcontext
from typing import Optional, Union

import torch.nn as nn
from transformers import PreTrainedModel
from trl import GKDTrainer as HFGKDTrainer
from trl import SFTTrainer as HFSFTTrainer
from trl.models.utils import prepare_deepspeed

from ..mixin import SwiftMixin
from .rlhf_mixin import RLHFTrainerMixin

del HFGKDTrainer.__init__
del HFSFTTrainer.__init__


class GKDTrainer(RLHFTrainerMixin, SwiftMixin, HFGKDTrainer):

    def __init__(self, model: Optional[Union[PreTrainedModel, nn.Module, str]] = None, *_args, **kwargs):
        teacher_model = kwargs.pop('teacher_model', None)
        super().__init__(model, *_args, **kwargs)
        args = kwargs['args']
        self.lmbda = args.lmbda
        self.temperature = args.temperature
        self.seq_kd = args.seq_kd
        self.generation_config = model.generation_config
        self._metrics = {'train': defaultdict(list), 'eval': defaultdict(list)}
        self._total_train_tokens = 0
        if self.is_deepspeed_enabled:
            self.teacher_model = prepare_deepspeed(teacher_model, self.accelerator)
        else:
            self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)
        # Initialize activation offloading context
        args.activation_offloading = False  # TODO: remove
        if args.activation_offloading:
            from trl.models import get_act_offloading_ctx_manager
            self.maybe_activation_offload_context = get_act_offloading_ctx_manager(model=self.model)
        else:
            self.maybe_activation_offload_context = nullcontext()

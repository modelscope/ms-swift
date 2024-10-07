# Copyright (c) Alibaba, Inc. and its affiliates.
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
import torch.amp as amp
import torch.nn as nn
from transformers import trainer
from accelerate.utils import gather_object
from transformers import PreTrainedModel, PreTrainedTokenizer
from trl.trainer import PPOv2Trainer as HFPPOTrainer, PPOTrainer, PPOv2Config
from trl.trainer.utils import print_rich_table
from trl import AutoModelForCausalLMWithValueHead
from swift.trainers import PushToMsHubMixin, RLHFTrainerMixin, SwiftMixin

class PPOTrainer(RLHFTrainerMixin, PushToMsHubMixin, SwiftMixin, HFPPOTrainer):

    def __init__(self, 
                 model: AutoModelForCausalLMWithValueHead, 
                 ref_model: AutoModelForCausalLMWithValueHead,
                 *_args, 
                 **kwargs):
        
        kwargs['policy'] = model
        kwargs['ref_policy'] = ref_model
        super().__init__(model, ref_model, *_args, **kwargs)

    def train(self, *args, **kwargs):
        HFPPOTrainer.train(self)
def patched_init(self, **kwargs):
    kwargs_to_pop = ['model', 'model_init', 'compute_metrics', 'preprocess_logits_for_metrics']
    for kwarg in kwargs_to_pop:
        kwargs.pop(kwarg, None)
    kwargs['config'] = kwargs.pop('args')
    original_init(self, **kwargs)
original_init = HFPPOTrainer.__init__
HFPPOTrainer.__init__ = patched_init


from trl.trainer.ppov2_trainer import PolicyAndValueWrapper
def patched_init(self, policy, value_model) -> None:
    # super(): __class__ cell not found
    nn.Module.__init__(self)
    self.policy = policy
    self.value_model = value_model
    if hasattr(value_model, 'pretrained_model'):
        self.critic_backbone = getattr(value_model.pretrained_model, value_model.pretrained_model.base_model_prefix)
    else:
        self.critic_backbone = getattr(value_model, value_model.base_model_prefix)

PolicyAndValueWrapper.__init__ = patched_init

origin_forward = PolicyAndValueWrapper.forward
def patched_forward(self, **kwargs):
    
    output = self.critic_backbone(
        **kwargs,
    )
    if hasattr(self, 'score'):
        logits = self.value_model.score(output.hidden_states[-1])
    elif hasattr(self, 'value_head'):
        logits = self.value_model.value_head(output.hidden_states[-1])
    else:
        raise Exception('No value head found')

    return self.policy(**kwargs), logits

PolicyAndValueWrapper.forward = patched_forward
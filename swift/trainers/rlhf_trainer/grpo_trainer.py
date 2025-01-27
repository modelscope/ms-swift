# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Optional, Union

import torch.nn as nn
from peft import PeftModel
from transformers import PreTrainedModel, GenerationConfig, AutoTokenizer
from trl import GRPOTrainer as HFGRPOTrainer

from ..mixin import SwiftMixin
from .rlhf_mixin import RLHFTrainerMixin

del HFGRPOTrainer.__init__

class GRPOTrainer(RLHFTrainerMixin, SwiftMixin, HFGRPOTrainer):

    def __init__(self,
                 model: Optional[Union[PreTrainedModel, nn.Module]] = None,
                 ref_model: Optional[Union[PreTrainedModel, nn.Module]] = None,
                 reward_model:Optional[Union[PreTrainedModel, nn.Module]] = None,
                 *_args,
                 **kwargs):
        
        args = kwargs['args']
        
        processing_class = kwargs.get('tokenizer', None) or kwargs.get('processing_class', None)
        reward_processing_class = kwargs.get('reward_processing_class', None)
        if reward_processing_class is None:
            reward_processing_class = AutoTokenizer.from_pretrained(reward_model.config._name_or_path)

        if reward_processing_class.pad_token_id is None:
            reward_processing_class.pad_token = reward_processing_class.eos_token
        self.reward_processing_class = reward_processing_class # TODO

        self.reward_model = reward_model
        self.reward_model.config.pad_token_id = reward_processing_class.pad_token_id


        kwargs['']
        self.max_prompt_length = args.max_prompt_length
        self.num_generations = args.num_generations

        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            num_return_sequences=self.num_generations,
            pad_token_id=processing_class.pad_token_id,
        )
        model.warnings_issued["estimate_tokens"] = True
        
        self._metrics = {"kl": [], "reward": [], "reward_std": []}

        super().__init__(model, ref_model, *_args, **kwargs)

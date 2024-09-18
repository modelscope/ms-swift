# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Optional, Union, Dict, Tuple,Any, List
import torch
import torch.amp as amp

import torch.nn as nn
from transformers import PreTrainedModel
from trl import RewardTrainer as HFRewardTrainer
from contextlib import nullcontext

from swift.trainers import PushToMsHubMixin, RLHFTrainerMixin, SwiftMixin

del HFRewardTrainer.__init__


class RewardTrainer(RLHFTrainerMixin, PushToMsHubMixin, SwiftMixin, HFRewardTrainer):

    def __init__(self,model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,*_args,**kwargs):
        ref_model = kwargs.get('ref_model')
        assert ref_model is None, 'RM does not require a ref_model.'
        self.args = kwargs['args'] # use in `compute_loss` and `visuualize_samples`
        self.use_reward_data_collator = True # disable warning
        super().__init__(model, ref_model, *_args, **kwargs)
        
        
    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        compute_loss_context_manager = amp.autocast("cuda") if self._peft_has_been_casted_to_bf16 else nullcontext()
        with compute_loss_context_manager:
            loss, metrics = self.get_batch_loss_metrics(model, inputs)

        # Make sure to move the loss to the device the original accumulating loss is at back in the `Trainer` class:
        loss = loss.to(self.args.device)

        if return_outputs:
            return (loss, metrics)
        return loss
    
    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]]
        ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        model_kwargs = batch.copy()
        labels = model_kwargs.pop('labels', None)
        if self.is_encoder_decoder:
            model_kwargs['labels'] = labels
        
        outputs = model(**model_kwargs, use_cache=False)
        model_kwargs['labels'] = labels
        if outputs.logits.shape[1] != labels.shape[1]:
            # for llava, the model returns logits for the entire sequence, including the image tokens
            # (placed before the text tokens)
            outputs.logits = outputs.logits[:, -labels.shape[1]:]


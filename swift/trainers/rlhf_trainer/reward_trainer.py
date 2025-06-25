# Copyright (c) Alibaba, Inc. and its affiliates.
from collections import defaultdict
from typing import Any, Dict, Tuple, Union

import pandas as pd
import torch
import torch.nn as nn
from accelerate.utils import gather_object
from transformers import PreTrainedModel
from trl import RewardTrainer as HFRewardTrainer
from trl.trainer.utils import print_rich_table

from ..mixin import SwiftMixin
from .rlhf_mixin import RLHFTrainerMixin

del HFRewardTrainer.__init__


class RewardTrainer(RLHFTrainerMixin, SwiftMixin, HFRewardTrainer):

    def compute_loss(self,
                     model: Union[PreTrainedModel, nn.Module],
                     inputs: Dict[str, Union[torch.Tensor, Any]],
                     return_outputs=False,
                     num_items_in_batch=None) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        inputs.pop('labels', None)  # not use
        attention_mask = inputs['attention_mask']
        batch_size = attention_mask.shape[0] // 2
        rewards = model(**inputs).logits
        rewards_chosen, rewards_rejected = torch.split(rewards, batch_size, dim=0)
        if 'margin' in inputs:
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected - inputs['margin']).mean()
        else:
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
        if self.args.center_rewards_coefficient is not None:
            loss += self.args.center_rewards_coefficient * torch.mean((rewards_chosen + rewards_rejected)**2)
        # compat transformers>=4.46.*
        if num_items_in_batch is not None and self.model_accepts_loss_kwargs:
            loss = loss / self.args.gradient_accumulation_steps
        if return_outputs:
            return loss, {
                'rewards_chosen': rewards_chosen,
                'rewards_rejected': rewards_rejected,
            }
        return loss

    def visualize_samples(self, num_print_samples: int):
        """
        Visualize the reward model logits prediction

        Args:
            num_print_samples (`int`, defaults to `4`):
                The number of samples to print. Set to `-1` to print all samples.
        """
        eval_dataloader = self.get_eval_dataloader()
        table = defaultdict(list)
        for _, inputs in enumerate(eval_dataloader):
            _, logits, _ = self.prediction_step(self.model, inputs, prediction_loss_only=False)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            sequence_lengths = ((torch.eq(attention_mask, 0).int().argmax(-1) - 1) % attention_mask.shape[1]).tolist()
            text = [self.template.safe_decode(tokens[:sequence_lengths[i]]) for i, tokens in enumerate(input_ids)]
            batch_size = input_ids.shape[0] // 2
            chosen_text, rejected_text = text[:batch_size], text[batch_size:]
            table['chosen_text'].extend(gather_object(chosen_text))
            table['rejected_text'].extend(gather_object(rejected_text))
            table['logits'].extend(
                gather_object([[round(inner_item, 4) for inner_item in item] for item in logits.tolist()]))
            if 0 <= num_print_samples <= len(table['chosen_text']):
                break
        df = pd.DataFrame(table)
        if self.accelerator.process_index == 0:
            print_rich_table(df[:num_print_samples])
            if 'wandb' in self.args.report_to:
                import wandb

                if wandb.run is not None:
                    wandb.log({'completions': wandb.Table(dataframe=df)})

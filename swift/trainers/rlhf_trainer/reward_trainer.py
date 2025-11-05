# Copyright (c) Alibaba, Inc. and its affiliates.
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Dict, Tuple, Union

import pandas as pd
import torch
import torch.nn as nn
from accelerate.utils import gather_object
from transformers import PreTrainedModel
from trl import RewardTrainer as HFRewardTrainer
from trl.trainer.utils import print_rich_table

from swift.utils import get_logger
from ..mixin import SwiftMixin
from .rlhf_mixin import RLHFTrainerMixin

del HFRewardTrainer.__init__

logger = get_logger()


class RewardTrainer(RLHFTrainerMixin, SwiftMixin, HFRewardTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            from trl.models import get_act_offloading_ctx_manager
            if self.args.activation_offloading:
                self.maybe_activation_offload_context = get_act_offloading_ctx_manager(model=self.model)
            else:
                self.maybe_activation_offload_context = nullcontext()
        except ImportError:
            self.maybe_activation_offload_context = nullcontext()
        self._metrics = {'train': defaultdict(list), 'eval': defaultdict(list)}

    def compute_loss(self,
                     model: Union[PreTrainedModel, nn.Module],
                     inputs: Dict[str, Union[torch.Tensor, Any]],
                     return_outputs=False,
                     num_items_in_batch=None) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        margin = inputs.pop('margin', None)
        attention_mask = inputs['attention_mask']
        batch_size = attention_mask.shape[0] // 2
        rewards = model(**inputs).logits
        rewards_chosen, rewards_rejected = torch.split(rewards, batch_size, dim=0)
        if margin is not None:
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected - margin).mean()
        else:
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
        mode = 'train' if self.model.training else 'eval'
        if self.args.center_rewards_coefficient is not None:
            center_rewards_loss = self.args.center_rewards_coefficient * torch.mean(
                (rewards_chosen + rewards_rejected)**2)
            loss += center_rewards_loss
            self.custom_metrics[mode]['center_rewards_loss'].update(center_rewards_loss.detach())
        # metrics
        rewards_chosen, rewards_rejected = rewards_chosen.detach(), rewards_rejected.detach()
        self.custom_metrics[mode]['rewards/chosen'].update(rewards_chosen.mean())
        self.custom_metrics[mode]['rewards/rejected'].update(rewards_rejected.mean())
        self.custom_metrics[mode]['rewards/accuracies'].update((rewards_chosen > rewards_rejected).float().mean())
        self.custom_metrics[mode]['rewards/margins'].update((rewards_chosen - rewards_rejected).mean())
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
            try:
                print_rich_table(df[:num_print_samples])
            except Exception as e:
                logger.error(e)
            if 'wandb' in self.args.report_to:
                import wandb

                if wandb.run is not None:
                    wandb.log({'completions': wandb.Table(dataframe=df)})

            if 'swanlab' in self.args.report_to:
                import swanlab
                if swanlab.get_run() is not None:
                    swanlab_table = swanlab.echarts.Table()
                    swanlab_table.add(headers=df.columns.tolist(), rows=df.values.tolist())
                    swanlab.log({'completions': swanlab_table})

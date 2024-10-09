# Copyright (c) Alibaba, Inc. and its affiliates.
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd
import torch
import torch.nn as nn
from accelerate.utils import gather_object
from transformers import PreTrainedModel
from trl import RewardTrainer as HFRewardTrainer
from trl.trainer.utils import print_rich_table

from swift.trainers import PushToMsHubMixin, RLHFTrainerMixin, SwiftMixin

del HFRewardTrainer.__init__


class RewardTrainer(RLHFTrainerMixin, PushToMsHubMixin, SwiftMixin, HFRewardTrainer):

    def __init__(self, model: Optional[Union[PreTrainedModel, nn.Module, str]] = None, *_args, **kwargs):
        ref_model = kwargs.pop('ref_model')
        assert ref_model is None, 'RM does not require a ref_model.'
        self.args = kwargs['args']
        self.use_reward_data_collator = True  # disable warning
        super().__init__(model, *_args, **kwargs)

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        model_kwargs = inputs.copy()
        labels = model_kwargs.pop('labels', None)
        if self.is_encoder_decoder:
            model_kwargs['labels'] = labels
        _, _, values = model(**model_kwargs, use_cache=False, return_dict=True)
        batch_size = model_kwargs['input_ids'].shape[0] // 2
        chosen_masks, rejected_masks = torch.split(model_kwargs['attention_mask'], batch_size, dim=0)
        chosen_rewards, rejected_rewards = torch.split(values, batch_size, dim=0)
        chosen_scores = chosen_rewards.gather(dim=-1, index=(chosen_masks.sum(dim=-1, keepdim=True) - 1)).squeeze()
        rejected_scores = rejected_rewards.gather(
            dim=-1, index=(rejected_masks.sum(dim=-1, keepdim=True) - 1)).squeeze()
        loss = -torch.nn.functional.logsigmoid(chosen_scores.float() - rejected_scores.float()).mean().to(
            self.args.device)
        if return_outputs:
            return loss, {
                'rewards_chosen': chosen_rewards,
                'rewards_rejected': rejected_rewards,
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
            batch_size = inputs['input_ids'].shape[0] // 2
            text = self.tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)
            chosen_text, rejected_text = text[:batch_size], text[batch_size:]
            table['chosen_text'].extend(gather_object(chosen_text))
            table['rejected_text'].extend(gather_object(rejected_text))
            table['logits'].extend(
                gather_object([[round(inner_item, 4) for inner_item in item] for item in logits.tolist()]))
            if num_print_samples >= 0 and len(table['chosen_text']) >= num_print_samples:
                break
        df = pd.DataFrame(table)
        if self.accelerator.process_index == 0:
            print_rich_table(df[:num_print_samples])
            if 'wandb' in self.args.report_to:
                import wandb

                if wandb.run is not None:
                    wandb.log({'completions': wandb.Table(dataframe=df)})

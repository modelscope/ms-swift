# Copyright (c) Alibaba, Inc. and its affiliates.
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from peft import PeftModel
from transformers import PreTrainedModel
from trl import KTOTrainer as HFKTOTrainer

from swift.llm import LLMDataset
from swift.trainers import PushToMsHubMixin, RLHFTrainerMixin, SwiftMixin

del HFKTOTrainer.__init__


def _add_kl_dataset(dataset: LLMDataset) -> LLMDataset:
    raw_dataset = dataset.data
    new_dataset = []
    for i in range(len(raw_dataset)):
        raw_data, kl_raw_data = raw_dataset[i], raw_dataset[(i + 1) % len(dataset)]
        KL_input_ids = raw_data['prompt_input_ids'] + kl_raw_data['answer_input_ids']
        KL_labels = raw_data['prompt_labels'] + kl_raw_data['answer_labels']
        new_dataset.append({
            'input_ids': raw_data['input_ids'],
            'labels': raw_data['labels'],
            'KL_input_ids': KL_input_ids,
            'KL_labels': KL_labels,
            'label': raw_data['label']
        })
    return LLMDataset(new_dataset)


class KTOTrainer(RLHFTrainerMixin, PushToMsHubMixin, SwiftMixin, HFKTOTrainer):

    def __init__(self,
                 model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
                 ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
                 *_args,
                 **kwargs):
        args = kwargs['args']
        args.disable_dropout = True
        self.desirable_weight = args.desirable_weight
        self.undesirable_weight = args.undesirable_weight
        self.precompute_ref_log_probs = args.precompute_ref_log_probs
        self.is_peft_model = isinstance(model, PeftModel)

        self.ref_adapter_name = args.ref_adapter_name
        # KL datasets
        train_dataset, eval_dataset = kwargs['train_dataset'], kwargs['eval_dataset']
        random_state = np.random.RandomState(args.data_seed)
        random_state.shuffle(train_dataset.data)
        random_state.shuffle(eval_dataset.data)
        train_dataset = _add_kl_dataset(train_dataset)
        eval_dataset = _add_kl_dataset(eval_dataset)
        label = train_dataset['label']
        num_desirable = max(sum(label), 1)
        num_undesirable = max(len(label) - num_desirable, 1)  # "label" is binary

        if num_desirable != num_undesirable:
            # The lower and upper bounds come from Eq. (8) of https://huggingface.co/papers/2402.01306
            des_weight_lower_bound = round((num_undesirable * self.undesirable_weight / num_desirable) * 1, 2)
            des_weight_upper_bound = round((num_undesirable * self.undesirable_weight / num_desirable) * 1.33, 2)
            und_weight_lower_bound = round((num_desirable * self.desirable_weight / num_undesirable) / 1.33, 2)
            und_weight_upper_bound = round((num_desirable * self.desirable_weight / num_undesirable) / 1, 2)

            des_weight_in_range = des_weight_lower_bound <= self.desirable_weight <= des_weight_upper_bound
            und_weight_in_range = und_weight_lower_bound <= self.undesirable_weight <= und_weight_upper_bound

            if not (des_weight_in_range or und_weight_in_range):
                warnings.warn(
                    f"""
            You have different amounts of desirable/positive and undesirable/negative examples but the
            weights on the desirable and undesirable losses don't seem to be in an ideal range. Based
            on your data, we recommend EITHER desirable_weight in [{des_weight_lower_bound}, '{des_weight_upper_bound}]
            or undesirable_weight in [{und_weight_lower_bound}, {und_weight_upper_bound}] (but NOT BOTH).
            See the documentation on how to optimally set these weights.""", UserWarning)
        kwargs['train_dataset'], kwargs['eval_dataset'] = train_dataset, eval_dataset
        super().__init__(model, ref_model, *_args, **kwargs)

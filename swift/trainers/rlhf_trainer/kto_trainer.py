# Copyright (c) Alibaba, Inc. and its affiliates.
import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch.nn as nn
from datasets import Dataset as HfDataset
from peft import PeftModel
from transformers import PreTrainedModel
from trl import KTOTrainer as HFKTOTrainer

from swift.llm import RowPreprocessor
from swift.utils import get_dist_setting, get_logger
from ..mixin import SwiftMixin
from .rlhf_mixin import RLHFTrainerMixin

logger = get_logger()

del HFKTOTrainer.__init__
del HFKTOTrainer.get_batch_samples


class KTOTrainer(RLHFTrainerMixin, SwiftMixin, HFKTOTrainer):

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
        if hasattr(args, 'loss_type'):
            self.loss_type = args.loss_type
        else:
            self.loss_type = 'kto'

        self.ref_adapter_name = None
        # Not all losses require a KL calculation
        self.calculate_KL = True
        if self.loss_type in ['apo_zero_unpaired']:
            self.calculate_KL = False
        train_dataset, eval_dataset = kwargs['train_dataset'], kwargs['eval_dataset']
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
                logger.info(f'desirable_weight: {self.desirable_weight}, undesirable_weight: {self.undesirable_weight}')
                warnings.warn(
                    f"""
            You have different amounts of desirable/positive and undesirable/negative examples but the
            weights on the desirable and undesirable losses don't seem to be in an ideal range. Based
            on your data, we recommend EITHER desirable_weight in [{des_weight_lower_bound}, '{des_weight_upper_bound}]
            or undesirable_weight in [{und_weight_lower_bound}, {und_weight_upper_bound}] (but NOT BOTH).
            See the documentation on how to optimally set these weights.""", UserWarning)
        kwargs['train_dataset'], kwargs['eval_dataset'] = train_dataset, eval_dataset
        super().__init__(model, ref_model, *_args, **kwargs)

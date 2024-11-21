# Copyright (c) Alibaba, Inc. and its affiliates.
import warnings
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
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
        super().__init__(model, ref_model, *_args, **kwargs)

    def forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        for key in [
                'completion_input_ids', 'completion_attention_mask', 'KL_completion_input_ids',
                'KL_completion_attention_mask'
        ]:
            if key not in batch:
                batch[key] = None
        is_kl = True

        def _add_data_hook(model: nn.Module, args, kwargs):
            nonlocal is_kl
            _data = batch['_data']
            if is_kl:
                data = [{k[len('KL_'):]: v
                         for k, v in inputs.items() if k.startswith('KL_completion_')} for inputs in _data]
            else:
                data = [{k: v for k, v in inputs.items() if k.startswith('completion_')} for inputs in _data]
            is_kl = not is_kl
            kwargs['_data'] = data
            return args, kwargs

        def _remove_prefix_hook(model: nn.Module, args, kwargs):
            kwargs = {k.replace('completion_', ''): v for k, v in kwargs.items()}
            return args, kwargs

        @contextmanager
        def _patch_model_call():
            handles = [model.register_forward_pre_hook(_remove_prefix_hook, with_kwargs=True)]
            if '_data' in batch:
                handles.append(model.register_forward_pre_hook(_add_data_hook, with_kwargs=True, prepend=True))

            yield
            for handle in handles:
                handle.remove()

        with _patch_model_call():
            return super().forward(model, batch)

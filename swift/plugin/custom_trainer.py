# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from swift.trainers import Trainer


class SequenceClassificationTrainer(Trainer):
    """A trainer for text-classification task"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_names = ['labels']

    def compute_loss(self, model, inputs, return_outputs=None, **kwargs):
        if 'label' in inputs:
            inputs['labels'] = torch.tensor(inputs.pop('label')).unsqueeze(1)
        return super().compute_loss(model, inputs, return_outputs=return_outputs)

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if 'label' in inputs:
            inputs['labels'] = torch.tensor(inputs.pop('label')).unsqueeze(1)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs)


# To train sequence classification tasks, uncomment this.
def custom_trainer_class(trainer_mapping, training_args_mapping):
    # trainer_mapping['train'] = 'swift.plugin.custom_trainer.SequenceClassificationTrainer'
    pass

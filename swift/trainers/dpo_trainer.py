from contextlib import contextmanager
from typing import Dict, List, Tuple, Union

import torch
from torch import nn
from trl import DPOTrainer as HFDPOTrainer

from swift.utils import get_logger
from .mixin import RLHFTrainerMixin, SwiftMixin
from .push_to_ms import PushToMsHubMixin

logger = get_logger()

del HFDPOTrainer.__init__


class DPOTrainer(RLHFTrainerMixin, PushToMsHubMixin, SwiftMixin, HFDPOTrainer):

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:

        model_kwargs = batch.copy()
        labels = model_kwargs.pop('labels', None)
        outputs = model(
            **model_kwargs,
            use_cache=False,
        )
        model_kwargs['labels'] = labels
        model_kwargs['chosen_labels'] = torch.zeros(model_kwargs['input_ids'].shape[0] // 2)  # just get shape
        for key in ['input_ids', 'attention_mask', 'labels']:
            model_kwargs[f'concatenated_{key}'] = model_kwargs.pop(key)
        @contextmanager
        def _patch_concatenated_forward():
            _old_concatenated_inputs = self.concatenated_inputs
            _old_model_call = model.__class__.__call__
            self.concatenated_inputs = lambda *args, **kwargs: model_kwargs
            model.__class__.__call__ = lambda *args, **kwargs: outputs
            yield
            self.concatenated_inputs = _old_concatenated_inputs
            model.__class__.__call__ = _old_model_call

        with _patch_concatenated_forward():
            return super().concatenated_forward(model, model_kwargs)

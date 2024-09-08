from contextlib import contextmanager
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

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
        model_kwargs.pop('concatenated_labels', None)
        outputs = model(
            **model_kwargs,
            use_cache=False,
        )

        @contextmanager
        def _patch_concatenated_forward():
            _old_concatenated_inputs = self.concatenated_inputs
            _old_model_call = model.__class__.__call__
            self.concatenated_inputs = lambda *args, **kwargs: batch
            model.__class__.__call__ = lambda *args, **kwargs: outputs
            yield
            self.concatenated_inputs = _old_concatenated_inputs
            model.__class__.__call__ = _old_model_call

        with _patch_concatenated_forward():
            batch['concatenated_input_ids'] = batch.pop('input_ids')
            batch['concatenated_attention_mask'] = batch.pop('attention_mask')
            return super().concatenated_forward(model, batch)

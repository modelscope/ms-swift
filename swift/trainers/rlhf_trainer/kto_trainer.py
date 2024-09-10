# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Dict, List, Union, Tuple
import torch.nn as nn
import torch
from trl import KTOTrainer as HFKTOTrainer
from swift.trainers import PushToMsHubMixin, RLHFTrainerMixin, SwiftMixin


del HFKTOTrainer.__init__

class KTOTrainer(RLHFTrainerMixin, PushToMsHubMixin, SwiftMixin, HFKTOTrainer):

    def forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        return super().forward(model, batch)


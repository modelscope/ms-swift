# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING

import torch

from .base import PeftTuner

if TYPE_CHECKING:
    from swift.arguments import SftArguments


class DummyTuner(PeftTuner):

    @staticmethod
    def prepare_model(args: 'SftArguments', model: torch.nn.Module) -> torch.nn.Module:
        return model

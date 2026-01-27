# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING

import torch
from peft import IA3Config, get_peft_model

from swift.model import ModelKeys
from swift.utils import find_all_linears
from .base import PeftTuner

if TYPE_CHECKING:
    from swift.arguments import SftArguments


# Here gives a simple example of IA3
class IA3Tuner(PeftTuner):

    @staticmethod
    def prepare_model(args: 'SftArguments', model: torch.nn.Module) -> torch.nn.Module:
        model_arch: ModelKeys = model.model_meta.model_arch
        ia3_config = IA3Config(
            target_modules=find_all_linears(model), feedforward_modules='.*' + model_arch.mlp.split('{}.')[1] + '.*')
        return get_peft_model(model, ia3_config)

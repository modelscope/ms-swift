# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING

import numpy as np
import torch

from .base import TrainerCallback

if TYPE_CHECKING:
    from swift.trainers import TrainingArguments, Trainer


class LISACallback(TrainerCallback):

    def __init__(self, args: 'TrainingArguments', trainer: 'Trainer'):
        assert args.tuner_type == 'full', 'LISA only supports full parameter training.'
        super().__init__(args, trainer)
        self.n_layers = args.lisa_activated_layers
        self.step_interval = args.lisa_step_interval
        self.model = self.trainer.model
        layers_name = None
        layers = None
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.ModuleList):
                layers_name = name
                layers = module
                break
        assert layers_name is not None
        self.layers_attribute = layers_name
        self.total_layers = len(layers)

        # Freeze all layers upon initialization
        self.freeze_all_layers()
        self.active_layers_indices = []
        self.switch_active_layers()

    def freeze_all_layers(self):
        layers = self.model.get_submodule(self.layers_attribute)
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False

    def on_step_begin(self, args, state, control, **kwargs):
        # Check if it's time to switch active layers, including at step 0
        if state.global_step % self.step_interval == 0 or state.global_step == 1:
            self.switch_active_layers()

    def switch_active_layers(self):
        # First, disable gradients for all layers
        self.freeze_all_layers()

        # Randomly select n_layers to activate
        layers = self.model.get_submodule(self.layers_attribute)
        self.active_layers_indices = np.random.choice(range(self.total_layers), self.n_layers, replace=False)
        # Enable gradients only for the selected layers
        for idx in self.active_layers_indices:
            for param in layers[idx].parameters():
                param.requires_grad = True

# Copyright (c) Alibaba, Inc. and its affiliates.
import types

import numpy as np
import torch
from transformers import TrainerCallback

from swift.utils import get_logger

logger = get_logger()


class TrainerAdapterCallback(TrainerCallback):

    def __init__(self, args):
        self.global_step = 0
        self.args = args

    # offload original_modules to cpu, to save memory
    def on_train_begin(self, _args, state, control, **kwargs):
        model = kwargs['model']
        if self.args.train_type == 'adalora':
            model.peft_config['default'].total_step = state.max_steps

            def zero_grad(_self, *args, **kwargs):
                _self.update_and_allocate(self.global_step + 1)
                _self._zero_grad(*args, **kwargs)

            model._zero_grad = model.zero_grad
            model.zero_grad = types.MethodType(zero_grad, model)

    def on_step_end(self, _args, state, control, **kwargs):
        if self.args.train_type == 'adalora':
            self.global_step = state.global_step


class DynamicLayerActivationCallback(TrainerCallback):

    def __init__(self, n_layers: int, step_interval: int, model: torch.nn.Module):
        super().__init__()
        self.n_layers = n_layers
        self.step_interval = step_interval
        self.model = model
        layers_name = None
        layers = None
        for name, module in model.named_modules():
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

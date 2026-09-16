# Copyright (c) ModelScope Contributors. All rights reserved.
import numpy as np
import torch
from accelerate.utils import extract_model_from_parallel
from typing import TYPE_CHECKING

from .base import TrainerCallback

if TYPE_CHECKING:
    from swift.trainers import Trainer, TrainingArguments


class LISACallback(TrainerCallback):

    def __init__(self, args: 'TrainingArguments', trainer: 'Trainer'):
        assert args.tuner_type == 'full', 'LISA only supports full parameter training.'
        super().__init__(args, trainer)
        self.n_layers = args.lisa_activated_layers
        self.step_interval = args.lisa_step_interval

    def on_train_begin(self, args, state, control, **kwargs):
        # Wait until the model exists and the optimizer has registered all layers.
        self.model = extract_model_from_parallel(kwargs['model'], keep_torch_compile=False)
        model_meta = getattr(self.model, 'model_meta', None)
        model_arch = getattr(model_meta, 'model_arch', None)
        module_list = getattr(model_arch, 'module_list', None)
        layer_prefixes = getattr(model_arch, 'language_model', None)
        if module_list:
            layer_prefixes = [module_list]
        if getattr(model_meta, 'is_multimodal', False) and not layer_prefixes:
            raise ValueError('LISA requires model_arch.module_list or language_model for multimodal models.')
        excluded_prefixes = [
            prefix for key in ('vision_tower', 'aligner', 'generator') for prefix in getattr(model_arch, key, [])
        ]
        layer_names = []
        for name, module in self.model.named_modules():
            if not isinstance(module, torch.nn.ModuleList) or (module_list and name != module_list):
                continue
            if layer_prefixes and not any(name == prefix or name.startswith(f'{prefix}.') for prefix in layer_prefixes):
                continue
            if any(name == prefix or name.startswith(f'{prefix}.') for prefix in excluded_prefixes):
                continue
            # Nested lists (e.g. MoE experts) belong to the enclosing layer stack.
            if not any(not parent or name.startswith(f'{parent}.') for parent in layer_names):
                layer_names.append(name)
        if len(layer_names) != 1:
            raise ValueError('LISA could not uniquely identify the layer list. '
                             f'Set model_arch.module_list explicitly. Candidates: {layer_names}')
        self.layers_attribute = layer_names[0]
        self.total_layers = len(self.model.get_submodule(self.layers_attribute))

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

# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field

import torch
from torch import nn
from transformers import TrainerCallback
import numpy as np
from swift.utils.logger import get_logger
from .utils import SwiftAdapter, SwiftConfig, SwiftOutput

logger = get_logger()


@dataclass
class LisaConfig(SwiftConfig):
    """
    The configuration class for the Lisa module.

    See https://arxiv.org/abs/2403.17919

    Args:
        lisa_activated_layers(`int`): The number of activated layers in LISA.
        lisa_step_interval(`int`): The step interval of LISA
    """
    lisa_activated_layers: int = field(
        default=2,
        metadata={
            "help": "The number of activated layers in LISA"
        }
    )
    lisa_step_interval: int = field(
        default=20,
        metadata={
            "help": "The step interval of LISA"
        }
    )

    def __post_init__(self):
        from .mapping import SwiftTuners
        self.swift_type = SwiftTuners.LISA


class LLaMAPro(SwiftAdapter):

    @staticmethod
    def prepare_model(model: nn.Module, config: LisaConfig,
                      adapter_name: str) -> SwiftOutput:
        """Prepare a model with `LisaConfig`"""

        class DynamicLayerActivationCallback(TrainerCallback):
            def __init__(self, n_layers, step_interval, model):
                super().__init__()
                self.n_layers = n_layers
                self.step_interval = step_interval
                self.model = model
                # Determine the way to access layers based on the model type
                if self.model.__class__.__name__ == 'LlamaForCausalLM':
                    self.layers_attribute = 'model.model.layers'  # Layer access path for LlamaForCausalLM
                else:
                    self.layers_attribute = 'model.transformer.h'  # General access path
                self.total_layers = len(
                    eval('self.' + self.layers_attribute))  # Dynamically execute to get the number of layers

                # Freeze all layers upon initialization
                self.freeze_all_layers()
                self.active_layers_indices = []

            def freeze_all_layers(self):
                layers = eval('self.' + self.layers_attribute)  # Dynamically execute to get layers
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
                layers = eval('self.' + self.layers_attribute)  # Re-fetch layer references
                self.active_layers_indices = np.random.choice(range(self.total_layers), self.n_layers,
                                                              replace=False)
                print(f"Activating layers at indices: {self.active_layers_indices} for the next steps.")

                # Enable gradients only for the selected layers
                for idx in self.active_layers_indices:
                    for param in layers[idx].parameters():
                        param.requires_grad = True

        # Instantiate the callback
        dynamic_layer_activation_callback = DynamicLayerActivationCallback(
            n_layers=config.lisa_activated_layers,  # Number of layers to activate
            step_interval=config.lisa_step_interval,  # Step interval to update active layers
            model=model.get_backend_model()
        )

        def state_dict_callback(state_dict, adapter_name):
            return state_dict

        def mark_trainable_callback(model):
            return

        return SwiftOutput(config, state_dict_callback,
                           mark_trainable_callback)

    @staticmethod
    def activate_adapter(module: torch.nn.Module,
                         adapter_name: str,
                         activate: bool,
                         offload: str = None):
        for sub_module in module.modules():
            if isinstance(sub_module, torch.nn.Embedding):
                sub_module.nef_activated = activate

    @staticmethod
    def has_additional_modules():
        return True

# Copyright (c) Alibaba, Inc. and its affiliates.
import re
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from types import MethodType
from typing import Dict, List, Optional

import torch
from torch import nn

from swift import get_logger
from .utils import ActivationMixin, SwiftAdapter, SwiftConfig, SwiftOutput
from ..llm.utils.utils import is_pyreft_available

logger = get_logger()


@dataclass
class LoReftConfig(SwiftConfig):
    """
    Train a model with LoReft.
    Paper: https://arxiv.org/pdf/2404.03592

    Args:
        model_type(`str`): The model_type to find down_proj/layers.
        layers (`layer numbers`): The layer number to inject.
        r(`int`): The rank of LoReft.
    """

    model_type: Optional[str] = None
    layers: Optional[List[int]] = None
    r: int = 4

    def __post_init__(self):
        from .mapping import SwiftTuners
        self.swift_type = SwiftTuners.LOREFT


class LoReft(SwiftAdapter):

    @staticmethod
    def prepare_model(model: nn.Module, config: LoReftConfig, adapter_name: str):
        if not is_pyreft_available():
            raise ImportError(f'Please install pyreft before using LoReFT: '
                              f'`pip install git+https://github.com/stanfordnlp/pyreft.git`')

        import pyreft
        from pyreft import ReftModel
        from pyreft.interventions import LowRankRotateLayer, LoreftIntervention

        def __getattr__(self, name: str):
            try:
                return super(ReftModel, self).__getattr__(name)
            except AttributeError:
                return getattr(self.model, name)
        
        ReftModel.__getattr__ = __getattr__

        def forward(self, x):
            self.to(x.device)
            return self.forward_origin(x)
        
        def forward2(self, base, source=None, subspaces=None):
            self.to(base.device)
            return self.forward_origin(base, source, subspaces)

        if not hasattr(LowRankRotateLayer, 'forward_origin'):
            LowRankRotateLayer.forward_origin = LowRankRotateLayer.forward
            LowRankRotateLayer.forward = forward
            LoreftIntervention.forward_origin = LoreftIntervention.forward
            LoreftIntervention.forward = forward2

        model_key_mapping = LoReft._get_model_key_mapping(config.model_type, config)
        logger.info(f'Applying LoReft to module: {model_key_mapping.module_list}')
        module_list: nn.ModuleList = model.get_submodule(model_key_mapping.module_list)
        representations = []
        for idx, layer in enumerate(module_list):
            if config.layers and idx not in config.layers:
                continue
            intervention_config = {
                "layer": idx, "component": model_key_mapping.module_list + f'[{idx}].output',
                "low_rank_dimension": config.r,
                "intervention": pyreft.LoreftIntervention(embed_dim=model.config.hidden_size,
                                                          low_rank_dimension=config.r)
            }
            representations.append(intervention_config)

        reft_config = pyreft.ReftConfig(representations=representations)
        reft_model = pyreft.get_reft_model(model, reft_config, set_device=False)
        reft_model.loreft_config = reft_model.config
        reft_model.config = reft_model.model.config

        def _pre_forward_hook(module, args, kwargs):
            # run intervened forward pass
            unit_locations = None
            if "intervention_locations" in kwargs:
                if kwargs["intervention_locations"].dim() == 3:
                    unit_locations = {"sources->base": (
                        None,
                        kwargs["intervention_locations"].permute(1, 0, 2).tolist()
                    )}
                else:
                    # this is dummy for lora only baseline
                    unit_locations = {"sources->base": (None, 0)}
            kwargs = {
                'base': {
                    "input_ids": kwargs["input_ids"],
                    "attention_mask": kwargs["attention_mask"]
                },
                'unit_locations': unit_locations,
                'labels': kwargs["labels"],
                'subspaces': kwargs["subspaces"].permute(1, 0, 2).tolist() if "subspaces" in kwargs else None
            }
            return args, kwargs

        def _post_forward_hook(module, inputs, outputs):
            return outputs[1]

        reft_model.register_forward_pre_hook(_pre_forward_hook, with_kwargs=True)
        reft_model.register_forward_hook(_post_forward_hook, with_kwargs=True)

        def save_callback(swift_model, model_dir, adapter_name):
            reft_model.save_intervention(save_directory=model_dir)
        
        def mark_trainable_callback(model):
            return

        def load_callback(swift_model, model_dir, adapter_name):
            reft_model.load_intervention(model_dir)

        return SwiftOutput(
            model=reft_model,
            config=config,
            mark_trainable_callback=mark_trainable_callback,
            save_callback=save_callback,
            load_callback=load_callback)

    @staticmethod
    def has_additional_modules():
        return True

    def activate_adapter(module: torch.nn.Module, adapter_name: str, activate: bool, offload: str = None):
        pass

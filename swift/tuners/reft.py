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
        model_key_mapping = LoReft._get_model_key_mapping(config.model_type, config)
        logger.info(f'Applying LoReft to module: {model_key_mapping.module_list}')
        module_list: nn.ModuleList = model.get_submodule(model_key_mapping.module_list)
        down_proj = model_key_mapping.down_proj
        representations = []
        for idx, layer in enumerate(module_list):
            intervention_config = {
                "layer": idx, "component": down_proj.replace('.{}.', f'[{idx}].'),
                "low_rank_dimension": config.r,
                "intervention": pyreft.LoreftIntervention(embed_dim=model.config.hidden_size,
                                                          low_rank_dimension=config.r)
            }
            representations.append(intervention_config)

        reft_config = pyreft.ReftConfig(representations=representations)
        reft_model = pyreft.get_reft_model(model, reft_config)

        def save_callback(swift_model, model_dir, adapter_name):
            reft_model.save(model_dir)

        def load_callback(swift_model, model_dir, adapter_name):
            swift_model.base_model = ReftModel.load(model_dir)

        return SwiftOutput(
            model=reft_model,
            config=config,
            save_callback=save_callback,
            load_callback=load_callback)

    @staticmethod
    def activate_adapter(module: torch.nn.Module, adapter_name: str, activate: bool, offload: str = None):
        name_list = [name for name, _ in module.named_modules(remove_duplicate=False)]
        for name in name_list:
            sub_module: nn.Module = module.get_submodule(name)
            if re.fullmatch(f'.*_part_{adapter_name}$', name):
                sub_module.activated = activate
                SwiftAdapter.save_memory(sub_module, adapter_name, name, activate, offload)

    @staticmethod
    def has_additional_modules():
        return True

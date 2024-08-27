# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass
from types import MethodType
from typing import List, Literal, Optional

import json
import torch
from torch import nn

from swift import get_logger
from .utils import SwiftAdapter, SwiftConfig, SwiftOutput

logger = get_logger()


@dataclass
class ReftConfig(SwiftConfig):
    """
    Train a model with Reft.
    Paper: https://arxiv.org/pdf/2404.03592

    Args:
        model_type(`Optional[str]`): The model_type to find down_proj/layers.
        layer_key(`Optional[str]`): Manually specify the layer key, for example `language_model.layers`.
        layers (`Optional[List[int]]`): The layer number to inject.
        r(`int`): The rank of Reft.
        intervention_type (`Literal['NoreftIntervention', 'LoreftIntervention',
                        'ConsreftIntervention', 'LobireftIntervention',
                        'DireftIntervention', 'NodireftIntervention']`): The intervention type,
                        default LoreftIntervention
        args (`Optional[str]`): Other reft_args in json-string format
    """

    model_type: Optional[str] = None
    layer_key: Optional[str] = None
    layers: Optional[List[int]] = None
    r: int = 4
    intervention_type: Literal['NoreftIntervention', 'LoreftIntervention', 'ConsreftIntervention',
                               'LobireftIntervention', 'DireftIntervention',
                               'NodireftIntervention'] = 'LoreftIntervention'
    args: Optional[str] = None

    def __post_init__(self):
        from .mapping import SwiftTuners
        self.swift_type = SwiftTuners.REFT
        if self.args:
            self.args = json.loads(self.args)
        else:
            self.args = {}


class Reft(SwiftAdapter):

    @staticmethod
    def prepare_model(model: nn.Module, config: ReftConfig, adapter_name: str):
        from swift.llm.utils.utils import is_pyreft_available
        if not is_pyreft_available():
            raise ImportError('Please install pyreft before using ReFT: '
                              '`pip install git+https://github.com/stanfordnlp/pyreft.git`')

        import pyreft
        from pyreft import ReftModel
        from pyreft.interventions import LowRankRotateLayer
        from pyreft import (
            NoreftIntervention,
            LoreftIntervention,
            ConsreftIntervention,
            LobireftIntervention,
            DireftIntervention,
            NodireftIntervention,
        )

        intervention_mapping = {
            'NoreftIntervention': NoreftIntervention,
            'LoreftIntervention': LoreftIntervention,
            'ConsreftIntervention': ConsreftIntervention,
            'LobireftIntervention': LobireftIntervention,
            'DireftIntervention': DireftIntervention,
            'NodireftIntervention': NodireftIntervention,
        }

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
            NoreftIntervention.forward_origin = NoreftIntervention.forward
            NoreftIntervention.forward = forward2
            LoreftIntervention.forward_origin = LoreftIntervention.forward
            LoreftIntervention.forward = forward2
            ConsreftIntervention.forward_origin = ConsreftIntervention.forward
            ConsreftIntervention.forward = forward2
            LobireftIntervention.forward_origin = LobireftIntervention.forward
            LobireftIntervention.forward = forward2
            DireftIntervention.forward_origin = DireftIntervention.forward
            DireftIntervention.forward = forward2
            NodireftIntervention.forward_origin = NodireftIntervention.forward
            NodireftIntervention.forward = forward2

        module_list_key = config.layer_key
        if module_list_key is None:
            model_key_mapping = Reft.get_model_key_mapping(config.model_type, config)
            module_list_key = model_key_mapping.module_list
        logger.info(f'Applying Reft to module: {module_list_key}')
        module_list: nn.ModuleList = model.get_submodule(module_list_key)
        representations = []
        for idx, layer in enumerate(module_list):
            if config.layers and idx not in config.layers:
                continue
            intervention_config = {
                'layer':
                idx,
                'component':
                module_list_key + f'[{idx}].output',
                'low_rank_dimension':
                config.r,
                'intervention':
                intervention_mapping[config.intervention_type](
                    embed_dim=model.config.hidden_size, low_rank_dimension=config.r, **config.args)
            }
            representations.append(intervention_config)

        reft_config = pyreft.ReftConfig(representations=representations)
        reft_model = pyreft.get_reft_model(model, reft_config, set_device=False)
        reft_model.reft_config = reft_model.config
        reft_model.config = reft_model.model.config

        def _pre_forward_hook(module, args, kwargs):
            if 'base' in kwargs:
                return args, kwargs

            if 'input_ids' not in kwargs:
                raise ValueError('Input does not contain `input_ids`, maybe the model does not support ReFT.')
            # run intervened forward pass
            unit_locations = None
            if 'intervention_locations' in kwargs:
                if kwargs['intervention_locations'].dim() == 3:
                    unit_locations = {
                        'sources->base': (None, kwargs['intervention_locations'].permute(1, 0, 2).tolist())
                    }
                else:
                    # this is dummy for lora only baseline
                    unit_locations = {'sources->base': (None, 0)}
            kwargs = {
                'base': {
                    'input_ids': kwargs['input_ids'],
                    'attention_mask': kwargs['attention_mask']
                },
                'unit_locations': unit_locations,
                'labels': kwargs['labels'],
                'subspaces': kwargs['subspaces'].permute(1, 0, 2).tolist() if 'subspaces' in kwargs else None
            }
            return args, kwargs

        def _post_forward_hook(module, args, kwargs, outputs):
            return outputs[1]

        def _generate(self, **kwargs):
            # run intervened forward pass
            unit_locations = None
            if 'intervention_locations' in kwargs:
                if kwargs['intervention_locations'].dim() == 3:
                    unit_locations = {
                        'sources->base': (None, kwargs['intervention_locations'].permute(1, 0, 2).tolist())
                    }
                else:
                    # this is dummy for lora only baseline
                    unit_locations = {'sources->base': (None, 0)}

            _kwargs = {
                'base': {
                    'input_ids': kwargs.pop('input_ids'),
                    'attention_mask': kwargs.pop('attention_mask')
                },
                'unit_locations': unit_locations,
                'subspaces': kwargs.pop('subspaces').permute(1, 0, 2).tolist() if 'subspaces' in kwargs else None
            }
            _kwargs = {**_kwargs, **kwargs}
            return self.generate_origin(**_kwargs)[1]

        reft_model.generate_origin = reft_model.generate
        reft_model.generate = MethodType(_generate, reft_model)
        reft_model.register_forward_pre_hook(_pre_forward_hook, with_kwargs=True)
        reft_model.register_forward_hook(_post_forward_hook, with_kwargs=True)

        def save_callback(swift_model, model_dir, adapter_name):
            reft_model.save_intervention(save_directory=model_dir, include_model=False)

        def mark_trainable_callback(model):
            return

        def load_callback(swift_model, model_dir, adapter_name):
            reft_model.load_intervention(model_dir, include_model=False)

        return SwiftOutput(
            model=reft_model,
            config=config,
            mark_trainable_callback=mark_trainable_callback,
            save_callback=save_callback,
            load_callback=load_callback)

    @staticmethod
    def has_additional_modules():
        return True

    @staticmethod
    def activate_adapter(module: torch.nn.Module, adapter_name: str, activate: bool, offload: str = None):
        assert activate, 'ReFT does not support deactivate'

# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect
import re
import types
from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch
from torch import nn
from transformers.activations import ACT2CLS

from swift import get_logger
from swift.tuners.utils import ActivationMixin, SwiftAdapter, SwiftConfig, SwiftOutput
from swift.utils.torch_utils import find_sub_module
from .scetuning_components import probe_output_hook

logger = get_logger()


@dataclass
class SCETuningConfig(SwiftConfig):
    """
    The configuration class for the SCEdit module.

    'SCEdit: Efficient and Controllable Image Diffusion Generation via Skip Connection Editing' by Jiang et al.(2023)
    See https://arxiv.org/abs/2312.11392

    Args:
        dims(`Union[List[int], int]`): The dimensions of the hidden states
        target_modules(`Union[List[str], str]`): The target module to be replaced, can a regex string
        hint_modules(`Union[List[str], str]`): The hint module to be replaced, can a regex string
        tuner_mode(`str`): Location of tuner operation.
        tuner_op(`str`): Tuner operation.
        down_ratio(`flaot`): The dim down ratio of tuner hidden state.
    """

    dims: Optional[Union[List[int], int]] = field(
        default=None, metadata={'help': 'The dimensions of the hidden states'})

    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={'help': 'The target module to be replaced, can be a regex string or name list of full match format'})

    hint_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={'help': 'The hint modules to be replaced, can be a regex string or name list of full match format'})

    tuner_mode: str = field(
        default='decoder',
        metadata={'help': 'Location of tuner operation. The tuner mode choices: encoder, decoder, and identity'})

    tuner_op: str = field(default='SCEAdapter', metadata={'help': 'The tuner ops choices: SCEAdapter'})

    down_ratio: float = field(default=1.0, metadata={'help': 'The dim down ratio of tuner hidden state'})

    def __post_init__(self):
        from swift.tuners.mapping import SwiftTuners
        self.swift_type = SwiftTuners.SCETUNING


class SCETuning(SwiftAdapter):

    @staticmethod
    def prepare_model(model: nn.Module, config: SCETuningConfig, adapter_name: str) -> SwiftOutput:
        """Prepare a model with `SCETuningConfig`"""
        module_keys = [key for key, _ in model.named_modules()]
        # 1. Matching the hint module
        hint_module_ins_list = []
        if config.hint_modules:
            if isinstance(config.hint_modules, list):
                for module_key in config.hint_modules:
                    assert module_key in module_keys
                    h_module = model.get_submodule(module_key)
                    logger.info(f'Matching hint module [{module_key}] of type {type(h_module)}')
                    if isinstance(h_module, (nn.ModuleList, nn.ModuleDict)):
                        logger.warning(
                            f'Type of {type(h_module)} may not be supported because of its customized forward')
                    h_module.register_forward_hook(probe_output_hook, with_kwargs=True)
                    hint_module_ins_list.append(h_module)
            else:
                for module_key in module_keys:
                    if re.fullmatch(config.hint_modules, module_key):
                        h_module = model.get_submodule(module_key)
                        logger.info(f'Matching hint module [{module_key}] of type {type(h_module)}')
                        if isinstance(h_module, (nn.ModuleList, nn.ModuleDict)):
                            logger.warning(
                                f'Type of {type(h_module)} may not be supported because of its customized forward')
                        h_module.register_forward_hook(probe_output_hook, with_kwargs=True)
                        hint_module_ins_list.append(h_module)
            if len(hint_module_ins_list) == 0:
                logger.error('Cannot match hint modules')

        def _get_module(module):
            if isinstance(module, nn.ModuleList):
                module = module[-1]
                return _get_module(module)
            return module

        # 2. Matching the target module
        target_module_ins_list = []
        assert config.target_modules is not None
        if isinstance(config.target_modules, list):
            for module_key in config.target_modules:
                assert module_key in module_keys
                t_module = model.get_submodule(module_key)
                logger.info(f'Matching target module [{module_key}] of type {type(t_module)}')
                target_module_ins_list.append(_get_module(t_module))
        else:
            for module_key in module_keys:
                if re.fullmatch(config.target_modules, module_key):
                    t_module = model.get_submodule(module_key)
                    logger.info(f'Matching target module [{module_key}] of type {type(t_module)}')
                    target_module_ins_list.append(_get_module(t_module))
        if len(target_module_ins_list) == 0:
            logger.error('Cannot match target modules')
        if len(hint_module_ins_list) > 0 and not len(hint_module_ins_list) == len(target_module_ins_list):
            logger.info("Target modules' length should be equal with hint modules.")
            assert len(hint_module_ins_list) == len(target_module_ins_list)
        if isinstance(config.dims, int):
            dims = [config.dims for _ in target_module_ins_list]
        else:
            assert len(config.dims) == len(target_module_ins_list)
            dims = config.dims

        # refactor forward function
        def _forward_encoder_mode(self, *args, **kwargs):
            args = getattr(self, f'forward_origin_{adapter_name}')(*args, **kwargs)
            args_type = type(args)
            if args_type is tuple:
                args = args[0]
            if hasattr(self, 'hint'):
                hint_out = self.hint.probe_output_data
                args_main = getattr(self, f'scetuner_{adapter_name}')(args, hint_out)
            else:
                args_main = getattr(self, f'scetuner_{adapter_name}')(args)
            if args_type is tuple:
                args_main = (args_main, )
            return args_main

        def _forward_decoder_mode(self, *args, **kwargs):
            args_type = type(args)
            if args_type is tuple:
                args_sub_tuner = args[0]
                args_sub_extra = args[1:]
            tuner_module = getattr(self, f'scetuner_{adapter_name}')
            args_hidden, args_res = torch.split(args_sub_tuner, args_sub_tuner.shape[1] - tuner_module.dim, 1)
            if hasattr(self, 'hint'):
                hint_out = self.hint.probe_output_data
                args_res_new = tuner_module(args_res, hint_out)
            else:
                args_res_new = tuner_module(args_res)
            args_sub_tuner_new = torch.cat([args_hidden, args_res_new], dim=1)
            if args_type is tuple:
                args_main = (args_sub_tuner_new, *args_sub_extra)

            args_main = getattr(self, f'forward_origin_{adapter_name}')(*args_main, **kwargs)
            return args_main

        # 3. inject the tuners
        for tuner_id, t_module in enumerate(target_module_ins_list):
            setattr(t_module, f'forward_origin_{adapter_name}', getattr(t_module, 'forward'))
            if config.tuner_mode in ('encoder', 'identity'):
                _forward = _forward_encoder_mode
            elif config.tuner_mode == 'decoder':
                _forward = _forward_decoder_mode
            else:
                raise Exception(f'Error tuner_mode: {config.tuner_mode}')
            setattr(t_module, 'forward', types.MethodType(_forward, t_module))
            tuner_op = SCETunerModule(
                name=config.tuner_op,
                adapter_name=adapter_name,
                module_key=str(tuner_id),
                dim=dims[tuner_id],
                tuner_length=int(dims[tuner_id] * config.down_ratio))
            setattr(t_module, f'scetuner_{adapter_name}', tuner_op)
            if len(hint_module_ins_list) > 0:
                setattr(t_module, 'hint', hint_module_ins_list[tuner_id])

        def state_dict_callback(state_dict, adapter_name):
            state_dict_new = {key: value for key, value in state_dict.items() if f'scetuner_{adapter_name}' in key}
            return state_dict_new

        def mark_trainable_callback(model):
            return

        return SwiftOutput(config, state_dict_callback, mark_trainable_callback)

    @staticmethod
    def activate_adapter(module: torch.nn.Module, adapter_name: str, activate: bool, offload: str = None):
        modules = find_sub_module(module, f'scetuner_{adapter_name}')
        for _module in modules:
            _module: ActivationMixin
            _module: nn.Module
            _module.set_activation(adapter_name, activate)
            SwiftAdapter.save_memory(_module, adapter_name, _module.module_key, activate, offload)


class SCETunerModule(nn.Module, ActivationMixin):

    def __init__(self,
                 name,
                 adapter_name,
                 module_key,
                 dim,
                 tuner_length,
                 tuner_type=None,
                 tuner_weight=None,
                 act_layer=nn.GELU,
                 zero_init_last=True,
                 use_bias=True):
        super(SCETunerModule, self).__init__()
        super(nn.Module, self).__init__(module_key)
        self.name = name
        self.adapter_name = adapter_name
        self.dim = dim
        if name == 'SCEAdapter':
            from .scetuning_components import SCEAdapter
            self.tuner_op = SCEAdapter(
                dim=dim,
                adapter_length=tuner_length,
                adapter_type=tuner_type,
                adapter_weight=tuner_weight,
                act_layer=act_layer)
        else:
            raise Exception(f'Error tuner op {name}')

    def forward(self, x, x_shortcut=None, use_shortcut=True, **kwargs):
        if not self.is_activated(self.adapter_name):
            return x
        if self.name == 'SCEAdapter':
            self.tuner_op.to(x.device)
            out = self.tuner_op(x)
        else:
            raise Exception(f'Error tuner op {self.name}')
        return out

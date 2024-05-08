# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import re
import types
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from swift import get_logger
from swift.utils.torch_utils import find_sub_module
from .restuning_components import ResTuner, detach_tensors, probe_input_pre_hook, probe_output_hook
from .utils import ActivationMixin, SwiftAdapter, SwiftConfig, SwiftOutput

logger = get_logger()


@dataclass
class ResTuningConfig(SwiftConfig):
    """
    The configuration class for the ResTuning module.

    ResTuning is a flexible parameter-efficient and memory-efficient tuning paradigm framework.
    'Res-Tuning: A Flexible and Efficient Tuning Paradigm via Unbinding Tuner from Backbone'
    by Jiang et al.(2023)
    See

    Args:
        dims(`Union[List[int], int]`): The dimensions of the hidden states
        root_modules(`str`): The root module to be replaced, can a regex string
        root_modules_hook(`str`): The hook type of root modules, can be "input" or "output"
        stem_modules(`Union[List[str], str]`): The stem modules to be replaced,
            can a regex string or name list of full match format
        stem_modules_hook(`Union[List[str], str]`): The hook type of stem modules, can be "input" or "output"
        target_modules(`str`): The target module to be replaced, can a regex string
        target_modules_hook(`str`): The hook type of target modules, can be "input" or "output"
        tuner_cfg(`Union[List[Dict], Dict, str]`): The configuration of the tuning module,
            can a string or customized config
        use_upsample(bool): Whether to use auxiliary upsample module
        upsample_out_channels(List[int]): The channels if `use_upsample`
        zero_init_last(bool): Use zero to initialize the last Linear in every sub tuner.

    """

    dims: Optional[Union[List[int], int]] = field(
        default=None, metadata={'help': 'The dimensions of the hidden states'})

    root_modules: str = field(
        default=None,
        metadata={
            'help':
            'The root module to be replaced, can a regex string (use the first matching module) or full match format'
        })

    root_modules_hook: str = field(
        default='input', metadata={'help': 'The hook type of root modules, can be "input" or "output"'})

    stem_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={'help': 'The stem modules to be replaced, can a regex string or name list of full match format'})

    stem_modules_hook: str = field(
        default='output', metadata={'help': 'The hook type of stem modules, can be "input" or "output"'})

    target_modules: str = field(
        default=None,
        metadata={
            'help':
            'The target module to be replaced, can a regex string (use the first matching module) or full match format'
        })

    target_modules_hook: str = field(
        default='input', metadata={'help': 'The hook type of target modules, can be "input" or "output"'})

    target_hidden_pos: Union[int, str] = field(
        default=None, metadata={'help': 'The position of the hidden state for target modules output'})

    tuner_cfg: Optional[Union[List[Dict], Dict, str]] = field(
        default=None, metadata={'help': 'The configuration of the tuning module, can a string or customized config'})

    use_upsample: bool = field(default=False, metadata={'help': 'Whether to use auxiliary upsample module'})

    upsample_out_channels: List[int] = field(
        default=None, metadata={'help': 'The number of output channels when "use_upsample" is set to "True"'})

    zero_init_last: bool = field(default=False, metadata={'help': 'Zero init last weight'})

    use_bypass: bool = field(default=True, metadata={'help': 'Whether to use bypass'})

    def __post_init__(self):
        from .mapping import SwiftTuners
        self.swift_type = SwiftTuners.RESTUNING
        self.target_hidden_pos = 0 if self.target_hidden_pos is None else self.target_hidden_pos


class ResTuning(SwiftAdapter):

    @staticmethod
    def prepare_model(model: nn.Module, config: ResTuningConfig, adapter_name: str) -> SwiftOutput:
        """Prepare a model with `ResTuningConfig`"""

        def _forward_seq(self, input, *args, **kwargs):
            for idx, module in enumerate(self):
                if idx >= len(self.origin_module_keys):
                    continue
                input = module(input)
            return input

        def _forward_target(self, *args, **kwargs):
            if self.target_modules_hook == 'input':
                if isinstance(self.target_hidden_pos, int):
                    args = list(args)
                    _arg = args[self.target_hidden_pos]
                else:
                    _arg = kwargs[self.target_hidden_pos]
                args_main = _forward_restuning(self, _arg)
                if isinstance(self.target_hidden_pos, int):
                    args[self.target_hidden_pos] = args_main
                else:
                    kwargs[self.target_hidden_pos] = args_main
                args_main = getattr(self, f'forward_origin_{adapter_name}')(*args, **kwargs)
            else:
                _args_main = getattr(self, f'forward_origin_{adapter_name}')(*args, **kwargs)
                _arg = _args_main[self.target_hidden_pos] if isinstance(_args_main, (tuple, list, dict)) else _args_main
                args_main = _forward_restuning(self, _arg)
                if type(_args_main) != type(args_main):
                    _args_main[self.target_hidden_pos] = args_main
                    args_main = _args_main
            return args_main

        def _forward_restuning(self, origin_arg):
            probe_results = []
            root_module_ins = self.root_module_ins_list[0]
            stem_module_ins_list = self.stem_module_ins_list
            top_module = model.get_submodule('')
            if root_module_ins:
                if root_module_ins.root_modules_hook == 'input':
                    probe_results.append(root_module_ins.probe_input_data)
                else:
                    probe_results.append(root_module_ins.probe_output_data)
            for i, st_mod in enumerate(stem_module_ins_list):
                if i == 0 and root_module_ins is None:
                    probe_results.append(st_mod.probe_input_data)
                if st_mod.stem_modules_hook == 'input':
                    probe_results.append(st_mod.probe_input_data)
                else:
                    probe_results.append(st_mod.probe_output_data)
            args_main = getattr(top_module, f'restuning_{adapter_name}')(probe_results, origin_arg)
            return args_main

        # 1. Matching the root module
        module_keys = [key for key, _ in model.named_modules()]
        root_module_ins_list = []
        if config.root_modules:
            for module_key in module_keys:
                if re.fullmatch(config.root_modules, module_key):
                    root_module = model.get_submodule(module_key)
                    logger.info(f'Matching root module [{module_key}] of type {type(root_module)}')
                    if isinstance(root_module, (nn.ModuleList, nn.ModuleDict)):
                        logger.warning(
                            f'Type of {type(root_module)} may not be supported because of its customized forward')
                    if config.root_modules_hook == 'input':
                        root_module.register_forward_pre_hook(probe_input_pre_hook)
                    else:
                        root_module.register_forward_hook(probe_output_hook)
                    root_module.root_modules_hook = config.root_modules_hook
                    root_module_ins_list.append(root_module)
                    break
            if len(root_module_ins_list) == 0:
                logger.error('Cannot match root modules')

        # 2. Matching the stem module
        stem_module_ins_list = []
        stem_module_ins_index = []
        for module_key in module_keys:
            if (isinstance(config.stem_modules, str) and re.fullmatch(config.stem_modules, module_key)) or \
                    (isinstance(config.stem_modules, list) and module_key in config.stem_modules):
                stem_module = model.get_submodule(module_key)
                if isinstance(config.stem_modules, list):
                    stem_module_ins_index.append(config.stem_modules.index(module_key))
                logger.info(f'Matching stem module [{module_key}] of type {type(stem_module)}')
                if isinstance(stem_module, (nn.ModuleList, nn.ModuleDict)):
                    logger.warning(
                        f'Type of {type(stem_module)} may not be supported because of its customized forward')
                if len(root_module_ins_list) == 0 and len(stem_module_ins_list) == 0:
                    stem_module.register_forward_pre_hook(probe_input_pre_hook)
                if config.stem_modules_hook == 'input':
                    stem_module.register_forward_pre_hook(probe_input_pre_hook)
                else:
                    stem_module.register_forward_hook(probe_output_hook)
                stem_module.stem_modules_hook = config.stem_modules_hook
                stem_module_ins_list.append(stem_module)
        if isinstance(config.stem_modules, list):
            stem_module_ins_list = [
                stem_module_ins_list[stem_module_ins_index.index(i)] for i in range(len(stem_module_ins_index))
            ]
        depth = len(stem_module_ins_list)
        if len(stem_module_ins_list) == 0:
            raise Exception('Cannot match source modules')

        # 3. Init restuning module
        if len(stem_module_ins_list) != 0:
            top_module = model.get_submodule('')
            restuning_module = ResTuningBypassModule(config.dims, depth, adapter_name, config.use_upsample,
                                                     config.upsample_out_channels, config.zero_init_last,
                                                     config.tuner_cfg)
            setattr(top_module, f'restuning_{adapter_name}', restuning_module)

        # 4. Matching the target module
        target_module_ins = None
        for module_key in module_keys:
            if re.fullmatch(config.target_modules, module_key):
                tgt_module = model.get_submodule(module_key)
                logger.info(f'Matching target module [{module_key}] of type {type(tgt_module)}')
                if isinstance(tgt_module, (nn.ModuleList, nn.ModuleDict)):
                    raise Exception(
                        f'Type of {type(tgt_module)} may not be supported because of its customized forward')

                tgt_module.target_modules_hook = config.target_modules_hook
                tgt_module.target_hidden_pos = config.target_hidden_pos
                tgt_module.root_module_ins_list = root_module_ins_list
                tgt_module.stem_module_ins_list = stem_module_ins_list
                target_module_ins = tgt_module

                if isinstance(tgt_module, nn.Sequential) and not hasattr(tgt_module, 'origin_module_keys'):
                    tgt_module.origin_module_keys = copy.deepcopy(list(tgt_module._modules.keys()))

                    setattr(tgt_module, f'forward_origin_{adapter_name}', types.MethodType(_forward_seq, tgt_module))
                else:
                    setattr(tgt_module, f'forward_origin_{adapter_name}', tgt_module.forward)
                tgt_module.forward = types.MethodType(_forward_target, tgt_module)
        if target_module_ins is None:
            raise Exception('Cannot match target modules')

        def state_dict_callback(state_dict, adapter_name):
            return {key: value for key, value in state_dict.items() if f'restuning_{adapter_name}' in key}

        def mark_trainable_callback(model):
            return

        return SwiftOutput(config, state_dict_callback, mark_trainable_callback)

    @staticmethod
    def activate_adapter(module: torch.nn.Module, adapter_name: str, activate: bool, offload: str = None):
        modules = find_sub_module(module, f'restuning_{adapter_name}')
        for _module in modules:
            _module: ActivationMixin
            _module: nn.Module
            _module.set_activation(adapter_name, activate)
            SwiftAdapter.save_memory(_module, adapter_name, _module.module_key, activate, offload)


class ResTuningBypassModule(nn.Module, ActivationMixin):
    """The implementation of ResTuningBypass method.
    """

    def __init__(
        self,
        dims,
        depth,
        adapter_name,
        use_upsample=False,
        upsample_out_channels=None,
        zero_init_last=False,
        tuner_cfg=None,
    ):
        super(ResTuningBypassModule, self).__init__()
        super(nn.Module, self).__init__('')
        self.adapter_name = adapter_name

        self.bypass_blocks = nn.Sequential(*[
            ResTunerBypassBlock(
                dim=dims[i] if isinstance(dims, list) else dims,
                layer_num=i,
                depth=depth,
                use_upsample=use_upsample,
                upsample_out_channels=upsample_out_channels[i] if isinstance(upsample_out_channels, list
                                                                             ) else upsample_out_channels,
                zero_init_last=zero_init_last,
                tuner_cfg=tuner_cfg[i] if isinstance(tuner_cfg, list) else tuner_cfg) for i in range(depth)
        ])

    def forward(self, x_list, origin_arg, **kwargs):
        if not self.is_activated(self.adapter_name):
            return origin_arg
        x_bypass = detach_tensors(x_list.pop(0))
        x_bypass = x_bypass[0] if isinstance(x_bypass, (list, tuple)) else x_bypass
        x_list = detach_tensors(x_list)
        x_list = [_x[0] if isinstance(_x, (list, tuple)) else _x for _x in x_list]
        for i, (bp_blk, x_stem) in enumerate(zip(self.bypass_blocks, x_list)):
            target_size = x_list[i + 1].shape[2:] if i < len(x_list) - 1 else None
            x_bypass = bp_blk(x_stem, x_bypass, target_size, **kwargs)
        return x_bypass


class ResTunerBypassBlock(nn.Module):

    def __init__(self, dim, layer_num=-1, depth=-1, use_upsample=False, zero_init_last=False, tuner_cfg=None, **kwargs):
        super().__init__()
        self.layer_num = layer_num
        self.depth = depth

        if isinstance(tuner_cfg, str):
            lateral_cfg = tuner_cfg
            vertical_cfg = tuner_cfg
            aux_cfg = 'upsample' if use_upsample and layer_num != depth - 1 else None
        elif isinstance(tuner_cfg, dict):
            lateral_cfg = tuner_cfg['lateral_cfg'] if 'lateral_cfg' in tuner_cfg else None
            vertical_cfg = tuner_cfg['vertical_cfg'] if 'vertical_cfg' in tuner_cfg else None
            aux_cfg = tuner_cfg['aux_cfg'] if 'aux_cfg' in tuner_cfg else None

        self.lateral_tuner = ResTuner(dim, layer_num, depth, zero_init_last, 'lateral', lateral_cfg, **kwargs)
        self.vertical_tuner = ResTuner(dim, layer_num, depth, zero_init_last, 'vertical', vertical_cfg, **kwargs)
        if aux_cfg and len(aux_cfg) != 0:
            self.aux_tuner = ResTuner(dim, layer_num, depth, zero_init_last, 'aux', aux_cfg, **kwargs)

    def forward(self, x_stem, x_bypass, target_size=None, **kwargs):
        x_lateral = self.lateral_tuner(x_stem)
        x_vertical = self.vertical_tuner(x_bypass)

        x_bypass_out = x_lateral + x_vertical
        if hasattr(self, 'aux_tuner'):
            x_bypass_out = self.aux_tuner(x_bypass_out, target_size)
        return x_bypass_out

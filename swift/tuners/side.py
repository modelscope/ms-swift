# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import re
import types
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import partial
from itertools import repeat
from typing import List, Union

import torch
from torch import nn

from swift.utils.logger import get_logger
from swift.utils.torch_utils import find_sub_module
from .utils import ActivationMixin, SwiftAdapter, SwiftConfig, SwiftOutput

logger = get_logger()


@dataclass
class SideConfig(SwiftConfig):
    """
    The configuration class for the side module.

    Side-Tuning only needs to train one side network and
    weights the output of pre-trained model and side network.
    'Side-Tuning: A Baseline for Network Adaptation via Additive Side Networks'
    by Zhang et al.(2019)
    See https://arxiv.org/abs/1912.13503

    Args:
        target_modules: The feedforward module to be replaced, in regex format
    """

    dim: int = field(default=None, metadata={'help': 'The dimension of the hidden states'})

    target_modules: str = field(
        default=None, metadata={'help': 'The target module to be replaced, in full match format'})

    side_module_name: str = field(default='fcn4', metadata={'help': 'The name of the additive side networks'})

    source_hidden_pos: Union[str, int] = field(
        default=0,
        metadata={
            'help': 'The position of the hidden state input to the target module, can be int (args) or str (kwargs)'
        })

    target_hidden_pos: Union[str, int] = field(
        default=0,
        metadata={
            'help': 'The position of the hidden state output from the target module, can be int (args) or str (kwargs)'
        })

    def __post_init__(self):
        from .mapping import SwiftTuners
        self.swift_type = SwiftTuners.SIDE


class Side(SwiftAdapter):

    @staticmethod
    def prepare_model(model: nn.Module, config: SideConfig, adapter_name: str) -> SwiftOutput:
        """Prepare a model with `SideConfig`"""
        module_keys = [key for key, _ in model.named_modules()]

        for module_key in module_keys:
            if re.fullmatch(config.target_modules, module_key):  # noqa
                tgt_module = model.get_submodule(module_key)
                logger.info(f'Matching target module [{module_key}] of type {type(tgt_module)}')
                if isinstance(tgt_module, (nn.ModuleList, nn.ModuleDict)):
                    raise Exception(
                        f'Type of {type(tgt_module)} may not be supported because of its customized forward')

                def _forward(self, *args, **kwargs):
                    args_main = getattr(self, f'forward_origin_{adapter_name}')(*args, **kwargs)

                    if isinstance(config.source_hidden_pos, int):
                        x = args[config.source_hidden_pos]
                    else:
                        x = kwargs[config.source_hidden_pos]

                    x_main = args_main[config.target_hidden_pos] \
                        if isinstance(args_main, (tuple, list, dict)) else args_main
                    out = getattr(self, f'side_{adapter_name}')(x, x_main)
                    if isinstance(args_main, (tuple, list, dict)):
                        args_main[config.target_hidden_pos] = out
                    else:
                        args_main = out
                    return args_main

                if isinstance(tgt_module, nn.Sequential) and not hasattr(tgt_module, 'tgt_module_keys'):
                    tgt_module.tgt_module_keys = copy.deepcopy(list(tgt_module._modules.keys()))

                    def forward_seq(self, input, *args, **kwargs):
                        for idx, module in enumerate(self):
                            if idx >= len(tgt_module.tgt_module_keys):
                                continue
                            input = module(input)
                        return input

                    setattr(tgt_module, f'forward_origin_{adapter_name}', types.MethodType(forward_seq, tgt_module))
                else:
                    setattr(tgt_module, f'forward_origin_{adapter_name}', tgt_module.forward)
                tgt_module.forward = types.MethodType(_forward, tgt_module)
                side_module = SideModule(config.dim, adapter_name, module_key, config.side_module_name)
                setattr(tgt_module, f'side_{adapter_name}', side_module)
                logger.info(f'Side modules(module_key): {module_key}.side_{adapter_name}')

        def state_dict_callback(state_dict, adapter_name):
            return {key: value for key, value in state_dict.items() if f'side_{adapter_name}' in key}

        def mark_trainable_callback(model):
            return

        return SwiftOutput(config, state_dict_callback, mark_trainable_callback)

    @staticmethod
    def activate_adapter(module: torch.nn.Module, adapter_name: str, activate: bool, offload: str = None):
        modules = find_sub_module(module, f'side_{adapter_name}')
        for _module in modules:
            _module: ActivationMixin
            _module: nn.Module
            _module.set_activation(adapter_name, activate)
            SwiftAdapter.save_memory(_module, adapter_name, _module.module_key, activate, offload)


class SideModule(nn.Module, ActivationMixin):
    """The implementation of vision side-tuning method.

    Side-Tuning only needs to train one side network and
    weights the output of pre-trained model and side network.
    'Side-Tuning: A Baseline for Network Adaptation via Additive Side Networks'
    by Zhang et al.(2019)
    See https://arxiv.org/abs/1912.13503

    Attributes:
        side_module_name: The name of the additive side networks.
    """

    def __init__(self, dim, adapter_name, module_key, side_module_name='fcn4'):
        super(SideModule, self).__init__()
        super(nn.Module, self).__init__(module_key)
        self.adapter_name = adapter_name

        side_module_name = side_module_name.lower()
        if side_module_name == 'fcn4':
            self.side_net = FCN4(out_dims=dim)
        elif side_module_name == 'mlp':
            self.side_net = Mlp(dim)
        elif side_module_name == 'alexnet':
            import torchvision
            mm = torchvision.models.alexnet(pretrained=True)
            self.side_net = nn.Sequential(
                OrderedDict([('features', mm.features), ('avgpool', mm.avgpool), ('flatten', nn.Flatten()),
                             ('fc', nn.Linear(9216, dim, bias=False))]))
        else:
            raise ValueError(f'Unsupported side_module_name: {side_module_name}')
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, x_main):
        if not self.is_activated(self.adapter_name):
            return x_main
        alpha_squashed = torch.sigmoid(self.alpha)
        x_side = self.side_net(x)
        x_out = alpha_squashed * x_main + (1 - alpha_squashed) * x_side
        return x_out


class FCN4(nn.Module):
    """The implementation of simple FCN4 network for side network.
    """

    def __init__(self, out_dims=-1, **kwargs):
        super(FCN4, self).__init__(**kwargs)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False, dilation=1), nn.GroupNorm(2, 16),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=0, bias=False, dilation=1), nn.GroupNorm(2, 16),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0, bias=False, dilation=1), nn.GroupNorm(2, 32),
            nn.ReLU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0, bias=False, dilation=1), nn.GroupNorm(2, 64),
            nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        if out_dims > 0:
            self.fc = nn.Linear(64, out_dims)
        else:
            self.fc = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        if self.fc is not None:
            x = self.fc(x)
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer.
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = tuple(repeat(bias, 2))
        drop_probs = tuple(repeat(drop, 2))
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

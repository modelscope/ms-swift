# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect
import re
import types
from dataclasses import dataclass, field
from typing import List, Union

import torch
from torch import nn
from transformers.activations import ACT2CLS

from swift.utils.torch_utils import find_sub_module, get_logger
from .utils import ActivationMixin, SwiftAdapter, SwiftConfig, SwiftOutput

logger = get_logger()


@dataclass
class AdapterConfig(SwiftConfig):
    """
    The configuration class for the adapter module.

    Adapters project input tokens by an MLP layer.
    'Parameter-Efficient Transfer Learning for NLP' by Houlsby et al.(2019)
    See http://arxiv.org/abs/1902.00751

    Args:
        dim(`int`): The dimension of the hidden states
        target_modules(`Union[str, List[str]]`): The feedforward module to be replaced.
            in regex format if this argument is str, else will match with `end with` if List[str].
        hidden_pos(`Union[str, int]`): The position of the hidden state to be passed into the adapter,
            can be int (args) or str (kwargs)
        method_name(`str`): The method to be replaced, default is `forward`
        adapter_length: The length of the adapter length (intermediate length)
        act_layer: The activation layer of the adapter
    """

    dim: int = field(default=None, metadata={'help': 'The dimension of the hidden states'})

    target_modules: Union[str, List[str]] = field(
        default=None,
        metadata={
            'help':
            'The feedforward module to be replaced. in regex format if this argument is str, '
            'else will match with `end with` if List[str].'
        })

    hidden_pos: Union[str, int] = field(
        default=None,
        metadata={
            'help': 'The position of the hidden state to be passed into the adapter, can be int (args) or str (kwargs)'
        })

    method_name: str = field(default='forward', metadata={'help': 'The method to be replaced, default is `forward`'})

    adapter_length: int = field(
        default=128, metadata={'help': 'The length of the adapter length (intermediate length)'})

    act_layer: str = field(default='gelu', metadata={'help': 'The activation layer of the adapter'})

    def __post_init__(self):
        from .mapping import SwiftTuners
        self.swift_type = SwiftTuners.ADAPTER


class Adapter(SwiftAdapter):

    @staticmethod
    def prepare_model(model: nn.Module, config: AdapterConfig, adapter_name: str) -> SwiftOutput:
        """Prepare a model with `AdapterConfig`"""
        module_keys = [key for key, _ in model.named_modules()]

        for module_key in module_keys:
            if isinstance(config.target_modules, str):
                target_module_found = re.fullmatch(config.target_modules, module_key)
            else:
                target_module_found = any(module_key.endswith(target_key) for target_key in config.target_modules)

            if target_module_found:  # noqa
                module = model.get_submodule(module_key)

                def _forward(self, *args, **kwargs):
                    args = getattr(self, f'forward_origin_{adapter_name}')(*args, **kwargs)
                    if isinstance(args, (tuple, list, dict)):
                        if isinstance(config.hidden_pos, int):
                            _type = type(args)
                            args = list(args)
                            args[config.hidden_pos] = getattr(self, f'adapter_{adapter_name}')(args[config.hidden_pos])
                            args = _type(args)
                        else:
                            args[config.hidden_pos] = getattr(self, f'adapter_{adapter_name}')(args[config.hidden_pos])
                    elif isinstance(args, torch.Tensor):
                        args = getattr(self, f'adapter_{adapter_name}')(args)
                    return args

                def _feed_forward_chunk(self, attention_output):
                    return _forward(self, attention_output)

                # TODO The `config.method_name` method should not be replaced twice.

                setattr(module, f'forward_origin_{adapter_name}', getattr(module, config.method_name))
                num_args_in_forward_chunk_fn = len(
                    inspect.signature(getattr(module, f'forward_origin_{adapter_name}')).parameters)
                if config.method_name == 'feed_forward_chunk' and num_args_in_forward_chunk_fn == 1:
                    setattr(module, config.method_name, types.MethodType(_feed_forward_chunk, module))
                else:
                    setattr(module, config.method_name, types.MethodType(_forward, module))
                adapter_module = AdapterModule(config.dim, adapter_name, module_key, config.adapter_length,
                                               ACT2CLS[config.act_layer])
                setattr(module, f'adapter_{adapter_name}', adapter_module)
                logger.info(f'Adapter modules(module_key): {module_key}.adapter_{adapter_name}')

        def state_dict_callback(state_dict, adapter_name: str, **kwargs):
            return {key: value for key, value in state_dict.items() if f'adapter_{adapter_name}' in key}

        def mark_trainable_callback(model):
            return

        return SwiftOutput(
            config=config, state_dict_callback=state_dict_callback, mark_trainable_callback=mark_trainable_callback)

    @staticmethod
    def activate_adapter(module: torch.nn.Module, adapter_name: str, activate: bool, offload: str = None):
        modules = find_sub_module(module, f'adapter_{adapter_name}')
        for _module in modules:
            _module: ActivationMixin
            _module: nn.Module
            _module.set_activation(adapter_name, activate)
            SwiftAdapter.save_memory(_module, adapter_name, _module.module_key, activate, offload)


class AdapterModule(nn.Module, ActivationMixin):
    """The implementation of adapter tuning method.

    Adapters project input tokens by an MLP layer.
    'Parameter-Efficient Transfer Learning for NLP' by Houlsby et al.(2019)
    See http://arxiv.org/abs/1902.00751

    Args:
        dim: An integer indicating the embedding dimension.
        adapter_length: An integer indicating the length of adapter tuning.
    """

    def __init__(
        self,
        dim,
        adapter_name,
        module_key,
        adapter_length=None,
        act_layer=nn.GELU,
    ):
        super(AdapterModule, self).__init__()
        super(nn.Module, self).__init__(module_key)
        self.dim = dim
        self.adapter_name = adapter_name
        self.adapter_length = adapter_length
        self.linear1 = nn.Linear(dim, adapter_length)
        self.act = act_layer()
        self.linear2 = nn.Linear(adapter_length, dim)
        self.init_weights()
        self._prepared = False
        self.mark_all_sub_modules_as_plugin()

    def init_weights(self):

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)

        self.apply(_init_weights)

    def forward(self, x, identity=None):
        if not self.is_activated(self.adapter_name):
            return x
        if not self._prepared:
            self.linear1.to(x.device)
            self.act.to(x.device)
            self.linear2.to(x.device)
            self._prepared = True

        x_dtype = x.dtype
        x = x.to(self.linear1.weight.dtype)
        out = self.linear2(self.act(self.linear1(x)))
        if identity is None:
            identity = x
        identity = identity.to(out.dtype)
        out = identity + out
        return out.to(x_dtype)

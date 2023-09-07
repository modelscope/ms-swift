# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect
import re
import types
from dataclasses import dataclass, field
from typing import Union

import torch
from torch import nn
from transformers.activations import ACT2CLS

from .utils import SwiftConfig, SwiftOutput


@dataclass
class AdapterConfig(SwiftConfig):
    """
    The configuration class for the adapter module.

    Adapters project input tokens by an MLP layer.
    'Parameter-Efficient Transfer Learning for NLP' by Houlsby et al.(2019)
    See http://arxiv.org/abs/1902.00751

    Args:
        dim: The dimension of the hidden states
        target_modules: The feedforward module to be replaced, in regex format
        hidden_pos: The position of the hidden state to passed into the adapter, can be int (args) or str (kwargs)
        method_name: The method to be replaced, default to replace the forward method
        adapter_length: The length of the adapter length (intermediate length)
        act_layer: The activation layer of the adapter
    """

    dim: int = field(
        default=None, metadata={'help': 'The dimension of the hidden states'})

    target_modules: str = field(
        default=None,
        metadata={
            'help': 'The feedforward module to be replaced, in regex format'
        })

    hidden_pos: Union[str, int] = field(
        default=None,
        metadata={
            'help':
            'The position of the hidden state to passed into the adapter, can be int (args) or str (kwargs)'
        })

    method_name: str = field(
        default='forward',
        metadata={
            'help':
            'The method to be replaced, default to replace the forward method'
        })

    adapter_length: int = field(
        default=128,
        metadata={
            'help': 'The length of the adapter length (intermediate length)'
        })

    act_layer: str = field(
        default='gelu',
        metadata={'help': 'The activation layer of the adapter'})

    def __post_init__(self):
        from .mapping import SwiftTuners
        self.swift_type = SwiftTuners.ADAPTER


class Adapter:

    @staticmethod
    def prepare_model(model: nn.Module, config: AdapterConfig) -> SwiftOutput:
        """Prepare a model with `AdapterConfig`"""
        module_keys = [key for key, _ in model.named_modules()]

        for module_key in module_keys:
            if isinstance(config.target_modules, str):
                target_module_found = re.fullmatch(config.target_modules,
                                                   module_key)
            else:
                target_module_found = any(
                    module_key.endswith(target_key)
                    for target_key in config.target_modules)

            if target_module_found:  # noqa
                module = model.get_submodule(module_key)

                def _forward(self, *args, **kwargs):
                    args = self.forward_origin(*args, **kwargs)
                    if isinstance(args, (tuple, list, dict)):
                        if isinstance(config.hidden_pos, int):
                            return args[0:config.hidden_pos] + args[
                                config.hidden_pos] + getattr(self, 'adapter')(args[config.hidden_pos]) \
                                + args[config.hidden_pos + 1:] # noqa
                        else:
                            kwargs[config.hidden_pos] = args[
                                config.hidden_pos] + getattr(self, 'adapter')(
                                    args[config.hidden_pos])
                    elif isinstance(args, torch.Tensor):
                        args = getattr(self, 'adapter')(args)
                    return args

                def _feed_forward_chunk(self, attention_output):
                    return _forward(self, attention_output)

                module.forward_origin = getattr(module, config.method_name)
                num_args_in_forward_chunk_fn = len(
                    inspect.signature(module.forward_origin).parameters)
                if config.method_name == 'feed_forward_chunk' and num_args_in_forward_chunk_fn == 1:
                    setattr(module, config.method_name,
                            types.MethodType(_feed_forward_chunk, module))
                else:
                    setattr(module, config.method_name,
                            types.MethodType(_forward, module))
                adapter_module = AdapterModule(config.dim,
                                               config.adapter_length,
                                               ACT2CLS[config.act_layer])
                setattr(module, 'adapter', adapter_module)

        def state_dict_callback(state_dict):
            return {
                key: value
                for key, value in state_dict.items() if 'adapter' in key
            }

        def mark_trainable_callback(model):
            return

        return SwiftOutput(config, state_dict_callback,
                           mark_trainable_callback)


class AdapterModule(nn.Module):
    """The implementation of adapter tuning method.

    Adapters project input tokens by an MLP layer.
    'Parameter-Efficient Transfer Learning for NLP' by Houlsby et al.(2019)
    See http://arxiv.org/abs/1902.00751

    Attributes:
        dim: An integer indicating the embedding dimension.
        adapter_length: An integer indicating the length of adapter tuning.
    """

    def __init__(
        self,
        dim,
        adapter_length=None,
        act_layer=nn.GELU,
    ):
        super(AdapterModule, self).__init__()
        self.dim = dim
        self.adapter_length = adapter_length
        # self.adapter_type = adapter_type
        self.ln1 = nn.Linear(dim, adapter_length)
        self.activate = act_layer()
        self.ln2 = nn.Linear(adapter_length, dim)
        self.init_weights()
        self._prepared = False

    def init_weights(self):

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)

        self.apply(_init_weights)

    def forward(self, x, identity=None):
        if not self._prepared:
            self.ln1.to(x.device)
            self.activate.to(x.device)
            self.ln2.to(x.device)
            self._prepared = True
        
        x_dtype = x.dtype
        x = x.to(self.ln1.weight.dtype)
        out = self.ln2(self.activate(self.ln1(x)))
        if identity is None:
            identity = x
        identity = identity.to(out.dtype)
        out = identity + out
        return out.to(x_dtype)

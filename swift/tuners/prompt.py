# Copyright (c) Alibaba, Inc. and its affiliates.

import re
import types
from dataclasses import dataclass, field
from typing import List, Union

import torch
from torch import nn

from swift import get_logger
from swift.utils.torch_utils import find_sub_module
from .utils import ActivationMixin, SwiftAdapter, SwiftConfig, SwiftOutput

logger = get_logger()


@dataclass
class PromptConfig(SwiftConfig):
    """
    The configuration class for the prompt module.

    Visual prompt tuning (VPT) is proposed to initialize tunable prompt tokens
    and prepend to the original tokens in the first layer or multiple layers.
    'Visual Prompt Tuning' by Jia et al.(2022)
    See https://arxiv.org/abs/2203.12119

    Here we apply the VPT to other fields.

    Args:
        dim(`Union[int, List[int]]`): The dimension of the hidden states, use list if there are up-sample blocks
            or down-sample blocks
        target_modules(str): The layer module to be replaced, in regex format
        embedding_pos(Union[str, int]): The position of the embedding tensor
        attention_mask_pos(Union[str, int]): The position of the attention mask
        attention_mask_value(Union[float, int, bool]): The value to pad to the attention mask
        prompt_length(int): The length of the prompt tokens
        attach_front(bool): When set to True, prompt is attached in front of the embedding
        extract_embedding(bool): Whether the embedding is extracted at final stage to keep the same dims with inputs
    """

    dim: Union[int, List[int]] = field(default=None, metadata={'help': 'The dimension of the hidden states'})

    target_modules: str = field(default=None, metadata={'help': 'The layer module to be replaced, in regex format'})

    embedding_pos: Union[str, int] = field(default=None, metadata={'help': 'The position of the embedding tensor'})

    attention_mask_pos: Union[str, int] = field(default=None, metadata={'help': 'The position of the attention mask'})

    attention_mask_value: Union[float, int, bool] = field(
        default=0., metadata={'help': 'The value to pad to the attention mask'})

    prompt_length: int = field(default=16, metadata={'help': 'The length of the prompt tokens'})

    attach_front: bool = field(
        default=True, metadata={'help': 'When set to True, prompt is attached in front of the embedding'})

    extract_embedding: bool = field(
        default=False,
        metadata={'help': 'Whether the embedding is extracted at final stage to keep the same dims with inputs'})

    def __post_init__(self):
        from .mapping import SwiftTuners
        self.swift_type = SwiftTuners.PROMPT


class Prompt(SwiftAdapter):

    @staticmethod
    def prepare_model(model: nn.Module, config: PromptConfig, adapter_name: str):
        module_keys = [key for key, _ in model.named_modules()]
        match_module_keys = []
        for module_key in module_keys:
            if isinstance(config.target_modules, str):
                target_module_found = re.fullmatch(config.target_modules, module_key)
            else:
                target_module_found = any(module_key.endswith(target_key) for target_key in config.target_modules)
            if target_module_found:  # noqa
                module = model.get_submodule(module_key)

                def _forward(self, *args, **kwargs):
                    if isinstance(config.embedding_pos, int):
                        input_embedding = args[config.embedding_pos]
                    else:
                        input_embedding = kwargs[config.embedding_pos]

                    input_embedding = getattr(self, f'prompt_{adapter_name}').forward(input_embedding)
                    if isinstance(config.embedding_pos, int):
                        args = type(args)(
                            args[0:config.embedding_pos] + (input_embedding, ) + args[config.embedding_pos + 1:])
                    else:
                        kwargs[config.embedding_pos] = input_embedding

                    if config.attention_mask_pos:
                        attention_mask = None
                        if isinstance(config.attention_mask_pos, int):
                            attention_mask = args[config.attention_mask_pos]
                        elif isinstance(config.attention_mask_pos, str):
                            attention_mask = kwargs[config.attention_mask_pos]

                        if attention_mask is not None:
                            attention_mask = getattr(self,
                                                     f'prompt_{adapter_name}').patch_attention_mask(attention_mask)
                        if isinstance(config.attention_mask_pos, int):
                            args = type(args)(
                                args[0:config.attention_mask_pos] + (attention_mask, )
                                + args[config.attention_mask_pos + 1:])
                        else:
                            kwargs[config.attention_mask_pos] = attention_mask

                    forward_output = getattr(self, f'forward_origin_{adapter_name}')(*args, **kwargs)
                    if config.extract_embedding:
                        forward_output = getattr(self, f'prompt_{adapter_name}').extract(forward_output)

                    return forward_output

                setattr(module, f'forward_origin_{adapter_name}', module.forward)
                module.forward = types.MethodType(_forward, module)
                if isinstance(config.dim, list):
                    input_dim = config.dim[len(match_module_keys)]
                else:
                    input_dim = config.dim
                prompt_module = PromptModule(input_dim, int(module_key.rsplit('.')[-1]), adapter_name, module_key,
                                             config.prompt_length, config.attention_mask_value, config.attach_front)
                setattr(module, f'prompt_{adapter_name}', prompt_module)
                logger.info(f'Prompt modules(module_key): {module_key}.prompt_{adapter_name}')
                match_module_keys.append(module_key)

        def state_dict_callback(state_dict, adapter_name):
            return {key: value for key, value in state_dict.items() if f'prompt_{adapter_name}' in key}

        def mark_trainable_callback(model):
            return

        return SwiftOutput(
            config=config, state_dict_callback=state_dict_callback, mark_trainable_callback=mark_trainable_callback)

    @staticmethod
    def activate_adapter(module: torch.nn.Module, adapter_name: str, activate: bool, offload: str = None):
        modules = find_sub_module(module, f'prompt_{adapter_name}')
        for _module in modules:
            _module: ActivationMixin
            _module: nn.Module
            _module.set_activation(adapter_name, activate)
            SwiftAdapter.save_memory(_module, adapter_name, _module.module_key, activate, offload)


class PromptModule(nn.Module, ActivationMixin):
    """The implementation of vision prompt tuning method.

    Visual prompt tuning (VPT) is proposed to initialize tunable prompt tokens
    and prepend to the original tokens in the first layer or multiple layers.
    'Visual Prompt Tuning' by Jia et al.(2022)
    See https://arxiv.org/abs/2203.12119

    Attributes:
        dim: An integer indicating the embedding dimension.
        layer_num: An integer indicating number of layers.
        prompt_length: An integer indicating the length of vision prompt tuning.
    """

    def __init__(self, dim, layer_num, adapter_name, module_key, prompt_length=None, mask_values=0., attach_front=True):
        super(PromptModule, self).__init__()
        super(nn.Module, self).__init__(module_key)
        self.dim = dim
        self.layer_num = layer_num
        self.adapter_name = adapter_name
        self.prompt_length = prompt_length
        self.mask_values = mask_values
        self.attach_front = attach_front
        self.prompt_token = nn.Parameter(torch.zeros(1, prompt_length, dim))
        nn.init.xavier_uniform_(self.prompt_token)
        self.mark_all_sub_modules_as_plugin()

    def forward(self, x):
        if not self.is_activated(self.adapter_name):
            return x
        prompt_token = self.prompt_token.expand(x.shape[0], -1, -1).to(x.device, x.dtype)

        if self.layer_num == 0:
            if self.attach_front:
                x = torch.cat((prompt_token, x), dim=1)
            else:
                x = torch.cat((x, prompt_token), dim=1)
        else:
            if self.attach_front:
                x = torch.cat((prompt_token, x[:, self.prompt_length:, :]), dim=1)
            else:
                x = torch.cat((x[:, :-self.prompt_length, :], prompt_token), dim=1)
        return x

    def patch_attention_mask(self, m):
        if not self.is_activated(self.adapter_name):
            return m
        prefix_attention_mask = torch.full((*m.shape[:-1], self.prompt_length), self.mask_values).to(m.device)
        if self.attach_front:
            return torch.cat((prefix_attention_mask, m), dim=-1)
        else:
            return torch.cat((m, prefix_attention_mask), dim=-1)

    def extract(self, x):
        if self.attach_front:
            return x[:, self.prompt_length:, :]
        else:
            return x[:, :-self.prompt_length, :]

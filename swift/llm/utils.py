from types import MethodType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from transformers import GenerationConfig, PreTrainedTokenizerBase

from swift.utils import deep_getattr, get_logger, upper_bound
from .model import ModelMeta, get_model_arch


logger = get_logger()

History = List[Union[Tuple[str, str], List[str]]]


def to_device(inputs: Any, device: torch.device) -> Any:
    """Move inputs to a device"""
    if callable(getattr(inputs, 'to', None)):
        return inputs.to(device=device)

    if isinstance(inputs, Mapping):
        res = {}
        for k, v in inputs.items():
            res[k] = to_device(v, device)
    elif isinstance(inputs, Sequence) and not isinstance(inputs, str):
        res = []
        for b in inputs:
            res.append(to_device(b, device))
    else:
        res = inputs
    return res


def limit_history_length(template: 'Template', query: str, history: Optional[History],
                         max_length: Optional[int]) -> Tuple[History, History]:
    """binary search"""
    if history is None:
        history = []
    if max_length is None:
        return [], history

    def compute_token_length(history_length: int) -> int:
        assert history_length != 0
        example = {'query': query, 'history': history[-history_length:]}
        input_ids = template.encode(example)[0]['input_ids']
        return len(input_ids)

    history_length = upper_bound(0, len(history), lambda mid: compute_token_length(mid) <= max_length)
    old_history = history[:len(history) - history_length]
    history = history[len(history) - history_length:]
    return old_history, history


def set_generation_config(model: nn.Module, generation_config: GenerationConfig) -> None:
    old_generation_config = getattr(model, 'generation_config', None)
    old_generation_priority_config = ['no_repeat_ngram_size', 'num_beams']
    if old_generation_config is not None:
        for k, old_v in old_generation_config.__dict__.items():
            if k.startswith('_'):
                continue
            v = getattr(generation_config, k, None)
            if k in old_generation_priority_config or old_v is not None and v is None:
                setattr(generation_config, k, old_v)
    model.generation_config = generation_config


def find_module_list(model) -> Optional[nn.ModuleList]:
    module_lists = []
    for m in model.modules():
        if hasattr(m, 'gradient_checkpointing'):
            return
        if isinstance(m, nn.ModuleList) and len(m) >= 10:
            module_lists.append(m)
    if module_lists:
        return max(module_lists, key=lambda x: len(x))


def _add_gradient_checkpointing(module_list):

    def _new_forward(self, *args, **kwargs):
        layer_ret = torch.utils.checkpoint.checkpoint(self.__old_forward, *args, **kwargs)
        return layer_ret

    for module in module_list:
        if hasattr(module, '_old_forward'):  # device_map
            __old_forward = module._old_forward
            module._old_forward = MethodType(_new_forward, module)
        else:
            __old_forward = module.forward
            module.forward = MethodType(_new_forward, module)
        module.__old_forward = __old_forward


def dynamic_gradient_checkpointing(model) -> None:
    model_meta: ModelMeta = model.model_meta
    model_arch = get_model_arch(model_meta.model_arch)
    tower_names = model_arch.language_model
    if model_meta.is_multimodal:
        tower_names += model_arch.vision_tower

    for tower_name in tower_names:
        model_tower = deep_getattr(model, tower_name)
        module_list = find_module_list(model_tower)
        if module_list is None:
            continue
        _add_gradient_checkpointing(module_list)
        logger.info(f'Automatically add gradient_checkpointing to {model_tower.__class__}.')


def history_to_messages(history: History,
                        system: Optional[str] = None,
                        roles: Optional[List[List[str]]] = None) -> 'Messages':
    """
    history: [['query1', 'response1'], ['query2', 'response2']]
        or [['query1', 'response1'], ['query2', None]]
    """
    messages = []
    if not roles:
        roles = [['user', 'assistant']] * len(history)
    else:
        assert len(roles) == len(history), f'len(roles): {len(roles)}, len(history): {len(history)}'
    if system is not None:
        messages.append({'role': 'system', 'content': system})

    for role, h in zip(roles, history):
        assert isinstance(h, (list, tuple))
        if h[0] is not None:
            messages.append({'role': role[0], 'content': h[0]})
        if h[1] is not None:
            messages.append({'role': role[1], 'content': h[1]})
    return messages


def messages_to_history(messages: 'Messages') -> Dict[str, Any]:
    system = None
    if messages[0]['role'] == 'system':
        system = messages[0]['content']
        messages = messages[1::]
    history = []
    history_roles = []
    for q, r in zip(messages[::2], messages[1::2]):
        history.append([q['content'], r['content']])
        history_roles.append([q['role'], r['role']])
    query = None
    query_role = None
    if len(messages) % 2 == 1:
        query = messages[-1]['content']
        query_role = messages[-1]['role']
    return {
        'history': history,
        'history_roles': history_roles,
        'query': query,
        'query_role': query_role,
        'system': system,
    }

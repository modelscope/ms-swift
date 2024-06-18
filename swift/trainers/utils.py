# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/transformers.

import inspect
from types import FunctionType, MethodType
from typing import Dict, List, Optional, Union

from torch.nn import Module
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import (EvaluationStrategy, FSDPOption, HPSearchBackend, HubStrategy, IntervalStrategy,
                                        SchedulerType)

from swift.llm.utils.template import Context, History, Template
from swift.utils import get_logger

try:
    # https://github.com/huggingface/transformers/pull/25702
    from transformers.trainer_utils import ShardedDDPOption
except ImportError:
    ShardedDDPOption = None

logger = get_logger()


def can_return_loss(model: Module) -> bool:
    """Check if a given model can return loss."""
    signature = inspect.signature(model.forward)
    for p in signature.parameters:
        if p == 'return_loss' and signature.parameters[p].default is True:
            return True
    return False


def find_labels(model: Module) -> List[str]:
    """Find the labels used by a given model."""
    model_name = model.__class__.__name__
    signature = inspect.signature(model.forward)
    if 'QuestionAnswering' in model_name:
        return [p for p in signature.parameters if 'label' in p or p in ('start_positions', 'end_positions')]
    else:
        return [p for p in signature.parameters if 'label' in p]


def get_function(method_or_function: Union[MethodType, FunctionType]) -> FunctionType:
    if isinstance(method_or_function, MethodType):
        method_or_function = method_or_function.__func__
    return method_or_function


def is_instance_of_ms_model(model: Module) -> bool:
    """avoid import modelscope: circular dependency problem"""
    for m_cls in model.__class__.__mro__:
        cls_name = m_cls.__name__
        cls_module = m_cls.__module__
        if cls_name == 'Model' and cls_module.startswith('modelscope'):
            return True
    return False


def concat_template(feature: Dict, template: Template):
    query: Optional[str] = feature.get('query', None)
    system: Optional[str] = feature.get('system', None)
    history: Optional[History] = feature.get('history', None)
    if history is None:
        history = []
    if system is None:
        if template.use_default_system:
            system = template.default_system
    else:
        assert template.system_prefix is not None, 'not support `system`'
    res_context_list: List[Context] = []
    compute_loss_idx: List[float] = []
    if system is None:
        assert template.prefix != template.system_prefix, f'template.prefix: {template.prefix}'
        prefix = template.prefix
    else:
        prefix = template.system_prefix
    template._concat_context_list(prefix, res_context_list, compute_loss_idx, system=system)
    for i, (q, r) in enumerate(history):
        template._concat_context_list(
            [
                *template.prompt,
                '{{RESPONSE}}',
                *template.chat_sep  # noqa
            ],
            res_context_list,
            compute_loss_idx,
            query=q,
            response=r,
            round0=i)  # noqa
    template._concat_context_list(template.prompt, res_context_list, compute_loss_idx, query=query, round0=len(history))
    res_context_list, compute_loss_idx = template._simplify_context_list(res_context_list, compute_loss_idx)

    return res_context_list, feature['response'], feature['rejected_response'], compute_loss_idx


def build_tokenized_answer(answer, template: Template):
    tgt_input_ids = template._encode_context_list([answer], [1.0])[0]
    tgt_input_ids += template._encode_context_list(template.suffix, [1.0])[0]
    return dict(
        input_ids=tgt_input_ids,
        attention_mask=[1] * len(tgt_input_ids),
    )

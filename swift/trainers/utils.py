# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/transformers.

import heapq
import inspect
from functools import partial
from types import FunctionType, MethodType
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from accelerate import PartialState
from datasets import Dataset as HfDataset
from datasets import IterableDataset as HFIterableDataset
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
DATASET_TYPE = Union[HfDataset, HFIterableDataset]


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


def preprocess_kto_dataset(example: Dict[str, List[Any]], template: Template):
    """
    preprocess KTO specific dataset with given template

    Args:
    batch: A dictionary containing:
        - prompt: The main prompt string
        - completion: The completion string
        - label: The label data
        - history (optional): A list of historical queries/responses
        - system (optional): A system string to use

    template: swift Template object

    Returns:
    A dictionary with encoded prompt, completion, and label.
    """
    query: str = example.get('query')
    history: Optional[History] = example.get('history', None)
    system: Optional[str] = example.get('system', None)
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
        template._concat_context_list([*template.prompt, '{{RESPONSE}}', *template.chat_sep],
                                      res_context_list,
                                      compute_loss_idx,
                                      query=q,
                                      response=r,
                                      round0=i)
    template._concat_context_list(template.prompt, res_context_list, compute_loss_idx, query=query, round0=len(history))
    res_context_list, compute_loss_idx = template._simplify_context_list(
        res_context_list, compute_loss_idx, example=example)
    prompt = ''.join(res_context_list)

    return {'prompt': prompt, 'completion': example['response'], 'label': example['label']}


def tokenize_paired_dataset(template: Template, examples: Dict[str, List[Any]], streaming: bool = False):
    model_inputs = {}
    chosen_example, rejected_example = examples.copy(), examples
    rejected_example['response'] = examples['rejected_response']
    if streaming:
        chosen_tokenized = template.encode(chosen_example)
        rejected_tokenized = template.encode(rejected_example)
    else:
        chosen_tokenized = template.encode(chosen_example)[0]
        rejected_tokenized = template.encode(rejected_example)[0]

    for prompt, tokenized in zip(['chosen', 'rejected'], [chosen_tokenized, rejected_tokenized]):
        for k, v in tokenized.items():
            model_inputs[f'{prompt}_{k}'] = v
        if f'{prompt}_attention_mask' not in model_inputs:
            model_inputs[f'{prompt}_attention_mask'] = [1] * len(tokenized['input_ids'])
    return model_inputs



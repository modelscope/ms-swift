# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/transformers.

import heapq
import inspect
from functools import partial
from types import FunctionType, MethodType
from typing import Any, Dict, List, Optional, Tuple, Union

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


def sort_by_max_length(dataset: HfDataset, num_dataset: int, is_encoder_decoder: bool = False) -> HfDataset:
    logger.info('sort by max length...')
    if not is_encoder_decoder:
        dataset_chosen_len = [len(d['chosen_input_ids']) for d in dataset]
        dataset_rejected_len = [len(d['rejected_input_ids']) for d in dataset]
        idx = heapq.nlargest(
            num_dataset,
            range(len(dataset_chosen_len)),
            key=lambda i: max(dataset_chosen_len[i], dataset_rejected_len[i]))
    else:
        dataset_len = [len(d['prompt_input_ids']) for d in dataset]
        idx = heapq.nlargest(num_dataset, range(len(dataset_len)), key=lambda i: dataset_len[i])
    return dataset.select(idx)


def get_preprocess_func(template: Template, rlhf_type: str, streaming: bool, is_encoder_decoder: bool):
    if rlhf_type == 'kto':
        # leave truncation in trainer for KTO
        return partial(preprocess_kto_dataset, template=template)
    else:
        return partial(
            tokenize_paired_dataset, template=template, streaming=streaming, is_encoder_decoder=is_encoder_decoder)


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


def tokenize_paired_dataset(
    examples: Dict[str, List[Any]],
    template: Template,
    streaming: bool = False,
    is_encoder_decoder: bool = False,
):
    model_inputs = {}
    chosen_example, rejected_example = examples.copy(), examples
    rejected_example['response'] = examples['rejected_response']

    chosen_tokenized = template.encode(chosen_example)[0] if not streaming else template.encode(chosen_example)
    rejected_tokenized = template.encode(rejected_example)[0] if not streaming else template.encode(rejected_example)

    for prompt, tokenized in zip(['chosen', 'rejected'], [chosen_tokenized, rejected_tokenized]):
        for k, v in tokenized.items():
            model_inputs[f'{prompt}_{k}'] = v
        if f'{prompt}_attention_mask' not in model_inputs:
            model_inputs[f'{prompt}_attention_mask'] = [1] * len(tokenized['input_ids'])
    return model_inputs


def get_preprocessed_rlhf_dataset(train_dataset: DATASET_TYPE, val_dataset: Optional[DATASET_TYPE], template: Template,
                                  rlhf_type, streaming: bool, is_encoder_decoder: bool,
                                  **kwargs) -> Tuple[DATASET_TYPE, Optional[DATASET_TYPE]]:
    """
    Preprocesses the RLHF datasets using the specified template and RLHF type.

    Args:
        train_dataset (DATASET_TYPE): The training dataset.
        val_dataset (Optional[DATASET_TYPE]): The validation dataset.
        template (Template): The template used for preprocessing.
        rlhf_type (Literal['dpo', 'orpo', 'simpo', 'kto', 'cpo']): The type of RLHF.
        dataset_enable_cache (bool, optional): Whether to enable cache. Defaults to True.
        **kwargs: kwargs for preprocess func.

    Returns:
        Tuple[DATASET_TYPE, Optional[DATASET_TYPE]]: The preprocessed training and validation datasets.
    """
    preprocess_func = get_preprocess_func(
        template=template, rlhf_type=rlhf_type, streaming=streaming, is_encoder_decoder=is_encoder_decoder)
    column_names = list(next(iter(train_dataset)).keys())
    with PartialState().local_main_process_first():
        train_dataset = train_dataset.map(preprocess_func, remove_columns=column_names, **kwargs)
        if val_dataset is not None:
            val_dataset = val_dataset.map(preprocess_func, remove_columns=column_names, **kwargs)
    return train_dataset, val_dataset

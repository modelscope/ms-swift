# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/transformers.

import heapq
import inspect
from copy import deepcopy
from functools import partial
from types import FunctionType, MethodType
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
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


def _convert_bfloat16_to_float32(data):
    if isinstance(data, torch.Tensor) and data.dtype == torch.bfloat16:
        return data.to(torch.float32)
    elif isinstance(data, list):
        return [_convert_bfloat16_to_float32(item) for item in data]
    return data


def get_preprocess_func(template: Template, rlhf_type, vision_keys: list, streaming: bool, max_length: int,
                        max_prompt_length: int, truncation_mode, is_encoder_decoder: bool):
    if rlhf_type == 'kto':
        # leave truncation in trainer for KTO
        return partial(preprocess_kto_dataset, template=template)
    else:
        return partial(
            tokenize_paired_dataset,
            template=template,
            vision_keys=vision_keys,
            streaming=streaming,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
            truncation_mode=truncation_mode,
            is_encoder_decoder=is_encoder_decoder)


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


def _truncate_tokens(
    chosen_tokens: Dict[str, List[int]],
    rejected_tokens: Dict[str, List[int]],
    prompt_tokens: Dict[str, List[int]],
    max_length: int,
    max_prompt_length: int,
    truncation_mode: Literal['keep_start', 'keep_end'] = 'keep_end',
) -> None:
    """
    Truncates the tokens in chosen, rejected
    and prompt sequences to ensure they fit within the maximum length constraints.
    """

    longer_response_length = max(len(chosen_tokens['input_ids']), len(rejected_tokens['input_ids']))

    # if combined sequence is too long, truncate the prompt
    for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
        if len(answer_tokens['prompt_input_ids']) + longer_response_length > max_length:
            if truncation_mode == 'keep_start':
                for k in ['prompt_input_ids', 'prompt_attention_mask']:
                    answer_tokens[k] = answer_tokens[k][:max_prompt_length]
            elif truncation_mode == 'keep_end':
                for k in ['prompt_input_ids', 'prompt_attention_mask']:
                    answer_tokens[k] = answer_tokens[k][-max_prompt_length:]

    # if that's still too long, truncate the response from the end
    for answer_tokens in [chosen_tokens, rejected_tokens]:
        if len(answer_tokens['prompt_input_ids']) + longer_response_length > max_length:
            for k in ['input_ids', 'attention_mask']:
                answer_tokens[k] = answer_tokens[k][:max_length - max_prompt_length]


def tokenize_paired_dataset(
    examples: Dict[str, List[Any]],
    template: Template,
    vision_keys: Optional[List[str]] = None,
    streaming: bool = False,
    max_length: int = 4096,
    max_prompt_length: int = 512,
    truncation_mode: Literal['keep_start', 'keep_end'] = 'keep_end',
    is_encoder_decoder: bool = False,
):
    model_inputs = {}
    prompt_example, chosen_example, rejected_example = deepcopy(examples), examples, deepcopy(examples)
    prompt_example['response'] = None
    rejected_example['response'] = examples['rejected_response']

    chosen_tokenized = template.encode(chosen_example)[0] if not streaming else template.encode(chosen_example)
    rejected_tokenized = template.encode(rejected_example)[0] if not streaming else template.encode(rejected_example)
    prompt_tokenized = template.encode(prompt_example)[0] if not streaming else template.encode(prompt_example)
    if not is_encoder_decoder:
        prompt_input_ids = prompt_tokenized['input_ids']
        prompt_tokenized = {
            'prompt_input_ids':
            prompt_input_ids,
            'prompt_attention_mask':
            prompt_tokenized['attention_mask'] if 'attention_mask' in prompt_tokenized else [1]
            * len(prompt_tokenized['input_ids']),
        }

        if 'attention_mask' not in chosen_tokenized:
            chosen_tokenized['attention_mask'] = [1] * len(chosen_tokenized['input_ids'])
            rejected_tokenized['attention_mask'] = [1] * len(rejected_tokenized['input_ids'])

        # take response tokens
        for tokenized in [chosen_tokenized, rejected_tokenized]:
            tokenized.update(prompt_tokenized)
            for k in ['input_ids', 'attention_mask']:
                tokenized[k] = tokenized[k][len(prompt_input_ids):]

        _truncate_tokens(chosen_tokenized, rejected_tokenized, prompt_tokenized, max_length, max_prompt_length,
                         truncation_mode)
        for prefix, tokenzied in zip(['chosen', 'rejected'], [chosen_tokenized, chosen_tokenized]):
            for k in ['input_ids', 'attention_mask']:
                model_inputs[f'{prefix}_{k}'] = tokenzied[k] + tokenized[f'prompt_{k}']
            model_inputs[f'{prefix}_labels'] = model_inputs[f'{prefix}_input_ids'][:]
            model_inputs[f'{prefix}_labels'][:len(tokenized['prompt_input_ids'])] = [-100] * len(
                tokenized['prompt_input_ids'])

        # we keep single vision related tokens to save memory, and convert to float32 for bfloat16
        if '_data' in chosen_tokenized and vision_keys is not None:
            for k in vision_keys:
                _data_key = f'vision_{k}'
                if k not in ['input_ids', 'labels']:
                    model_inputs[_data_key] = (_convert_bfloat16_to_float32(chosen_tokenized['_data'][k]))
        # fix glm4v
        elif vision_keys is not None:
            for k in vision_keys:
                if k not in ['input_ids', 'labels']:
                    _data_key = f'vision_{k}'
                    model_inputs[_data_key] = (_convert_bfloat16_to_float32(chosen_tokenized[k]))
    else:
        model_inputs['chosen_labels'] = chosen_tokenized['input_ids']
        model_inputs['rejected_labels'] = rejected_tokenized['input_ids']
        model_inputs['prompt_input_ids'] = prompt_tokenized['input_ids']
        model_inputs['prompt_attention_mask'] = prompt_tokenized['attention_mask']
        if '_data' in chosen_tokenized and vision_keys is not None:
            for k in vision_keys:
                _data_key = f'vision_{k}'
                if k not in ['input_ids', 'labels']:
                    model_inputs[_data_key] = (_convert_bfloat16_to_float32(chosen_tokenized['_data'][k]))

    return model_inputs


def get_preprocessed_rlhf_dataset(train_dataset: DATASET_TYPE, val_dataset: Optional[DATASET_TYPE], template: Template,
                                  rlhf_type, vision_keys: Optional[list], streaming: bool, max_length: int,
                                  max_prompt_length: int, truncation_mode: Literal['keep_start', 'keep_end'],
                                  is_encoder_decoder: bool, **kwargs) -> Tuple[DATASET_TYPE, Optional[DATASET_TYPE]]:
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
        template=template,
        rlhf_type=rlhf_type,
        vision_keys=vision_keys,
        streaming=streaming,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        truncation_mode=truncation_mode,
        is_encoder_decoder=is_encoder_decoder)
    column_names = list(next(iter(train_dataset)).keys())
    with PartialState().local_main_process_first():
        train_dataset = train_dataset.map(preprocess_func, remove_columns=column_names, **kwargs)
        if val_dataset is not None:
            val_dataset = val_dataset.map(preprocess_func, remove_columns=column_names, **kwargs)
    return train_dataset, val_dataset


def patch_trl(is_vision_model: bool = False):
    from .callback import DefaultFlowCallbackNew, PrinterCallbackNew, ProgressCallbackNew
    from transformers import trainer

    trainer.DEFAULT_PROGRESS_CALLBACK = ProgressCallbackNew
    trainer.DEFAULT_CALLBACKS = [DefaultFlowCallbackNew]
    trainer.PrinterCallback = PrinterCallbackNew

    # fix encoder-decoder error
    if is_vision_model:
        patch_datacollator()
        patch_dataset_map()

    patch_itds_map()


def patch_datacollator():
    import torch
    from trl.trainer.utils import DPODataCollatorWithPadding, pad
    if not hasattr(DPODataCollatorWithPadding, '_old_call'):  # Avoid double patching
        from torch.nn.utils.rnn import pad_sequence
        from functools import wraps

        old_call = DPODataCollatorWithPadding.__call__

        @wraps(old_call)
        def new_call(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
            padded_batch = {}
            for k in features[0].keys():
                if k.endswith(('_input_ids', '_attention_mask', '_labels', '_pixel_values', '_images')):
                    if self.is_encoder_decoder:
                        to_pad = [torch.LongTensor(ex[k]) for ex in features]

                        if (k.startswith('prompt')) and (k.endswith('input_ids')):
                            if self.pad_token_id is None:
                                raise ValueError(
                                    'Padding is enabled, but the tokenizer is not configured with a padding token.'
                                    ' Explicitly set `tokenizer.pad_token`'
                                    ' (e.g. `tokenizer.pad_token = tokenizer.eos_token`)'
                                    ' before calling the trainer.')
                            padding_value = self.pad_token_id
                        elif k.endswith('_attention_mask'):
                            padding_value = 0
                        elif k.startswith(('chosen', 'rejected', 'completion')) or ('decoder' in k):
                            padding_value = self.label_pad_token_id
                        # patch here
                        elif k.endswith('_pixel_values'):
                            padding_value = 0
                        else:
                            raise ValueError(f"Unexpected key in batch '{k}'")
                        padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                    else:
                        # Set padding value based on the key
                        if k.endswith('_input_ids'):
                            if self.pad_token_id is None:
                                raise ValueError(
                                    'Padding is enabled, but the tokenizer is not configured with a padding token.'
                                    ' Explicitly set `tokenizer.pad_token`'
                                    ' (e.g. `tokenizer.pad_token = tokenizer.eos_token`)'
                                    ' before calling the trainer.')
                            padding_value = self.pad_token_id
                        elif k.endswith('_labels'):
                            padding_value = self.label_pad_token_id
                        elif k.endswith('_attention_mask'):
                            padding_value = 0
                        elif k.endswith(('_pixel_values', '_images')):
                            padding_value = 0
                        else:
                            raise ValueError(f"Unexpected key in batch '{k}'")

                        # Set padding side based on the key
                        if k in ['prompt_input_ids', 'prompt_attention_mask']:
                            padding_side = 'left'
                        else:
                            padding_side = 'right'

                        # Set the dtype
                        if k.endswith(('_pixel_values', '_images')):
                            dtype = torch.float32  # will be downcasted if necessary by the Trainer
                        else:
                            dtype = torch.int64

                        # Convert to tensor and pad
                        to_pad = [torch.tensor(ex[k], dtype=dtype) for ex in features]
                        padded_batch[k] = pad(to_pad, padding_value=padding_value, padding_side=padding_side)
                elif k.endswith('_logps'):
                    # the cached reference model logprobs
                    padded_batch[k] = torch.tensor([ex[k] for ex in features])
                else:
                    padded_batch[k] = [ex[k] for ex in features]

            return padded_batch

        DPODataCollatorWithPadding.__call__ = new_call
        DPODataCollatorWithPadding._old_call = old_call


def patch_itds_map():
    # resolve conflict with `num_proc` in iterable_dataset map func
    from datasets import IterableDataset
    from functools import wraps

    if not hasattr(IterableDataset, '_old_map'):  # Avoid double patching
        old_map = IterableDataset.map

        @wraps(old_map)
        def new_map(self, *args, **kwargs):
            kwargs.pop('num_proc', None)
            kwargs.pop('writer_batch_size', None)
            return old_map(self, *args, **kwargs)

        IterableDataset.map = new_map
        IterableDataset._old_map = old_map


def patch_dataset_map():
    original_map = HfDataset.map
    if not hasattr(HfDataset, '_old_map'):

        def patched_map(self, function, **kwargs):
            if 'writer_batch_size' not in kwargs:
                kwargs['writer_batch_size'] = 10
            return original_map(self, function, **kwargs)

        HfDataset.map = patched_map
        HfDataset._old_map = original_map

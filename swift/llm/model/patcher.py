# Copyright (c) Alibaba, Inc. and its affiliates.
from contextlib import contextmanager
from functools import wraps
from types import MethodType
from typing import List

import torch
import torch.nn as nn
from accelerate.utils import find_device
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

from swift.llm import to_device
from swift.utils import get_logger
from .model_arch import get_model_arch
from .utils import HfConfigFactory

logger = get_logger()


def patch_fixed_device(module: torch.nn.Module, device):
    """Move the output to the specific device"""

    def get_device_hook(device):

        def _device_hook(module, input, output):
            return to_device(output, device)

        return _device_hook

    module.register_forward_hook(get_device_hook(device))


def patch_output_clone(module: torch.nn.Module):
    """Clone the output, to avoid the inplace problem"""

    def _clone_hook(module, input, output):
        return output.requires_grad_(True).clone()

    module.register_forward_hook(_clone_hook)


def patch_output_to_input_device(module: torch.nn.Module):
    """Patch the module, to make sure the output is in the same device with the input.

    Args:
        module: The module to be patched
    """

    def _output_to_input_device_hook(module, args, kwargs, output):
        device = find_device(args) or find_device(kwargs)
        return to_device(output, device)

    module.register_forward_hook(_output_to_input_device_hook, with_kwargs=True)


@contextmanager
def patch_device_map():
    _get_no_split_modules = PreTrainedModel._get_no_split_modules

    def _new_get_no_split_modules(self, device_map: str):
        for module in self.modules():
            if isinstance(module, PreTrainedModel) and module._no_split_modules is None:
                module.__class__._no_split_modules = []
        return _get_no_split_modules(self, device_map)

    PreTrainedModel._get_no_split_modules = _new_get_no_split_modules
    try:
        yield
    finally:
        PreTrainedModel._get_no_split_modules = _get_no_split_modules


@contextmanager
def patch_ignore_check_imports():
    import transformers.dynamic_module_utils as td

    def _check_imports(filename) -> List[str]:
        return td.get_relative_imports(filename)

    _old_check_imports = td.check_imports
    td.check_imports = _check_imports
    try:
        yield
    finally:
        td.check_imports = _old_check_imports


def _patch_sequence_classification(model, model_meta):
    hidden_size = HfConfigFactory.get_config_attr(model.config, 'hidden_size')
    initializer_range = HfConfigFactory.get_config_attr(model.config, 'initializer_range')

    lm_heads = ['lm_head', 'output', 'embed_out', 'output_layer']
    llm_prefix = getattr(get_model_arch(model_meta.model_arch), 'language_model', None)
    if llm_prefix:
        llm_model = getattr(model, llm_prefix[0])
    else:
        llm_model = model
    if 'CausalLM' not in llm_model.__class__.__name__:
        llm_model = model
    llm_model.num_labels = model.config.num_labels
    llm_model.score = nn.Linear(hidden_size, llm_model.num_labels, bias=False, dtype=llm_model.dtype)
    if llm_model.score.weight.device == torch.device('meta'):
        llm_model.score.to_empty(device='cpu')
    llm_model.score.weight.data.normal_(mean=0.0, std=initializer_range)
    for lm_head in lm_heads:
        if hasattr(llm_model, lm_head):
            setattr(llm_model, lm_head, nn.Identity())
            break

    origin_forward = llm_model.forward.__func__

    @wraps(origin_forward)
    def new_forward(self, *args, **kwargs):
        labels = kwargs.pop('labels', None)
        return_dict = kwargs.pop('return_dict', None)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        input_ids = kwargs.get('input_ids')
        inputs_embeds = kwargs.get('inputs_embeds')

        output = origin_forward(self, *args, **kwargs)
        output.logits = output.logits.to(self.score.weight.dtype)
        logits = self.score(output.logits)
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError('Cannot handle batch sizes > 1 if no padding token is defined.')
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = 'regression'
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = 'single_label_classification'
                else:
                    self.config.problem_type = 'multi_label_classification'

            if self.config.problem_type == 'regression':
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == 'single_label_classification':
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == 'multi_label_classification':
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits, ) + output[1:]
            return ((loss, ) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )

    llm_model.forward = MethodType(new_forward, llm_model)


@contextmanager
def patch_automodel_for_sequence_classification(model_meta):
    from_pretrained = PreTrainedModel.from_pretrained.__func__

    @classmethod
    def _new_from_pretrained(cls, *args, **kwargs):
        cls_name = cls.__name__
        cls_name = cls_name.split('For', 1)[0]
        cls_name += 'ForSequenceClassification'
        cls = type(cls_name, (cls, ), {})  # new_cls
        __init__ = cls.__init__

        def __new_init__(self, *args, **kwargs):
            __init__(self, *args, **kwargs)
            _patch_sequence_classification(self, model_meta)

        cls.__init__ = __new_init__
        res = from_pretrained(cls, *args, **kwargs)
        cls.__init__ = __init__
        return res

    PreTrainedModel.from_pretrained = _new_from_pretrained

    try:
        yield
    finally:
        PreTrainedModel.from_pretrained = classmethod(from_pretrained)


@contextmanager
def patch_automodel_for_awq():
    from_pretrained = PreTrainedModel.from_pretrained.__func__

    @classmethod
    def _new_from_pretrained(cls, *args, **kwargs):
        kwargs.pop('use_cache', None)
        return from_pretrained(cls, *args, **kwargs)

    PreTrainedModel.from_pretrained = _new_from_pretrained

    try:
        yield
    finally:
        PreTrainedModel.from_pretrained = classmethod(from_pretrained)

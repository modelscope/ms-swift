# Copyright (c) Alibaba, Inc. and its affiliates.
from contextlib import contextmanager
from functools import wraps
from types import MethodType
from typing import Dict, List, Optional, Union

import accelerate
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from accelerate.utils import find_device
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import PreTrainedModel, dynamic_module_utils, trainer
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

from swift.llm import to_device, to_float_dtype
from swift.utils import get_dist_setting, get_logger, is_mp_ddp, safe_ddp_context, use_torchacc
from swift.utils.torch_utils import _get_max_memory, _sync_max_memory, get_device_count
from .model_arch import get_model_arch
from .utils import HfConfigFactory

logger = get_logger()


def patch_fixed_float_dtype(module: torch.nn.Module, dtype):
    """Patch the module, to make sure the consisitent dtype."""

    def get_float_dtype_hook(dtype):

        def _float_dtype_hook(module, input, output):
            return to_float_dtype(output, dtype)

        return _float_dtype_hook

    module.register_forward_hook(get_float_dtype_hook(dtype))


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


def patch_output_normalizer(module: torch.nn.Module, model_meta):

    def lm_head_forward(self, hidden_states):
        return hidden_states

    lm_heads = ['lm_head', 'output', 'embed_out', 'output_layer']
    llm_prefix = getattr(get_model_arch(model_meta.model_arch), 'language_model', None)
    if llm_prefix:
        llm_model = getattr(module, llm_prefix[0])
    else:
        llm_model = module

    if 'CausalLM' not in llm_model.__class__.__name__:
        llm_model = module

    found = False
    for lm_head in lm_heads:
        if hasattr(llm_model, lm_head):
            getattr(llm_model, lm_head).forward = MethodType(lm_head_forward, getattr(llm_model, lm_head))
            found = True
            break

    assert found, 'Cannot find the proper lm_head name'

    def forward(self, input_ids: torch.LongTensor = None, attention_mask=None, *args, **kwargs):

        outputs = self.forward_origin(input_ids=input_ids, attention_mask=attention_mask, *args, **kwargs)
        hidden_states = outputs.logits
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            embeddings = hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.shape[0]
            embeddings = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return {
            'last_hidden_state': embeddings.contiguous(),
        }

    llm_model.forward_origin = llm_model.forward
    llm_model.forward = MethodType(forward, llm_model)


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
    if 'CausalLM' not in llm_model.__class__.__name__:  # fix qwen2_vl
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
def patch_automodel(automodel_class, model_info):
    from_pretrained = PreTrainedModel.from_pretrained.__func__

    @classmethod
    def _new_from_pretrained(cls, *args, **kwargs):
        if 'AutoAWQFor' in automodel_class.__name__:
            kwargs.pop('use_cache', None)
        if model_info.quant_method == 'gptq':
            cls.main_input_name = 'input_ids'
        return from_pretrained(cls, *args, **kwargs)

    PreTrainedModel.from_pretrained = _new_from_pretrained

    try:
        yield
    finally:
        PreTrainedModel.from_pretrained = classmethod(from_pretrained)


_mp_ddp_patched = False


def patch_mp_ddp():
    """Patch ddp with device_map.
    After patching, the ddp can run with the device_map.
    This should be called before any training starts.
    """
    global _mp_ddp_patched
    if is_mp_ddp() and not _mp_ddp_patched:
        _mp_ddp_patched = True
        from accelerate.utils.modeling import get_balanced_memory, infer_auto_device_map

        @wraps(infer_auto_device_map)
        def _infer_auto_device_map_patch(model: nn.Module,
                                         max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,
                                         **kwargs) -> Dict[str, Union[int, str, torch.device]]:
            """The auxiliary function for supports MP + DDP. Monkey Patching.
            add feat in accelerate to support MP + DDP"""
            verbose = kwargs.pop('verbose', False)
            n_gpu = get_device_count()
            _, local_rank, _, local_world_size = get_dist_setting()
            device_ids = list(range(local_rank, n_gpu, local_world_size))
            max_memory = _get_max_memory(device_ids)
            max_memory = _sync_max_memory(max_memory)
            max_memory = get_balanced_memory(model, max_memory, low_zero=False, **kwargs)
            max_memory = {k: v for k, v in max_memory.items() if v > 0}
            return infer_auto_device_map(model, max_memory, verbose=verbose, **kwargs)

        _old_ddp_init = DDP.__init__
        accelerate.accelerator.torch.nn.parallel.DistributedDataParallel.__init__ = (
            lambda self, model, device_ids, output_device, *args, **kwargs: _old_ddp_init(self, model, *args, **kwargs))
        transformers.modeling_utils.get_balanced_memory = lambda *args, **kwargs: None
        transformers.modeling_utils.infer_auto_device_map = _infer_auto_device_map_patch

    if is_mp_ddp() or use_torchacc():
        _old_accelerator_init = trainer.Accelerator.__init__
        trainer.Accelerator.__init__ = (lambda self, device_placement=False, *args, **kwargs: _old_accelerator_init(
            self, device_placement=device_placement, *args, **kwargs))
        trainer.Accelerator.verify_device_map = lambda *args, **kwargs: False


@contextmanager
def patch_get_dynamic_module():
    origin_get_cached_module_file = dynamic_module_utils.get_cached_module_file

    def new_get_cached_module_file(pretrained_model_name_or_path, *args, **kwargs):
        with safe_ddp_context(hash_id=str(pretrained_model_name_or_path)):
            return origin_get_cached_module_file(pretrained_model_name_or_path, *args, **kwargs)

    dynamic_module_utils.get_cached_module_file = new_get_cached_module_file
    try:
        yield
    finally:
        dynamic_module_utils.get_cached_module_file = origin_get_cached_module_file

# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import os
from contextlib import contextmanager
from functools import wraps
from types import MethodType
from typing import Any, Dict, List, Optional, Union

import accelerate
import torch
import torch.nn as nn
import transformers
from accelerate.utils import find_device
from packaging import version
from peft import PeftModel
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import PreTrainedModel, dynamic_module_utils, trainer
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

from swift.llm import deep_getattr, to_device, to_float_dtype
from swift.utils import get_dist_setting, get_logger, is_mp, is_mp_ddp, safe_ddp_context
from swift.utils.torch_utils import (_get_max_memory, _sync_max_memory, get_cu_seqlens_from_position_ids,
                                     get_device_count, get_position_ids_from_cu_seqlens)
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


def patch_get_input_embeddings(model, embedding_keys: str):

    def get_input_embeddings(self) -> nn.Module:
        return deep_getattr(model, embedding_keys)

    model.get_input_embeddings = MethodType(get_input_embeddings, model)


def patch_output_normalizer(module: torch.nn.Module, model_meta):

    def lm_head_forward(self, hidden_states):
        return hidden_states

    lm_heads = ['lm_head', 'output', 'embed_out', 'output_layer']
    llm_model = get_lm_head_model(module, model_meta=model_meta)

    found = False
    for lm_head in lm_heads:
        if hasattr(llm_model, lm_head):
            getattr(llm_model, lm_head).forward = MethodType(lm_head_forward, getattr(llm_model, lm_head))
            found = True
            break

    assert found, 'Cannot find the proper lm_head name'

    def _output_embedding_hook(module, args, kwargs, output):
        attention_mask = kwargs['attention_mask']
        hidden_states = output.logits
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

    llm_model.register_forward_hook(_output_embedding_hook, with_kwargs=True)


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


def get_lm_head_model(model, model_meta=None, lm_heads=None):
    if isinstance(model, PeftModel):
        model = model.model
    model_meta = model_meta or model.model_meta
    lm_heads = lm_heads or ['lm_head']
    llm_prefix_list = getattr(model_meta.model_arch, 'language_model', None)
    prefix_list = []
    if llm_prefix_list:
        prefix_list = llm_prefix_list[0].split('.')

    current_model = model
    for prefix in prefix_list:
        current_model = getattr(current_model, prefix)
        for lm_head in lm_heads:
            if hasattr(current_model, lm_head):
                return current_model
    return model


def transformers_seq_cls_forward(self, *args, origin_forward, **kwargs):
    labels = kwargs.pop('labels', None)
    return_dict = kwargs.pop('return_dict', None)
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    input_ids = kwargs.get('input_ids')
    inputs_embeds = kwargs.get('inputs_embeds')

    output = origin_forward(*args, **kwargs)
    if hasattr(output, 'logits'):
        output.logits = output.logits.to(self.score.weight.dtype)
    elif 'last_hidden_state' in output:
        output.logits = output['last_hidden_state'].to(self.score.weight.dtype)
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
        if output.get('attention_mask') is not None:
            # When use padding_free in seq_cls tasks, `revert_padding_free` will add a attention_mask in the output
            batch_size = output.get('attention_mask').shape[0]
            sequence_lengths = output.get('attention_mask').sum(dim=1) - 1
        elif input_ids is not None:
            # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
            sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
            sequence_lengths = sequence_lengths % input_ids.shape[-1]
            sequence_lengths = sequence_lengths.to(logits.device)
        elif kwargs.get('attention_mask') is not None:
            sequence_lengths = kwargs['attention_mask'].sum(dim=1) - 1
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


def _patch_sequence_classification(model, model_meta):
    hidden_size = HfConfigFactory.get_config_attr(model.config, 'hidden_size')
    initializer_range = HfConfigFactory.get_config_attr(model.config, 'initializer_range')

    lm_heads = ['lm_head', 'output', 'embed_out', 'output_layer']
    llm_model = get_lm_head_model(model, model_meta, lm_heads)
    llm_model.num_labels = model.config.num_labels
    llm_model.score = nn.Linear(hidden_size, llm_model.num_labels, bias=False, dtype=llm_model.dtype)
    if llm_model.score.weight.device == torch.device('meta'):
        llm_model.score.to_empty(device='cpu')
    llm_model.score.weight.data.normal_(mean=0.0, std=initializer_range)
    for lm_head in lm_heads:
        if hasattr(llm_model, lm_head):
            setattr(llm_model, lm_head, nn.Identity())
            break

    origin_forward = llm_model.forward

    @wraps(origin_forward.__func__)
    def new_forward(self, *args, **kwargs):
        return transformers_seq_cls_forward(self, *args, origin_forward=origin_forward, **kwargs)

    llm_model.forward = MethodType(new_forward, llm_model)


@contextmanager
def patch_automodel_for_sequence_classification(model_info=None,
                                                model_meta=None,
                                                patch_from_pretrained=True,
                                                patch_missing_init=True,
                                                **kwargs):
    """
    Context manager for patching AutoModel sequence classification.

    Args:
        model_info: Model information
        model_meta: Model metadata
        patch_from_pretrained (bool): Whether to patch PreTrainedModel.from_pretrained
        patch_missing_init (bool): Whether to patch missing __init__ methods
        **kwargs: Additional keyword arguments
    """
    model_config = kwargs.get('model_config', None)
    from_pretrained = PreTrainedModel.from_pretrained.__func__

    # Patch 1: from_pretrained method
    _new_from_pretrained = None
    if patch_from_pretrained:

        @classmethod
        def _new_from_pretrained(cls, *args, **kwargs):
            __init__ = cls.__init__

            def __new_init__(self, *args, **kwargs):
                __init__(self, *args, **kwargs)
                _patch_sequence_classification(self, model_meta)

            cls.__init__ = __new_init__
            if hasattr(cls, '_tp_plan'):  # fix tp_plan
                cls._tp_plan = cls._tp_plan or {}
            res = from_pretrained(cls, *args, **kwargs)
            cls.__init__ = __init__
            return res

    # Patch 2: missing __init__ methods
    # https://github.com/modelscope/ms-swift/pull/5820
    patched_classes = []
    if patch_missing_init:

        def get_all_subclasses(cls, include_root=True):
            subclass_list = []

            def recurse(cl):
                for subclass in cl.__subclasses__():
                    subclass_list.append(subclass)
                    recurse(subclass)

            recurse(cls)

            ret = set(subclass_list)
            if include_root:
                ret.add(cls)
            return ret

        def create_default_init(cls):
            """Create a default __init__ method that calls super().__init__"""

            def default_init(self, *args, **kwargs):
                super(cls, self).__init__(*args, **kwargs)

            return default_init

        if model_config is not None:
            # we should import in advance so that get_all_subclasses can find the class
            archs = model_config.architectures
            for arch in archs:
                try:
                    getattr(transformers, arch)
                except AttributeError:
                    continue

        for subclass in get_all_subclasses(torch.nn.modules.module.Module):
            if '__init__' not in subclass.__dict__:
                subclass.__init__ = create_default_init(subclass)
                patched_classes.append(subclass)

    if patch_from_pretrained:
        PreTrainedModel.from_pretrained = _new_from_pretrained

    try:
        yield
    finally:
        # Restore patches
        if patch_from_pretrained:
            PreTrainedModel.from_pretrained = classmethod(from_pretrained)

        if patch_missing_init:
            for subclass in patched_classes:
                try:
                    if '__init__' in subclass.__dict__:
                        del subclass.__init__
                except (AttributeError, TypeError):
                    pass


@contextmanager
def patch_automodel(model_info, model_meta, automodel_class, return_dummy_model, **kwargs):
    from_pretrained = PreTrainedModel.from_pretrained.__func__

    @classmethod
    def _new_from_pretrained(cls, *args, **kwargs):
        if 'AutoAWQFor' in automodel_class.__name__:
            kwargs.pop('use_cache', None)
        if model_info.quant_method == 'gptq':
            cls.main_input_name = 'input_ids'
        if hasattr(cls, '_tp_plan'):  # fix tp_plan
            cls._tp_plan = cls._tp_plan or {}
        if return_dummy_model:
            origin_torch_dtype = torch.get_default_dtype()
            torch.set_default_dtype(kwargs['config'].torch_dtype)
            model = cls(copy.deepcopy(kwargs['config']))
            torch.set_default_dtype(origin_torch_dtype)
        else:
            model = from_pretrained(cls, *args, **kwargs)
        return model

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
    if _mp_ddp_patched:
        return
    _mp_ddp_patched = True
    if is_mp_ddp():
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
        transformers.modeling_utils.get_balanced_memory = lambda *args, **kwargs: {}
        transformers.modeling_utils.infer_auto_device_map = _infer_auto_device_map_patch

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


@contextmanager
def patch_tp_plan(load_model: bool):
    if not load_model or not is_mp() or version.parse(
            transformers.__version__) < version.parse('4.50') or 'WORLD_SIZE' not in os.environ:
        yield
        return
    logger.info('Patch tp_plan.')
    WORLD_SIZE = os.environ.get('WORLD_SIZE')
    os.environ['_PATCH_WORLD_SIZE'] = WORLD_SIZE
    os.environ.pop('WORLD_SIZE')
    yield
    os.environ['WORLD_SIZE'] = WORLD_SIZE


def revert_padding_free(outputs: Dict[str, Any], inputs: Dict[str, Any], padding_side='left'):
    hidden_state_key = None
    if 'last_hidden_state' in outputs:
        hidden_state_key = 'last_hidden_state'
    elif 'logits' in outputs:
        hidden_state_key = 'logits'
    elif 'token_embeddings' in outputs:
        hidden_state_key = 'token_embeddings'

    if hidden_state_key is None:
        raise NotImplementedError()
    last_hidden_state = outputs[hidden_state_key]
    last_hidden_state = last_hidden_state.squeeze(dim=0)
    if 'cu_seq_lens_q' in inputs:
        position_ids = get_position_ids_from_cu_seqlens(inputs['cu_seq_lens_q'])
    elif 'position_ids' in inputs and inputs['position_ids'].shape[0] == 1:
        position_ids = inputs['position_ids']
    else:
        raise ValueError(
            "revert_padding_free requires 'cu_seq_lens_q' or 'position_ids' in inputs, but neither was found.")

    seq_lengths = []
    pos = position_ids[0]
    resets = torch.where(pos[1:] < pos[:-1])[0] + 1

    if len(resets) == 0:
        # Only one sequence in this batch item
        seq_lengths = [pos.max().item() + 1]
    else:
        # Multiple sequences
        start = 0
        for end in resets:
            seq_lengths.append(end - start)
            start = end
        seq_lengths.append(pos.shape[0] - start)

    max_length = max(seq_lengths)
    unpacked_logits = []
    attention_mask = []

    start = 0
    for length in seq_lengths:
        seq_state = last_hidden_state[start:start + length]
        mask = torch.ones((seq_state.shape[0])).to(last_hidden_state.device)
        padding = torch.zeros(
            (max_length - length, last_hidden_state.shape[-1])).to(last_hidden_state.dtype).to(last_hidden_state.device)
        attention_padding = torch.zeros((max_length - length)).to(last_hidden_state.device)
        # re-padding
        if padding_side == 'left':
            seq_state = torch.cat((padding, seq_state), dim=0)
            mask = torch.cat((attention_padding, mask), dim=0)
        else:
            seq_state = torch.cat((seq_state, padding), dim=0)
            mask = torch.cat((mask, attention_padding), dim=0)
        unpacked_logits.append(seq_state)
        attention_mask.append(mask)
        start += length
    outputs[hidden_state_key] = torch.stack(unpacked_logits, dim=0)
    inputs['attention_mask'] = torch.stack(attention_mask, dim=0).to(torch.int64)
    outputs['attention_mask'] = inputs['attention_mask']
    return outputs


@contextmanager
def patch_attach_align_device_hook_on_blocks():
    from accelerate import big_modeling
    origin_attach_align_device_hook_on_blocks = big_modeling.attach_align_device_hook_on_blocks

    def attach_align_device_hook_on_blocks(*args, **kwargs):
        return

    big_modeling.attach_align_device_hook_on_blocks = attach_align_device_hook_on_blocks
    try:
        yield
    finally:
        big_modeling.attach_align_device_hook_on_blocks = origin_attach_align_device_hook_on_blocks

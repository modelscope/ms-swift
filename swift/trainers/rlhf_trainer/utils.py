# Copyright (c) Alibaba, Inc. and its affiliates.
import functools
import math
import os
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from functools import partial
from io import BytesIO
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, TypeVar, Union

import datasets
import torch
import torch.nn.functional as F
from msgspec import field
from peft.tuners import lora
from peft.tuners.lora import LoraLayer
from PIL import Image
from pydantic import BaseModel, field_validator
from torch import nn
from torch.utils.data import DataLoader, RandomSampler

from swift.utils import gc_collect, get_logger, is_swanlab_available, is_vllm_available, is_wandb_available
from swift.utils.torch_utils import get_torch_device

if is_wandb_available():
    import wandb
if is_swanlab_available():
    import swanlab

if TYPE_CHECKING:
    from swift.llm.utils import Messages
T = TypeVar('T')

TensorLoRARequest = None
if is_vllm_available():
    from vllm.lora.request import LoRARequest

    class TensorLoRARequest(LoRARequest):
        peft_config: dict = field(default=None)
        lora_tensors: dict = field(default=None)
        lora_embeddings: Optional[Dict[str, torch.Tensor]] = None

        @property
        def config(self):
            return self.peft_config

        @property
        def embeddings(self):
            return self.lora_embeddings


# code borrowed from verl/verl/utils/memory_utils.py
def aggressive_empty_cache(force_sync: bool = True, max_retries: int = 3) -> None:
    """
    More aggressive GPU memory cleanup function, tries to release PyTorch reserved
    but unallocated memory.

    Args:
        force_sync: Whether to force device synchronization
        max_retries: Maximum number of retries
    """
    logger = get_logger()

    device = get_torch_device()
    if not hasattr(device, 'is_available') or not device.is_available():
        return

    for attempt in range(max_retries):
        # Record memory status before cleanup
        before_reserved = device.memory_reserved()
        before_allocated = device.memory_allocated()

        # Run garbage collection
        gc_collect()

        # Clear PyTorch cache
        device.empty_cache()

        # Force synchronization (optional)
        if force_sync:
            device.synchronize()

        # Record memory status after cleanup
        after_reserved = device.memory_reserved()
        after_allocated = device.memory_allocated()

        # Calculate freed memory
        reserved_freed = before_reserved - after_reserved
        allocated_freed = before_allocated - after_allocated

        logger.info(f'Memory cleanup attempt {attempt + 1}: Freed {reserved_freed / 1024**3:.2f} GB reserved, '
                    f'{allocated_freed / 1024**3:.2f} GB allocated')

        # Stop retrying if little memory was freed
        if reserved_freed < 1024**3:  # less than 1GB
            break


def prepare_deepspeed(model, accelerator, deepspeed_config=None, deepspeed_plugin=None, training_args=None):
    """
    Prepares the model for DeepSpeed inference or evaluation by initializing it with the appropriate configuration.

    Args:
        model: The model to prepare
        accelerator: The accelerator instance
        deepspeed_config: Optional deepspeed config. If provided, use this instead of accelerator's plugin.
        deepspeed_plugin: Optional DeepSpeedPlugin. If provided, use this instead of accelerator's plugin.
        training_args: Optional training arguments for resolving "auto" values in config

    Returns:
        The prepared DeepSpeed model
    """
    try:
        import deepspeed
        import os
        from copy import deepcopy
        from transformers.integrations.deepspeed import HfTrainerDeepSpeedConfig
        from accelerate.utils import DeepSpeedPlugin
    except ImportError:
        pass

    # Determine which config to use and create HfTrainerDeepSpeedConfig
    if deepspeed_config is not None:
        # Use provided config - need to wrap it with HfTrainerDeepSpeedConfig to handle "auto" values
        if isinstance(deepspeed_config, dict):
            # Create HfTrainerDeepSpeedConfig which will handle "auto" values
            hf_ds_config = HfTrainerDeepSpeedConfig(deepspeed_config)

            # Process the config with training args to resolve "auto" values
            if training_args is not None:
                hf_ds_config.trainer_config_process(training_args)

            # Create a DeepSpeedPlugin with the processed config
            temp_plugin = DeepSpeedPlugin(hf_ds_config=hf_ds_config)
            config_kwargs = deepcopy(temp_plugin.deepspeed_config)
        else:
            raise ValueError(f'deepspeed_config should be a dict, got {type(deepspeed_config)}')
    elif deepspeed_plugin is not None:
        # Use provided plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)
    else:
        # Use accelerator's plugin (default behavior)
        deepspeed_plugin = accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

    stage = config_kwargs['zero_optimization']['stage']

    if model is not None:
        hidden_size = (
            max(model.config.hidden_sizes) if getattr(model.config, 'hidden_sizes', None) else getattr(
                model.config, 'hidden_size', None))
        if hidden_size is not None and stage == 3:
            # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache
            # @ step 0: expected module 1, but got module 0`
            # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
            config_kwargs.update({
                'zero_optimization.reduce_bucket_size': hidden_size * hidden_size,
                'zero_optimization.stage3_param_persistence_threshold': 10 * hidden_size,
                'zero_optimization.stage3_prefetch_bucket_size': 0.9 * hidden_size * hidden_size,
            })

    # If ZeRO-3 is used, we shard both the active and reference model.
    # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO
    # disabled (stage 0)
    if stage != 3:
        config_kwargs['zero_optimization']['stage'] = 0

    # CRITICAL: Save and clear DeepSpeed-related environment variables before initialization
    # These environment variables (set during student model's DeepSpeed init) can override our config!
    # Reference: https://github.com/microsoft/DeepSpeed/issues/xxxx
    env_vars_to_clear = [
        'DEEPSPEED_ZERO_STAGE',
        'DEEPSPEED_CONFIG',
        'DEEPSPEED_CONFIG_FILE',
    ]
    saved_env = {}
    for env_var in env_vars_to_clear:
        if env_var in os.environ:
            saved_env[env_var] = os.environ[env_var]
            del os.environ[env_var]

    try:
        # Explicitly pass args=None to ensure no args.deepspeed_config interference
        model, *_ = deepspeed.initialize(args=None, model=model, config=config_kwargs)
        model.eval()

    finally:
        # Restore environment variables
        for env_var, value in saved_env.items():
            os.environ[env_var] = value

    return model


@contextmanager
def memory_time_profiling_context(
    name: str = 'Operation',
    enable_profiling: bool = True,
    sync_cuda: bool = True,
    reset_peak_stats: bool = True,
):
    """
    General-purpose memory and time profiling context manager (pure monitoring, no execution).

    Records memory usage and execution time when entering and exiting the context, but does not
    handle any actual model loading/offloading operations.

    Args:
        name: Operation name for logging identification
        enable_profiling: Whether to enable profiling records
        sync_cuda: Whether to synchronize CUDA before recording (ensures accuracy with slight overhead)
        reset_peak_stats: Whether to reset peak memory statistics on exit
    """
    if not enable_profiling:
        yield
        return

    logger = get_logger()

    # ===== Entry phase: Record initial state =====
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()

    gc_collect()

    # Record initial memory state
    memory_before = torch.cuda.memory_allocated() / 1024**3  # GiB
    memory_reserved_before = torch.cuda.memory_reserved() / 1024**3
    max_memory_before = torch.cuda.max_memory_allocated() / 1024**3

    logger.info(f'[{name}] Before: '
                f'Allocated = {memory_before:.2f} GiB, '
                f'Reserved = {memory_reserved_before:.2f} GiB, '
                f'Peak = {max_memory_before:.2f} GiB')

    # Start timing
    start_time = time.perf_counter()

    yield

    # Synchronize and clean up memory before measuring (important for offload operations)
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    gc_collect()

    # ===== Exit phase: Record final state =====
    # Calculate elapsed time (before cleanup to measure actual operation time)
    elapsed_time = time.perf_counter() - start_time

    # Record final memory state
    memory_after = torch.cuda.memory_allocated() / 1024**3
    memory_reserved_after = torch.cuda.memory_reserved() / 1024**3
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    memory_change = memory_after - memory_before

    logger.info(f'[{name}] After: '
                f'Allocated = {memory_after:.2f} GiB, '
                f'Reserved = {memory_reserved_after:.2f} GiB, '
                f'Peak = {peak_memory:.2f} GiB, '
                f'Change = {memory_change:+.2f} GiB, '
                f'Time = {elapsed_time:.2f}s')

    # Reset peak memory statistics for next cycle
    if reset_peak_stats and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def round_robin(num_reqs, num_workers):
    """Distribute requests evenly across workers using round-robin algorithm.

    Args:
        num_reqs (int): Total number of requests to distribute
        num_workers (int): Number of available workers

    Returns:
        list: A list of lists where each sublist contains the request indices
                assigned to that particular node
    """
    distribution = [[] for _ in range(num_workers)]
    for idx in range(num_reqs):
        worker_id = idx % num_workers
        distribution[worker_id].append(idx)
    return distribution


@contextmanager
def patch_lora_merge(model, parameter_group=None):
    """Patch LoraLayer's merge and get_delta_weight methods for controlled merging.

    Args:
        model: The PEFT model to patch
        parameter_group: Optional list of parameter names to restrict merging

    Yields:
        The patched model (context manager ensures cleanup)
    """
    from peft.tuners.tuners_utils import check_adapters_to_merge

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        if parameter_group and all(self.name not in pg for pg in parameter_group):
            return  # Skip if not in target parameter group
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if self.use_dora.get(active_adapter, False):
                    self.lora_magnitude_vector[active_adapter].weight.data = \
                        self.lora_magnitude_vector[active_adapter].weight.data.to(base_layer.weight.device)

        return self.merge_origin(safe_merge, adapter_names)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        # Ensure tensors are on correct device
        if isinstance(self, lora.Embedding):
            self.lora_embedding_A[adapter].data = self.lora_embedding_A[adapter].data.to(self.base_layer.weight.device)
            self.lora_embedding_B[adapter].data = self.lora_embedding_B[adapter].data.to(self.base_layer.weight.device)
        else:
            self.lora_A[adapter].weight.data = self.lora_A[adapter].weight.data.to(self.base_layer.weight.device)
            self.lora_B[adapter].weight.data = self.lora_B[adapter].weight.data.to(self.base_layer.weight.device)
        return self.get_delta_weight_origin(adapter).to(self.base_layer.weight.device)

    def _cache_pop(self, key: str) -> Any:
        value = self._caches.pop(key).to(self.base_layer.weight.device)
        return value

    # Patch all LoraLayer instances
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            module.name = name
            if not hasattr(module, 'merge_origin') and hasattr(module, 'base_layer'):
                module.merge_origin = module.merge
                module.merge = MethodType(merge, module)
                module.get_delta_weight_origin = module.get_delta_weight
                module.get_delta_weight = MethodType(get_delta_weight, module)
                module._cache_pop_origin = module._cache_pop
                module._cache_pop = MethodType(_cache_pop, module)

    try:
        yield model
    finally:
        # Cleanup: restore original methods
        for module in model.modules():
            if isinstance(module, LoraLayer):
                if hasattr(module, 'merge_origin'):
                    module.merge = module.merge_origin
                    del module.merge_origin
                    module.get_delta_weight = module.get_delta_weight_origin
                    del module.get_delta_weight_origin
                    module._cache_pop = module._cache_pop_origin
                    del module._cache_pop_origin


@contextmanager
def patch_lora_unmerge(model):

    def unmerge_patched(self):
        if not self.merged:
            return
        # Move magnitude vectors to correct device first
        for adapter in list(self.merged_adapters):
            if self.use_dora.get(adapter, False):
                self.lora_magnitude_vector[adapter].weight.data = \
                    self.lora_magnitude_vector[adapter].weight.data.to(self.base_layer.weight.device)

        return self.unmerge_origin()

    for module in model.modules():
        if isinstance(module, LoraLayer) and not hasattr(module, 'unmerge_origin'):
            module.unmerge_origin = module.unmerge
            module.unmerge = MethodType(unmerge_patched, module)

    try:
        yield model
    finally:
        for module in model.modules():
            if isinstance(module, LoraLayer) and hasattr(module, 'unmerge_origin'):
                module.unmerge = module.unmerge_origin
                del module.unmerge_origin


@contextmanager
def patch_profiling_context(trainer, name: str):
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    duration = end_time - start_time

    profiling_metrics = {f'profiling/Time taken: {trainer.__class__.__name__}.{name}': duration}

    if 'wandb' in trainer.args.report_to and wandb.run is not None and trainer.accelerator.is_main_process:
        wandb.log(profiling_metrics)

    if 'swanlab' in trainer.args.report_to and swanlab.get_run() is not None and trainer.accelerator.is_main_process:
        swanlab.log(profiling_metrics)


def patch_profiling_decorator(func):

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        with patch_profiling_context(self, func.__name__):
            return func(self, *args, **kwargs)

    return wrapper


class _ForwardRedirection:
    """Implements the `forward-redirection`.
    Taken from Pytorch-lightning:
    https://github.com/Lightning-AI/pytorch-lightning/blob/02311d03fb982560246eead7c08104481fac9579/src/lightning/pytorch/strategies/strategy.py#L602
    A method call to a wrapped module gets rerouted through the wrapper's `forward` method instead.
    """

    def __call__(self, wrapper_module: nn.Module, original_module: nn.Module, method: callable, *args: Any,
                 **kwargs: Any):
        """Reroutes a method call through the `wrapper_module`'s `forward` method.
        Args:
            wrapper_module: The module that has `original_module` wrapped.
            original_module: The module that was wrapped inside `wrapper_module`.
            method_name: The name of the method that should be called on the `original_module` after inputs get
                redirected through the `wrapper_module`'s `forward` method.
            *args: The positional arguments to the method `method_name`. They will get passed to a patched
                `forward` method instead.
            **kwargs: The keyword arguments to the method `method_name`. They will get passed to a patched
                `forward` method instead.
        """
        original_forward = original_module.forward

        def wrapped_forward(*_args: Any, **_kwargs: Any) -> Any:
            # Unpatch ourselves immediately before calling the method `method_name`
            # because itself may want to call the real `forward`
            original_module.forward = original_forward  # type: ignore[method-assign]
            # Call the actual method e.g. `.training_step(...)`
            out = method(*_args, **_kwargs)
            self.on_after_inner_forward(wrapper_module, original_module)
            return out

        # Patch the original_module's forward so we can redirect the arguments back to the real method
        original_module.forward = wrapped_forward  # type: ignore[method-assign]

        wrapper_output = wrapper_module(*args, **kwargs)
        self.on_after_outer_forward(wrapper_module, original_module)
        return wrapper_output

    def on_after_inner_forward(self, wrapper_module: nn.Module, original_module: nn.Module) -> None:
        pass

    def on_after_outer_forward(self, wrapper_module: nn.Module, original_module: nn.Module) -> None:
        pass


def entropy_from_logits(logits, chunk_size: int = 1) -> torch.Tensor:
    """
    Compute the Shannon entropy (in nats) for each row of *logits* without
    materialising the full soft-max in memory.
    The batch dimension is processed in chunks of size `chunk_size` so that
    only a subset of rows is expanded to probabilities at any one time.
    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`. Entropy is taken along the last axis; all
            leading dimensions are preserved.
        chunk_size (`int`, *optional*, defaults to `1`):
            Number of rows to process per iteration.
    Returns:
        `torch.Tensor`:
            Entropy values with shape `logits.shape[:-1]`.
    """
    per_token_entropies = []
    for logits_chunk in logits.split(chunk_size, dim=0):
        logps = F.log_softmax(logits_chunk, dim=-1)
        chunk_entropy = -(torch.exp(logps) * logps).sum(-1)
        per_token_entropies.append(chunk_entropy)
    return torch.cat(per_token_entropies, dim=0)


def load_pil_img(img) -> Image:
    if isinstance(img, (list, tuple)):
        if len(img) == 1:
            img = img[0]
        else:
            raise ValueError('Image list must contain a single image.')

    if isinstance(img, Image.Image):
        return img
    if isinstance(img, str):
        return Image.open(img)

    if not isinstance(img, dict):
        raise ValueError("Image must be a PIL Image, a file path, or a dictionary with 'bytes' or 'path' key.")

    if 'bytes' in img and img['bytes'] is not None:
        return Image.open(BytesIO(img['bytes']))
    elif 'path' in img and img['path'] is not None:
        return Image.open(img['path'])
    else:
        raise ValueError("Image dictionary must contain either 'bytes' or 'path' key.")


def replace_assistant_response_with_ids(messages: 'Messages',
                                        completion_ids: List[Union[int, List[int]]],
                                        loss_mask: Optional[List[List[int]]] = None) -> 'Messages':  # noqa
    """
    Replace assistant messages in a conversation with token IDs (and optional loss masks).

    This function traverses the messages in reverse order and replaces the content of
    assistant-role messages with the given `completion_ids`. If `loss_mask` is provided,
    each assistant message content will be replaced by a dictionary containing both the
    token IDs and the corresponding loss mask.

    Args:
        messages:
            List of message dictionaries representing a conversation history.
        completion_ids:
            Either:
              - A single list of token IDs, e.g. [1, 2, 3]
              - A list of completion sequences, e.g. [[1, 2], [3, 4]]
        loss_mask (optional):
            Loss mask(s) aligned with `completion_ids`.
            Must satisfy:
              - Same outer length as `completion_ids`
              - Each inner list has the same length as the corresponding completion_ids sequence
            Example:
              completion_ids = [[1, 2], [3, 4]]
              loss_mask      = [[1, 1], [1, 0]]

    Returns:
        The modified messages list, where assistant responses are replaced by:
          - A list of token IDs if `loss_mask` is None
          - A dict with keys:
              - "input_ids": List[int]
              - "loss_scale": List[int]
            if `loss_mask` is provided.

    Example:
        >>> messages = [
        ...     {"role": "user", "content": "Hello"},
        ...     {"role": "assistant", "content": "Hi there"}
        ... ]
        >>> replace_assistant_response_with_ids(messages, [1, 2, 3])
        [{'role': 'user', 'content': 'Hello'},
         {'role': 'assistant', 'content': [1, 2, 3]}]

        >>> replace_assistant_response_with_ids(messages,
        ...     completion_ids=[[1, 2, 3]],
        ...     loss_mask=[[1, 1, 0]])
        [{'role': 'user', 'content': 'Hello'},
         {'role': 'assistant', 'content': {'input_ids': [1, 2, 3], 'loss_scale': [1, 1, 0]}}]
    """
    # Normalize input to always be list of lists
    if isinstance(completion_ids[0], int):
        completion_ids = [completion_ids]
    if loss_mask and isinstance(loss_mask[0], int):
        loss_mask = [loss_mask]

    if loss_mask:
        assert (
            len(completion_ids) == len(loss_mask)
            and all(len(ids) == len(mask) for ids, mask in zip(completion_ids, loss_mask))
        ), f'completion_ids and loss_mask must have the same length, but got {len(completion_ids)} and {len(loss_mask)}'

    remaining_completions = len(completion_ids)
    completion_index = 0

    for message in reversed(messages):
        if message['role'] != 'assistant':
            continue

        if completion_index >= remaining_completions:
            break

        # Assign completion IDs (starting from last)
        if loss_mask:
            message['content'] = {
                'loss_scale': loss_mask[-1 - completion_index],
                'token_ids': completion_ids[-1 - completion_index]
            }
        else:
            message['content'] = completion_ids[-1 - completion_index]

        completion_index += 1

    return messages


def patch_save_last_checkpoint():
    import trl
    from packaging import version
    if version.parse(trl.__version__) >= version.parse('0.20'):
        return

    # patch to fix save last_checkpoint https://github.com/modelscope/ms-swift/pull/4969
    from trl.trainer.grpo_trainer import RepeatSampler
    if not hasattr(RepeatSampler, 'old_len_func'):
        origin_len_func = RepeatSampler.__len__

        def patched_len(self) -> int:
            return (self.num_samples // self.batch_size) * self.batch_size * self.mini_repeat_count * self.repeat_count

        RepeatSampler.__len__ = patched_len
        RepeatSampler.old_len_func = origin_len_func


def get_gather_if_zero3_context(trainer):
    deepspeed_plugin = trainer.accelerator.state.deepspeed_plugin
    zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
    if zero_stage_3:
        import deepspeed
        gather_if_zero3 = deepspeed.zero.GatheredParameters
    else:
        gather_if_zero3 = nullcontext
    return gather_if_zero3


def patch_vllm_load_adapter():
    from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
    from vllm.lora.models import LoRAModel
    from vllm.lora.utils import get_adapter_absolute_path

    try:
        from vllm.transformers_utils.tokenizer_group import TokenizerGroup
    except ImportError:
        # removed in https://github.com/vllm-project/vllm/pull/24078
        TokenizerGroup = None

    def patched_load_adapter(self: LRUCacheWorkerLoRAManager, lora_request: TensorLoRARequest) -> LoRAModel:
        """
        code borrowed from verl.utils.vllm.utils.py
        based on vllm.lora.worker_manager.WorkerLoRAManager._load_adapter, support load adapter with lora tensors
        Reason:
        VLLM does not support adding LoRA from tensors directly. It only supports adding LoRA via file paths.
        To synchronize the LoRA tensors of the actor model, we need to find a workaround to enable VLLM to
        load memory-based LoRA tensors.
        """
        try:
            supported_lora_modules = self._adapter_manager.supported_lora_modules
            packed_modules_mapping = self._adapter_manager.packed_modules_mapping
            expected_lora_modules: list[str] = []
            for module in supported_lora_modules:
                if module in packed_modules_mapping:
                    expected_lora_modules.extend(packed_modules_mapping[module])
                else:
                    expected_lora_modules.append(module)
            expected_lora_modules = list(set(expected_lora_modules))
            # this is the patch
            lora_tensors = None
            from vllm.lora.peft_helper import PEFTHelper
            if isinstance(lora_request, TensorLoRARequest):
                peft_config = lora_request.peft_config
                lora_tensors = lora_request.lora_tensors
                peft_helper = PEFTHelper.from_dict(peft_config)
            else:
                lora_path = get_adapter_absolute_path(lora_request.lora_path)
                peft_helper = PEFTHelper.from_local_dir(lora_path, self.max_position_embeddings)
            # Validates the LoRA configuration against requirements before
            # loading weights, throwing an exception if validation fails.
            peft_helper.validate_legal(self.lora_config)
            # For some models like Qwen2VL, we need to use hf_to_vllm_mapper
            # to ensure correct loading of lora weights.
            model = self._adapter_manager.model
            hf_to_vllm_mapper = getattr(model, 'hf_to_vllm_mapper', None)
            if isinstance(lora_request, TensorLoRARequest):  # this is the patch
                lora = self._lora_model_cls.from_lora_tensors(
                    lora_model_id=lora_request.lora_int_id,
                    tensors=lora_tensors,
                    peft_helper=peft_helper,
                    device='cpu',
                    dtype=self.lora_config.lora_dtype,
                    embeddings=None,
                    target_embedding_padding=self.vocab_size + self.lora_config.lora_extra_vocab_size,
                    embedding_modules=self.embedding_modules,
                    embedding_padding_modules=self.embedding_padding_modules,
                    weights_mapper=hf_to_vllm_mapper,
                )
            else:
                lora = self._lora_model_cls.from_local_checkpoint(
                    lora_path,
                    expected_lora_modules,
                    peft_helper=peft_helper,
                    lora_model_id=lora_request.lora_int_id,
                    device='cpu',
                    dtype=self.lora_config.lora_dtype,
                    target_embedding_padding=self.vocab_size + self.lora_config.lora_extra_vocab_size,
                    embedding_modules=self.embedding_modules,
                    embedding_padding_modules=self.embedding_padding_modules,
                    weights_mapper=hf_to_vllm_mapper,
                )
        except Exception as e:
            raise e
        if lora.extra_vocab_size > self.lora_config.lora_extra_vocab_size:
            raise ValueError(f'LoRA added vocab size {lora.extra_vocab_size} is greater than '
                             f'lora_extra_vocab_size {self.lora_config.lora_extra_vocab_size}.')
        return lora

    def patched_get_lora_tokenizer(self: TokenizerGroup, lora_request: LoRARequest):
        # since we pass dummy path, skip get tokenizer from path
        return self.tokenizer

    if not hasattr(LRUCacheWorkerLoRAManager, '_old_load_adapter'):
        _old_load_adapter = LRUCacheWorkerLoRAManager._load_adapter
        LRUCacheWorkerLoRAManager._load_adapter = patched_load_adapter
        LRUCacheWorkerLoRAManager._old_load_adapter = _old_load_adapter
        if TokenizerGroup is not None:
            TokenizerGroup._old_get_lora_tokenizer = TokenizerGroup.get_lora_tokenizer
            TokenizerGroup.get_lora_tokenizer = patched_get_lora_tokenizer


# FlattenedTensor, code borrowed from sglang/srt/weight_sync/tensor_bucket.py
class FlattenedTensorMetadata(BaseModel):
    """Metadata for a tensor in a flattened bucket"""
    name: str
    shape: Tuple[int, ...]
    dtype: str
    start_idx: int
    end_idx: int
    numel: int

    @field_validator('shape', mode='before')
    @classmethod
    def ensure_shape_tuple(cls, v: Any) -> Tuple[int, ...]:
        # accept tuple/list, torch.Size, or other iterable of ints
        if torch is not None and isinstance(v, torch.Size):
            return tuple(int(x) for x in v)
        if isinstance(v, (list, tuple)):
            return tuple(int(x) for x in v)
        if isinstance(v, Iterable):
            return tuple(int(x) for x in v)
        raise ValueError('shape must be an iterable of ints (e.g. tuple/list/torch.Size)')

    @field_validator('dtype', mode='before')
    @classmethod
    def ensure_dtype_str(cls, v: Any) -> str:
        # accept torch.dtype or str
        if torch is not None and isinstance(v, torch.dtype):
            return str(v)
        if isinstance(v, str):
            return v
        raise ValueError('dtype must be a torch.dtype or str')


class FlattenedTensorBucket:
    """
    A bucket that flattens multiple tensors into a single tensor for efficient processing
    while preserving all metadata needed for reconstruction.
    """

    def __init__(
        self,
        named_tensors: List[Tuple[str, torch.Tensor]] = None,
        flattened_tensor: torch.Tensor = None,
        metadata: List[FlattenedTensorMetadata] = None,
    ):
        """
        Initialize a tensor bucket from a list of named tensors OR from pre-flattened data.
        Args:
            named_tensors: List of (name, tensor) tuples (for creating new bucket)
            flattened_tensor: Pre-flattened tensor (for reconstruction)
            metadata: Pre-computed metadata (for reconstruction)
        """
        if named_tensors is not None:
            # Create bucket from named tensors
            self.metadata: List[FlattenedTensorMetadata] = [None] * len(named_tensors)
            self.flattened_tensor: torch.Tensor = None

            if not named_tensors:
                raise ValueError('Cannot create empty tensor bucket')

            # First pass: compute total size and metadata
            current_idx = 0
            total_numel = 0
            for i, (name, tensor) in enumerate(named_tensors):
                numel = tensor.numel()
                metadata_obj = FlattenedTensorMetadata(
                    name=name,
                    shape=tuple(tensor.shape),
                    dtype=str(tensor.dtype),
                    start_idx=current_idx,
                    end_idx=current_idx + numel,
                    numel=numel,
                )
                self.metadata[i] = metadata_obj
                current_idx += numel
                total_numel += numel

            # Pre-allocate the final flattened tensor to avoid intermediate copies
            # Use the dtype and device of the first tensor
            first_tensor = named_tensors[0][1]
            self.flattened_tensor = torch.empty(total_numel, dtype=first_tensor.dtype, device=first_tensor.device)

            # Second pass: copy data directly into pre-allocated tensor
            for meta, (name, tensor) in zip(self.metadata, named_tensors):
                self.flattened_tensor[meta.start_idx:meta.end_idx].copy_(tensor.flatten())
        else:
            # Initialize from pre-flattened data
            if flattened_tensor is None or metadata is None:
                raise ValueError('Must provide either named_tensors or both flattened_tensor and metadata')
            self.flattened_tensor = flattened_tensor
            self.metadata = metadata

    def get_flattened_tensor(self) -> torch.Tensor:
        """Get the flattened tensor containing all bucket tensors"""
        return self.flattened_tensor

    def get_metadata(self) -> List[FlattenedTensorMetadata]:
        """Get metadata for all tensors in the bucket"""
        return self.metadata

    def reconstruct_tensors(self) -> Dict[str, torch.Tensor]:
        """
        Reconstruct original tensors from flattened tensor with optimized performance.
        Uses memory-efficient operations to minimize allocations and copies.
        """
        # preallocate the result list
        reconstructed = {}

        for meta in self.metadata:
            tensor = self.flattened_tensor[meta.start_idx:meta.end_idx].reshape(meta.shape)
            dtype = getattr(torch, meta.dtype.split('.')[-1])
            # batch dtype conversion (if needed)
            if tensor.dtype != dtype:
                tensor = tensor.to(dtype)

            reconstructed[meta.name] = tensor

        return reconstructed


def identity_data_collator(features):
    """Identity data collator that returns features as-is without any processing."""
    return features


def mu_schedule_function(global_step: int, mu_warmup_steps: int, mu_decay_steps: int, mu_peak: float,
                         mu_valley: float) -> float:
    """
    Computes a cosine decay schedule with a warmup phase for the mu parameter.

    Args:
        global_step: Current global training step
        mu_warmup_steps: Number of warmup steps
        mu_decay_steps: Number of decay steps
        mu_peak: Peak value of mu during warmup
        mu_valley: Final value of mu after decay

    Returns:
        Current mu value based on the schedule
    """
    # Warmup
    if global_step < mu_warmup_steps:
        return (global_step / mu_warmup_steps) * mu_peak

    # Decay
    if global_step >= (mu_warmup_steps + mu_decay_steps):
        return mu_valley

    adjusted_step = global_step - mu_warmup_steps
    cosine_decay = 0.5 * (1 + math.cos(math.pi * adjusted_step / mu_decay_steps))
    decayed_mu = (mu_peak - mu_valley) * cosine_decay + mu_valley
    return decayed_mu


def create_cyclic_iterator(iterable):
    """Create a cyclic iterator that repeats the iterable indefinitely."""
    while True:
        for x in iterable:
            yield x


def get_chord_sft_dataloader(trainer,
                             dataset,
                             description,
                             batch_size,
                             sampler_fn=None,
                             is_training=False,
                             dataloader_key=None) -> DataLoader:
    """
    Create a DataLoader from the given dataset for CHORD SFT training.
    Mimics transformers.trainers._get_dataloader.

    Args:
        trainer: The trainer instance
        dataset: The dataset to create DataLoader from
        description: Description of the dataset (e.g., 'Training')
        batch_size: Batch size for the DataLoader
        sampler_fn: Optional sampler function
        is_training: Whether this is for training
        dataloader_key: Optional dataloader key

    Returns:
        Prepared DataLoader
    """
    data_collator = identity_data_collator
    if isinstance(dataset, datasets.Dataset):
        dataset = trainer._remove_unused_columns(dataset, description=description)
    else:
        data_collator = trainer._get_collator_with_removed_columns(data_collator, description=description)

    dataloader_params = {
        'batch_size': batch_size,
        'collate_fn': data_collator,
        'num_workers': trainer.args.dataloader_num_workers,
        'pin_memory': trainer.args.dataloader_pin_memory,
        'persistent_workers': trainer.args.dataloader_persistent_workers,
    }

    if not isinstance(dataset, torch.utils.data.IterableDataset):
        if sampler_fn is not None:
            dataloader_params['sampler'] = sampler_fn(dataset)
        dataloader_params['drop_last'] = trainer.args.dataloader_drop_last
        dataloader_params['prefetch_factor'] = trainer.args.dataloader_prefetch_factor
        if is_training:
            from swift.utils import seed_worker
            dataloader_params['worker_init_fn'] = partial(
                seed_worker, num_workers=trainer.args.dataloader_num_workers, rank=trainer.args.process_index)

    dataloader = trainer.accelerator.prepare(DataLoader(dataset, **dataloader_params))
    return dataloader


def make_chord_sft_dataset(trainer, chord_sft_dataset):
    """
    Create and setup CHORD SFT dataset iterator for the trainer.

    Args:
        trainer: The trainer instance
        chord_sft_dataset: The CHORD SFT dataset
    """
    trainer.chord_sft_dataset = chord_sft_dataset
    if trainer.chord_sft_dataset:
        chord_sft_dataloader = get_chord_sft_dataloader(
            trainer=trainer,
            dataset=chord_sft_dataset,
            description='Training',
            batch_size=trainer.args.chord_sft_per_device_train_batch_size,
            sampler_fn=RandomSampler,
            is_training=True,
        )
        return create_cyclic_iterator(chord_sft_dataloader)


def compute_chord_loss(trainer, grpo_loss: torch.Tensor) -> torch.Tensor:
    """
    Compute CHORD loss combining GRPO loss with SFT loss.

    Args:
        trainer: The trainer instance
        grpo_loss: The GRPO loss tensor

    Returns:
        Combined CHORD loss tensor
    """
    from swift.trainers import per_token_loss_func
    from swift.llm import to_device

    current_step = trainer.state.global_step
    mu = mu_schedule_function(current_step, trainer.args.chord_mu_warmup_steps, trainer.args.chord_mu_decay_steps,
                              trainer.args.chord_mu_peak, trainer.args.chord_mu_valley)
    chord_sft_loss = torch.tensor(0.0, device=grpo_loss.device, dtype=grpo_loss.dtype)
    if mu > 0:
        sft_inputs = next(trainer.chord_sft_iterator)
        sft_inputs = to_device(trainer.template.data_collator(sft_inputs), trainer.accelerator.device)

        labels = sft_inputs.pop('labels')
        loss_scale = sft_inputs.pop('loss_scale', None)
        outputs = trainer.model(**sft_inputs)
        chord_sft_loss = per_token_loss_func(outputs, labels)

        if trainer.args.chord_enable_phi_function:
            per_token_probs = torch.exp(-chord_sft_loss)
            phi = per_token_probs * (1 - per_token_probs)
            chord_sft_loss *= phi

        if loss_scale is not None:
            loss_scale = torch.roll(loss_scale, shifts=-1, dims=-1).view(-1)
            chord_sft_loss *= loss_scale

        num_items_in_batch = (labels[:, 1:] != -100).sum()
        chord_sft_loss = chord_sft_loss.sum() / num_items_in_batch
    else:
        assert mu == 0
        chord_sft_loss = torch.tensor(0.0, device=grpo_loss.device, dtype=grpo_loss.dtype)
    loss = (1 - mu) * grpo_loss + mu * chord_sft_loss
    return loss


_EXPANDABLE_SEGMENTS_SET = 'expandable_segments' in os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')


def set_expandable_segments(enable: bool) -> None:
    """
    Enable or disable expandable segments for CUDA memory allocation.

    This function provides a safe way to configure CUDA expandable segments without
    overriding user preferences. It only takes effect when the user has previously
    set the PYTORCH_CUDA_ALLOC_CONF environment variable, ensuring that explicit
    user configurations are respected.

    Expandable segments allow PyTorch to grow memory pools dynamically, which can
    help prevent out-of-memory (OOM) errors during long-running reinforcement
    learning training sessions by reducing memory fragmentation.

    Args:
        enable (bool): Whether to enable expandable segments. When True, allows
            CUDA memory pools to expand dynamically to reduce fragmentation and
            mitigate OOM issues.

    Note:
        - Only takes effect if PYTORCH_CUDA_ALLOC_CONF was previously set by the user
        - Requires CUDA to be available
        - Changes apply to both the PyTorch allocator settings and environment variable

    Example:
        >>> # Only works if user has already set PYTORCH_CUDA_ALLOC_CONF
        >>> set_expandable_segments(True)  # Enable to help with OOM issues
        >>> set_expandable_segments(False) # Disable for more predictable memory usage
    """
    global _EXPANDABLE_SEGMENTS_SET
    if not _EXPANDABLE_SEGMENTS_SET:
        return
    if torch.cuda.is_available():
        torch.cuda.memory._set_allocator_settings(f'expandable_segments:{enable}')
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f'expandable_segments:{enable}'


def peft_config_to_dict(peft_config):
    if not isinstance(peft_config, dict):
        peft_config = asdict(peft_config)
    # turn set to list to serializable
    if 'target_modules' in peft_config and isinstance(peft_config['target_modules'], set):
        peft_config['target_modules'] = list(peft_config['target_modules'])

    return peft_config


def _create_parameter_buckets(named_params, bucket_size_mb=512):
    """Create parameter buckets for efficient processing"""
    buckets = []
    current_bucket = []
    current_size = 0
    bucket_size_bytes = bucket_size_mb * 1024 * 1024

    for name, param in named_params:
        param_size = param.numel() * param.element_size()

        # If adding this param would exceed bucket size, process current bucket first
        if current_size + param_size > bucket_size_bytes and current_bucket:
            buckets.append(current_bucket)
            current_bucket = []
            current_size = 0

        current_bucket.append((name, param))
        current_size += param_size

    # Process remaining parameters in the last bucket
    if current_bucket:
        buckets.append(current_bucket)

    return buckets


def _process_bucket_with_flattened_tensor(trainer, bucket_params):
    """Process a bucket of parameters using FlattenedTensorBucket for efficiency"""
    if not bucket_params:
        return

    # Create FlattenedTensorBucket for efficient processing
    bucket = FlattenedTensorBucket(named_tensors=bucket_params)
    metadatas = bucket.get_metadata()
    flattened_tensor = bucket.get_flattened_tensor()

    # Use the new flattened parameter update method
    # If not available, fall back to individual parameter updates
    try:
        trainer.vllm_client.update_flattened_params(metadatas, flattened_tensor)
    except AttributeError:
        # Fallback to individual parameter updates
        reconstructed = bucket.reconstruct_tensors()
        for name, param in reconstructed.items():
            trainer.vllm_client.update_named_param(name, param)

    # Clean up
    del bucket, metadatas, flattened_tensor


def get_even_process_data(trainer, global_data: List[T]) -> List[T]:
    """
    Evenly splits `global_data` among all processes.

    Each process receives a contiguous chunk of data. If `len(global_data)` is not
    perfectly divisible by the number of processes, the first `remainder` processes
    will receive one additional item.

    Args:
        global_data (List[T]): The full list of data to be distributed.

    Returns:
        List[T]: The subset of `global_data` assigned to this process.
    """
    num_procs = trainer.accelerator.num_processes
    proc_idx = trainer.accelerator.process_index
    total = len(global_data)

    base_size = total // num_procs
    remainder = total % num_procs

    # Calculate the number of samples that need to be padded
    # This ensures all processes have the same number of samples for gather operations
    trainer.rollout_pad_count = 0
    if remainder > 0 and proc_idx >= remainder:
        # Processes with extra samples need padding
        trainer.rollout_pad_count = 1

    if proc_idx < remainder:
        start = proc_idx * (base_size + 1)
        end = start + base_size + 1
    else:
        start = remainder * (base_size + 1) + (proc_idx - remainder) * base_size
        end = start + base_size

    return global_data[start:end]

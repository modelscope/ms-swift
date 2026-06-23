# Copyright (c) ModelScope Contributors. All rights reserved.
import datasets
import functools
import ipaddress
import math
import os
import re
import socket
import time
import torch
import torch.nn.functional as F
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from datetime import timedelta
from functools import partial
from io import BytesIO
from msgspec import field
from packaging import version
from peft.tuners.lora import LoraLayer
from PIL import Image
from pydantic import BaseModel, field_validator
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from transformers.utils import is_torch_npu_available
from types import MethodType
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypeVar, Union

from swift.rl_core.data import GRPOBatch, OnPolicySample
from swift.template import Messages, Template
from swift.tuners.lora import LoraConfig
from swift.utils import (gc_collect, get_cu_seqlens_from_position_ids, get_logger, get_packed_seq_params,
                         get_torch_device, is_swanlab_available, is_vllm_available, is_wandb_available, swanlab_get_run,
                         synchronize, to_device)

if is_wandb_available():
    import wandb
if is_swanlab_available():
    import swanlab

T = TypeVar('T')

logger = get_logger()

_ipv6_patch_applied = False

# Constants for the RL training LoRA adapter identity.
VLLM_LORA_INT_ID = 111
VLLM_LORA_NAME = 'swift_lora'
VLLM_LORA_PATH = 'swift_dummy_lora_path'


def broadcast_tensor_for_vllm_weight_sync(communicator, tensor: torch.Tensor, src: int) -> None:
    if is_torch_npu_available():
        device_module = get_torch_device()
        with device_module.device(communicator.device):
            communicator.broadcast(tensor, src=src, stream=device_module.current_stream())
    else:
        communicator.broadcast(tensor, src=src, stream=getattr(get_torch_device(), 'current_stream', lambda: None)())


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
else:
    TensorLoRARequest = None


def chunk_list(lst: list, n: int) -> list[list]:
    """
    Split list `lst` into `n` evenly distributed sublists.

    Example:
    ```python
    >>> chunk_list([1, 2, 3, 4, 5, 6], 2)
    [[1, 2, 3], [4, 5, 6]]

    >>> chunk_list([1, 2, 3, 4, 5, 6], 4)
    [[1, 2], [3, 4], [5], [6]]

    >>> chunk_list([1, 2, 3, 4, 5, 6], 8)
    [[1], [2], [3], [4], [5], [6], [], []]
    ```
    """
    k, r = divmod(len(lst), n)
    return [lst[i * k + min(i, r):(i + 1) * k + min(i + 1, r)] for i in range(n)]


def is_valid_ipv6_address(address: str) -> bool:
    """Check if the given address is a valid IPv6 address."""
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


def format_host_for_url(host: str) -> str:
    """Format host for URL - wrap IPv6 addresses in brackets."""
    if is_valid_ipv6_address(host):
        return f'[{host}]'
    return host


def resolve_hostname(hostname: str) -> str:
    """Resolve hostname to IP address, supporting both IPv4 and IPv6.

    Uses socket.getaddrinfo() which supports both IPv4 and IPv6,
    unlike socket.gethostbyname() which only supports IPv4.
    """
    # If it's already an IP address (IPv4 or IPv6), return as-is
    try:
        ipaddress.ip_address(hostname)
        return hostname
    except ValueError:
        pass

    # Resolve hostname using getaddrinfo (supports both IPv4 and IPv6)
    try:
        addr_info = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        if addr_info:
            # Return the first resolved address
            return addr_info[0][4][0]
    except socket.gaierror:
        pass

    # Fallback to original hostname if resolution fails
    return hostname


def patch_stateless_process_group_for_ipv6():
    """Apply monkey patch to vLLM's StatelessProcessGroup.create to support IPv6.

    The original implementation hardcodes socket.AF_INET which only supports IPv4.
    This patch detects IPv6 addresses at runtime and uses socket.AF_INET6 accordingly.
    For IPv4 addresses, it falls back to the original implementation.

    This function is idempotent - calling it multiple times is safe.
    """
    global _ipv6_patch_applied

    if _ipv6_patch_applied:
        return

    if not is_vllm_available():
        return

    import inspect
    from vllm.distributed.utils import StatelessProcessGroup

    # vLLM >= 0.19.0: create() accepts listen_socket and handles TCPStore internally
    _has_listen_socket_param = 'listen_socket' in inspect.signature(StatelessProcessGroup.create).parameters

    # Save original method for fallback
    _original_create = StatelessProcessGroup.create

    @staticmethod
    def _patched_stateless_pg_create(
        host: str,
        port: int,
        rank: int,
        world_size: int,
        data_expiration_seconds: int = 3600,
        store_timeout: int = 300,
        **kwargs,
    ) -> StatelessProcessGroup:
        """Patched version of StatelessProcessGroup.create that supports IPv6.

        For IPv4 addresses, falls back to the original implementation.
        """
        # If not IPv6, use original implementation
        if not is_valid_ipv6_address(host):
            return _original_create(
                host=host,
                port=port,
                rank=rank,
                world_size=world_size,
                data_expiration_seconds=data_expiration_seconds,
                store_timeout=store_timeout,
                **kwargs,
            )

        # IPv6 path: create an AF_INET6 socket
        launch_server = rank == 0
        if launch_server:
            listen_socket = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
            listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listen_socket.bind((host, port))
            listen_socket.listen()
        else:
            listen_socket = None

        if _has_listen_socket_param:
            # vLLM >= 0.19.0: pass listen_socket to create(), which handles
            # TCPStore creation and returns StatelessProcessGroup without socket field
            kwargs.pop('listen_socket', None)
            return _original_create(
                host=host,
                port=port,
                rank=rank,
                world_size=world_size,
                data_expiration_seconds=data_expiration_seconds,
                store_timeout=store_timeout,
                listen_socket=listen_socket,
                **kwargs,
            )
        else:
            # vLLM < 0.19.0: manually create TCPStore and pass socket to constructor
            from torch.distributed import TCPStore
            listen_fd = listen_socket.fileno() if listen_socket else None
            store = TCPStore(
                host_name=host,
                port=port,
                world_size=world_size,
                is_master=launch_server,
                timeout=timedelta(seconds=store_timeout),
                use_libuv=False,
                master_listen_fd=listen_fd,
            )
            return StatelessProcessGroup(
                rank=rank,
                world_size=world_size,
                store=store,
                socket=listen_socket,
                data_expiration_seconds=data_expiration_seconds,
            )

    # Apply the monkey patch to vLLM
    StatelessProcessGroup.create = _patched_stateless_pg_create

    _ipv6_patch_applied = True


# Apply IPv6 patch at module load time
patch_stateless_process_group_for_ipv6()


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
        from accelerate.utils import DeepSpeedPlugin
        from copy import deepcopy
        from transformers.integrations.deepspeed import HfTrainerDeepSpeedConfig
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
    if sync_cuda:
        synchronize()

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
    if sync_cuda:
        synchronize()
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
    """Patch LoraLayer.merge to support selective merging by ``parameter_group``.

    peft's ``merge_adapter()`` merges the whole model; this patch lets us merge only the
    layers whose name is in ``parameter_group`` (used by the vLLM weight-sync, which merges
    one parameter group at a time within its DeepSpeed Zero3 gather context).

    Before merging, each target adapter's sublayers (lora_A/B, and the DoRA
    ``lora_magnitude_vector``) are aligned to the base-layer device via peft's
    type-agnostic ``_move_adapter_to_device_of_base_layer``. This is correct for
    Linear/Embedding/Conv as well as parameter-based ``ParamWrapper`` (MoE experts via
    ``target_parameters``), whose base layer has no ``.weight`` and which overrides the
    method to use ``get_param().device``. This replaces the previous hand-rolled,
    Linear-only device handling that hard-coded ``base_layer.weight.device``.

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
        for active_adapter in check_adapters_to_merge(self, adapter_names) or []:
            # Align adapter sublayers (lora_A/B, DoRA magnitude, ...) to the base device.
            # Type-agnostic: ParamWrapper overrides this to use get_param().device.
            self._move_adapter_to_device_of_base_layer(active_adapter)
        return self.merge_origin(safe_merge, adapter_names)

    # Patch all LoraLayer instances
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            module.name = name
            if not hasattr(module, 'merge_origin') and hasattr(module, 'base_layer'):
                module.merge_origin = module.merge
                module.merge = MethodType(merge, module)

    try:
        yield model
    finally:
        # Cleanup: restore original methods
        for module in model.modules():
            if isinstance(module, LoraLayer) and hasattr(module, 'merge_origin'):
                module.merge = module.merge_origin
                del module.merge_origin


@contextmanager
def patch_lora_unmerge(model):
    """Patch LoraLayer.unmerge to align adapter sublayers to the base device first.

    Mirrors ``patch_lora_merge``'s device handling (via peft's type-agnostic
    ``_move_adapter_to_device_of_base_layer``) so unmerge works under DeepSpeed Zero3 /
    offload and for parameter-based ``ParamWrapper`` layers.
    """

    def unmerge_patched(self):
        if not self.merged:
            return
        # Move adapter sublayers (incl. DoRA magnitude) to the base device first
        for adapter in list(self.merged_adapters):
            self._move_adapter_to_device_of_base_layer(adapter)
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
def profiling_context(trainer, name: str):
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    duration = end_time - start_time

    if trainer is None:
        return

    profiling_metrics = {f'profiling/Time taken: {trainer.__class__.__name__}.{name}': duration}

    is_main_process = False
    if hasattr(trainer, 'accelerator'):
        is_main_process = trainer.accelerator.is_main_process
    elif hasattr(trainer, 'is_main_process'):
        is_main_process = trainer.is_main_process

    if 'wandb' in trainer.args.report_to and wandb.run is not None and is_main_process:
        wandb.log(profiling_metrics, commit=False)

    if 'swanlab' in trainer.args.report_to and swanlab_get_run() is not None and is_main_process:
        swanlab.log(profiling_metrics)


def profiling_decorator(func):

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        with profiling_context(self, func.__name__):
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


def get_response_prefix_ids(template: Template, sample_enable_thinking: Optional[bool] = None) -> Optional[List[int]]:
    effective = sample_enable_thinking if sample_enable_thinking is not None else template.enable_thinking
    if effective is True:
        prefix_str = template.template_meta.thinking_prefix
    elif effective is False:
        prefix_str = template.template_meta.non_thinking_prefix
    else:
        return None
    if prefix_str:
        return template.tokenizer.encode(prefix_str, add_special_tokens=False)
    return None


def encode_sample(sample: OnPolicySample, template: Template, *, encode_prompt_only: bool = False) -> Dict[str, Any]:
    """Encode a sample into a template.encode output dict.

    Does NOT mutate ``sample.messages`` — works on a copy from
    ``to_template_dict()`` so the sample's original messages are preserved
    for logging / reward computation / reuse across steps_per_generation.

    Per-sample ``enable_thinking``: the response prefix (thinking or
    non-thinking) is computed per-sample from
    ``sample.extra['chat_template_kwargs']['enable_thinking']``, falling back
    to the template's global setting.  This keeps the trainer sequence
    aligned with the rollout sequence for both thinking and non-thinking
    prefixes.
    """
    data = sample.to_template_dict()
    if sample.response_token_ids:
        loss_mask = sample.response_loss_mask or None
        msgs = data.get('messages')
        if msgs is not None:
            msgs = [m.copy() for m in msgs]
        ctk = sample.extra.get('chat_template_kwargs') or {}
        sample_et = ctk.get('enable_thinking')
        prefix_ids = get_response_prefix_ids(template, sample_enable_thinking=sample_et)
        logger.debug(f'[encode_sample] uuid={sample.request_id} '
                     f'sample_enable_thinking={sample_et} global={template.enable_thinking} '
                     f'prefix_ids={prefix_ids}')
        data['messages'] = replace_assistant_response_with_ids(
            msgs, sample.response_token_ids, loss_mask, non_thinking_prefix_ids=prefix_ids)

    if encode_prompt_only:
        messages = data.get('messages', [])
        if messages and messages[-1].get('role') == 'assistant':
            data = {**data, 'messages': messages[:-1] + [{**messages[-1], 'content': None}]}

    encoded = template.encode(data, return_length=True)
    return encoded


def replace_assistant_response_with_ids(messages: 'Messages',
                                        completion_ids: List[Union[int, List[int]]],
                                        loss_mask: Optional[List[List[int]]] = None,
                                        non_thinking_prefix_ids: Optional[List[int]] = None) -> 'Messages':  # noqa
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

    # Inject the non-thinking prefix (e.g. '<think>\n\n</think>\n\n') into the LAST assistant turn.
    # When enable_thinking false, the engine prepends non_thinking_prefix before generation
    # so completion_ids here are generated with the non-thinking prefix, inject here
    if non_thinking_prefix_ids:
        n_prefix = len(non_thinking_prefix_ids)
        last_ids = list(completion_ids[-1])
        # Skip if the response already starts with the prefix (avoid double injection).
        if last_ids[:n_prefix] != list(non_thinking_prefix_ids):
            if loss_mask is None:
                loss_mask = [[1] * len(ids) for ids in completion_ids]
            completion_ids[-1] = list(non_thinking_prefix_ids) + last_ids
            loss_mask[-1] = [0] * n_prefix + list(loss_mask[-1])

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


def parse_prompt_logprobs(response, topk: int) -> Tuple[List[List[float]], List[List[int]]]:
    raw = response.prompt_logprobs or []
    lps: List[List[float]] = []
    ixs: List[List[int]] = []
    for pos_lp in raw[1:]:
        sorted_items = sorted(pos_lp.items(), key=lambda x: -x[1]['logprob'])[:topk]
        lp_row = [info['logprob'] for _, info in sorted_items]
        ix_row = [int(tid) for tid, _ in sorted_items]
        lps.append(lp_row)
        ixs.append(ix_row)
    return lps, ixs


def assemble_teacher_topk_logprobs(
    parsed: List[Tuple[List[List[float]], List[List[int]]]],
    batch_size: int,
    seq_len: int,
    cu_seqlens: Optional[List[int]],
    topk: int,
    device: torch.device,
    offsets: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    is_packed = cu_seqlens is not None

    if is_packed:
        total_len = seq_len
        out_lp = torch.full((total_len, topk), float('-inf'), dtype=torch.float32)
        out_ix = torch.zeros(total_len, topk, dtype=torch.long)
        num_seqs = len(cu_seqlens) - 1
        assert len(parsed) == num_seqs, f'parsed length {len(parsed)} != num_seqs {num_seqs}'
        for i in range(num_seqs):
            start, end = cu_seqlens[i], cu_seqlens[i + 1]
            lps, ixs = parsed[i]
            length = min(len(lps), end - start)
            if length <= 0:
                continue
            out_lp[start:start + length] = torch.tensor(lps[:length], dtype=torch.float32)
            out_ix[start:start + length] = torch.tensor(ixs[:length], dtype=torch.long)
        return out_lp.unsqueeze(0).to(device), out_ix.unsqueeze(0).to(device)

    out_lp = torch.full((batch_size, seq_len, topk), float('-inf'), dtype=torch.float32)
    out_ix = torch.zeros(batch_size, seq_len, topk, dtype=torch.long)
    assert len(parsed) == batch_size, f'parsed length {len(parsed)} != batch_size {batch_size}'
    for idx in range(batch_size):
        lps, ixs = parsed[idx]
        P = len(lps)
        start = offsets[idx] if offsets is not None else 0
        length = min(P, seq_len - start)
        if length <= 0:
            continue
        out_lp[idx, start:start + length] = torch.tensor(lps[:length], dtype=torch.float32)
        out_ix[idx, start:start + length] = torch.tensor(ixs[:length], dtype=torch.long)
    return out_lp.to(device), out_ix.to(device)


def patch_save_last_checkpoint():
    import trl
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


def get_gather_if_zero3_context(trainer, is_zero3: Optional[bool] = None):
    deepspeed_plugin = trainer.accelerator.state.deepspeed_plugin
    zero_stage_3 = is_zero3 if is_zero3 is not None else (deepspeed_plugin is not None
                                                          and deepspeed_plugin.zero_stage == 3)

    if zero_stage_3:
        import deepspeed
        gather_if_zero3 = deepspeed.zero.GatheredParameters
    else:
        gather_if_zero3 = nullcontext
    return gather_if_zero3


def prepare_fsdp(model, accelerator, evaluation_mode: bool = True):
    """Prepare a model with FSDP wrapping

    This function wraps a model with the appropriate FSDP mechanism based on
    the accelerator configuration. It's designed for auxiliary models like
    ref_model, teacher_model, or reward_model that need to be FSDP-wrapped
    to prevent mixing DTensor (main model) with regular Tensor (auxiliary model).

    Args:
        model: The model to wrap with FSDP.
        accelerator: The accelerator instance from trainer.
        evaluation_mode: Whether to set the model to evaluation mode. Defaults to True.
            When True, the model is frozen BEFORE FSDP wrapping to avoid float32 upcast,
            which saves significant memory for evaluation-only models.

    Returns:
        The FSDP-wrapped model.
    """
    if evaluation_mode:
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

    if getattr(accelerator, 'is_fsdp2', False):
        # FSDP2 uses fully_shard API with DTensor
        from accelerate.utils.fsdp_utils import fsdp2_prepare_model
        model = fsdp2_prepare_model(accelerator, model)
    else:
        # FSDP1 uses FullyShardedDataParallel wrapper
        from trl.models.utils import prepare_fsdp as trl_prepare_fsdp
        model = trl_prepare_fsdp(model, accelerator)

    return model


_moe_model_registry_cache = None


def _get_moe_model_registry():

    global _moe_model_registry_cache
    if _moe_model_registry_cache is not None:
        return _moe_model_registry_cache

    import importlib

    moe_model_configs = [
        ('vllm.model_executor.models.deepseek_v2', ('DeepseekV2ForCausalLM', 'DeepseekV3ForCausalLM'), 'mlp'),
        ('vllm.model_executor.models.mixtral', ('MixtralForCausalLM', ), 'block_sparse_moe'),
        ('vllm.model_executor.models.qwen2_moe', ('Qwen2MoeForCausalLM', ), 'mlp'),
        ('vllm.model_executor.models.qwen3_moe', ('Qwen3MoeForCausalLM', ), 'mlp'),
        ('vllm.model_executor.models.qwen3_vl_moe', ('Qwen3MoeLLMForCausalLM', ), 'mlp'),
        ('vllm.model_executor.models.qwen3_5', ('Qwen3_5MoeForCausalLM', ), 'mlp'),
        ('vllm.model_executor.models.qwen3_next', ('Qwen3NextForCausalLM', ), 'mlp'),
        ('vllm.model_executor.models.kimi_vl', ('KimiVLForConditionalGeneration', ), 'mlp'),
    ]

    supported_moe_models = []
    mlp_attr_mapping = {}

    for module_path, class_names, mlp_attr in moe_model_configs:
        try:
            module = importlib.import_module(module_path)
            for class_name in class_names:
                if hasattr(module, class_name):
                    model_class = getattr(module, class_name)
                    supported_moe_models.append(model_class)
                    mlp_attr_mapping[model_class] = mlp_attr
        except (ImportError, AttributeError, RuntimeError):
            pass

    _moe_model_registry_cache = (supported_moe_models, mlp_attr_mapping)
    return _moe_model_registry_cache


def patch_vllm_moe_model_weight_loader(model):
    """
    Patch vLLM MoE model to add weight_loader attribute to expert weights.

    This is a workaround for a bug in vLLM 0.8.2 where MoE weights (w13_weight, w2_weight)
    don't have the weight_loader attribute, causing AttributeError during weight loading.
    Code adapted from verl/verl/utils/vllm/patch.py

    Args:
        model: The vLLM model to patch.
    """
    # Check if already patched (idempotent). On NPU/vLLM-Ascend, sleep/wake
    # and full-model reload can recreate expert Parameters while keeping this
    # model-level flag, so the loader needs to be reattached every reload.
    if getattr(model, '_swift_moe_weight_loader_patched', False) and not is_torch_npu_available():
        return

    supported_moe_models, mlp_attr_mapping = _get_moe_model_registry()

    if not supported_moe_models:
        return

    original_model = model
    original_model_type = type(model)

    # Handle NPU ACLGraphWrapper (for vllm_ascend compatibility)
    if hasattr(model, 'runnable') and 'ACLGraphWrapper' in str(original_model_type):
        model = model.runnable
        original_model_type = type(model)

    # Get inner model (either model.model or model.language_model)
    inner_model = getattr(model, 'model', None) or getattr(model, 'language_model', None)
    if inner_model is None:
        # Model structure not recognized, skip patching
        return

    if not isinstance(model, tuple(supported_moe_models)) and not isinstance(inner_model, tuple(supported_moe_models)):
        return

    # Handle Qwen3-VL MoE structure
    if type(inner_model).__name__ == 'Qwen3MoeLLMForCausalLM':
        inner_model = inner_model.model
    if type(inner_model).__name__ == 'Qwen3_5MoeForCausalLM':
        inner_model = inner_model.model

    # Check if inner_model has layers attribute
    if not hasattr(inner_model, 'layers'):
        return

    def maybe_patch_vllm_ascend_moe_expert_weight_loader(experts, name, param):
        quant_method = getattr(experts, 'quant_method', None)
        if not is_torch_npu_available() or not type(quant_method).__module__.startswith('vllm_ascend'):
            return
        from swift.model.npu_patch.vllm_ascend import (patch_vllm_ascend_moe_expert_weight_loader,
                                                       use_vllm_ascend_moe_preprocessed_weight)
        patch_vllm_ascend_moe_expert_weight_loader(
            experts,
            name,
            param,
            load_preprocessed_weight=use_vllm_ascend_moe_preprocessed_weight(original_model),
        )

    for layer in inner_model.layers:
        mlp_attr = mlp_attr_mapping.get(original_model_type, 'mlp')

        mlp = getattr(layer, mlp_attr, None)
        if not mlp:
            continue

        experts = getattr(mlp, 'experts', None)
        if not experts or not hasattr(experts, 'weight_loader'):
            continue

        # Patch the weight loaders for MoE expert weights
        for name, param in mlp.named_parameters():
            if 'w13_weight' in name or 'w2_weight' in name:
                if not hasattr(param, 'weight_loader'):
                    param.weight_loader = experts.weight_loader
                maybe_patch_vllm_ascend_moe_expert_weight_loader(experts, name, param)

    # Mark the model as patched (for idempotency)
    original_model._swift_moe_weight_loader_patched = True


def finish_vllm_weight_reload(vllm_model, model_config, target_device):
    if vllm_model is None or model_config is None or target_device is None:
        return
    if is_torch_npu_available():
        from swift.model.npu_patch.vllm_ascend import should_skip_vllm_ascend_moe_post_load
        if should_skip_vllm_ascend_moe_post_load(vllm_model):
            return
    try:
        from vllm.model_executor.model_loader.utils import process_weights_after_loading
        process_weights_after_loading(vllm_model, model_config, target_device)
    except Exception:
        return


_cached_reverse_renamings = None


def _build_reverse_renamings(model):
    """Build and cache reverse WeightRenaming rules for the given model.

    Only one model type goes through weight sync per training run, so a single
    module-level variable suffices. Returns None if no renamings apply.
    """
    global _cached_reverse_renamings
    if _cached_reverse_renamings is not None:
        return _cached_reverse_renamings

    try:
        from transformers.core_model_loading import WeightRenaming
    except ImportError:
        return None

    weight_conversions = getattr(model, '_weight_conversions', None)
    if weight_conversions is None:
        try:
            from transformers.conversion_mapping import get_model_conversion_mapping
            weight_conversions = get_model_conversion_mapping(model, add_legacy=False)
        except Exception:
            return None
    if not weight_conversions:
        return None

    renamings = [c for c in weight_conversions if isinstance(c, WeightRenaming)]
    if not renamings:
        return None

    # Reverse order before inverting, matching transformers' own revert_weight_conversion
    # (core_model_loading.py) which reverses the list so that chained renamings undo
    # in the correct order.
    try:
        _cached_reverse_renamings = [c.reverse_transform() for c in renamings[::-1]]
    except Exception as e:
        logger = get_logger()
        logger.warning(f'Failed to build reverse renamings for {type(model).__name__}: {e}')
        return None

    return _cached_reverse_renamings


def revert_runtime_names_to_checkpoint(model, state_dict):
    """Map HF runtime param names back to HF checkpoint names before vLLM weight sync.

    transformers>=5 may rename checkpoint keys to different *runtime* module names
    (e.g. gemma4_unified: checkpoint ``model.vision_embedder.*`` -> runtime
    ``model.embed_vision.*``). vLLM's ``hf_to_vllm_mapper`` is built around the
    checkpoint names (the same path used by ``vllm serve``), so online weight sync
    that sends runtime names can land on the wrong vLLM module and raise
    "There is no module or parameter named ...".

    We revert only the *renaming* part (``WeightRenaming``). Tensor-level
    ``WeightConverter`` ops (e.g. MoE fuse/split) are intentionally skipped and
    left to the existing MoE weight-loader patch, which expects the fused runtime
    layout.

    Safe by construction: models whose vLLM mapper accepts runtime names also
    carry the checkpoint-name rules (required by ``vllm serve``), so reverting to
    checkpoint names still maps correctly. Any failure or absence of conversions
    is a no-op that returns the input unchanged.
    """
    # Unwrap PEFT to reach the underlying transformers model that holds conversions.
    if hasattr(model, 'get_base_model'):
        try:
            model = model.get_base_model()
        except Exception:
            pass

    reverse_renamings = _build_reverse_renamings(model)
    if not reverse_renamings:
        return state_dict

    try:
        from transformers.core_model_loading import rename_source_key
    except ImportError:
        return state_dict

    new_state_dict = {}
    for name, param in state_dict.items():
        try:
            new_name, _ = rename_source_key(name, reverse_renamings, [])
        except Exception:
            new_name = name
        new_state_dict[new_name] = param
    return new_state_dict


def patch_vllm_load_adapter():
    from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
    try:
        from vllm.lora.models import LoRAModel
    except ImportError:
        # vllm >= 0.13 https://github.com/vllm-project/vllm/pull/30253
        from vllm.lora.lora_model import LoRAModel
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

            lora_request_kwargs = {
                'peft_helper': peft_helper,
                'lora_model_id': lora_request.lora_int_id,
                'device': 'cpu',
                'dtype': self.lora_config.lora_dtype,
                'weights_mapper': hf_to_vllm_mapper,
            }
            if hasattr(self, 'embedding_padding_modules'):
                lora_request_kwargs['embedding_modules'] = self.embedding_modules
                lora_request_kwargs['embedding_padding_modules'] = self.embedding_padding_modules
            else:
                lora_request_kwargs['model_vocab_size'] = self.vocab_size
            if hasattr(self.lora_config, 'lora_extra_vocab_size'):
                # lora_extra_vocab_size is removed in vllm >= 0.12
                # https://github.com/vllm-project/vllm/issues/23474
                lora_request_kwargs['target_embedding_padding'] = (
                    self.vocab_size + self.lora_config.lora_extra_vocab_size)

            if isinstance(lora_request, TensorLoRARequest):
                lora = self._lora_model_cls.from_lora_tensors(
                    tensors=lora_tensors,
                    **lora_request_kwargs,
                )
            else:
                lora = self._lora_model_cls.from_local_checkpoint(
                    lora_path,
                    expected_lora_modules,
                    **lora_request_kwargs,
                )
        except Exception as e:
            raise e
        if hasattr(self.lora_config, 'lora_extra_vocab_size'):
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


def expand_vllm_param_name_aliases(param_names: set[str]) -> set[str]:
    stacked_mappings = [
        (re.compile(r'\bqkv_proj\b'), ('q_proj', 'k_proj', 'v_proj', 'q', 'k', 'v')),
        (re.compile(r'\bgate_up_proj\b'), ('gate_proj', 'up_proj')),
        (re.compile(r'\bin_proj_ba\b'), ('in_proj_b', 'in_proj_a')),
        (re.compile(r'\blanguage_model\.model\b'), ('model.language_model', )),
        (re.compile(r'^visual\.'), ('model.visual.', )),
    ]

    def _expand_once(keys: set[str]) -> set[str]:
        expanded = set(keys)
        for key in keys:
            for pattern, aliases in stacked_mappings:
                if pattern.search(key):
                    for alias in aliases:
                        expanded.add(pattern.sub(alias, key))
        return expanded

    # Two passes allow chained replacement:
    # e.g. language_model.model + qkv_proj -> model.language_model + q_proj
    expanded = _expand_once(param_names)
    expanded = _expand_once(expanded)
    return expanded


def add_base_layer_suffix_by_param_names(weight_iterator: Iterable[Tuple[str, Any]],
                                         vllm_param_names: set[str]) -> Iterable[Tuple[str, Any]]:
    """Map HF dense param names to vLLM LoRA-wrapped modules (*.base_layer.weight / .bias)."""
    for name, tensor in weight_iterator:
        if '.base_layer.' in name or '.' not in name:
            yield name, tensor
            continue
        if name in vllm_param_names:
            yield name, tensor
            continue
        module_name, param_type = name.rsplit('.', 1)
        if param_type in {'weight', 'bias'}:
            bl = f'{module_name}.base_layer.{param_type}'
            if bl in vllm_param_names:
                name = bl
        yield name, tensor


# FlattenedTensor, code borrowed from sglang/srt/weight_sync/tensor_bucket.py
class FlattenedTensorMetadata(BaseModel):
    """Metadata for a tensor in a flattened bucket"""
    name: str
    shape: Tuple[int, ...]
    dtype: str
    start_idx: int
    end_idx: int
    numel: int

    @field_validator('dtype', mode='before')
    @classmethod
    def ensure_dtype_str(cls, v: Any) -> str:
        # accept torch.dtype or str
        if torch is not None and isinstance(v, torch.dtype):
            return str(v)
        if isinstance(v, str):
            return v
        raise ValueError('dtype must be a torch.dtype or str')


class TensorMetadata(BaseModel):
    """Metadata for a single tensor."""
    name: str
    shape: Tuple[int, ...]
    dtype: str
    numel: int


class UpdateFlattenedAdapterRequest(BaseModel):
    peft_config: LoraConfig
    metadatas: List[FlattenedTensorMetadata]


class UpdateFlattenedParamsRequest(BaseModel):
    metadatas: List[FlattenedTensorMetadata]


class UpdateAdapterRequest(BaseModel):
    """Request for non-flattened adapter weight update"""
    peft_config: LoraConfig
    lora_tensors_metadata: List[TensorMetadata]


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
        if named_tensors is not None:
            if not named_tensors:
                raise ValueError('Cannot create empty tensor bucket')

            self.metadata: List[FlattenedTensorMetadata] = [None] * len(named_tensors)
            flattened_chunks: List[torch.Tensor] = [None] * len(named_tensors)
            current_byte = 0

            for i, (name, tensor) in enumerate(named_tensors):
                flat_u8 = tensor.flatten().view(torch.uint8)
                flattened_chunks[i] = flat_u8

                numel = flat_u8.numel()
                self.metadata[i] = FlattenedTensorMetadata(
                    name=name,
                    shape=tuple(tensor.shape),
                    dtype=str(tensor.dtype),
                    start_idx=current_byte,
                    end_idx=current_byte + numel,
                    numel=numel,
                )
                current_byte += numel

            self.flattened_tensor = torch.cat(flattened_chunks, dim=0)
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
            dtype = getattr(torch, meta.dtype.split('.')[-1])
            byte_slice = self.flattened_tensor[meta.start_idx:meta.end_idx]
            tensor = byte_slice.view(dtype).reshape(meta.shape)
            reconstructed[meta.name] = tensor
        return reconstructed


def identity_data_collator(features, **kwargs):
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

    current_step = trainer.state.global_step
    mu = mu_schedule_function(current_step, trainer.args.chord_mu_warmup_steps, trainer.args.chord_mu_decay_steps,
                              trainer.args.chord_mu_peak, trainer.args.chord_mu_valley)
    chord_sft_loss = torch.tensor(0.0, device=grpo_loss.device, dtype=grpo_loss.dtype)
    if mu > 0:
        sft_inputs = next(trainer.chord_sft_iterator)
        sft_inputs = to_device(sft_inputs, 'cpu')
        sft_inputs = to_device(trainer.template.data_collator(sft_inputs), trainer.accelerator.device)

        labels = sft_inputs.pop('labels')
        loss_scale = sft_inputs.pop('loss_scale', None)
        outputs = trainer.model(**sft_inputs)
        chord_sft_loss = per_token_loss_func(outputs, labels)

        if trainer.args.chord_enable_phi_function:
            per_token_probs = torch.exp(-chord_sft_loss.detach())
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


def check_vllm_version_ge(min_version: str) -> bool:
    """check if the vllm version is greater than or equal to the minimum version"""
    if not is_vllm_available():
        return False
    import vllm
    vllm_version = vllm.__version__
    # if dev version, regard it as latest version
    if vllm_version is None or 'dev' in vllm_version:
        return True
    return version.parse(vllm_version) >= version.parse(min_version)


def vllm_supports_lora_load_inplace() -> bool:
    """True when vLLM LoRARequest supports load_inplace (replaces same lora_int_id without remove_lora).

    Introduced in vLLM v0.15.0 (see vllm/lora/request.py). Older versions require remove_lora before add_lora
    when reusing a stable adapter id.
    """
    return check_vllm_version_ge('0.15.0')


# ============================================================================
# Padding-free utilities
# ============================================================================


def pad_logps_back_to_batch(logps_rmpad: Optional[torch.Tensor],
                            position_ids: Optional[torch.Tensor] = None,
                            logits_to_keep: int = None,
                            batch_size: int = None,
                            seq_lengths: Optional[torch.Tensor] = None,
                            dtype: Optional[torch.dtype] = None,
                            pad_value: float = -1e10) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Restore padding-free logprobs back to [batch_size, seq_len] shape with LEFT PADDING.

    - Input: logps in rmpad format [1, total_nnz] or None
    - Output: logps in batch format [batch_size, max_seq_len] with data right-aligned

    Args:
        logps_rmpad: [1, total_nnz] per-token log probabilities in padding_free format or None
        position_ids: [1, total_nnz] position ids to determine sequence boundaries (deprecated, use seq_lengths)
        logits_to_keep: number of tokens to keep per sequence (= max_seq_len)
        batch_size: number of sequences in the batch
        seq_lengths: [batch_size] actual sequence lengths (preferred over position_ids)
        dtype: optional dtype for output, defaults to logps_rmpad.dtype
        pad_value: value to use for padding positions (default: -1e10 for logps, use 0.0 for masks)

    Returns:
        logps_padded: [batch_size, logits_to_keep] padded log probabilities (left-padded, data right-aligned) or None
        valid_mask: [batch_size, logits_to_keep] mask indicating valid (non-padding) positions or None
    """
    if logps_rmpad is None:
        return None, None

    if dtype is None:
        dtype = logps_rmpad.dtype

    device = logps_rmpad.device

    # Determine sequence lengths
    if seq_lengths is not None:
        # Use provided seq_lengths directly - they should already be adjusted
        # by the caller (e.g., in _generate_and_score_completions)
        # DO NOT adjust again here to avoid double adjustment
        pass
    else:
        # Fallback: infer from position_ids
        cu_seqlens = get_cu_seqlens_from_position_ids(position_ids)

        # Adjust cu_seqlens for logits_to_keep if needed
        total_length = cu_seqlens[-1].item()
        if total_length > logits_to_keep:
            # Adjust the first sequence length
            adjustment = total_length - logits_to_keep
            cu_seqlens = cu_seqlens - adjustment
            cu_seqlens[0] = 0  # First element should always be 0

        # Compute actual sequence lengths
        seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]

    # Compute cumulative sequence lengths
    cu_seqlens = torch.cumsum(torch.cat([torch.tensor([0], device=device), seq_lengths]), dim=0)
    max_seq_len = logits_to_keep  # All sequences will be padded to this length

    # Initialize output tensors with padding value
    logps_padded = torch.full((batch_size, max_seq_len), pad_value, dtype=dtype, device=device)
    valid_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.float32, device=device)

    # Unflatten: assign each sequence's logps to the corresponding row
    # Use LEFT PADDING (right-align the data) to match the standard padding convention
    logps_flat = logps_rmpad.squeeze(0)  # [total_nnz]

    for i in range(batch_size):
        start_idx = cu_seqlens[i].item()
        end_idx = cu_seqlens[i + 1].item()
        seq_len = int(seq_lengths[i].item())

        actual_end_idx = min(end_idx, len(logps_flat))
        actual_len = actual_end_idx - start_idx

        if actual_len <= 0:
            continue

        # Left padding: place data at the RIGHT side of the row
        # pad_len is the number of padding tokens at the beginning
        pad_len = max_seq_len - seq_len

        if actual_len < seq_len:
            # Input data is shorter than expected seq_len
            # This happens when logps_flat doesn't have enough data
            # Place actual data at the rightmost positions
            data_pad_len = max_seq_len - actual_len
            logps_padded[i, data_pad_len:] = logps_flat[start_idx:actual_end_idx]
            valid_mask[i, data_pad_len:] = 1.0
        else:
            # Normal case: seq_len tokens of data
            logps_padded[i, pad_len:] = logps_flat[start_idx:end_idx]
            valid_mask[i, pad_len:] = 1.0

    return logps_padded, valid_mask


def build_completion_mask_and_seq_lengths(
    labels: torch.Tensor,
    batch_size: int,
    *,
    padding_free: bool = False,
    encoded_batch: Optional[dict] = None,
    device: Optional[torch.device] = None,
    logits_to_keep: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Build completion_mask and seq_lengths from labels, shared by HF / Megatron / Ray GRPO paths.

    Two frame conventions, selected by ``logits_to_keep``:

    - ``logits_to_keep is None`` -> full-sequence frame + roll (Megatron / Ray):
      ``completion_mask = roll(labels,-1) != -100``, shape ``[B, T_full]``.
    - ``logits_to_keep is int`` -> completion-region frame, no roll (HF):
      ``completion_mask = labels[:, -ltk:] != -100``, shape ``[B, ltk]``; the per-sample
      ``seq_lengths`` (padding_free) carries the first-sentence prompt adjustment so it
      matches HF's ``num_logits_to_keep`` logps frame.

    Args:
        labels: Label tensor from data collator.
        batch_size: Number of samples in the batch.
        padding_free: Whether padding-free (rmpad) mode is used.
        encoded_batch: The full encoded batch dict (needed for cu_seq_lens / attention_mask / position_ids).
        device: Target device for output tensors.
        logits_to_keep: Region width for the HF frame; ``None`` selects the full-sequence frame.

    Returns:
        (completion_mask, seq_lengths, max_seq_len) where:
        - completion_mask: ``[B, max_seq_len]`` bool tensor
        - seq_lengths: ``[B]`` int tensor of per-sample lengths
        - max_seq_len: int
    """
    if device is None:
        device = labels.device
    if encoded_batch is None:
        encoded_batch = {}

    if logits_to_keep is None:
        # Full-sequence frame + roll (Megatron / Ray)
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
        if padding_free:
            if 'cu_seq_lens_q' in encoded_batch:
                cu = encoded_batch['cu_seq_lens_q']
            else:
                cu = get_packed_seq_params(encoded_batch['position_ids'])['cu_seq_lens_q']
            seq_lengths = cu[1:] - cu[:-1]
            max_seq_len = int(seq_lengths.max().item())
            completion_mask_rmpad = (rolled_labels != -100).float()
            completion_mask, _ = pad_logps_back_to_batch(
                logps_rmpad=completion_mask_rmpad,
                logits_to_keep=max_seq_len,
                batch_size=batch_size,
                seq_lengths=seq_lengths,
                pad_value=0.0)
            completion_mask = completion_mask.bool()
        else:
            attention_mask = encoded_batch.get('attention_mask')
            if attention_mask is not None:
                if attention_mask.dim() == 4:
                    attention_mask = attention_mask[:, 0, 0, :]
                seq_lengths = attention_mask.sum(dim=-1).to(torch.int64)
            else:
                seq_lengths = torch.full((batch_size, ), labels.shape[-1], dtype=torch.int64, device=device)
            max_seq_len = labels.shape[-1]
            completion_mask = (rolled_labels != -100)
        return completion_mask, seq_lengths, max_seq_len

    # Completion-region frame, no roll (HF); aligns with num_logits_to_keep logps frame.
    completion_mask_raw = labels[:, -logits_to_keep:] != -100
    max_seq_len = logits_to_keep
    if padding_free:
        position_ids = encoded_batch.get('text_position_ids')
        if position_ids is None:
            position_ids = encoded_batch.get('position_ids')
        position_ids = position_ids.squeeze()
        lengths = torch.diff(
            torch.cat([(position_ids == 0).nonzero(as_tuple=True)[0],
                       torch.tensor([len(position_ids)]).to(position_ids.device)]))
        total_lengths = lengths.sum()
        # The first sentence has its prompt portion removed due to logits_to_keep
        lengths[0] = lengths[0] - (total_lengths - logits_to_keep)
        seq_lengths = lengths
        completion_mask, _ = pad_logps_back_to_batch(
            logps_rmpad=completion_mask_raw.float(),
            logits_to_keep=logits_to_keep,
            batch_size=batch_size,
            seq_lengths=lengths,
            pad_value=0.0)
        completion_mask = completion_mask.bool()
    else:
        completion_mask = completion_mask_raw
        # Non-padding-free HF frame: every row spans the full region. Return real
        # per-sample lengths (= region width) instead of an empty tensor, so callers
        # can treat seq_lengths uniformly regardless of padding_free.
        seq_lengths = torch.full((batch_size, ), logits_to_keep, dtype=torch.int64, device=device)
    return completion_mask, seq_lengths, max_seq_len


def build_rollout_logps(
    rollout_logprobs_list: List[Optional[List[List[float]]]],
    completion_mask: torch.Tensor,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Convert per-sample ``rollout_logprobs`` into a [B, T] tensor aligned with completion_mask.

    Data-structure agnostic: callers pass a list of per-sample nested logprobs
    (``List[List[float]]`` per sample, or ``None``).

    Returns None if logprobs are missing or counts don't match.
    """
    lp_list = list(rollout_logprobs_list)
    if not all(lp is not None and lp for lp in lp_list):
        return None

    batch_size, seq_len = completion_mask.shape
    rollout_per_token_logps = torch.zeros(batch_size, seq_len, dtype=torch.float32, device=device)
    for i, nested_lp in enumerate(lp_list):
        flat_lps = [lp for turn_lps in nested_lp for lp in turn_lps]
        if not flat_lps:
            continue
        if any(lp is None for lp in flat_lps):
            return None
        completion_count = int(completion_mask[i].sum().item())
        if len(flat_lps) == completion_count + 1:
            flat_lps = flat_lps[:completion_count]
        if len(flat_lps) != completion_count:
            return None
        completion_indices = completion_mask[i].nonzero(as_tuple=True)[0]
        rollout_per_token_logps[i, completion_indices] = torch.tensor(flat_lps, dtype=torch.float32, device=device)
    return rollout_per_token_logps


def _normalize_routed_experts_tensor(value: Any) -> torch.Tensor:
    routed = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if routed.dim() >= 4 and routed.shape[0] == 1:
        routed = routed.squeeze(0)
    if routed.dim() < 2:
        raise ValueError(f'Invalid routed_experts shape: {tuple(routed.shape)}')
    return routed.to(dtype=torch.int64)


def _pad_or_trim_routed_experts(routed: torch.Tensor, target_len: int, *, padding_right: bool) -> torch.Tensor:
    current_len = int(routed.shape[0])
    if current_len == target_len:
        return routed
    if current_len > target_len:
        return routed[:target_len] if padding_right else routed[-target_len:]
    pad_len = target_len - current_len
    pad = [0] * (2 * routed.dim())
    if padding_right:
        pad[2 * (routed.dim() - 1) + 1] = pad_len
    else:
        pad[2 * (routed.dim() - 1)] = pad_len
    return torch.nn.functional.pad(routed, tuple(pad), 'constant', 0)


def build_routed_experts_batch(
    samples: List[OnPolicySample],
    *,
    seq_lengths: torch.Tensor,
    max_seq_len: int,
    template: Template,
    device: torch.device,
    router_replay_mode: str = 'disabled',
) -> Optional[torch.Tensor]:
    """Build the batched ``routed_experts`` model input from per-sample R3 routing.

    Shared by all backends. Each ``sample`` is an :class:`OnPolicySample` carrying
    ``routed_experts`` (per-sample, seq-first) and ``encoded['length']``. Returns
    ``None`` when no sample provides routing (and mode is not ``R3``).
    """
    if not samples or all(getattr(s, 'routed_experts', None) is None for s in samples):
        return None

    padding_right = template.padding_side == 'right'
    n_samples = len(samples)
    current_seq_lengths = seq_lengths
    if seq_lengths.size(0) > n_samples:
        current_seq_lengths = seq_lengths[:n_samples].clone()
        current_seq_lengths[n_samples - 1] = seq_lengths[n_samples - 1:].sum()

    routed_tensors: List[torch.Tensor] = []
    for sample, cur_seq_len in zip(samples, current_seq_lengths):
        routed_value = getattr(sample, 'routed_experts', None)
        if routed_value is None:
            if router_replay_mode == 'R3':
                raise AssertionError('When router_replay_mode = R3, routed_experts must be in rollout data')
            return None
        routed = _normalize_routed_experts_tensor(routed_value)
        expected_len = (sample.encoded or {}).get('length')
        experts_seq_len = int(routed.shape[0])
        if router_replay_mode == 'R3' and expected_len is not None:
            if experts_seq_len not in (expected_len, expected_len - 1):
                raise AssertionError(f'The seq_len of routed_experts({experts_seq_len}) does not match encoded length '
                                     f'({expected_len}); expected same length or one less.')
        target_len = int(cur_seq_len.item()) if template.padding_free else max_seq_len
        routed = _pad_or_trim_routed_experts(routed, target_len, padding_right=padding_right)
        routed_tensors.append(routed)

    if template.padding_free:
        return torch.cat(routed_tensors, dim=0).unsqueeze(0).to(device=device)
    return torch.stack(routed_tensors).to(device=device)


def collate_to_grpo_micro_batch(
    samples: List[OnPolicySample],
    template: Template,
    *,
    device: torch.device,
    padding_to: Optional[int] = None,
    router_replay_mode: str = 'disabled',
    use_logits_to_keep: bool = False,
) -> Tuple[Dict[str, Any], GRPOBatch]:
    """Collate ``List[OnPolicySample]`` into ``(model_inputs, grpo_batch)``.

    The single shared collate used by HF / Megatron / Megatron-Ray. Splits the
    per-sample world into two batch-level halves:

    - ``model_inputs`` (dict): ``data_collator([s.encoded ...])`` plus batch-computed
      model extras (``routed_experts``). A clean whitelist — ``model(**model_inputs)``
      needs no key filtering.
    - ``grpo_batch`` (:class:`GRPOBatch`): ``completion_mask`` / ``truncated_mask`` /
      ``seq_lengths`` / ``rollout_per_token_logps`` derived purely from the collated batch.

    ``use_logits_to_keep`` selects the completion_mask frame (see
    :func:`build_completion_mask_and_seq_lengths`):
    - ``False`` -> full-sequence frame + roll (Megatron / Ray).
    - ``True`` -> completion-region frame (HF); ``logits_to_keep`` is computed from
      the collated labels and stored on ``grpo_batch`` for the HF logps path.

    Backend-specific signals (``old_per_token_logps`` / ``ref_per_token_logps`` /
    ``advantages`` / ``num_items_in_batch``) are filled by the caller afterwards —
    non-Ray via batch forward, Ray by stacking per-sample remote results. No
    distributed communication happens here.
    """
    encoded_list = [s.encoded for s in samples]
    model_inputs = to_device(template.data_collator(encoded_list, padding_to=padding_to), device)

    labels = model_inputs['labels']
    batch_size = len(samples)
    logits_to_keep = None
    if use_logits_to_keep:
        logits_to_keep = int((labels.shape[-1] - (torch.ne(labels, -100).int().argmax(-1))).max().item())
    completion_mask, seq_lengths, max_seq_len = build_completion_mask_and_seq_lengths(
        labels,
        batch_size,
        padding_free=template.padding_free,
        encoded_batch=model_inputs,
        device=device,
        logits_to_keep=logits_to_keep,
    )
    truncated_mask = torch.tensor([bool(s.is_truncated) for s in samples], dtype=torch.bool, device=device)
    rollout_per_token_logps = build_rollout_logps([s.rollout_logprobs for s in samples], completion_mask, device)

    routed_experts = build_routed_experts_batch(
        samples,
        seq_lengths=seq_lengths,
        max_seq_len=max_seq_len,
        template=template,
        device=device,
        router_replay_mode=router_replay_mode,
    )
    if routed_experts is not None:
        model_inputs['routed_experts'] = routed_experts

    grpo_batch = GRPOBatch(
        completion_mask=completion_mask,
        truncated_mask=truncated_mask,
        seq_lengths=seq_lengths,
        rollout_per_token_logps=rollout_per_token_logps,
        logits_to_keep=logits_to_keep,
    )
    return model_inputs, grpo_batch


def resolve_reward_funcs(
    reward_funcs_cfg: list,
    args: Any = None,
) -> Tuple[List[Any], List[str]]:
    """Resolve reward function configs into callables and their names.

    Shared between ``MegatronGRPOTrainer._prepare_rewards`` and
    ``GRPOTrainer._prepare_rewards``.

    Returns:
        (reward_funcs, reward_func_names)
    """
    import asyncio
    import inspect

    from swift.rewards import orms

    reward_funcs = list(reward_funcs_cfg)
    for i, reward_func in enumerate(reward_funcs):
        if isinstance(reward_func, str) and reward_func in orms:
            reward_funcs[i] = orms[reward_func](args=args) if args is not None else orms[reward_func]()
        elif not callable(reward_func) and not isinstance(reward_func, str):
            raise ValueError(f'reward_function {reward_func} is not implemented in swift.rewards')

    names = []
    for func in reward_funcs:
        if inspect.isfunction(func):
            names.append(func.__name__)
        else:
            names.append(func.__class__.__name__)

    return reward_funcs, names


def make_reward_weights(
    reward_weights_cfg: Optional[List[float]],
    num_funcs: int,
    device: torch.device,
) -> torch.Tensor:
    """Build reward weight tensor, validating length against the reward
    function count."""
    if reward_weights_cfg is not None:
        if len(reward_weights_cfg) != num_funcs:
            raise ValueError(f'Number of reward weights ({len(reward_weights_cfg)}) must '
                             f'match number of reward functions ({num_funcs})')
        return torch.tensor(reward_weights_cfg, dtype=torch.float32, device=device)
    return torch.ones(num_funcs, dtype=torch.float32, device=device)

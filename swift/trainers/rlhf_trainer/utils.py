# Copyright (c) Alibaba, Inc. and its affiliates.
import functools
import math
import os
import time
from contextlib import contextmanager
from functools import partial
from io import BytesIO
from types import MethodType
from typing import TYPE_CHECKING, Any, List, Optional, Union

import datasets
import torch
import torch.nn.functional as F
from peft.tuners import lora
from peft.tuners.lora import LoraLayer
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from transformers import Trainer

from swift.utils import is_swanlab_available, is_wandb_available

if is_wandb_available():
    import wandb
if is_swanlab_available():
    import swanlab

if TYPE_CHECKING:
    from swift.llm.utils import Messages


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

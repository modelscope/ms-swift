# Copyright (c) ModelScope Contributors. All rights reserved.
# ruff: noqa: C901
"""Keep a multimodal model's vision path *used* on text-only batches, without forking ``forward``.

Under DeepSpeed ZeRO-3 every partitioned parameter must take part in each forward pass, or the
all-gather / reduce-scatter collectives across ranks fall out of step and training hangs. On a
text-only micro-batch the stock ``forward`` skips the vision tower entirely, so its parameters --
and, for models like Qwen3-VL, the deepstack mergers wired into intermediate layers -- go dark on
that rank while other ranks still gather them.

Legacy swift solved this by *copying the whole ``forward``* into ``swift/model/models`` and
splicing an ``image_features.mean() * 0`` term in the middle. That fork pins a dozen transformers
internals (``get_rope_index`` signature, ``Qwen3VLModelOutputWithPast`` fields, the
``_deepstack_process`` method name, ...) and rots on every transformers release.

This module keeps the same *maths* -- "run the vision path, contribute exactly zero" -- but moves
it entirely outside transformers internals:

  * the **data side** feeds a tiny dummy image plus matching placeholder tokens (masked out of the
    loss), so the *stock* forward runs the whole vision path exactly as it would for a real image,
    and every vision parameter is therefore used;
  * this module zeroes the **output of the aligner family** -- the single chokepoint every vision
    feature passes through on its way into the LLM (image embeds *and* deepstack embeds). The dummy
    then contributes nothing to the hidden states, so the loss is uncontaminated.

Because the aligner sits downstream of the vision tower and upstream of the LLM, zeroing it keeps
the tower alive (it still ran) while blocking all contamination. The only model-specific knowledge
required is the aligner module names -- and those are already declared on :class:`ModelArch`, so
one mechanism covers every multimodal family instead of a per-model forward fork.

The zeroing must be *conditional*: the aligner is shared, so a blanket ``* 0`` would also wipe the
features of real images in a mixed micro-batch. The hang only occurs when an entire micro-batch is
text-only (``pixel_values is None``), which is also the only case the collator injects a dummy, so
the collator flips :attr:`VisionKeepAlive.active` for exactly those forwards.
"""

from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Sequence, Tuple, Union

import torch
import torch.nn as nn

from swift.dev.data_format import VISION_DUMMY_FLAG

__all__ = [
    "DUMMY_FLAG",
    "MediaGraphPath",
    "MultimodalGraphAdapter",
    "VisionKeepAlive",
    "apply_vision_keep_alive",
    "inject_dummy_media",
]

# Marker the data side leaves on a collated batch to tell the model side "this forward carries an
# injected dummy image -- zero the aligner". Popped by the pre-hook before it reaches the model.
DUMMY_FLAG = VISION_DUMMY_FLAG


def _deep_getattr(root: nn.Module, dotted: str) -> nn.Module:
    obj = root
    for attr in dotted.split("."):
        obj = getattr(obj, attr)
    return obj


def _leaf_modules(root: nn.Module, prefix: str) -> Iterator[nn.Module]:
    """Yield the module(s) whose ``forward`` actually runs for one aligner prefix.

    A plain module (a merger) runs its own ``forward``, so it is the hook target. A container
    (``deepstack_merger_list`` is an ``nn.ModuleList``) never runs its own ``forward`` -- its
    elements are called one by one inside the model -- so the elements are the hook targets.
    """
    try:
        module = _deep_getattr(root, prefix)
    except AttributeError:
        return
    if isinstance(module, (nn.ModuleList, nn.Sequential)):
        yield from module
    else:
        yield module


def _zero(output):
    """Multiply an output by zero while keeping its autograd connection."""
    if isinstance(output, torch.Tensor):
        return output * 0
    if isinstance(output, (tuple, list)):
        return type(output)(_zero(o) for o in output)
    return output


@dataclass(frozen=True)
class MediaGraphPath:
    name: str
    media_type: str
    input_keys: Tuple[str, ...]
    tower_paths: Tuple[str, ...]
    aligner_paths: Tuple[str, ...]


class MultimodalGraphAdapter:
    def __init__(self, paths: Sequence[MediaGraphPath], processor, image_token_id: int):
        self.paths = tuple(paths)
        self.processor = processor
        self.image_token_id = image_token_id

    def _make_image_dummy(self) -> Dict:
        from PIL import Image

        image_processor = self.processor.image_processor
        dummy = dict(image_processor(images=[Image.new("RGB", (32, 32))], return_tensors="pt"))
        grid = dummy.get("image_grid_thw")
        if grid is None:
            raise RuntimeError("The multimodal graph adapter requires image_grid_thw from the image processor.")
        merge_size = getattr(image_processor, "merge_size", 1)
        dummy["num_tokens"] = int((grid.prod(dim=-1) // (merge_size**2)).sum())
        return dummy

    def inject_missing_paths(self, batch: Dict) -> Dict:
        for path in self.paths:
            if path.media_type != "image":
                raise NotImplementedError(f"No dummy injector is registered for {path.media_type!r}.")
            if not any(batch.get(key) is not None for key in path.input_keys):
                inject_dummy_media(
                    batch, make_dummy=self._make_image_dummy, image_token_id=self.image_token_id, flag_key=DUMMY_FLAG
                )
        return batch

    def install_zero_hooks(self, model: nn.Module) -> "VisionKeepAlive":
        aligners = [name for path in self.paths for name in path.aligner_paths]
        return apply_vision_keep_alive(model, aligners)


class VisionKeepAlive:
    """A toggle over a set of forward hooks that zero the aligner outputs when :attr:`active`.

    Created by :func:`apply_vision_keep_alive`; the collator (or the training step) sets
    ``active = True`` for a text-only-plus-dummy forward and restores it afterwards, e.g. via the
    :meth:`activated` context manager.
    """

    def __init__(self) -> None:
        self.active: bool = False
        self._flag_key: str = DUMMY_FLAG
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def _hook(self, module: nn.Module, args, output):
        return _zero(output) if self.active else output

    def _pre_hook(self, module: nn.Module, args, kwargs):
        # Runs once at the top model, before any aligner sub-module. Reads (and removes) the data
        # side's marker so the model never sees an unexpected kwarg, and arms the zeroing for this
        # forward only.
        self.active = bool(kwargs.pop(self._flag_key, False))
        return args, kwargs

    def _post_hook(self, module: nn.Module, args, kwargs, output):
        # Disarm as soon as the top model returns, so a subsequent real-image forward is never
        # zeroed even if it somehow arrives without going through the pre-hook.
        self.active = False
        return output

    @contextmanager
    def activated(self):
        prev, self.active = self.active, True
        try:
            yield
        finally:
            self.active = prev

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def apply_vision_keep_alive(
    model: nn.Module, aligner: Union[str, Sequence[str]], flag_key: str = DUMMY_FLAG
) -> VisionKeepAlive:
    """Hook every aligner module of ``model`` so its output can be zeroed on demand.

    Also installs one pre-hook / post-hook on ``model`` itself so the zeroing is driven entirely by
    the data side: :func:`inject_dummy_media` marks a batch with ``flag_key`` and the pre-hook arms
    the zeroing for that forward, the post-hook disarms it. No manual toggling in the training loop.

    Args:
        model: The loaded multimodal model.
        aligner: The aligner module-name prefix(es), i.e. ``ModelArch.aligner``.
        flag_key: The batch key the data side sets to request zeroing.

    Returns:
        A :class:`VisionKeepAlive` handle (also usable manually via
        :meth:`VisionKeepAlive.activated`).
    """
    if isinstance(aligner, str):
        aligner = [aligner]
    state = VisionKeepAlive()
    state._flag_key = flag_key
    for prefix in aligner:
        for module in _leaf_modules(model, prefix):
            state._handles.append(module.register_forward_hook(state._hook))
    state._handles.append(model.register_forward_pre_hook(state._pre_hook, with_kwargs=True))
    state._handles.append(model.register_forward_hook(state._post_hook, with_kwargs=True))
    return state


def inject_dummy_media(
    batch: Dict, *, make_dummy: Callable[[], Dict], image_token_id: int, flag_key: str = DUMMY_FLAG
) -> Dict:
    """Data side of the keep-alive: give a text-only batch a dummy image so the stock forward runs.

    A batch that already carries images is returned untouched -- the vision path runs naturally and
    no keep-alive is needed. Otherwise a single dummy image is spliced into the first sample: its
    ``N`` placeholder tokens (``N`` == the dummy's feature count, to satisfy the stock forward's
    ``n_image_tokens == n_features`` assertion) are appended with ``attention_mask=1`` and
    ``labels=-100``; the other rows get ``N`` padding positions with ``attention_mask=0`` so the
    batch stays rectangular. The batch is marked with ``flag_key`` for the model-side pre-hook.

    Args:
        batch: A collated batch dict (``input_ids`` required; ``attention_mask`` / ``labels``
            extended when present).
        make_dummy: Builds one dummy image; returns ``{'pixel_values', 'image_grid_thw',
            'num_tokens'}`` (extra keys are copied through, e.g. ``pixel_values_videos``).
        image_token_id: The id of the image placeholder token for this model.
        flag_key: The batch key to set so the model side arms zeroing.

    Returns:
        The same ``batch`` dict, mutated in place.
    """
    if batch.get("pixel_values") is not None or batch.get("pixel_values_videos") is not None:
        return batch
    input_ids = batch["input_ids"]
    # Invariant the model-side blanket `*0` relies on: a flagged batch is *wholly* dummy. A batch
    # with no pixel_values must therefore carry no image placeholders either -- otherwise arming the
    # zeroing would silently wipe features the stock forward still expects. Fail loudly instead.
    if bool((input_ids == image_token_id).any()):
        raise ValueError(
            "inject_dummy_media: batch has image placeholder tokens but no pixel_values; "
            "the wholly-text-only invariant is violated."
        )
    dummy = make_dummy()
    n = dummy["num_tokens"]

    def _extend(key: str, first_row_fill: int, other_rows_fill: int) -> None:
        tensor = batch.get(key)
        if tensor is None:
            return
        pad = torch.full((*tensor.shape[:-1], n), other_rows_fill, dtype=tensor.dtype, device=tensor.device)
        if tensor.dim() == 2:
            pad[0].fill_(first_row_fill)
        else:
            pad[..., 0, :].fill_(first_row_fill)
        batch[key] = torch.cat([tensor, pad], dim=-1)

    _extend("input_ids", image_token_id, 0)
    _extend("attention_mask", 1, 0)
    _extend("labels", -100, -100)
    _extend("completion_mask", 0, 0)
    _extend("loss_scale", 0, 0)
    _extend("replay_mask", 0, 0)
    _extend("token_type_ids", 0, 0)
    _extend("mm_token_type_ids", 1, 0)
    position_ids = batch.get("position_ids")
    if position_ids is not None:
        step = torch.arange(1, n + 1, dtype=position_ids.dtype, device=position_ids.device)
        batch["position_ids"] = torch.cat([position_ids, position_ids[..., -1:] + step], dim=-1)

    for key in ("pixel_values", "image_grid_thw"):
        if key in dummy:
            batch[key] = dummy[key]
    for key, value in dummy.items():
        if key not in ("num_tokens", "pixel_values", "image_grid_thw"):
            batch[key] = value
    batch[flag_key] = True
    return batch

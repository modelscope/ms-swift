# Copyright (c) ModelScope Contributors. All rights reserved.
# ruff: noqa: C901
"""Inject dummy images into text-only multimodal micro-batches.

Swift appends loss-masked image tokens and marks the batch with
``VISION_DUMMY_FLAG``. Twinkle owns the conditional aligner-output hooks.
Batches containing real media are left unchanged.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple

import torch

from swift.dev.data_format import VISION_DUMMY_FLAG

__all__ = [
    "MediaGraphPath",
    "MultimodalGraphAdapter",
    "inject_dummy_media",
]


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
                inject_dummy_media(batch, make_dummy=self._make_image_dummy, image_token_id=self.image_token_id)
        return batch


def inject_dummy_media(
    batch: Dict, *, make_dummy: Callable[[], Dict], image_token_id: int, flag_key: str = VISION_DUMMY_FLAG
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

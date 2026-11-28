# Copyright (c) ModelScope Contributors. All rights reserved.
"""CHORD's expert data and two losses on one trainable model."""

from __future__ import annotations
import math
from copy import copy, deepcopy
from typing import TYPE_CHECKING, Callable, Iterator

if TYPE_CHECKING:
    from twinkle.data_format import InputFeature

    from swift.dev.config import DatasetConfig, RLConfig
    from swift.dev.recipe.preparation import TrainingModel
    from swift.dev.template import Template


def load_chord_features(rl_config: RLConfig, dataset_config: DatasetConfig, template: Template) -> list[InputFeature]:
    """Encode expert rows once without changing the run's dataset config."""
    if not rl_config.chord_sft_dataset:
        return []
    from swift.dev.builder import load_prompt_only_rows

    chord_config = copy(dataset_config)
    chord_config.dataset = list(rl_config.chord_sft_dataset)
    chord_config.val_dataset = []
    rows = load_prompt_only_rows(chord_config, None, split_dataset_ratio=0.0)
    features = [template.encode(row) for row in rows]
    if not features:
        raise ValueError("chord_sft_dataset produced no encodable rows.")
    return features


def expert_batches(features: list[InputFeature], batch_size: int, *, offset: int) -> Iterator[list[InputFeature]]:
    """Cycle defensive copies from the cursor restored by the recipe's micro-step count."""
    while True:
        batch = [deepcopy(features[(offset + i) % len(features)]) for i in range(batch_size)]
        offset += batch_size
        yield batch


def chord_forward_backward(
    model: TrainingModel,
    rl_config: RLConfig,
    forward_kwargs: dict,
    chord_fn: Callable[[], list[InputFeature]],
    *,
    step: int,
    num_micro_batches: int,
) -> None:
    """Accumulate both weighted losses; only the second forward finalizes gradients."""
    overlap = {"loss_key", "loss_weight", "finalize_gradients", "kind"} & set(forward_kwargs)
    if overlap:
        raise ValueError(f"forward kwargs duplicate explicit loss controls: {sorted(overlap)}")
    warmup = int(rl_config.chord_mu_warmup_steps or 0)
    decay = int(rl_config.chord_mu_decay_steps or 0)
    peak = float(rl_config.chord_mu_peak or 0.0)
    valley = float(rl_config.chord_mu_valley or 0.0)
    if warmup > 0 and step < warmup:
        mu = peak * step / warmup
    elif decay == 0 or step >= warmup + decay:
        mu = valley
    else:
        mu = valley + (peak - valley) * 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / decay))
    model.forward_backward(**forward_kwargs, loss_key="grpo", loss_weight=1.0 - mu, finalize_gradients=False)
    sft_inputs = chord_fn()
    sft_micro_batch_size = len(sft_inputs) // model.device_mesh.data_world_size
    for _ in range(num_micro_batches - 1):
        sft_inputs.extend(chord_fn())
    model.forward_backward(
        inputs=sft_inputs,
        gradient_accumulation_steps=1,
        kind="sft",
        micro_batch_size=sft_micro_batch_size,
        loss_key="sft",
        loss_weight=mu,
        finalize_gradients=True,
        **{
            key: forward_kwargs[key]
            for key in ("dynamic_batching", "max_tokens_per_micro_batch", "packing_algorithm", "micro_batch_order")
            if key in forward_kwargs
        },
    )

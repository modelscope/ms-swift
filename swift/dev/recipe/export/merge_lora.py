# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
import os
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from swift.dev.config import CheckpointConfig, DatasetConfig, ModelConfig, TemplateConfig, TunerConfig


def run_merge_lora(
    model_config: ModelConfig,
    tuner_config: TunerConfig,
    *,
    template_config: Optional[TemplateConfig] = None,
    checkpoint_config: Optional[CheckpointConfig] = None,
    output_dir: Optional[str] = None,
    device_map: Optional[Any] = None,
    replace_if_exists: bool = False,
    parallel_size: int = 1,
) -> str:
    """Merge ``tuner_config.adapters`` into the base model and save it; returns the output directory.

    Defaults to ``'{first adapter}-merged'`` so a merge next to the checkpoint needs no argument;
    an existing directory is left untouched unless ``replace_if_exists`` (merging is idempotent,
    and accidentally re-writing multi-GB weights is expensive).
    """
    from swift.dev.components import MergeComponent

    component = MergeComponent(
        model_config,
        tuner_config,
        template_config=template_config,
        checkpoint_config=checkpoint_config,
        output_dir=output_dir,
        device_map=device_map,
        replace_if_exists=replace_if_exists,
        parallel_size=parallel_size)
    return component().path


def run_to_peft_format(adapter: str, output_dir: str) -> str:
    """Convert a Swift-format adapter to PEFT's directory layout."""
    from swift.tuners import swift_to_peft_format

    os.makedirs(output_dir, exist_ok=True)
    return swift_to_peft_format(adapter, output_dir)


def run_push_to_hub(folder_path: str, checkpoint_config: CheckpointConfig,
                    dataset_config: DatasetConfig, commit_message: str) -> str:
    """Upload one produced model directory using the configured Hub implementation."""
    if not checkpoint_config.hub_model_id:
        raise ValueError('--hub_model_id is required with --push_to_hub.')
    if not os.path.isdir(folder_path):
        raise ValueError(f'Hub upload source is not a directory: {folder_path}')
    from swift.dev.utils.hub import get_hub

    get_hub(dataset_config.use_hf).push_to_hub(
        checkpoint_config.hub_model_id,
        folder_path,
        token=dataset_config.hub_token,
        private=checkpoint_config.hub_private_repo,
        revision=checkpoint_config.hub_revision,
        commit_message=commit_message)
    return folder_path

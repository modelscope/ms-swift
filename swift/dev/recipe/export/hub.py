# Copyright (c) ModelScope Contributors. All rights reserved.
"""The export chain's Hub upload step: publish one produced model directory.

Not a weight transform -- it moves an already-written directory to the Hub, so it stays a
plain function beside the transform recipes rather than becoming a component.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from swift.dev.config import CheckpointConfig, DatasetConfig


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

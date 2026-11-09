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
    """Upload one produced model directory through twinkle's ``HubOperation``.

    The same channel the training-side pusher uses (``build_hub_pusher``): dev's own hub classes
    are download-only by design, so the previous ``get_hub(use_hf).push_to_hub`` call hit a
    method that does not exist and only failed after the whole export had already run.
    ``HubOperation`` honors ``private`` / ``revision`` and dispatches ModelScope vs HF off the
    repo_id prefix; ``use_hf`` selects that prefix.
    """
    if not checkpoint_config.hub_model_id:
        raise ValueError('--hub_model_id is required with --push_to_hub.')
    if not os.path.isdir(folder_path):
        raise ValueError(f'Hub upload source is not a directory: {folder_path}')
    from twinkle.hub import HubOperation

    repo_id = checkpoint_config.hub_model_id
    if dataset_config.use_hf and not repo_id.startswith(('hf://', 'ms://')):
        repo_id = f'hf://{repo_id}'
    HubOperation.push_to_hub(
        repo_id,
        folder_path,
        token=dataset_config.hub_token,
        private=checkpoint_config.hub_private_repo,
        revision=checkpoint_config.hub_revision or 'master',
        commit_message=commit_message)
    return folder_path

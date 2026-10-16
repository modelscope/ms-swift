"""Adapter-format and Hub recipes, plus the ``run_merge_lora`` entry point.

``run_merge_lora`` is the name the CLI (``swift merge``, ``swift export --merge_lora``) and the
infer/deploy merge paths call; the merge itself lives in
:class:`swift.dev.components.MergeComponent`. This module keeps the procedural signature and returns
a plain ``str``, so those callers are unaffected by the component's typed artifact. It also keeps the
``_lora`` spelling the flag and the chained-export contract are known by, while the component is
named for the operation alone.
"""
from __future__ import annotations
import os
from typing import TYPE_CHECKING, Any, Optional

from swift.dev.components import MergeComponent

if TYPE_CHECKING:
    from swift.dev.config import CheckpointConfig, ModelConfig, TemplateConfig, TunerConfig


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

    Defaults to ``'{first adapter}-merged'`` (legacy's convention) so a merge next to the checkpoint
    needs no extra argument. An existing directory is left untouched unless ``replace_if_exists``,
    because merging is idempotent and re-writing multi-GB weights by accident is expensive.
    """
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


def run_push_to_hub(folder_path: str, checkpoint_config, dataset_config, commit_message: str) -> str:
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

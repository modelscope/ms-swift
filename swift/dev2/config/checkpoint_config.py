# Copyright (c) ModelScope Contributors. All rights reserved.
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class CheckpointConfig:
    """Model saving, resumption, and checkpoint management settings."""

    # === Save Strategy ===
    output_dir: str = 'output'
    save_strategy: Literal['steps', 'epoch', 'no', 'best'] = 'steps'
    # Integer interval, or a ratio in (0, 1) resolved against total optimizer steps in plan_steps.
    save_steps: float = 500.0
    save_total_limit: Optional[int] = None
    safe_serialization: bool = True
    max_shard_size: str = '5GB'
    save_only_model: bool = False

    # === Resume ===
    resume_from_checkpoint: Optional[str] = None
    resume_only_model: bool = False
    ignore_data_skip: bool = False

    # === Hub ===
    # The single home for hub credentials and target repo, so training and export push the same way.
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_private_repo: bool = False
    hub_strategy: Literal['end', 'every_save', 'checkpoint', 'all_checkpoints'] = 'every_save'
    hub_revision: Optional[str] = None
    # Off by default so uploading does not stall training; on to notice a dropped push.
    hub_always_push: bool = False

    # === Save/load selectivity ===
    # no_save_optim folds into save_optimizer on both backends; the rng/load trio is megatron-only.
    no_save_optim: bool = False
    no_save_rng: bool = False
    no_load_optim: bool = False
    no_load_rng: bool = False

    # === Save mechanics ===
    # mcore-native dist-ckpt knobs: no consumer under the unified HF-format runtime, refused.
    async_save: bool = False
    # The runtime always emits HF-format safetensors on both backends; False is rejected.
    save_safetensors: bool = True

    # === Misc ===
    add_version: bool = True
    # Restore model/template/quantization metadata from a checkpoint args.json (training default
    # off; inference and artifact CLIs enable it unless explicitly set).
    load_args: bool = False
    load_data_args: bool = False  # with load_args, also restore DatasetConfig fields

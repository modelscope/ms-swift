"""Checkpoint saving and resumption configuration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class CheckpointConfig:
    """Model saving, resumption, and checkpoint management settings."""

    # === Save Strategy ===
    output_dir: str = 'output'
    save_strategy: Literal['steps', 'epoch', 'no', 'best'] = 'steps'
    save_steps: int = 500
    save_total_limit: Optional[int] = None
    safe_serialization: bool = True
    max_shard_size: str = '5GB'
    save_on_each_node: bool = False
    save_only_model: bool = False

    # === Resume ===
    resume_from_checkpoint: Optional[str] = None
    resume_only_model: bool = False
    ignore_data_skip: bool = False

    # === Hub ===
    # The single home for hub credentials and target repo. ConvertConfig deliberately holds none of
    # these -- only the commit message -- so that a training push and an export push cannot end up
    # pointing at different repos.
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_private_repo: bool = False
    hub_strategy: Literal['end', 'every_save', 'checkpoint', 'all_checkpoints'] = 'every_save'
    hub_revision: Optional[str] = None
    #: Block at each push until the previous one finishes. Off by default so uploading does not stall
    #: training; turn it on when a dropped push would go unnoticed.
    hub_always_push: bool = False

    # === Megatron: save/load selectivity ===
    # Megatron saves weights, optimizer state and RNG state as separate pieces, so each can be skipped.
    # Dropping the optimizer makes a checkpoint much smaller and no longer resumable; dropping the RNG
    # state makes a resumed run's dropout and data order differ from an uninterrupted one.
    no_save_optim: bool = False
    no_save_rng: bool = False
    no_load_optim: bool = False
    no_load_rng: bool = False

    # === Megatron: save mechanics ===
    #: Write the checkpoint from a background thread so training continues during the write. The step
    #: after a save stops waiting on disk; a crash mid-write leaves that checkpoint incomplete.
    async_save: bool = False
    #: Save in safetensors rather than Megatron's torch format.
    save_safetensors: bool = True
    #: Keep one writer process alive across saves instead of starting one each time.
    use_persistent_ckpt_worker: bool = False
    #: Store distributed-optimizer state so it can be reloaded under any parallel layout, rather than
    #: only the one that wrote it. This is what makes a checkpoint resumable after changing TP/PP.
    dist_ckpt_optim_fully_reshardable: bool = False
    #: The same, in a form that uses less memory while resharding and more time.
    distrib_optim_fully_reshardable_mem_efficient: bool = False
    #: Write in the pre-0.14 mcore layout, for a checkpoint that has to be read by an older mcore.
    dist_ckpt_save_pre_mcore_014: bool = False

    # === Misc ===
    add_version: bool = True
    create_checkpoint_symlink: bool = False
    use_flash_ckpt: bool = False
    load_args: bool = False
    load_data_args: bool = False

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
    # Background write after a frozen CPU staging snapshot; completion markers land post-write.
    # Incompatible with save_total_limit.
    async_save: bool = False
    # The runtime always emits HF-format safetensors on both backends; False is rejected.
    save_safetensors: bool = True

    # === Misc ===
    add_version: bool = True
    # Restore model/template/quantization metadata from a checkpoint args.json (training default
    # off; inference and artifact CLIs enable it unless explicitly set).
    load_args: bool = False
    load_data_args: bool = False  # with load_args, also restore DatasetConfig fields


# === Checkpoint args.json contract ===
# The write side (recipe/utils.py::write_ckpt_args_json) iterates these key sets, so the reader
# here and the writer there share one definition and cannot drift apart.
#
# Keys applied regardless of the reader's current value. An entry is a plain field name, or a
# (json_key, field_name) pair when a persisted key kept its historical spelling (tuner_type).
CHECKPOINT_FORCE_LOAD_FIELDS = {
    'model_config': ('task_type',),
    'quantize_config': ('bnb_4bit_quant_type', 'bnb_4bit_use_double_quant'),
    'tuner_config': (('tuner_type', 'tuner'),),
}
# DatasetConfig fields restored only under --load_data_args (the reader writes them onto its
# DatasetConfig directly).
CHECKPOINT_DATA_FIELDS = (
    'dataset', 'val_dataset', 'cached_dataset', 'cached_val_dataset', 'split_dataset_ratio', 'data_seed',
    'dataset_num_proc', 'load_from_cache_file', 'dataset_shuffle', 'val_dataset_shuffle', 'streaming', 'interleave_prob',
    'stopping_strategy', 'shuffle_buffer_size', 'download_mode', 'columns', 'strict', 'remove_unused_columns',
    'disable_auto_column_mapping', 'model_name', 'model_author', 'custom_dataset_info')
# Keys applied only when the reader's own value is None/empty, indexed by _config_mapping's
# '<prefix>_config' names.
CHECKPOINT_LOAD_FIELDS = {
    'model_config': (
        'model', 'model_type', 'model_revision', 'torch_dtype', 'attn_impl', 'experts_impl',
        'new_special_tokens', 'num_labels', 'problem_type', 'rope_scaling', 'max_model_len'),
    'plugin_config': ('external_plugins', ),
    'quantize_config': (
        'quant_method', 'quant_bits', 'hqq_axis', 'bnb_4bit_compute_dtype', 'bnb_4bit_quant_storage'),
    'template_config': (
        'template', 'system', 'truncation_strategy', 'use_chat_template', 'response_prefix'),
}

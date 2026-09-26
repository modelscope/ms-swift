"""Builders: config -> object construction glue.

The single place that maps swift's atomic Configs onto constructors.
"""
from __future__ import annotations

from .dataset import build_dataset, load_prompt_rows, split_prompt_and_reference, to_trajectory
from .model import (build_device_mesh, build_device_mesh_if_dp, build_hf_device_mesh, build_model, is_megatron_backend,
                    load_model_processor)
from .sampler import (build_engine_args, build_sampler, is_pooling_task, pooled_data, pooling_task_for, sampled_texts,
                      to_pooling_params, to_sampling_params)
from .template import build_template

__all__ = [
    'build_model', 'build_template', 'build_dataset', 'is_megatron_backend', 'build_device_mesh',
    'build_device_mesh_if_dp', 'build_hf_device_mesh', 'load_model_processor', 'build_engine_args', 'build_sampler',
    'to_sampling_params', 'sampled_texts', 'to_pooling_params', 'pooled_data', 'pooling_task_for', 'is_pooling_task',
    'load_prompt_rows', 'to_trajectory', 'split_prompt_and_reference'
]

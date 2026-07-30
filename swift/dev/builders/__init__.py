"""Builders: config -> object construction glue.

The single place that maps swift's atomic Configs onto constructors.
"""
from __future__ import annotations

from .dataset import build_dataset
from .model import build_device_mesh, build_model, is_megatron_backend
from .template import build_template

__all__ = ['build_model', 'build_template', 'build_dataset', 'is_megatron_backend', 'build_device_mesh']

# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared policy-gradient training, including estimators that require a critic."""

from .run_policy import run_policy

__all__ = ["run_policy"]

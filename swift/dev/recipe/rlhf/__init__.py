# Copyright (c) ModelScope Contributors. All rights reserved.
# The offline RLHF family (dpo / kto / cpo / orpo / simpo / rm): assembly builds, run_dpo trains.
from __future__ import annotations

from swift.dev.recipe.rlhf.assembly import build_preference_objects
from swift.dev.recipe.rlhf.run_dpo import run_dpo

__all__ = ['build_preference_objects', 'run_dpo']

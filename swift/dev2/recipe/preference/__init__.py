# Copyright (c) ModelScope Contributors. All rights reserved.
# The offline preference family (dpo / kto / cpo / orpo / simpo / rm): assembly builds, run_dpo trains.
from __future__ import annotations

from swift.dev2.recipe.preference.assembly import build_preference_objects
from swift.dev2.recipe.preference.run_dpo import run_dpo

__all__ = ['build_preference_objects', 'run_dpo']

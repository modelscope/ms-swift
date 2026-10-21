# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelArtifact:
    """A model directory on disk: weights + config + processor, loadable on its own.

    Frozen because an artifact names a finished result: a component that wants a different
    directory produces a new artifact rather than editing this one.
    """

    path: str


@dataclass(frozen=True)
class DatasetArtifact:
    """An encoded dataset store: the train split plus an optional val split.

    ``train_dir`` / ``val_dir`` are standalone ``load_from_disk`` subdirectories -- what
    ``DatasetConfig.cached_dataset`` / ``cached_val_dataset`` consume (the SUBDIRECTORY, not the
    parent output_dir). Frozen like ``ModelArtifact``: it names a finished result.
    """

    train_dir: str
    val_dir: Optional[str] = None

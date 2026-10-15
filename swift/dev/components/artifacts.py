# Copyright (c) ModelScope Contributors. All rights reserved.
"""Typed artifacts: the values that flow between recipe components.

A component consumes artifacts and produces artifacts. Naming the value (rather than passing a bare
``str`` directory around) makes the data flow explicit: ``MergeComponent`` returns a ``ModelArtifact``
naming the directory it wrote, and a downstream component takes that artifact as its model source
instead of reading a config field someone mutated in place.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelArtifact:
    """A model directory on disk: weights + config + processor, loadable on its own.

    ``path`` is the directory produced by a weight transform (merge / quantize / convert) or by
    training. It is frozen because an artifact names a finished result -- a component that wants a
    different directory produces a new artifact rather than editing this one.
    """
    path: str


@dataclass(frozen=True)
class DatasetArtifact:
    """An encoded dataset store on disk: the train split plus an optional val split.

    ``train_dir`` / ``val_dir`` are standalone ``load_from_disk`` subdirectories, which is what
    ``DatasetConfig.cached_dataset`` / ``cached_val_dataset`` consume (they take the SUBDIRECTORY, not
    the parent output_dir). ``val_dir`` is None when there is no val split. Frozen for the same reason
    as ``ModelArtifact``: it names a finished result.
    """
    train_dir: str
    val_dir: Optional[str] = None

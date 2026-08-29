# Copyright (c) ModelScope Contributors. All rights reserved.
"""The dataloader on its way out, kept until twinkle's can carry every dataset dev builds.

`legacy` because the replacement is decided, not because this is broken: twinkle's ``DataLoader`` is
a remote class, so under Ray each worker iterates its own copy and no batch is ever pickled to a
driver. What is here is a plain torch DataLoader living in the driver process, which means one
process reads for every data-parallel rank and ships each batch out over the wire -- the decode work
for eight GPUs' worth of images landing on one CPU. That gap cannot be closed from this side.

What still keeps it alive is the dataset, not the loader. Handing a dataset to a remote DataLoader
means the dataset has to reach the worker, and only the HuggingFace-native ones survive the trip:
``LazyLLMDataset`` carries an encode callable, and both packing datasets hold an ``mp.Queue`` and
live worker processes. Those are dev's main paths. twinkle's documented way out is to pass a factory
instead of an instance, so the dataset is built inside the worker -- which is a change to how
``build_dataset`` composes the encode chain, not to anything in this package.

So the order is: make the encode chain constructible worker-side, then move the map-style path to
twinkle (``resumable.py`` goes away with it -- twinkle tracks consumed samples and resumes by
``(epoch, offset)`` itself), and only then decide the iterable path, where both frameworks currently
waste the same critical path in different ways.

Two things in here should not be carried across, whatever happens to the rest:
- ``_build_device_mesh_dataloader`` is unreachable (``build_dataset`` only builds a device mesh when
  ``dp_shard_in_loader``, and that branch returns before reaching it) and duplicates twinkle's
  ``EpochSampler`` + ``DeviceMeshSampler`` -- without the per-epoch reseed, so it reads the same
  order every epoch.
- The iterable path multiplies the base batch width by the data-parallel size on top of a dispatcher
  that already reads one batch per rank, so each rank receives a batch that many times too wide.
"""
from .factory import build_dataloader, identity_collate
from .resumable import ResumableDataLoaderWrapper

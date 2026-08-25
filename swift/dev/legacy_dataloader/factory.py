# Copyright (c) ModelScope Contributors. All rights reserved.
import logging
import torch
import torch.distributed as dist
from functools import partial
from torch.utils.data import DataLoader, Dataset, IterableDataset
from twinkle.dataloader.device_mesh_sampler import DeviceMeshSampler
from typing import Any, Callable, List, Optional, Union

from swift.dataloader.dispatcher import DataLoaderDispatcher
from swift.dataloader.shard import BatchSamplerShard, DataLoaderShard
from .resumable import ResumableDataLoaderWrapper

logger = logging.getLogger(__name__)


def identity_collate(batch):
    """Collate that returns the batch as a ``list[InputFeature]`` unchanged.

    We run collation inside our InputProcessor pipeline.
    """
    return list(batch)


class _MegatronDPBatchSampler(BatchSamplerShard):
    """BatchSamplerShard that shards by DATA-PARALLEL rank instead of global rank.

    Why this exists: under mode='local' (torchrun) there is no driver to scatter a global batch,
    so the dataloader itself must shard by data-parallel rank (see the run_sft/mode design notes).
    The base BatchSamplerShard derives rank as ``dist.get_rank() // tp_size`` -- correct only for
    pure TP, NOT for the full TP/PP/CP/EP layout, where TP/PP/CP/EP members must get the SAME data
    and only the DP dimension differs.

    The DP coordinate is passed IN (from builders.build_device_mesh) rather than read from Megatron's
    mpu. Reading mpu here was a real bug: run_sft builds the dataloader before build_model, and mpu
    answers 0 until the model initializes the parallel state, so ``total_samples // dp_world_size``
    divided by zero and every mode='local' Megatron run died before training. The DeviceMesh is the
    same object the model hands to mpu.initialize_model_parallel, so its dp coordinate and mpu's DP
    rank are the same by construction (pinned by test_mesh_dp_matches_megatron_rank_generator).

    Everything else (stateless-across-epochs shuffle via set_epoch, per_device_train_batch_size
    yield granularity, __len__) is inherited unchanged, so the ResumableDataLoaderWrapper resume
    contract and the SFTLoop._fit_megatron grouping semantics are preserved -- only the shard
    rank/size is corrected. (Ray mode does not use this: there slice_dp shards on the driver.)

    ``data_sharding`` (Megatron-only, mirrors legacy MegatronPretrainingRandomSampler) additionally
    changes the shuffle SCOPE: instead of permuting globally and then striding by DP rank, each rank
    permutes only its own contiguous bucket. Callers must not combine it with group_by_length
    (build_dataset downgrades that combination with a warning, matching legacy).
    """

    def __init__(self, *args, dp_rank: int, dp_world_size: int, data_sharding: bool = False, **kwargs):
        self._dp_rank = dp_rank
        self._dp_world_size = dp_world_size
        super().__init__(*args, **kwargs)
        self.data_sharding = data_sharding
        if data_sharding and self.group_by_length:
            raise ValueError('data_sharding is incompatible with group_by_length (the length-grouped '
                             'order is global, so it cannot be restricted to a per-rank bucket).')

    @property
    def rank(self):
        return self._dp_rank

    @property
    def world_size(self):
        return self._dp_world_size

    def __iter__(self):
        # Only data_sharding needs a different index order; everything else defers to the base.
        # Legacy equivalent: MegatronPretrainingRandomSampler.__iter__ data_sharding branch
        # (swift/megatron/trainers/batch_sampler.py:121-129) -- shard first, then shuffle inside
        # the rank's own bucket, so no global permutation is materialized.
        if not (self.shuffle and self.data_sharding):
            yield from super().__iter__()
            return

        generator = torch.Generator()
        generator.manual_seed(self.curr_seed)
        # Bucket size differs from legacy ON PURPOSE: legacy floors the bucket to a multiple of
        # micro_batch_size ((total // (mbs*dp)) * mbs, batch_sampler.py:122), dropping up to
        # dp*mbs-1 samples per epoch; we use the base class's total_samples (== total // dp), which
        # keeps every sample and stays consistent with the non-data_sharding path. Example
        # (total=50, mbs=4, dp=2): legacy bucket 24 (2 samples dropped) vs ours 25. Do not "fix"
        # this back to legacy's flooring -- it silently discards data.
        bucket_size = self.total_samples
        start_idx = self.rank * bucket_size
        random_idx = torch.randperm(bucket_size, generator=generator).tolist()

        batch = []
        for idx in random_idx:
            batch.append(start_idx + idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if not self.drop_last and len(batch) > 0:
            yield batch


def _mesh_dp_coords(device_mesh) -> tuple:
    """(dp_rank, dp_world_size) from the DeviceMesh, which needs neither dist nor mpu.

    Required (not optional-with-fallback) on the loader-side DP sharding path: a fallback to global
    rank would silently give TP/PP/CP peers DIFFERENT data, which corrupts training instead of
    failing. dp_rank is None when this process is outside the mesh, which is equally unusable here.
    """
    if device_mesh is None:
        raise ValueError('dp_shard_in_loader=True requires a device_mesh: the dataloader must shard by '
                         'data-parallel rank, and the mesh is the only pre-initialization source for it '
                         '(build_dataset passes builders.build_device_mesh(distributed_config)).')
    dp_rank, dp_world_size = device_mesh.dp_rank, device_mesh.dp_world_size
    if dp_rank is None or not dp_world_size:
        raise ValueError(f'device_mesh has no usable data-parallel coordinate for this process '
                         f'(dp_rank={dp_rank}, dp_world_size={dp_world_size}); it was likely built for a '
                         f'different world size than the one this process runs in.')
    return dp_rank, dp_world_size


def _is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _get_world_size(tp_size: int = 1) -> int:
    if _is_dist_initialized():
        return dist.get_world_size() // tp_size
    return 1


def _get_rank(tp_size: int = 1) -> int:
    if _is_dist_initialized():
        return dist.get_rank() // tp_size
    return 0


def _seed_worker(worker_id: int, num_workers: int = 0, rank: int = 0):
    # int() the composed seed: rank may arrive as a numpy integer (DeviceMesh.dp_rank comes from an
    # np.arange mesh, so it is np.int64), and random.seed rejects numpy scalars outright with "The
    # only supported seed types are ...". Only reachable with dataloader_num_workers > 0, which is
    # why it stayed latent -- dev defaults to no workers while legacy defaults to 4.
    init_seed = torch.initial_seed() % 2**32
    worker_seed = int(num_workers * rank + init_seed + worker_id)
    import numpy as np
    import random
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**32))
    torch.manual_seed(worker_seed)


def build_dataloader(
    dataset: Union[Dataset, IterableDataset],
    collate_fn: Callable,
    *,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = False,
    data_seed: int = 42,
    group_by_length: bool = False,
    lengths: Optional[List[int]] = None,
    data_sharding: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    sequence_parallel_size: int = 1,
    tp_size: int = 1,
    device: Optional[Any] = None,
    device_mesh: Optional[Any] = None,
    resumable: bool = False,
    consumed_samples: int = 0,
    dp_shard_in_loader: bool = False,
) -> Union[DataLoader, ResumableDataLoaderWrapper]:
    """Build a distributed dataloader with optional resumable state.

    Routing: SP path -> Iterable path -> Map-style path.

    dp_shard_in_loader: when True, the dataloader itself owns data-parallel sharding -- the map-style
    path uses _MegatronDPBatchSampler (shard by DP rank, not global rank) and the iterable path uses
    MegatronDataLoaderDispatcher (scatter on the DP group). Required whenever no upstream component
    splits the global batch by DP (see build_dataset, which derives it from the config), and it needs
    ``device_mesh`` for the DP coordinate. When False, DP sharding is handled elsewhere (a driver-side
    slice_dp scatter, or the global-rank BatchSamplerShard for the transformers path).
    """
    if group_by_length and lengths is None:
        raise ValueError('lengths must be provided when group_by_length=True')
    if group_by_length and not shuffle:
        raise ValueError('shuffle must be True when group_by_length=True')
    # data_sharding is a DP-group concept (shard first, then shuffle within the shard); it has no
    # meaning without loader-side DP sharding, so reject it rather than ignore it silently.
    if data_sharding and not dp_shard_in_loader:
        raise ValueError('data_sharding requires the Megatron backend (dp_shard_in_loader=True): it '
                         'reshuffles within a data-parallel shard.')

    # Route 1: Sequence Parallel -- deliberately unimplemented for this phase. The prior SP path was
    # dead code (build_dataset never passes sequence_parallel_size>1) and untested, and it depended
    # on swift.sequence_parallel global state that could silently rot. Rather than ship a
    # never-exercised path that others might assume is correct, fail fast until the dedicated SP
    # chapter wires and tests it end to end.
    if sequence_parallel_size > 1:
        raise NotImplementedError('SP dataloader is not implemented in this phase. sequence_parallel_size>1 will be '
                                  'supported by the dedicated sequence-parallel work; until then the SP data path is '
                                  'intentionally disabled to avoid a silently-wrong, untested branch.')

    # Route 2: IterableDataset
    if isinstance(dataset, IterableDataset):
        dataloader = _build_iterable_dataloader(
            dataset=dataset,
            collate_fn=collate_fn,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            device=device,
            dp_shard_in_loader=dp_shard_in_loader,
            device_mesh=device_mesh)
        return ResumableDataLoaderWrapper(dataloader, consumed_samples=consumed_samples) if resumable else dataloader

    # Route 3: Map-style
    dataloader = _build_map_dataloader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        data_seed=data_seed,
        group_by_length=group_by_length,
        lengths=lengths,
        data_sharding=data_sharding,
        num_workers=num_workers,
        pin_memory=pin_memory,
        tp_size=tp_size,
        device=device,
        device_mesh=device_mesh,
        dp_shard_in_loader=dp_shard_in_loader)
    return ResumableDataLoaderWrapper(dataloader, consumed_samples=consumed_samples) if resumable else dataloader


def _build_iterable_dataloader(dataset,
                               collate_fn,
                               batch_size,
                               num_workers,
                               pin_memory,
                               device,
                               dp_shard_in_loader=False,
                               device_mesh=None):
    if dp_shard_in_loader:
        # Scatter on the DP group, not WORLD. The dispatcher overrides
        # group=mpu.get_data_parallel_group() (resolved when it iterates, by which time the model
        # has initialized mpu); rank0 of each DP group reads dp_world_size batches and scatters one
        # per DP rank (TP/PP/CP members get the same data). The batch WIDTH is needed now, though,
        # so it comes from the DeviceMesh rather than from the not-yet-initialized mpu.
        from swift.megatron.trainers.utils import MegatronDataLoaderDispatcher
        _, dp_world_size = _mesh_dp_coords(device_mesh)
        base = DataLoader(
            dataset,
            batch_size=batch_size * dp_world_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory)
        return MegatronDataLoaderDispatcher(base, device=device)
    world_size = _get_world_size()
    base = DataLoader(
        dataset,
        batch_size=batch_size * world_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory)
    return DataLoaderDispatcher(base, device=device)


def _build_map_dataloader(dataset,
                          collate_fn,
                          batch_size,
                          shuffle,
                          drop_last,
                          data_seed,
                          group_by_length,
                          lengths,
                          num_workers,
                          pin_memory,
                          tp_size,
                          device,
                          device_mesh,
                          dp_shard_in_loader=False,
                          data_sharding=False):
    # Shard by DP rank (see _MegatronDPBatchSampler): this path owns DP sharding because no upstream
    # component scatters the global batch (slice_dp is a no-op under mode='local'). The DP coordinate
    # comes from the DeviceMesh, so worker_init_fn seeds per DP rank too -- reading it from mpu here
    # used to return 0 for every rank (mpu is not initialized until build_model), which both broke
    # the sampler and gave all ranks the same worker seed.
    if dp_shard_in_loader:
        dp_rank, dp_world_size = _mesh_dp_coords(device_mesh)
        batch_sampler = _MegatronDPBatchSampler(
            total_samples=len(dataset),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            data_seed=data_seed,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            group_by_length=group_by_length,
            lengths=lengths,
            data_sharding=data_sharding)
        return DataLoaderShard(
            dataset,
            device=device,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=partial(_seed_worker, num_workers=num_workers, rank=dp_rank))

    rank = _get_rank(tp_size)

    if device_mesh is not None:
        return _build_device_mesh_dataloader(
            dataset=dataset,
            collate_fn=collate_fn,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            data_seed=data_seed,
            group_by_length=group_by_length,
            lengths=lengths,
            num_workers=num_workers,
            pin_memory=pin_memory,
            device=device,
            device_mesh=device_mesh)

    batch_sampler = BatchSamplerShard(
        total_samples=len(dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        data_seed=data_seed,
        tp_size=tp_size,
        group_by_length=group_by_length,
        lengths=lengths)

    return DataLoaderShard(
        dataset,
        device=device,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=partial(_seed_worker, num_workers=num_workers, rank=rank))


def _build_device_mesh_dataloader(dataset, collate_fn, batch_size, shuffle, drop_last, data_seed, group_by_length,
                                  lengths, num_workers, pin_memory, device, device_mesh):
    from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler

    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(data_seed)
        if group_by_length:
            from torch.utils.data import SubsetRandomSampler
            from transformers.trainer_pt_utils import get_length_grouped_indices
            indices = get_length_grouped_indices(lengths, batch_size, generator=generator)
            base_sampler = SubsetRandomSampler(indices)
        else:
            base_sampler = RandomSampler(dataset, generator=generator)
    else:
        base_sampler = SequentialSampler(dataset)

    base_batch_sampler = BatchSampler(base_sampler, batch_size=batch_size, drop_last=drop_last)
    mesh_batch_sampler = DeviceMeshSampler(base_batch_sampler, device_mesh, min_batch_size=None, skip_samples=0)

    dp_rank = getattr(device_mesh, 'data_rank', 0)
    return DataLoaderShard(
        dataset,
        device=device,
        batch_sampler=mesh_batch_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=partial(_seed_worker, num_workers=num_workers, rank=dp_rank))

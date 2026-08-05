# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import torch.distributed as dist
from tqdm import tqdm

from swift.utils import get_device, get_logger, to_device

logger = get_logger()


class DataLoaderDispatcher:
    """Feed every rank of a data-parallel group from a streaming dataset.

    Two modes:
    - dispatch (default): rank 0 iterates the dataset and scatters one batch to each rank.
    - shard (`shard_state` is set): the dataset has already been split into per-rank blocks, so
      each rank iterates its own share and nothing is scattered. The split is bound here because
      this is the layer that knows the data-parallel group (see `StreamingShardState`).
    """

    def __init__(self, base_dataloader, device=None, skip_batches: int = 0, shard_state=None):
        self.base_dataloader = base_dataloader
        self.device = device
        self.skip_batches = skip_batches
        self.shard_state = shard_state

    @property
    def rank(self):
        return dist.get_rank(self.group) if dist.is_initialized() else 0

    @property
    def world_size(self):
        return dist.get_world_size(self.group) if dist.is_initialized() else 1

    @property
    def group(self):
        return dist.group.WORLD if dist.is_initialized() else 1

    def __iter__(self):
        if self.shard_state is None:
            yield from self._iter_dispatch()
        else:
            yield from self._iter_shard()

    def _to_device(self, data):
        return to_device(data, self.device) if self.device else data

    # dispatch: rank 0 prepares every rank's batch

    def _scatter_object_list(self, inputs):
        if not dist.is_initialized():
            return inputs[0]
        outputs = [None]
        global_src_rank = dist.get_global_rank(self.group, 0)
        dist.scatter_object_list(outputs, inputs, global_src_rank, group=self.group)
        return outputs[0]

    def _skip_batches(self, base_iter):
        if self.rank == 0 and self.skip_batches > 0:
            for _ in tqdm(range(self.skip_batches), dynamic_ncols=True, desc='Skip Batches: '):
                [next(base_iter) for _ in range(self.world_size)]

    def _iter_dispatch(self):
        base_iter = iter(self.base_dataloader)
        self._skip_batches(base_iter)
        while True:
            if self.rank == 0:
                try:
                    data = [next(base_iter) for _ in range(self.world_size)]
                except StopIteration:
                    data = [None] * self.world_size
                data = self._scatter_object_list(data)
            else:
                data = self._scatter_object_list(None)
            if data is None:
                break
            yield self._to_device(data)

    # shard: every rank prepares its own batch

    @property
    def _sync_device(self):
        backend = dist.get_backend(self.group)
        if 'nccl' in backend or 'hccl' in backend:
            # collectives on these backends need the tensor on the accelerator. get_device()
            # rather than get_current_device(): the latter returns a bare index, which torch
            # resolves to cuda even on npu
            return self.device or get_device()
        return None

    @property
    def _shard_block_size(self) -> int:
        """How many consecutive raw samples make up one shard block.

        Sizing a block like the amount of raw data the loader consumes per emitted item means
        each rank ends up with the blocks rank 0 would have scattered to it, so without packing
        the per-rank sample sequence is unchanged. With packing it only lines the packing
        windows up -- rank 0 scatters individual packs, not whole packing rounds -- so the
        samples are the same but their grouping into packs is not.
        """
        # IterablePackingDataset consumes packing_interval raw samples per packing round
        packing_interval = getattr(self.base_dataloader.dataset, 'packing_interval', None)
        return packing_interval or self.base_dataloader.batch_size or 1

    def _any_rank_exhausted(self, exhausted: bool) -> bool:
        """Stop every rank as soon as one of the shards runs out, to keep the ranks in step."""
        if not dist.is_initialized():
            return exhausted
        flag = torch.tensor([exhausted], dtype=torch.uint8, device=self._sync_device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=self.group)
        return bool(flag.item())

    def _skip_shard_batches(self, base_iter):
        # each rank replays only its own shard, so resuming is world_size times cheaper
        for _ in tqdm(range(self.skip_batches), dynamic_ncols=True, desc='Skip Batches: ', disable=self.rank != 0):
            try:
                next(base_iter)
            except StopIteration:
                break

    def _iter_shard(self):
        self.shard_state.bind(self.rank, self.world_size, self._shard_block_size)
        logger.info_once(
            f'streaming_shard is enabled: each rank preprocesses 1/{self.world_size} of the stream instead of '
            f'rank0 preprocessing everything and scattering the batches. {self.shard_state}',
            hash_id='streaming_shard')
        base_iter = iter(self.base_dataloader)
        self._skip_shard_batches(base_iter)
        while True:
            try:
                data = next(base_iter)
            except StopIteration:
                data = None
            if self._any_rank_exhausted(data is None):
                break
            yield self._to_device(data)

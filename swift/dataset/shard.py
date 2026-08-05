# Copyright (c) ModelScope Contributors. All rights reserved.
from functools import partial
from typing import Optional, Tuple

SHARD_STATE_ATTR = 'streaming_shard_state'


class StreamingShardState:
    """How a streaming dataset is split across the ranks of a data-parallel group.

    Rank `r` keeps the blocks of `block_size` consecutive samples whose block index is
    `r` modulo `world_size`. Blocks rather than single samples because a block sized like
    the amount of raw data the loader consumes per emitted item reproduces the grouping
    rank 0 would have scattered, which keeps the data each rank sees unchanged.

    None of the three values are known when the dataset is built: Megatron-SWIFT initializes
    its model parallel state after the dataset is prepared, with sequence parallel the relevant
    group is the dp group rather than WORLD, and the batch size lives on the dataloader. They
    are filled in by the dataloader before iteration starts, and the filter reads them lazily.
    """

    def __init__(self):
        self.rank = 0
        self.world_size = 1
        self.block_size = 1

    def bind(self, rank: int, world_size: int, block_size: int) -> None:
        self.rank = rank
        self.world_size = world_size
        self.block_size = block_size

    def __repr__(self) -> str:
        return (f'StreamingShardState(rank={self.rank}, world_size={self.world_size}, '
                f'block_size={self.block_size})')


def _keep_sample(example, idx: int, state: StreamingShardState) -> bool:
    return idx // state.block_size % state.world_size == state.rank


def shard_streaming_dataset(dataset) -> Tuple[object, StreamingShardState]:
    """Split a streaming dataset so that each rank encodes only its own share.

    Must be applied before the encode/packing stage, otherwise the samples belonging to other
    ranks are still encoded before being dropped. Every rank reads the raw stream in full; only
    the expensive encoding is split, which keeps the split independent of the number and size of
    the underlying files.
    """
    state = StreamingShardState()
    dataset = dataset.filter(partial(_keep_sample, state=state), with_indices=True)
    return dataset, state


def set_shard_state(dataset, state: StreamingShardState) -> None:
    setattr(dataset, SHARD_STATE_ATTR, state)


def get_shard_state(dataset) -> Optional[StreamingShardState]:
    return getattr(dataset, SHARD_STATE_ATTR, None)

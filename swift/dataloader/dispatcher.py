# Copyright (c) ModelScope Contributors. All rights reserved.
import logging
import torch.distributed as dist
from tqdm import tqdm

from swift.utils import to_device

logger = logging.getLogger(__name__)


def compute_streaming_skip_batches(
    consumed_samples: int,
    micro_batch_size: int,
    dp_world_size: int,
    num_generations: int = 1,
) -> int:
    """Number of dispatcher batches to skip when resuming a streaming run.

    One `DataLoaderDispatcher` logical step consumes exactly
    `micro_batch_size * dp_world_size` samples: rank 0 draws one base batch of
    `micro_batch_size` per rank and scatters one to each. Training checkpoints
    count global samples (`consumed_train_samples`), so the number of logical
    steps already consumed is `consumed_samples // (micro_batch_size *
    dp_world_size)`. A non-divisible remainder is replayed rather than skipped
    past, so unseen data is never dropped.

    For GRPO/GKD, `consumed_train_samples` counts *generated* samples while the
    dataloader iterates unrepeated prompts, so callers pass `num_generations`
    to normalize first, mirroring the non-streaming sampler path. A count that
    is not divisible by `num_generations` is an invalid checkpoint state.
    """
    if num_generations > 1:
        if consumed_samples % num_generations:
            raise RuntimeError(f'consumed_train_samples ({consumed_samples}) must be divisible by '
                               f'num_generations ({num_generations})')
        consumed_samples //= num_generations
    samples_per_step = micro_batch_size * dp_world_size
    if consumed_samples <= 0 or samples_per_step <= 0:
        return 0
    return consumed_samples // samples_per_step


class DataLoaderDispatcher:
    """Scatter base batches from rank 0 and apply a resume offset to the first pass.

    Exact continuation requires the dispatcher to reach the offset within that
    pass. If a finite stream ends before the full offset is consumed, every rank
    ends the pass and the dispatcher discards the remaining offset. A cyclic
    caller then starts at the beginning of its next pass and can replay samples
    already consumed before the checkpoint.
    """

    def __init__(self, base_dataloader, device=None, skip_batches: int = 0):
        self.base_dataloader = base_dataloader
        self.device = device
        self.skip_batches = skip_batches

    @property
    def rank(self):
        return dist.get_rank(self.group) if dist.is_initialized() else 0

    @property
    def world_size(self):
        return dist.get_world_size(self.group) if dist.is_initialized() else 1

    @property
    def group(self):
        return dist.group.WORLD if dist.is_initialized() else 1

    def _scatter_object_list(self, inputs):
        if self.world_size == 1:
            return inputs[0]
        outputs = [None]
        global_src_rank = dist.get_global_rank(self.group, 0)
        dist.scatter_object_list(outputs, inputs, global_src_rank, group=self.group)
        return outputs[0]

    def _skip_batches(self, base_iter):
        if self.rank == 0 and self.skip_batches > 0:
            for _ in tqdm(range(self.skip_batches), dynamic_ncols=True, desc='Skip Batches: '):
                try:
                    [next(base_iter) for _ in range(self.world_size)]
                except StopIteration:
                    # Only rank 0 pulls from the stream, so an unhandled
                    # StopIteration here would leave the other ranks waiting
                    # for their first scatter. Consume what is left and let
                    # the main loop terminate every rank via the regular
                    # end-of-iteration path instead.
                    logger.warning(f'skip_batches ({self.skip_batches}) exceeds the batches available in this pass; '
                                   'the remaining offset is dropped and this pass yields no data.')
                    break
            # The resume offset applies once: a cyclic/restarted iterator must
            # read its later epochs in full instead of skipping again.
            self.skip_batches = 0

    def set_epoch(self, epoch: int):
        dataset = self.base_dataloader.dataset
        if hasattr(dataset, 'set_epoch'):
            dataset.set_epoch(epoch)

    def __iter__(self):
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
            if self.device:
                data = to_device(data, self.device)
            yield data

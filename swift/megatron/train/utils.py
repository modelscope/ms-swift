# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
from megatron.core import mpu

from swift.llm import DataLoaderDispatcher


class MegatronDataLoaderDispatcher(DataLoaderDispatcher):

    @property
    def group(self):
        return mpu.get_data_parallel_group()


def build_streaming_dataloader(args, dataset, collate_fn):
    from megatron.training.training import cyclic_iter
    base_dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_size=args.micro_batch_size,
        prefetch_factor=args.dataloader_prefetch_factor,
        persistent_workers=args.dataloader_persistent_workers,
    )
    return iter(cyclic_iter(MegatronDataLoaderDispatcher(base_dataloader)))

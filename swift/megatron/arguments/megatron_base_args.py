# Copyright (c) ModelScope Contributors. All rights reserved.
import os
from dataclasses import dataclass

from swift.arguments import BaseArguments
from swift.utils import get_logger
from .megatron_args import MegatronArguments

logger = get_logger()


@dataclass
class MegatronBaseArguments(MegatronArguments, BaseArguments):

    def _init_megatron_args(self):
        MegatronArguments.__post_init__(self)

    def __post_init__(self):
        self.sequence_parallel_size = self.context_parallel_size
        if self.packing:
            self.padding_free = True
        BaseArguments.__post_init__(self)
        self._init_megatron_args()
        if self.streaming:
            if self.dataloader_num_workers > 1:
                self.dataloader_num_workers = 1
                logger.info('Using streaming dataset, setting args.dataloader_num_workers to 1.')

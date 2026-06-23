# Copyright (c) ModelScope Contributors. All rights reserved.
import os
from dataclasses import dataclass

from swift.arguments import BaseArguments
from swift.utils import get_logger
from .megatron_args import MegatronArguments, RLHFMegatronArgumentsMixin

logger = get_logger()


@dataclass
class MegatronBaseArguments(MegatronArguments, BaseArguments):

    # true for ray pipeline to skip distributed init
    skip_megatron_init: bool = False

    def _init_distributed(self):
        if self.skip_megatron_init:
            return
        super()._init_distributed()

    def _init_output_dir(self):
        if self.skip_megatron_init:
            if self.output_dir is None:
                self.output_dir = f'megatron_output/{self.model_suffix}'
            os.makedirs(self.output_dir, exist_ok=True)
            return
        super()._init_output_dir()

    def _init_megatron_args(self):
        MegatronArguments.__post_init__(self)

    def __post_init__(self):
        self.sequence_parallel_size = self.context_parallel_size
        if self.packing:
            self.padding_free = True
        BaseArguments.__post_init__(self)
        self.seq_length = self.packing_length or self.max_length
        if self.skip_megatron_init:
            RLHFMegatronArgumentsMixin.__post_init__(self)
        else:
            self._init_megatron_args()
        if self.streaming:
            if self.dataloader_num_workers > 1:
                self.dataloader_num_workers = 1
                logger.info('Using streaming dataset, setting args.dataloader_num_workers to 1.')

# Copyright (c) ModelScope Contributors. All rights reserved.
import os
from dataclasses import dataclass

from swift.arguments import BaseArguments
from swift.utils import add_version_to_work_dir, get_logger, init_process_group, is_last_rank, to_abspath
from .megatron_args import MegatronArguments

logger = get_logger()


@dataclass
class MegatronBaseArguments(MegatronArguments, BaseArguments):

    def _init_output_dir(self):
        init_process_group(backend=self.ddp_backend, timeout=self.ddp_timeout)
        if self.output_dir is None:
            self.output_dir = f'megatron_output/{self.model_suffix}'
        self.output_dir = to_abspath(self.output_dir)
        if self.add_version:
            self.output_dir = add_version_to_work_dir(self.output_dir)
            logger.info(f'args.output_dir: {self.output_dir}')
        if is_last_rank():
            os.makedirs(self.output_dir, exist_ok=True)

    def __post_init__(self):
        self.sequence_parallel_size = self.context_parallel_size
        if self.packing:
            self.padding_free = True
        BaseArguments.__post_init__(self)
        self._init_output_dir()
        MegatronArguments.__post_init__(self)
        if self.streaming:
            self.dataloader_type = 'external'
            if self.dataloader_num_workers > 1:
                self.dataloader_num_workers = 1
                logger.info('Using streaming dataset, setting args.dataloader_num_workers to 1.')

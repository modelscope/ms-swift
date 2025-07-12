# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from dataclasses import dataclass

from swift.llm import BaseArguments
from swift.llm.argument.base_args import to_abspath
from swift.utils import add_version_to_work_dir, get_logger, init_process_group, is_master
from ..model import get_megatron_model_meta
from .megatron_args import MegatronArguments

logger = get_logger()


@dataclass
class MegatronTrainArguments(MegatronArguments, BaseArguments):
    add_version: bool = True
    # dataset
    lazy_tokenize: bool = False

    def init_model_args(self, config):
        self.megatron_model_meta = get_megatron_model_meta(self.model_type)
        kwargs = self.megatron_model_meta.convert_hf_config(config)
        logger.info(f'megatron_config: {kwargs}')
        for k, v in kwargs.items():
            if getattr(self, k) is None:
                setattr(self, k, v)
        MegatronArguments.__post_init__(self)
        self.extra_args = self.parse_to_megatron()

    def _init_save(self):
        init_process_group(backend=self.ddp_backend, timeout=self.ddp_timeout)
        if self.save is None:
            self.save = f'megatron_output/{self.model_suffix}'
        self.save = to_abspath(self.save)
        if self.add_version:
            self.save = add_version_to_work_dir(self.save)
            logger.info(f'args.save: {self.save}')
        if is_master():
            os.makedirs(self.save, exist_ok=True)

    def __post_init__(self):
        self.sequence_parallel_size = self.context_parallel_size
        self.load = to_abspath(self.load, check_path_exist=True)
        BaseArguments.__post_init__(self)
        self._init_save()
        self._check_packing()
        self.seq_length = self.seq_length or self.max_length
        if self.streaming:
            self.dataloader_type = 'external'
            if self.num_workers > 1:
                self.num_workers = 1
                logger.info('Using streaming dataset, setting args.num_workers to 1.')

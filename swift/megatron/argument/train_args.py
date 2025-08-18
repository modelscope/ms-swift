# Copyright (c) Alibaba, Inc. and its affiliates.
import math
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

    def init_model_args(self, tokenizer, config):
        self.megatron_model_meta = get_megatron_model_meta(self.model_type)
        kwargs = self.megatron_model_meta.convert_hf_config(config)
        if self.new_special_tokens and kwargs['padded_vocab_size'] < len(tokenizer):
            kwargs['padded_vocab_size'] = math.ceil(len(tokenizer) / 128) * 128
            self.initialize_embedding = True
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
        if self.packing:
            self.padding_free = True
        self.load = to_abspath(self.load, check_path_exist=True)
        BaseArguments.__post_init__(self)
        if len(self.dataset) == 0 and len(self.cached_dataset) == 0:
            raise ValueError(f'self.dataset: {self.dataset}, self.cached_dataset: {self.cached_dataset}. '
                             'Please input the training dataset.')
        self._init_save()
        self.seq_length = self.seq_length or self.max_length
        if self.streaming:
            self.dataloader_type = 'external'
            if self.num_workers > 1:
                self.num_workers = 1
                logger.info('Using streaming dataset, setting args.num_workers to 1.')
        if self.load is None and self.no_initialization:
            raise ValueError('You did not pass `--load`, so you need to set `--no_initialization false` '
                             'to allow the model to initialize weights properly.')

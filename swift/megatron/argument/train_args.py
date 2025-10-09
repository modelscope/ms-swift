# Copyright (c) Alibaba, Inc. and its affiliates.
import math
import os
from dataclasses import dataclass

import json

from swift.llm import BaseArguments
from swift.llm.argument.base_args import to_abspath
from swift.utils import add_version_to_work_dir, get_logger, init_process_group, is_master
from ..model import get_megatron_model_meta
from .megatron_args import MegatronArguments

logger = get_logger()


@dataclass
class MegatronTrainArguments(MegatronArguments, BaseArguments):
    add_version: bool = True
    load_args: bool = False

    def init_model_args(self, tokenizer, config):
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
        self.extra_args['model_info'] = self.model_info
        self.extra_args['model_meta'] = self.model_meta
        self.extra_args['megatron_model_meta'] = self.megatron_model_meta

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

    def _init_ckpt_dir(self, adapters=None):
        super()._init_ckpt_dir(adapters)
        if self.ckpt_dir and self.model is None:
            args_path = os.path.join(self.ckpt_dir, 'args.json')
            if not os.path.exists(args_path):
                return
            with open(args_path, 'r', encoding='utf-8') as f:
                old_args = json.load(f)
            self.model = old_args.get('model')

    def __post_init__(self):
        self.sequence_parallel_size = self.context_parallel_size
        if self.packing:
            self.padding_free = True
        self.load = to_abspath(self.load, check_path_exist=True)
        BaseArguments.__post_init__(self)
        self.megatron_model_meta = get_megatron_model_meta(self.model_type)
        if len(self.dataset) == 0 and len(self.cached_dataset) == 0:
            raise ValueError(f'self.dataset: {self.dataset}, self.cached_dataset: {self.cached_dataset}. '
                             'Please input the training dataset.')
        self._init_save()
        self.seq_length = self.seq_length or self.packing_length or self.max_length
        if self.streaming:
            self.dataloader_type = 'external'
            if self.num_workers > 1:
                self.num_workers = 1
                logger.info('Using streaming dataset, setting args.num_workers to 1.')
        if self.load is None and self.no_initialization:
            raise ValueError('You did not pass `--load`, so you need to set `--no_initialization false` '
                             'to allow the model to initialize weights properly.')
        if self.cached_dataset and self.context_parallel_size > 1:
            raise ValueError('`cached_dataset` does not support context parallelism.')

# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from dataclasses import dataclass

import torch

from swift.llm import BaseArguments
from swift.llm.argument.base_args import to_abspath
from swift.utils import add_version_to_work_dir, get_logger, init_process_group, is_local_master
from ..model import get_megatron_model_meta
from .megatron_args import MegatronArguments

logger = get_logger()


@dataclass
class MegatronTrainArguments(MegatronArguments, BaseArguments):
    add_version: bool = True
    lazy_tokenize: bool = False

    def init_model_args(self, config):
        self.megatron_model_meta = get_megatron_model_meta(self.model)
        kwargs = self.megatron_model_meta.load_config(config)
        for k, v in kwargs.items():
            setattr(self, k, v)
        MegatronArguments.__post_init__(self)
        self.extra_args = self.parse_to_megatron()

    def _init_save(self):
        init_process_group()
        if self.save is not None:
            return
        self.save = f'megatron_output/{self.model_suffix}'
        if self.add_version:
            self.save = add_version_to_work_dir(self.save)
            logger.info(f'args.save: {self.save}')
        self.save = to_abspath(self.save)
        if is_local_master():
            os.makedirs(self.save, exist_ok=True)

    def _init_mixed_precision(self):
        if self.torch_dtype == torch.bfloat16:
            self.bf16 = True
        elif self.torch_dtype == torch.float16:
            self.fp16 = True
            if self.apply_query_key_layer_scaling is None:
                self.apply_query_key_layer_scaling = True

    def __post_init__(self):
        BaseArguments.__post_init__(self)
        self._init_mixed_precision()
        self._init_save()
        if self.load is None:
            self.load = self.megatron_model

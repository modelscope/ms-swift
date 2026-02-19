# Copyright (c) ModelScope Contributors. All rights reserved.
import os
from dataclasses import dataclass

import json

from swift.utils import add_version_to_work_dir, get_logger, init_process_group, is_last_rank, to_abspath
from .megatron_base_args import MegatronBaseArguments

logger = get_logger()


@dataclass
class MegatronSftArguments(MegatronBaseArguments):
    add_version: bool = True
    load_args: bool = False

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

    def _init_ckpt_dir(self, adapters=None):
        super()._init_ckpt_dir(adapters)
        if self.ckpt_dir and self.model is None:
            args_path = os.path.join(self.ckpt_dir, 'args.json')
            if not os.path.exists(args_path):
                return
            with open(args_path, 'r', encoding='utf-8') as f:
                old_args = json.load(f)
            self.model = old_args.get('model')

    def _init_megatron_args(self):
        self._init_output_dir()
        super()._init_megatron_args()

    def __post_init__(self):
        self.mcore_model = to_abspath(self.mcore_model, check_path_exist=True)
        super().__post_init__()
        if len(self.dataset) == 0 and len(self.cached_dataset) == 0:
            raise ValueError(f'self.dataset: {self.dataset}, self.cached_dataset: {self.cached_dataset}. '
                             'Please input the training dataset.')
        if self.tensorboard_dir is None and self.output_dir is not None:
            self.tensorboard_dir = f'{self.output_dir}/runs'
        self.tensorboard_dir = to_abspath(self.tensorboard_dir)
        if self.mcore_model is None and self.model is None and not self.perform_initialization:
            raise ValueError('You did not pass `--mcore_model/--model` to read weights, so you need to set '
                             '`--perform_initialization true` to allow the model to initialize weights properly.')

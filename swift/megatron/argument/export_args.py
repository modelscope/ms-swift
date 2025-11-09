# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from dataclasses import dataclass
from typing import Optional

from swift.llm import HfConfigFactory
from swift.llm.argument.base_args import to_abspath
from swift.utils import get_logger
from .megatron_args import MegatronArguments
from .megatron_base_args import MegatronBaseArguments

logger = get_logger()


@dataclass
class MegatronExportArguments(MegatronBaseArguments):
    to_mcore: bool = False
    to_hf: bool = False
    test_convert_precision: bool = False
    test_convert_dtype: str = 'float32'
    exist_ok: bool = False
    merge_lora: Optional[bool] = None

    def _init_save(self):
        if self.save is None:
            ckpt_dir = self.ckpt_dir or f'./{self.model_suffix}'
            ckpt_dir, ckpt_name = os.path.split(ckpt_dir)
            if self.to_mcore:
                suffix = 'mcore'
            elif self.to_hf:
                suffix = 'hf'
            self.save = os.path.join(ckpt_dir, f'{ckpt_name}-{suffix}')

        self.save = to_abspath(self.save)
        if not self.exist_ok and os.path.exists(self.save):
            raise FileExistsError(f'args.save: `{self.save}` already exists.')
        logger.info(f'args.save: `{self.save}`')

    def __post_init__(self):
        super().__post_init__()
        self._init_save()
        self.test_convert_dtype = HfConfigFactory.to_torch_dtype(self.test_convert_dtype)
        extra_config = MegatronArguments.load_args_config(self.ckpt_dir)
        extra_config['adapter_load'] = self.adapter_load
        if self.load:
            extra_config['load'] = self.load
        for k, v in extra_config.items():
            setattr(self, k, v)
        if self.to_hf or self.to_mcore:
            self._init_convert()
            if self.model_info.is_moe_model is not None and self.tensor_model_parallel_size > 1:
                self.sequence_parallel = True
                logger.info('Settting args.sequence_parallel: True')
            if self.merge_lora is None:
                self.merge_lora = self.to_hf

    def _init_convert(self):
        convert_kwargs = {
            'no_save_optim': True,
            'no_save_rng': True,
            'no_load_optim': True,
            'no_load_rng': True,
            'finetune': True,
            'attention_backend': 'unfused',
            'padding_free': False,
        }
        for k, v in convert_kwargs.items():
            setattr(self, k, v)
        if self.model_info.is_moe_model:
            self.moe_grouped_gemm = True

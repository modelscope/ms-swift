# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from dataclasses import dataclass
from typing import Literal, Optional

import torch.distributed as dist

from swift.utils import get_logger, is_dist
from .base_args import BaseArguments, to_abspath
from .merge_args import MergeArguments
from .tuner_args import adapters_can_be_merged

logger = get_logger()


@dataclass
class ExportArguments(BaseArguments, MergeArguments):
    output_dir: Optional[str] = None

    to_peft_format: bool = False
    # awq/gptq
    quant_n_samples: int = 256
    quant_seqlen: int = 2048
    quant_device_map: str = 'auto'  # e.g. 'cpu', 'auto'
    quant_batch_size: int = 1

    # ollama
    to_ollama: bool = False
    gguf_file: Optional[str] = None

    # push to ms hub
    push_to_hub: bool = False
    # 'user_name/repo_name' or 'repo_name'
    hub_model_id: Optional[str] = None
    hub_private_repo: bool = False
    commit_message: str = 'update files'

    # megatron
    to_megatron: bool = False
    to_hf: bool = False
    tp: int = 1
    pp: int = 1

    def _init_quant(self):

        if self.quant_bits > 0:
            if self.quant_method is None:
                raise ValueError('Please specify the quantization method using `--quant_method awq/gptq`.')
            if len(self.dataset) == 0:
                raise ValueError(f'self.dataset: {self.dataset}, Please input the quant dataset.')

    def _init_output_dir(self):
        if self.ckpt_dir is None:
            ckpt_dir = self.model_info.model_dir
        ckpt_dir, ckpt_name = os.path.split(model_dir)
        if self.to_peft_format:
            suffix = 'peft'
        elif self.merge_lora:
            suffix = 'merged'
        elif self.quant_bits > 0:
            suffix = f'{self.quant_method}-int{self.quant_bits}'
        elif self.to_ollama:
            suffix = 'ollama'
        elif self.to_megatron:
            suffix = f'tp{self.tp}-pp{self.pp}'
        elif self.to_hf:
            suffix = 'hf'
        self.output_dir = os.path.join(ckpt_dir, f'{ckpt_name}-{suffix}')

        logger.info(f'Setting args.output_dir: {self.output_dir}')

        self.output_dir = to_abspath(self.output_dir)
        assert not os.path.exists(self.output_dir), (f'args.output_dir: {self.output_dir} already exists.')

    def __post_init__(self):
        super().__post_init__()
        self._init_output_dir()
        if self.quant_bits > 0:
            self._init_quant()
        elif self.to_ollama:
            assert self.train_type in ['full'] + adapters_can_be_merged()
            if self.train_type != 'full':
                self.merge_lora = True

        elif self.to_megatron or self.to_hf:
            os.environ['RANK'] = '0'
            os.environ['LOCAL_RANK'] = '0'
            os.environ['WORLD_SIZE'] = '1'
            os.environ['LOCAL_WORLD_SIZE'] = '1'
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
            assert is_dist(), 'Please start in distributed mode.'
            dist.init_process_group(backend='nccl')

    def _init_torch_dtype(self) -> None:
        if self.quant_bits > 0 and self.torch_dtype is None:
            self.torch_dtype = 'float16'
            logger.info(f'Setting args.torch_dtype: {self.torch_dtype}')
        super()._init_torch_dtype()

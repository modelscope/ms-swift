# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.distributed as dist

from swift.utils import get_logger, init_process_group, set_default_ddp_config
from .base_args import BaseArguments, to_abspath
from .merge_args import MergeArguments

logger = get_logger()


@dataclass
class ExportArguments(MergeArguments, BaseArguments):
    """
    ExportArguments is a dataclass that inherits from BaseArguments and MergeArguments.

    Args:
        output_dir (Optional[str]): Directory where the output will be saved.
        quant_n_samples (int): Number of samples for quantization.
        max_length (int): Sequence length for quantization.
        quant_batch_size (int): Batch size for quantization.
        to_ollama (bool): Flag to indicate export model to ollama format.
        push_to_hub (bool): Flag to indicate if the output should be pushed to the model hub.
        hub_model_id (Optional[str]): Model ID for the hub.
        hub_private_repo (bool): Flag to indicate if the hub repository is private.
        commit_message (str): Commit message for pushing to the hub.
        to_peft_format (bool): Flag to indicate if the output should be in PEFT format.
            This argument is useless for now.
    """
    output_dir: Optional[str] = None

    # awq/gptq
    quant_method: Literal['awq', 'gptq', 'bnb', 'fp8'] = None
    quant_n_samples: int = 256
    max_length: int = 2048
    quant_batch_size: int = 1
    group_size: int = 128

    # ollama
    to_ollama: bool = False

    # megatron
    to_mcore: bool = False
    to_hf: bool = False
    mcore_model: Optional[str] = None
    thread_count: Optional[int] = None
    test_convert_precision: bool = False

    # push to ms hub
    push_to_hub: bool = False
    # 'user_name/repo_name' or 'repo_name'
    hub_model_id: Optional[str] = None
    hub_private_repo: bool = False
    commit_message: str = 'update files'
    # compat
    to_peft_format: bool = False
    exist_ok: bool = False

    def _init_output_dir(self):
        if self.output_dir is None:
            ckpt_dir = self.ckpt_dir or f'./{self.model_suffix}'
            ckpt_dir, ckpt_name = os.path.split(ckpt_dir)
            if self.to_peft_format:
                suffix = 'peft'
            elif self.quant_method:
                suffix = f'{self.quant_method}'
                if self.quant_bits is not None:
                    suffix += f'-int{self.quant_bits}'
            elif self.to_ollama:
                suffix = 'ollama'
            elif self.merge_lora:
                suffix = 'merged'
            elif self.to_mcore:
                suffix = 'mcore'
            elif self.to_hf:
                suffix = 'hf'
            else:
                return

            self.output_dir = os.path.join(ckpt_dir, f'{ckpt_name}-{suffix}')

        self.output_dir = to_abspath(self.output_dir)
        if not self.exist_ok and os.path.exists(self.output_dir):
            raise FileExistsError(f'args.output_dir: `{self.output_dir}` already exists.')
        logger.info(f'args.output_dir: `{self.output_dir}`')

    def __post_init__(self):
        if self.quant_batch_size == -1:
            self.quant_batch_size = None
        if self.quant_bits and self.quant_method is None:
            raise ValueError('Please specify the quantization method using `--quant_method awq/gptq/bnb`.')
        if self.quant_method and self.quant_bits is None and self.quant_method != 'fp8':
            raise ValueError('Please specify `--quant_bits`.')
        if self.quant_method in {'gptq', 'awq'} and self.torch_dtype is None:
            self.torch_dtype = torch.float16
        if self.to_mcore or self.to_hf:
            self.mcore_model = to_abspath(self.mcore_model, check_path_exist=True)
            if not dist.is_initialized():
                set_default_ddp_config()
                init_process_group(backend=self.ddp_backend, timeout=self.ddp_timeout)

        BaseArguments.__post_init__(self)
        self._init_output_dir()
        if self.quant_method in {'gptq', 'awq'} and len(self.dataset) == 0:
            raise ValueError(f'self.dataset: {self.dataset}, Please input the quant dataset.')

# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from dataclasses import dataclass, field
from typing import Literal, Optional

import torch.distributed as dist

from swift.utils import get_logger, is_dist
from .base_args import BaseArguments, to_abspath
from .merge_args import MergeArguments

logger = get_logger()


@dataclass
class ExportArguments(MergeArguments, BaseArguments):
    """
    ExportArguments is a dataclass that inherits from BaseArguments and MergeArguments.

    Args:
        output_dir (Optional[str]): Directory where the output will be saved.
        to_peft_format (bool): Flag to indicate if the output should be in PEFT format.
            This argument is useless for now.
        quant_n_samples (int): Number of samples for quantization.
        quant_seqlen (int): Sequence length for quantization.
        quant_device_map (str): Device map for quantization, e.g., 'cpu', 'auto'.
        quant_batch_size (int): Batch size for quantization.
        to_ollama (bool): Flag to indicate export model to ollama format.
        gguf_file (Optional[str]): Path to the GGUF file when exporting to ollama format.
        push_to_hub (bool): Flag to indicate if the output should be pushed to the model hub.
        hub_model_id (Optional[str]): Model ID for the hub.
        hub_private_repo (bool): Flag to indicate if the hub repository is private.
        commit_message (str): Commit message for pushing to the hub.
        to_megatron (bool): Flag to indicate export model to megatron format.
        to_hf (bool): Flag to indicate export model to hugging face format.
        tp (int): Tensor parallelism degree.
        pp (int): Pipeline parallelism degree.
    """
    ckpt_dir: Optional[str] = field(default=None, metadata={'help': '/path/to/your/vx-xxx/checkpoint-xxx'})
    output_dir: Optional[str] = None
    device_map: str = 'auto'  # e.g. 'cpu', 'auto'
    safe_serialization: bool = True
    max_shard_size: str = '5GB'

    to_peft_format: bool = False
    # awq/gptq
    quant_n_samples: int = 256
    quant_seqlen: int = 2048
    quant_batch_size: int = 1
    group_size: int = 128

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
        ckpt_dir = self.ckpt_dir
        if ckpt_dir is None:
            ckpt_dir = self.model_info.model_dir
        ckpt_dir, ckpt_name = os.path.split(ckpt_dir)
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
        else:
            raise ValueError(f'args: {self}')
        self.output_dir = os.path.join(ckpt_dir, f'{ckpt_name}-{suffix}')

        logger.info(f'Setting args.output_dir: {self.output_dir}')

        self.output_dir = to_abspath(self.output_dir)
        assert not os.path.exists(self.output_dir), f'args.output_dir: {self.output_dir} already exists.'

    def __post_init__(self):
        super().__post_init__()
        self._init_output_dir()
        if self.quant_bits > 0:
            self._init_quant()
        elif self.to_ollama:
            assert self.train_type in ['full'] + self.adapters_can_be_merged

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

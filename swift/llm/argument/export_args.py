# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from dataclasses import dataclass, field
from typing import List, Literal, Optional

import torch
import torch.distributed as dist

from swift.llm import HfConfigFactory
from swift.utils import get_logger, init_process_group, set_default_ddp_config
from .base_args import BaseArguments, to_abspath
from .merge_args import MergeArguments

logger = get_logger()


@dataclass
class ExportArguments(MergeArguments, BaseArguments):
    """ExportArguments is a dataclass that inherits from BaseArguments and MergeArguments.

    Args:
        output_dir (Optional[str]): Directory to save the exported results. Defaults to None, which automatically sets
            a path with an appropriate suffix.
        quant_method (Optional[str]): The quantization method. Can be 'awq', 'gptq', 'bnb', 'fp8', or 'gptq_v2'.
            Defaults to None. See examples for more details.
        quant_n_samples (int): Number of samples for GPTQ/AWQ calibration. Defaults to 256.
        quant_batch_size (int): The batch size for quantization. Defaults to 1.
        group_size (int): The group size for quantization. Defaults to 128.
        to_cached_dataset (bool): Whether to tokenize and export the dataset in advance as a cached dataset. Defaults
            to False. Note: You can specify the validation set content through
            `--split_dataset_ratio` or `--val_dataset`.
        to_ollama (bool): Whether to generate the `Modelfile` required by Ollama. Defaults to False.
        to_mcore (bool): Whether to convert Hugging Face format weights to Megatron-Core format. Defaults to False.
        to_hf (bool): Whether to convert Megatron-Core format weights to Hugging Face format. Defaults to False.
        mcore_model (Optional[str]): The path to the Megatron-Core format model. Defaults to None.
        mcore_adapters (List[str]): A list of adapter paths for the Megatron-Core format model. Defaults to [].
        thread_count (Optional[int]): The number of model shards when `to_mcore` is True. Defaults to None, which
            automatically sets the number based on the model size to keep the largest shard under 10GB.
        test_convert_precision (bool): Whether to test the precision error of weight conversion between Hugging Face
            and Megatron-Core formats. Defaults to False.
        test_convert_dtype (str): The dtype to use for the conversion precision test. Defaults to 'float32'.
        push_to_hub (bool): Whether to push the output to the Model Hub. Defaults to False. See examples for more
            details.
        hub_model_id (Optional[str]): The model ID for pushing to the Hub (e.g., 'user_name/repo_name' or 'repo_name').
            Defaults to None.
        hub_private_repo (bool): Whether the Hub repository is private. Defaults to False.
        commit_message (str): The commit message for pushing to the Hub. Defaults to 'update files'.
        to_peft_format (bool): Whether to export in PEFT format. This argument is for compatibility and currently has
            no effect. Defaults to False.
        exist_ok (bool): If the output_dir exists, do not raise an exception and overwrite its contents. Defaults to
            False.
    """
    output_dir: Optional[str] = None

    # awq/gptq
    quant_method: Literal['awq', 'gptq', 'bnb', 'fp8', 'gptq_v2'] = None
    quant_n_samples: int = 256
    quant_batch_size: int = 1
    group_size: int = 128

    # cached_dataset
    to_cached_dataset: bool = False
    template_mode: Literal['train', 'rlhf', 'kto'] = 'train'

    # ollama
    to_ollama: bool = False

    # megatron
    to_mcore: bool = False
    to_hf: bool = False
    mcore_model: Optional[str] = None
    mcore_adapters: List[str] = field(default_factory=list)
    thread_count: Optional[int] = None
    test_convert_precision: bool = False
    test_convert_dtype: str = 'float32'

    # push to ms hub
    push_to_hub: bool = False
    # 'user_name/repo_name' or 'repo_name'
    hub_model_id: Optional[str] = None
    hub_private_repo: bool = False
    commit_message: str = 'update files'
    # compat
    to_peft_format: bool = False
    exist_ok: bool = False

    def load_args_from_ckpt(self) -> None:
        if self.to_cached_dataset:
            return
        super().load_args_from_ckpt()

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
            elif self.to_cached_dataset:
                suffix = 'cached_dataset'
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
        if isinstance(self.mcore_adapters, str):
            self.mcore_adapters = [self.mcore_adapters]
        if self.quant_bits and self.quant_method is None:
            raise ValueError('Please specify the quantization method using `--quant_method awq/gptq/bnb`.')
        if self.quant_method and self.quant_bits is None and self.quant_method != 'fp8':
            raise ValueError('Please specify `--quant_bits`.')
        if self.quant_method in {'gptq', 'awq'} and self.torch_dtype is None:
            self.torch_dtype = torch.float16
        if self.to_mcore or self.to_hf:
            if self.merge_lora:
                self.merge_lora = False
                logger.warning('`swift export --to_mcore/to_hf` does not support the `--merge_lora` parameter. '
                               'To export LoRA delta weights, please use `megatron export`')

            self.mcore_model = to_abspath(self.mcore_model, check_path_exist=True)
            if not dist.is_initialized():
                set_default_ddp_config()
                init_process_group(backend=self.ddp_backend, timeout=self.ddp_timeout)

        BaseArguments.__post_init__(self)
        self._init_output_dir()
        self.test_convert_dtype = HfConfigFactory.to_torch_dtype(self.test_convert_dtype)
        if self.quant_method in {'gptq', 'awq'} and len(self.dataset) == 0:
            raise ValueError(f'self.dataset: {self.dataset}, Please input the quant dataset.')
        if self.to_cached_dataset:
            self.lazy_tokenize = False
            if self.packing:
                raise ValueError('Packing will be handled during training; here we only perform tokenization '
                                 'in advance, so you do not need to set up packing separately.')
            assert not self.streaming, 'not supported'

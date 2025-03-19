# Copyright (c) Alibaba, Inc. and its affiliates.
import datetime as dt
import os
from dataclasses import dataclass
from typing import Literal, Optional, Union

import torch.distributed as dist

from swift.utils import get_logger, init_process_group, is_dist
from .base_args import BaseArguments, to_abspath
from .base_args.model_args import ModelArguments
from .merge_args import MergeArguments

logger = get_logger()


@dataclass
class LmdeployArguments:
    """
    LmdeployArguments is a dataclass that holds the configuration for lmdeploy.

    Args:
        tp (int): Tensor parallelism size. Default is 1.
        session_len(Optional[int]): The session length, default None.
        cache_max_entry_count (float): Maximum entry count for cache. Default is 0.8.
        quant_policy (int): Quantization policy, e.g., 4, 8. Default is 0.
        vision_batch_size (int): Maximum batch size in VisionConfig. Default is 1.
    """

    # lmdeploy
    tp: int = 1
    session_len: Optional[int] = None
    cache_max_entry_count: float = 0.8
    quant_policy: int = 0  # e.g. 4, 8
    vision_batch_size: int = 1  # max_batch_size in VisionConfig

    def get_lmdeploy_engine_kwargs(self):
        kwargs = {
            'tp': self.tp,
            'session_len': self.session_len,
            'cache_max_entry_count': self.cache_max_entry_count,
            'quant_policy': self.quant_policy,
            'vision_batch_size': self.vision_batch_size
        }
        if dist.is_initialized():
            kwargs.update({'devices': [dist.get_rank()]})
        return kwargs


@dataclass
class VllmArguments:
    """
    VllmArguments is a dataclass that holds the configuration for vllm.

    Args:
        gpu_memory_utilization (float): GPU memory utilization. Default is 0.9.
        tensor_parallel_size (int): Tensor parallelism size. Default is 1.
        pipeline_parallel_size(int): Pipeline parallelism size. Default is 1.
        max_num_seqs (int): Maximum number of sequences. Default is 256.
        max_model_len (Optional[int]): Maximum model length. Default is None.
        disable_custom_all_reduce (bool): Flag to disable custom all-reduce. Default is False.
        enforce_eager (bool): Flag to enforce eager execution. Default is False.
        limit_mm_per_prompt (Optional[str]): Limit multimedia per prompt. Default is None.
        vllm_max_lora_rank (int): Maximum LoRA rank. Default is 16.
        enable_prefix_caching (bool): Flag to enable automatic prefix caching. Default is False.
    """
    # vllm
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    max_num_seqs: int = 256
    max_model_len: Optional[int] = None
    disable_custom_all_reduce: bool = False
    enforce_eager: bool = False
    limit_mm_per_prompt: Optional[Union[dict, str]] = None  # '{"image": 5, "video": 2}'
    vllm_max_lora_rank: int = 16
    enable_prefix_caching: bool = False

    def __post_init__(self):
        self.limit_mm_per_prompt = ModelArguments.parse_to_dict(self.limit_mm_per_prompt)

    def get_vllm_engine_kwargs(self):
        adapters = self.adapters
        if hasattr(self, 'adapter_mapping'):
            adapters = adapters + list(self.adapter_mapping.values())
        kwargs = {
            'gpu_memory_utilization': self.gpu_memory_utilization,
            'tensor_parallel_size': self.tensor_parallel_size,
            'pipeline_parallel_size': self.pipeline_parallel_size,
            'max_num_seqs': self.max_num_seqs,
            'max_model_len': self.max_model_len,
            'disable_custom_all_reduce': self.disable_custom_all_reduce,
            'enforce_eager': self.enforce_eager,
            'limit_mm_per_prompt': self.limit_mm_per_prompt,
            'max_lora_rank': self.vllm_max_lora_rank,
            'enable_lora': len(adapters) > 0,
            'max_loras': max(len(adapters), 1),
            'enable_prefix_caching': self.enable_prefix_caching,
        }
        if dist.is_initialized():
            kwargs.update({'device': dist.get_rank()})
        return kwargs


@dataclass
class InferArguments(MergeArguments, VllmArguments, LmdeployArguments, BaseArguments):
    """
    InferArguments is a dataclass that extends BaseArguments, MergeArguments, VllmArguments, and LmdeployArguments.
    It is used to define the arguments required for model inference.

    Args:
        ckpt_dir (Optional[str]): Directory to the checkpoint. Default is None.
        infer_backend (Literal): Backend to use for inference. Default is 'pt'.
            Allowed values are 'vllm', 'pt', 'lmdeploy'.
        result_path (Optional[str]): Directory to store inference results. Default is None.
        max_batch_size (int): Maximum batch size for the pt engine. Default is 1.
        val_dataset_sample (Optional[int]): Sample size for validation dataset. Default is None.
    """
    infer_backend: Literal['vllm', 'pt', 'lmdeploy'] = 'pt'

    result_path: Optional[str] = None
    metric: Literal['acc', 'rouge'] = None
    # for pt engine
    max_batch_size: int = 1
    ddp_backend: Optional[str] = None

    # only for inference
    val_dataset_sample: Optional[int] = None

    def _get_result_path(self, folder_name: str) -> str:
        result_dir = self.ckpt_dir or f'result/{self.model_suffix}'
        os.makedirs(result_dir, exist_ok=True)
        result_dir = to_abspath(os.path.join(result_dir, folder_name))
        os.makedirs(result_dir, exist_ok=True)
        time = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
        return os.path.join(result_dir, f'{time}.jsonl')

    def _init_result_path(self, folder_name: str) -> None:
        if self.result_path is not None:
            self.result_path = to_abspath(self.result_path)
            return
        self.result_path = self._get_result_path(folder_name)
        logger.info(f'args.result_path: {self.result_path}')

    def _init_stream(self):
        self.eval_human = not (self.dataset and self.split_dataset_ratio > 0 or self.val_dataset)

        if self.stream and self.num_beams != 1:
            self.stream = False
            logger.info('Setting args.stream: False')

    def _init_ddp(self):
        if not is_dist():
            return
        assert not self.eval_human and not self.stream
        self._init_device()
        init_process_group(self.ddp_backend)

    def __post_init__(self) -> None:
        BaseArguments.__post_init__(self)
        VllmArguments.__post_init__(self)
        self._init_result_path('infer_result')
        self._init_eval_human()
        self._init_stream()
        self._init_ddp()

    def _init_eval_human(self):
        if len(self.dataset) == 0 and len(self.val_dataset) == 0:
            eval_human = True
        else:
            eval_human = False
        self.eval_human = eval_human
        logger.info(f'Setting args.eval_human: {self.eval_human}')

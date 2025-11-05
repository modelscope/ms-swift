# Copyright (c) Alibaba, Inc. and its affiliates.
import datetime as dt
import os
from dataclasses import dataclass
from typing import Literal, Optional

import torch.distributed as dist

from swift.trainers import VllmArguments
from swift.utils import get_logger, init_process_group, is_dist
from .base_args import BaseArguments, to_abspath
from .merge_args import MergeArguments

logger = get_logger()


@dataclass
class LmdeployArguments:
    """
    LmdeployArguments is a dataclass that holds the configuration for lmdeploy.

    Args:
        lmdeploy_tp (int): Tensor parallelism size. Default is 1.
        lmdeploy_session_len(Optional[int]): The session length, default None.
        lmdeploy_cache_max_entry_count (float): Maximum entry count for cache. Default is 0.8.
        lmdeploy_quant_policy (int): Quantization policy, e.g., 4, 8. Default is 0.
        lmdeploy_vision_batch_size (int): Maximum batch size in VisionConfig. Default is 1.
    """

    # lmdeploy
    lmdeploy_tp: int = 1
    lmdeploy_session_len: Optional[int] = None
    lmdeploy_cache_max_entry_count: float = 0.8
    lmdeploy_quant_policy: int = 0  # e.g. 4, 8
    lmdeploy_vision_batch_size: int = 1  # max_batch_size in VisionConfig

    def get_lmdeploy_engine_kwargs(self):
        kwargs = {
            'tp': self.lmdeploy_tp,
            'session_len': self.lmdeploy_session_len,
            'cache_max_entry_count': self.lmdeploy_cache_max_entry_count,
            'quant_policy': self.lmdeploy_quant_policy,
            'vision_batch_size': self.lmdeploy_vision_batch_size
        }
        if dist.is_initialized():
            kwargs.update({'devices': [dist.get_rank()]})
        return kwargs


@dataclass
class SglangArguments:
    sglang_tp_size: int = 1
    sglang_pp_size: int = 1
    sglang_dp_size: int = 1
    sglang_ep_size: int = 1
    sglang_enable_ep_moe: bool = False
    sglang_mem_fraction_static: Optional[float] = None
    sglang_context_length: Optional[int] = None
    sglang_disable_cuda_graph: bool = False
    sglang_quantization: Optional[str] = None
    sglang_kv_cache_dtype: str = 'auto'
    sglang_enable_dp_attention: bool = False
    sglang_disable_custom_all_reduce: bool = True

    def get_sglang_engine_kwargs(self):
        kwargs = {
            'tp_size': self.sglang_tp_size,
            'pp_size': self.sglang_pp_size,
            'dp_size': self.sglang_dp_size,
            'ep_size': self.sglang_ep_size,
            'enable_ep_moe': self.sglang_enable_ep_moe,
            'mem_fraction_static': self.sglang_mem_fraction_static,
            'context_length': self.sglang_context_length,
            'disable_cuda_graph': self.sglang_disable_cuda_graph,
            'quantization': self.sglang_quantization,
            'kv_cache_dtype': self.sglang_kv_cache_dtype,
            'enable_dp_attention': self.sglang_enable_dp_attention,
            'disable_custom_all_reduce': self.sglang_disable_custom_all_reduce,
        }
        if self.task_type == 'embedding':
            kwargs['task_type'] = 'embedding'
        return kwargs


@dataclass
class InferArguments(MergeArguments, LmdeployArguments, SglangArguments, VllmArguments, BaseArguments):
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
        reranker_use_activation (bool): reranker use activation after calculating. Default is True.
    """
    infer_backend: Literal['vllm', 'pt', 'sglang', 'lmdeploy'] = 'pt'

    result_path: Optional[str] = None
    write_batch_size: int = 1000
    metric: Literal['acc', 'rouge'] = None
    # for pt engine
    max_batch_size: int = 1

    # only for inference
    val_dataset_sample: Optional[int] = None

    # for reranker
    reranker_use_activation: bool = True

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
        if self.stream is None:
            self.stream = self.eval_human
        if self.stream and self.num_beams != 1:
            self.stream = False
            logger.info('Setting args.stream: False')

    def _init_ddp(self):
        if not is_dist():
            return
        eval_human = getattr(self, 'eval_human', False)
        assert not eval_human and not self.stream, (
            'In DDP scenarios, interactive interfaces and streaming output are not supported.'
            f'args.eval_human: {eval_human}, args.stream: {self.stream}')
        self._init_device()
        init_process_group(backend=self.ddp_backend, timeout=self.ddp_timeout)

    def __post_init__(self) -> None:
        BaseArguments.__post_init__(self)
        VllmArguments.__post_init__(self)
        self._init_result_path('infer_result')
        self._init_eval_human()
        self._init_ddp()

    def _init_eval_human(self):
        if len(self.dataset) == 0 and len(self.val_dataset) == 0:
            eval_human = True
        else:
            eval_human = False
        self.eval_human = eval_human
        logger.info(f'Setting args.eval_human: {self.eval_human}')

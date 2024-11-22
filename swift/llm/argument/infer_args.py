# Copyright (c) Alibaba, Inc. and its affiliates.
import datetime as dt
import os
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional, Tuple

import json
from transformers.utils.versions import require_version

from swift.llm import MODEL_MAPPING, TEMPLATE_MAPPING, ModelInfo, PtLoRARequest, get_template_meta
from swift.utils import get_logger, is_lmdeploy_available, is_vllm_available
from .base_args import BaseArguments, to_abspath
from .merge_args import MergeArguments

logger = get_logger()


@dataclass
class LmdeployArguments:
    """
    LmdeployArguments is a dataclass that holds the configuration for lmdeploy.

    Args:
        tp (int): Tensor parallelism degree. Default is 1.
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
        disable_custom_all_reduce (bool): Flag to disable custom all-reduce. Default is True.
        enforce_eager (bool): Flag to enforce eager execution. Default is False.
        limit_mm_per_prompt (Optional[str]): Limit multimedia per prompt. Default is None.
        vllm_max_lora_rank (int): Maximum LoRA rank. Default is 16.
        lora_modules (List[str]): List of LoRA modules. Default is an empty list.
        max_logprobs (int): Maximum log probabilities. Default is 20.
    """
    # vllm
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    max_num_seqs: int = 256
    max_model_len: Optional[int] = None
    disable_custom_all_reduce: bool = True  # Default values different from vllm
    enforce_eager: bool = False
    limit_mm_per_prompt: Optional[str] = None  # '{"image": 10, "video": 5}'
    vllm_max_lora_rank: int = 16
    lora_modules: List[str] = field(default_factory=list)
    max_logprobs: int = 20

    def __post_init__(self):
        self.vllm_enable_lora = len(self.lora_modules) > 0
        self.vllm_max_loras = max(len(self.lora_modules), 1)


@dataclass
class InferArguments(MergeArguments, VllmArguments, LmdeployArguments, BaseArguments):
    """
    InferArguments is a dataclass that extends BaseArguments, MergeArguments, VllmArguments, and LmdeployArguments.
    It is used to define the arguments required for model inference.

    Args:
        infer_backend (Literal): Backend to use for inference. Default is 'pt'.
            Allowed values are 'vllm', 'pt', 'lmdeploy'.
        ckpt_dir (Optional[str]): Directory to the checkpoint. Default is None.
        max_batch_size (int): Maximum batch size for the pt engine. Default is 1.
        val_dataset_sample (Optional[int]): Sample size for validation dataset. Default is None.
        result_dir (Optional[str]): Directory to store inference results. Default is None.
        save_result (bool): Flag to indicate if results should be saved. Default is True.
        stream (Optional[bool]): Flag to indicate if streaming should be enabled. Default is None.
    """
    ckpt_dir: Optional[str] = field(default=None, metadata={'help': '/path/to/your/vx-xxx/checkpoint-xxx'})
    infer_backend: Literal['vllm', 'pt', 'lmdeploy'] = 'pt'
    max_batch_size: int = 1  # for pt engine

    # only for inference
    val_dataset_sample: Optional[int] = None
    result_dir: Optional[str] = field(default=None, metadata={'help': '/path/to/your/infer_result'})
    save_result: bool = True
    stream: Optional[bool] = None

    def _init_result_dir(self, folder_name: str = 'infer_result') -> None:
        self.result_path = None
        if not self.save_result:
            return

        if self.result_dir is None:
            if self.ckpt_dir is None:
                if hasattr(self, 'model_info'):
                    result_dir = self.model_info.model_dir
                else:
                    result_dir = './'
            else:
                result_dir = self.ckpt_dir
            result_dir = os.path.join(result_dir, folder_name)
        else:
            result_dir = self.result_dir
        result_dir = to_abspath(result_dir)
        os.makedirs(result_dir, exist_ok=True)
        self.result_dir = result_dir
        time = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
        self.result_path = os.path.join(result_dir, f'{time}.jsonl')
        logger.info(f'args.result_path: {self.result_path}')

    def _init_stream(self):
        self.eval_human = not (self.dataset and self.split_dataset_ratio > 0 or self.val_dataset)
        if self.stream is None:
            self.stream = self.eval_human

        if self.template:
            template_meta = get_template_meta(self.template)
            if self.num_beams != 1 or not template_meta.support_stream:
                self.stream = False
                logger.info('Setting args.stream: False')

    def _init_weight_type(self):
        if self.ckpt_dir and os.path.exists(os.path.join(self.ckpt_dir, 'adapter_config.json')):
            self.weight_type = 'adapter'
        else:
            self.weight_type = 'full'

    def __post_init__(self) -> None:
        if self.ckpt_dir and self.load_args:
            self.load_args_from_ckpt(self.ckpt_dir)
        self._init_weight_type()
        BaseArguments.__post_init__(self)
        VllmArguments.__post_init__(self)
        MergeArguments.__post_init__(self)
        self._parse_lora_modules()

        self._init_result_dir()
        self._init_stream()
        self._init_eval_human()
        if self.ckpt_dir is None:
            self.train_type = 'full'

    def _init_eval_human(self):
        if len(self.dataset) == 0 and len(self.val_dataset) == 0:
            eval_human = True
        else:
            eval_human = False
        self.eval_human = eval_human
        logger.info(f'Setting args.eval_human: {self.eval_human}')

    def _parse_lora_modules(self) -> None:
        if len(self.lora_modules) == 0:
            self.lora_request_list = []
            return
        assert self.infer_backend in {'vllm', 'pt'}
        if self.infer_backend == 'vllm':
            from vllm.lora.request import LoRARequest
            lora_request_cls = LoRARequest
        elif self.infer_backend == 'pt':
            lora_request_cls = PtLoRARequest

        lora_request_list = []
        for i, lora_module in enumerate(self.lora_modules):
            lora_name, lora_local_path = lora_module.split('=')
            lora_request_list.append(lora_request_cls(lora_name, i + 1, lora_local_path))
        self.lora_request_list = lora_request_list

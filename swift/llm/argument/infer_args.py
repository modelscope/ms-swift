# Copyright (c) Alibaba, Inc. and its affiliates.
import datetime as dt
import os
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional, Tuple

import json
from transformers.utils.versions import require_version

from swift.llm import MODEL_MAPPING, TEMPLATE_MAPPING, ModelInfo
from swift.utils import get_logger, is_lmdeploy_available, is_vllm_available
from .base_args import BaseArguments, to_abspath
from .merge_args import MergeArguments
from .tuner_args import adapters_can_be_merged

logger = get_logger()


@dataclass
class VllmArguments:
    # vllm
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    max_num_seqs: int = 256
    max_model_len: Optional[int] = None
    disable_custom_all_reduce: bool = True  # Default values different from vllm
    enforce_eager: bool = False
    limit_mm_per_prompt: Optional[str] = None  # '{"image": 10, "video": 5}'
    vllm_enable_lora: bool = False
    vllm_max_lora_rank: int = 16
    lora_modules: List[str] = field(default_factory=list)
    max_logprobs: int = 20


@dataclass
class LmdeployArguments:
    # lmdeploy
    tp: int = 1
    cache_max_entry_count: float = 0.8
    quant_policy: int = 0  # e.g. 4, 8
    vision_batch_size: int = 1  # max_batch_size in VisionConfig


@dataclass
class InferArguments(BaseArguments, MergeArguments, VllmArguments, LmdeployArguments):
    infer_backend: Literal['vllm', 'pt', 'lmdeploy', None] = None
    ckpt_dir: Optional[str] = field(default=None, metadata={'help': '/path/to/your/vx-xxx/checkpoint-xxx'})

    val_dataset_sample: int = -1
    result_dir: Optional[str] = field(default=None, metadata={'help': '/path/to/your/infer_result'})
    save_result: bool = True

    # other
    stream: Optional[bool] = None

    def _init_result_dir(self) -> None:
        self.result_path = None
        if not self.save_result:
            return

        if self.result_dir is None:
            if self.ckpt_dir is None:
                result_dir = self.model_info.model_dir
            else:
                result_dir = self.ckpt_dir
            result_dir = os.path.join(result_dir, 'infer_result')
        else:
            result_dir = self.result_dir
        result_dir = to_abspath(result_dir)
        os.makedirs(result_dir, exist_ok=True)
        self.result_dir = result_dir
        time = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
        self.result_path = os.path.join(result_dir, f'{time}.jsonl')
        logger.info(f'args.result_path: {self.result_path}')

    def _init_stream(self):
        self.eval_dataset = bool(self.dataset and self.split_dataset_ratio > 0 or self.val_dataset)
        if self.stream is None:
            self.stream = not self.eval_dataset
        template_info = TEMPLATE_MAPPING[self.template]
        if self.num_beams != 1 or not template_info.get('stream', True):
            self.stream = False
            logger.info('Setting args.stream: False')

    def __post_init__(self) -> None:
        BaseArguments.__post_init__(self)
        MergeArguments.__post_init__(self)

        self._init_result_dir()
        self._init_stream()
        self._init_eval_human()
        self.prepare_infer_backend()

    def _init_eval_human(self):
        if len(self.dataset) == 0 and len(self.val_dataset) == 0:
            eval_human = True
        else:
            eval_human = False
        self.eval_human = eval_human
        logger.info(f'Setting args.eval_human: {self.eval_human}')

    def prepare_infer_backend(self):
        model_info = MODEL_MAPPING.get(self.model_type, {})
        support_vllm = model_info.get('support_vllm', False)
        support_lmdeploy = model_info.get('support_lmdeploy', False)
        self.lora_request_list = None
        if self.infer_backend == 'auto':
            self.infer_backend = 'pt'
            if is_vllm_available() and support_vllm and not self.is_multimodal:
                if ((self.train_type == 'full' or self.train_type in adapters_can_be_merged() and self.merge_lora)
                        and self.quantization_bit == 0):
                    self.infer_backend = 'vllm'
                if self.vllm_enable_lora:
                    self.infer_backend = 'vllm'
            if is_lmdeploy_available() and support_lmdeploy and self.is_multimodal:
                if ((self.train_type == 'full' or self.train_type == 'lora' and self.merge_lora)
                        and self.quantization_bit == 0):
                    self.infer_backend = 'lmdeploy'
        if self.infer_backend == 'vllm':
            require_version('vllm')
            if not support_vllm:
                logger.warning(f'vllm not support `{self.model_type}`')
            if self.train_type == 'lora' and not self.vllm_enable_lora:
                assert self.merge_lora, ('To use vLLM, you need to provide the complete weight parameters. '
                                         'Please set `--merge_lora true`.')
        if self.infer_backend == 'lmdeploy':
            require_version('lmdeploy')
            assert self.quantization_bit == 0, 'lmdeploy does not support bnb.'
            if not support_lmdeploy:
                logger.warning(f'lmdeploy not support `{self.model_type}`')
            if self.train_type == 'lora':
                assert self.merge_lora, ('To use LMDeploy, you need to provide the complete weight parameters. '
                                         'Please set `--merge_lora true`.')

        if (self.infer_backend == 'vllm' and self.vllm_enable_lora
                or self.infer_backend == 'pt' and isinstance(self, DeployArguments) and self.train_type == 'lora'):
            assert self.ckpt_dir is not None
            self.lora_modules.append(f'default-lora={self.ckpt_dir}')
            self.lora_request_list, self.use_dora = self._parse_lora_modules(self.lora_modules,
                                                                             self.infer_backend == 'vllm')

    @staticmethod
    def _parse_lora_modules(lora_modules: List[str], use_vllm: bool) -> Tuple[List[Any], bool]:
        VllmLoRARequest = None
        if use_vllm:
            try:
                from .vllm_utils import LoRARequest as VllmLoRARequest
            except ImportError:
                logger.warning('The current version of VLLM does not support `enable_lora`. Please upgrade VLLM.')
                raise

        # TODO:move
        @dataclass
        class PtLoRARequest:
            lora_name: str
            lora_int_id: int
            lora_local_path: str

        LoRARequest = VllmLoRARequest if use_vllm else PtLoRARequest
        lora_request_list = []
        use_dora_list = []
        for i, lora_module in enumerate(lora_modules):
            lora_name, lora_local_path = lora_module.split('=')
            with open(os.path.join(lora_local_path, 'adapter_config.json'), 'r') as f:
                _json = json.load(f)
                use_dora_list.append(_json.get('use_dora', False))
            lora_request_list.append(LoRARequest(lora_name, i + 1, lora_local_path))
        if any(use_dora_list) and len(lora_modules) > 1:
            raise ValueError('Dora does not support inference with other loras')
        elif not any(use_dora_list):
            use_dora = False
        else:
            use_dora = True
        return lora_request_list, use_dora

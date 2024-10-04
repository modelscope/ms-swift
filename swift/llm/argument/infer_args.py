# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional, Tuple

import json
from transformers.utils.versions import require_version

from swift.llm import MODEL_MAPPING, TEMPLATE_MAPPING, multimodal_keys
from swift.tuners import swift_to_peft_format
from swift.utils import get_logger, is_lmdeploy_available, is_vllm_available
from .base_args import BaseArguments
from .tuner_args import adapters_can_be_merged

logger = get_logger()


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


class LmdeployArguments:
    # lmdeploy
    tp: int = 1
    cache_max_entry_count: float = 0.8
    quant_policy: int = 0  # e.g. 4, 8
    vision_batch_size: int = 1  # max_batch_size in VisionConfig


class MergeArguments:
    merge_lora: bool = False
    merge_device_map: Optional[str] = None


@dataclass
class InferArguments(BaseArguments, MergeArguments, VllmArguments, LmdeployArguments):
    infer_backend: Literal['auto', 'vllm', 'pt', 'lmdeploy'] = 'auto'
    ckpt_dir: Optional[str] = field(default=None, metadata={'help': '/path/to/your/vx-xxx/checkpoint-xxx'})
    result_dir: Optional[str] = field(default=None, metadata={'help': '/path/to/your/infer_result'})

    load_args_from_ckpt_dir: bool = True
    load_dataset_config: bool = False
    eval_human: Optional[bool] = None
    show_dataset_sample: int = -1
    save_result: bool = True

    # other
    stream: bool = True
    verbose: Optional[bool] = None
    overwrite_generation_config: bool = False

    def __post_init__(self) -> None:
        BaseArguments.__post_init__(self)

        if self.eval_human is None:
            if len(self.dataset) == 0 and len(self.val_dataset) == 0:
                self.eval_human = True
            else:
                self.eval_human = False
            logger.info(f'Setting args.eval_human: {self.eval_human}')
        elif self.eval_human is False and len(self.dataset) == 0 and len(self.val_dataset) == 0:
            raise ValueError('Please provide the dataset or set `--load_dataset_config true`.')

        self.handle_infer_backend()
        self.handle_merge_device_map()

    def handle_ckpt_dir(self):
        if self.ckpt_dir is None:
            if self.ckpt_dir is None:
                # The original model is the complete model.
                self.train_type = 'full'
            if self.load_args_from_ckpt_dir:
                self.load_args_from_ckpt_dir = False
        if self.load_args_from_ckpt_dir:
            self.load_from_ckpt_dir()
        else:
            assert self.load_dataset_config is False, 'You need to first set `--load_args_from_ckpt_dir true`.'

    def handle_dataset(self):
        pass

    def handle_infer_backend(self):
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

        template_info = TEMPLATE_MAPPING[self.template_type]
        if self.num_beams != 1 or not template_info.get('stream', True):
            self.stream = False
            logger.info('Setting args.stream: False')
        self.infer_media_type = template_info.get('infer_media_type', 'none')
        if self.infer_media_type == 'none' and self.is_multimodal:
            self.infer_media_type = 'interleave'
        self.media_type = template_info.get('media_type', 'image')
        self.media_key = multimodal_keys.get(self.media_type, 'images')

    def handle_merge_device_map(self):
        if self.merge_device_map is None:
            self.merge_device_map = 'cpu'

    @staticmethod
    def _parse_lora_modules(lora_modules: List[str], use_vllm: bool) -> Tuple[List[Any], bool]:
        VllmLoRARequest = None
        if use_vllm:
            try:
                from .vllm_utils import LoRARequest as VllmLoRARequest
            except ImportError:
                logger.warning('The current version of VLLM does not support `enable_lora`. Please upgrade VLLM.')
                raise

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
            lora_local_path = swift_to_peft_format(lora_local_path)
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

    @staticmethod
    def check_ckpt_dir_correct(ckpt_dir) -> bool:
        """Check the checkpoint dir is correct, which means it must contain a `configuration.json` file.
        Args:
            ckpt_dir: The checkpoint dir
        Returns:
            A bool value represents the dir is valid or not.
        """
        if not os.path.exists(ckpt_dir):
            return False
        return os.path.isfile(os.path.join(ckpt_dir, 'configuration.json'))


@dataclass
class DeployArguments(InferArguments):
    host: str = '0.0.0.0'
    port: int = 8000
    api_key: Optional[str] = None
    ssl_keyfile: Optional[str] = None
    ssl_certfile: Optional[str] = None

    owned_by: str = 'swift'
    served_model_name: Optional[str] = None
    verbose: bool = True  # Whether to log request_info
    log_interval: int = 10  # Interval for printing global statistics

# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import os
import sys
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Set, Tuple, Union

import numpy as np
import torch
from datasets import Dataset as HfDataset
from datasets import IterableDataset as HfIterableDataset
from datasets import concatenate_datasets
from modelscope import HubApi
from modelscope.hub.api import ModelScopeConfig
from swift.llm.utils.model import dtype_mapping
from swift.utils.module_mapping import MODEL_KEYS_MAPPING
from transformers.utils import is_torch_bf16_gpu_available, is_torch_cuda_available, is_torch_npu_available, strtobool
from transformers.utils.versions import require_version

from swift.llm.model.config import ConfigReader
from swift.llm.model.loader import MODEL_MAPPING, safe_snapshot_download
from swift.utils import (get_logger, get_dist_setting)
from swift.utils.env import use_hf_hub

logger = get_logger()
DATASET_TYPE = Union[HfDataset, HfIterableDataset]

dtype_mapping_reversed = {v: k for k, v in dtype_mapping.items()}


class ArgumentsBase:

    sft_type: str = 'lora'

    @classmethod
    def _check_path(cls,
                    value: Union[str, List[str]],
                    k: Optional[str] = None,
                    check_exist_path_set: Optional[Set[str]] = None) -> Union[str, List[str]]:
        if check_exist_path_set is None:
            check_exist_path_set = set()
        if isinstance(value, str):
            value = os.path.expanduser(value)
            value = os.path.abspath(value)
            if k in check_exist_path_set and not os.path.exists(value):
                if k is not None:
                    raise FileNotFoundError(f"`{k}`: '{value}'")
                else:
                    raise FileNotFoundError(f"path: '{value}'")
        elif isinstance(value, list):
            res = []
            for v in value:
                res.append(cls._check_path(v, k, check_exist_path_set))
            value = res
        return value

    def handle_path(self: Union['SftArguments', 'InferArguments']) -> None:
        check_exist_path = ['ckpt_dir', 'resume_from_checkpoint', 'custom_register_path']
        maybe_check_exist_path = ['model_id_or_path', 'custom_dataset_info']
        if isinstance(self, SftArguments):
            check_exist_path.append('deepspeed_config_path')
            maybe_check_exist_path.append('deepspeed')

        for k in maybe_check_exist_path:
            v = getattr(self, k)
            if isinstance(v, str) and v is not None and (v.startswith('~') or v.startswith('/') or os.path.exists(v)):
                check_exist_path.append(k)
        check_exist_path_set = set(check_exist_path)
        other_path = ['output_dir', 'logging_dir']
        for k in check_exist_path + other_path:
            value = getattr(self, k, None)
            if value is None:
                continue
            value = self._check_path(value, k, check_exist_path_set)
            setattr(self, k, value)


@dataclass
class GenerationArguments:

    # generation config
    max_new_tokens: int = 2048
    do_sample: Optional[bool] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    num_beams: int = 1
    stop_words: List[str] = field(default_factory=list)

    def handle_do_sample(self, is_training, infer_backend) -> None:
        if self.temperature == 0:
            self.do_sample = False
        if self.do_sample is False and (is_training or (not is_training and infer_backend == 'pt')):
            # fix warning
            self.temperature = 1.
            self.top_p = 1.
            self.top_k = 50
            logger.info('Due to do_sample=False, the following settings are applied: args.temperature: '
                        f'{self.temperature}, args.top_p: {self.top_p}, args.top_k: {self.top_k}.')


@dataclass
class QuantizeArguments:
    # note: bf16 and quantization have requirements for gpu architecture
    # awq, gptq, and aqlm need to be pre-quantized models,
    # while bnb, hqq, and eetq can be quantized during SFT using the original models.
    quant_method: Literal['bnb', 'hqq', 'eetq', 'awq', 'gptq', 'aqlm'] = None
    quantization_bit: Literal[0, 1, 2, 3, 4, 8] = 0  # hqq: 1,2,3,4,8. bnb: 4,8
    hqq_axis: Literal[0, 1] = 0
    hqq_dynamic_config_path: Optional[str] = None
    bnb_4bit_comp_dtype: Literal['fp16', 'bf16', 'fp32', 'AUTO'] = 'AUTO'
    bnb_4bit_quant_type: Literal['fp4', 'nf4'] = 'nf4'
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_storage: Optional[str] = None

    def select_bnb(self, dtype) -> Tuple[Optional[torch.dtype], bool, bool]:
        if self.bnb_4bit_comp_dtype == 'AUTO':
            self.bnb_4bit_comp_dtype = dtype

        if self.bnb_4bit_comp_dtype != 'AUTO':
            bnb_4bit_compute_dtype = dtype_mapping_reversed[self.bnb_4bit_comp_dtype]
            assert bnb_4bit_compute_dtype in {torch.float16, torch.bfloat16, torch.float32}
        else:
            bnb_4bit_compute_dtype = None
        quantization_bit = self.quantization_bit
        if self.quant_method == 'bnb':
            if quantization_bit == 4:
                require_version('bitsandbytes')
                load_in_4bit, load_in_8bit = True, False
            elif quantization_bit == 8:
                require_version('bitsandbytes')
                load_in_4bit, load_in_8bit = False, True
            else:
                logger.warning('bnb only support 4/8 bits quantization, you should assign --quantization_bit 4 or 8,\
                    Or specify another quantization method; No quantization will be performed here.')
                load_in_4bit, load_in_8bit = False, False
        else:
            load_in_4bit, load_in_8bit = False, False

        return bnb_4bit_compute_dtype, load_in_4bit, load_in_8bit

    def is_quant_model(self, model_type, model_id_or_path, revision) -> bool:
        # Check if the model is gptq, awq, aqlm model. Do not check for other quantization situations such as bnb.
        if model_type is not None:
            for k in ['int4', 'int8', 'awq', 'aqlm']:
                if k in model_type:
                    return True

        bits = ConfigReader.read_config('quantization_config.bits', model_type, model_id_or_path, revision)
        if bits:
            return True


@dataclass
class ModelArguments:

    # You can specify the model by either using the model_type or model_id_or_path.
    model_type: Optional[str] = field(
        default=None, metadata={'help': f'model_type choices: {list(MODEL_MAPPING.keys())}'})
    model_id_or_path: Optional[str] = None
    ckpt_dir: Optional[str] = None
    resume_from_checkpoint: Optional[str] = None
    model_revision: Optional[str] = None

    model_kwargs: Optional[str] = None
    use_flash_attn: Optional[bool] = None
    # rope-scaling
    rope_scaling: Literal['linear', 'dynamic'] = None

    dtype: Literal['bf16', 'fp16', 'fp32', 'AUTO'] = 'AUTO'

    local_repo_path: Optional[str] = None

    device_map_config: Optional[str] = None
    device_max_memory: List[str] = field(default_factory=list)

    def _load_json_or_path(self, key) -> None:
        value = getattr(self, key)
        if isinstance(value, str):
            if os.path.exists(value):  # local path
                with open(value, 'r') as f:
                    value = json.load(f)
            else:  # json str
                value = json.loads(value)
        setattr(self, key, value)

    def prepare_model_extra_args(self):
        if self.model_kwargs is None:
            self.model_kwargs = {}
        self._load_json_or_path('model_kwargs')
        for k, v in self.model_kwargs.items():
            k = k.upper()
            os.environ[k] = str(v)

    def prepare_device_map_args(self):
        self._load_json_or_path('device_map_config')
        _, local_rank, _, local_world_size = get_dist_setting()
        # compat mp&ddp
        if local_world_size > 1 and isinstance(self.device_map_config, dict) and local_rank > 0:
            for k, v in self.device_map_config.items():
                if isinstance(v, int):
                    self.device_map_config[k] += local_rank

    def get_additional_saved_files(self, model_type: str) -> List[str]:
        files_mapping = {
            'qwen-vl': ['SimSun.ttf'],
            'qwen-audio': ['mel_filters.npz'],
            'yi-vl': ['vit'],
            'minicpm-v-v2_6-chat': ['modeling_navit_siglip.py']
        }
        for key, files_list in files_mapping.items():
            if key in model_type:
                return files_list
        return []

    def get_model_group(self):
        # This model_type is used to map the model structure
        model_type = self.model_type or self.model_id_or_path
        model_group = None
        for key in MODEL_KEYS_MAPPING.keys():
            if key in model_type.lower():
                model_group = key
                break
        return model_group

    def check_flash_attn(self) -> None:
        model_info = MODEL_MAPPING.get(self.model_type, {})
        support_flash_attn = model_info.get('support_flash_attn', False)
        if self.use_flash_attn and not support_flash_attn:
            logger.warning(f'use_flash_attn: {self.use_flash_attn}, ' f'but support_flash_attn: {support_flash_attn}')

    def is_multimodal(self) -> bool:
        if self.model_type is None:
            return False
        model_info = MODEL_MAPPING.get(self.model_type, {})
        tags = model_info.get('tags') or []
        return 'multi-modal' in tags

    def select_dtype(self, is_training, sft_type) -> Tuple[Optional[torch.dtype], bool, bool]:
        if not is_torch_cuda_available() and not is_torch_npu_available():
            # cpu
            if self.dtype == 'AUTO':
                self.dtype = 'fp32'
                logger.info(f'Setting args.dtype: {self.dtype}')
            assert self.dtype != 'fp16', 'The CPU does not support matrix multiplication with FP16.'
            if self.dtype == 'fp32':
                return torch.float32, False, False
            elif self.dtype == 'bf16':
                return torch.bfloat16, False, True
            else:
                raise ValueError(f'args.dtype: {self.dtype}')
        # cuda, npu
        if self.dtype == 'AUTO':
            if not is_torch_bf16_gpu_available():
                self.dtype = 'fp16'
            else:
                model_torch_dtype = MODEL_MAPPING[self.model_type].get('torch_dtype')
                if model_torch_dtype is not None:
                    self.dtype = dtype_mapping[model_torch_dtype]
                elif is_training:
                    self.dtype = 'bf16'
                else:
                    return None, False, False

        torch_dtype = dtype_mapping_reversed[self.dtype]

        assert torch_dtype in {torch.float16, torch.bfloat16, torch.float32}
        if torch_dtype == torch.float16:
            if is_training and sft_type == 'full':
                self.dtype = 'fp32'
                torch_dtype = torch.float32
                logger.warning(
                    'Fine-tuning with full parameters does not support fp16, and is prone to NaN. '
                    'We will use the fp32 & AMP approach, which consumes approximately twice the memory of bf16.')
                logger.info(f'Setting torch_dtype: {torch_dtype}')
            fp16, bf16 = True, False
        elif torch_dtype == torch.bfloat16:
            support_bf16 = is_torch_bf16_gpu_available()
            if not support_bf16:
                logger.warning(f'support_bf16: {support_bf16}')
            fp16, bf16 = False, True
        else:
            fp16, bf16 = False, False
        return torch_dtype, fp16, bf16

    def set_model_type(self, is_training) -> None:
        if self.model_id_or_path is not None:
            model_mapping_reversed = {}
            for k, v in MODEL_MAPPING.items():
                if use_hf_hub():
                    model_id = v.get('hf_model_id')
                else:
                    model_id = v.get('model_id_or_path')
                if model_id is None:
                    continue
                model_id = model_id.lower()
                model_mapping_reversed[model_id] = k
            model_id_or_path = self.model_id_or_path
            model_id_or_path_lower = model_id_or_path.lower()

            if self.model_type is None and model_id_or_path_lower in model_mapping_reversed:
                model_type = model_mapping_reversed[model_id_or_path_lower]
                assert self.model_type is None or self.model_type == model_type
                self.model_type = model_type
                logger.info(f'Setting args.model_type: {model_type}')
            else:
                if (not is_training and 'checkpoint-' in model_id_or_path
                        and 'merged' not in model_id_or_path and self.ckpt_dir is None):
                    raise ValueError('Please use `--ckpt_dir vx-xxx/checkpoint-xxx` to use the checkpoint.')

        model_info = MODEL_MAPPING.get(self.model_type, {})
        if self.model_revision is not None:
            model_info['revision'] = self.model_revision
            logger.info(f"Setting model_info['revision']: {self.model_revision}")
        elif use_hf_hub():
            model_info['revision'] = 'main'
        self.model_revision = model_info['revision']
        if self.model_id_or_path is None:
            self.model_id_or_path = model_info['hf_model_id'] if use_hf_hub() else model_info['model_id_or_path']
        requires = model_info.get('requires', [])
        for require in requires:
            require_version(require)

    def __post_init__(self):
        if self.rope_scaling:
            logger.info(f'rope_scaling is set to {self.rope_scaling}, please remember to set max_length')


def load_from_ckpt_dir(args, is_training, is_export_hf) -> None:
    if is_training:
        ckpt_dir = args.resume_from_checkpoint
    else:
        ckpt_dir = args.ckpt_dir
    sft_args_path = os.path.join(ckpt_dir, 'sft_args.json')
    export_args_path = os.path.join(ckpt_dir, 'export_args.json')
    from_sft_args = os.path.exists(sft_args_path)
    if not os.path.exists(sft_args_path) and not os.path.exists(export_args_path):
        logger.warning(f'{sft_args_path} not found')
        return
    args_path = sft_args_path if from_sft_args else export_args_path
    with open(args_path, 'r', encoding='utf-8') as f:
        old_args = json.load(f)

    imported_keys = [
        'model_type', 'model_revision', 'template_type', 'dtype', 'quant_method', 'quantization_bit',
        'bnb_4bit_comp_dtype', 'bnb_4bit_quant_type', 'bnb_4bit_use_double_quant', 'model_id_or_path',
        'custom_register_path', 'custom_dataset_info'
    ]
    if (is_training and args.train_backend == 'megatron') or is_export_hf:
        imported_keys += ['tp', 'pp']
    if not is_training:
        imported_keys += ['sft_type', 'rope_scaling', 'system']
        if getattr(args, 'load_dataset_config', False) and from_sft_args:
            imported_keys += [
                'dataset', 'val_dataset', 'dataset_seed', 'dataset_test_ratio', 'check_dataset_strategy',
                'self_cognition_sample', 'model_name', 'model_author', 'train_dataset_sample', 'val_dataset_sample'
            ]
    for key in imported_keys:
        if not hasattr(args, key):
            continue
        value = getattr(args, key)
        old_value = old_args.get(key)
        if old_value is None:
            continue
        if key in {'dataset', 'val_dataset'} and len(value) > 0:
            continue
        if key in {
                'system', 'quant_method', 'model_id_or_path', 'custom_register_path', 'custom_dataset_info',
                'dataset_seed'
        } and value is not None:
            continue
        if key in {'template_type', 'dtype'} and value != 'AUTO':
            continue
        setattr(args, key, old_value)

    if args.sft_type == 'full' or args.train_backend == 'megatron':
        args.model_id_or_path = args.resume_from_checkpoint


def prepare_ms_hub(args) -> None:
    hub_token = args.hub_token
    if hub_token is None:
        hub_token = os.environ.get('MODELSCOPE_API_TOKEN')
    if hub_token:
        api = HubApi()
        api.login(hub_token)
    if not hasattr(args, 'push_to_hub') or not args.push_to_hub:
        return
    args.hub_token = hub_token
    assert ModelScopeConfig.get_token() is not None, 'Please enter hub_token'
    if args.hub_model_id is None:
        args.hub_model_id = f'{args.model_type}-{args.sft_type}'
        logger.info(f'Setting hub_model_id: {args.hub_model_id}')
    logger.info('hub login successful!')

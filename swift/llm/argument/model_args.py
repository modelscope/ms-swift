# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple, Union

import json
import torch
from datasets import Dataset as HfDataset
from datasets import IterableDataset as HfIterableDataset
from transformers.utils import is_torch_bf16_gpu_available, is_torch_cuda_available, is_torch_npu_available
from transformers.utils.versions import require_version

from swift.llm.model.config import ConfigReader
from swift.llm.model.loader import MODEL_MAPPING
from swift.llm.model.model import dtype_mapping
from swift.utils import get_dist_setting, get_logger
from swift.utils.env import use_hf_hub
from swift.utils.module_mapping import MODEL_KEYS_MAPPING

logger = get_logger()
DATASET_TYPE = Union[HfDataset, HfIterableDataset]

dtype_mapping_reversed = {v: k for k, v in dtype_mapping.items()}


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

    def handle_do_sample(self) -> None:
        """Change the arguments because the training/pt infer/lmdeploy infer/vllm infer
        need different arguments when do_sample=False"""
        if self.temperature == 0:
            self.do_sample = False
        from swift.llm.argument import InferArguments, SftArguments
        if self.do_sample is False and (isinstance(self, SftArguments) or
                                        (isinstance(self, InferArguments) and self.infer_backend == 'pt')):
            # fix warning
            self.temperature = 1.
            self.top_p = 1.
            self.top_k = 50
            logger.info('Due to do_sample=False, the following settings are applied: args.temperature: '
                        f'{self.temperature}, args.top_p: {self.top_p}, args.top_k: {self.top_k}.')

    def __post_init__(self):
        self.handle_do_sample()


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

    def select_bnb(self) -> Tuple[Optional[torch.dtype], bool, bool]:
        """Find proper arguments when doing model quantization"""
        if self.quantization_bit > 0 and self.quant_method is None:
            if self.quantization_bit == 4 or self.quantization_bit == 8:
                logger.info('Since you have specified quantization_bit as greater than 0 '
                            "and have not designated a quant_method, quant_method will be set to 'bnb'.")
                self.quant_method = 'bnb'
            else:
                self.quant_method = 'hqq'
                logger.info('Since you have specified quantization_bit as greater than 0 '
                            "and have not designated a quant_method, quant_method will be set to 'hqq'.")

        if self.bnb_4bit_comp_dtype == 'AUTO':
            self.bnb_4bit_comp_dtype = self.dtype

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

        self.bnb_4bit_compute_dtype, self.load_in_4bit, self.load_in_8bit = (bnb_4bit_compute_dtype, load_in_4bit,
                                                                             load_in_8bit)

    def is_quant_model(self: Union['SftArguments', 'InferArguments']) -> bool:
        """Judge if the current model has already been a quantized model"""
        # Check if the model is gptq, awq, aqlm model. Do not check for other quantization situations such as bnb.
        if self.model_type is not None:
            for k in ['int4', 'int8', 'awq', 'aqlm']:
                if k in self.model_type:
                    return True

        model_path = self.model_id_or_path or self.resume_from_checkpoint or self.ckpt_dir
        bits = ConfigReader.read_config('quantization_config.bits', self.model_type, model_path, self.model_revision)
        if bits:
            return True

    def __post_init__(self: Union['SftArguments', 'InferArguments']):
        self.select_bnb()


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
        """Load content from json file set them into self attributes"""
        value = getattr(self, key)
        if isinstance(value, str):
            if os.path.exists(value):  # local path
                with open(value, 'r') as f:
                    value = json.load(f)
            else:  # json str
                value = json.loads(value)
        setattr(self, key, value)

    def prepare_model_extra_args(self):
        """Prepare model args and set them to the env"""
        if self.model_kwargs is None:
            self.model_kwargs = {}
        self._load_json_or_path('model_kwargs')
        for k, v in self.model_kwargs.items():
            k = k.upper()
            os.environ[k] = str(v)

    def prepare_device_map_args(self):
        """Prepare device map args"""
        self._load_json_or_path('device_map_config')
        _, local_rank, _, local_world_size = get_dist_setting()
        # compat mp&ddp
        if local_world_size > 1 and isinstance(self.device_map_config, dict) and local_rank > 0:
            for k, v in self.device_map_config.items():
                if isinstance(v, int):
                    self.device_map_config[k] += local_rank

    def get_additional_saved_files(self) -> List[str]:
        """Some models have extra files need to be saved, list them here"""
        files_mapping = {
            'qwen-vl': ['SimSun.ttf'],
            'qwen-audio': ['mel_filters.npz'],
            'yi-vl': ['vit'],
            'minicpm-v-v2_6-chat': ['modeling_navit_siglip.py']
        }
        for key, files_list in files_mapping.items():
            if key in self.model_type:
                return files_list
        return []

    def get_model_group(self):
        """Find the model group. This is used to find the model structure"""
        model_type = (self.model_type or self.model_id_or_path).replace('-', '_')
        model_group = None
        for key in MODEL_KEYS_MAPPING.keys():
            if key in model_type.lower():
                model_group = key
                break
        return model_group

    def check_flash_attn(self) -> None:
        """Some models do not support flash-attention"""
        model_info = MODEL_MAPPING.get(self.model_type, {})
        support_flash_attn = model_info.get('support_flash_attn', False)
        if self.use_flash_attn and not support_flash_attn:
            logger.warning(f'use_flash_attn: {self.use_flash_attn}, ' f'but support_flash_attn: {support_flash_attn}')

    @property
    def is_multimodal(self) -> bool:
        """Is multi modal models?"""
        if self.model_type is None:
            return False
        model_info = MODEL_MAPPING.get(self.model_type, {})
        tags = model_info.get('tags') or []
        return 'multi-modal' in tags

    def select_dtype(self) -> Tuple[Optional[torch.dtype], bool, bool]:
        """If dtype is `AUTO`, find a proper dtype by the sft_type/GPU"""
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
        from swift.llm.argument import SftArguments
        if self.dtype == 'AUTO':
            if not is_torch_bf16_gpu_available():
                self.dtype = 'fp16'
            else:
                model_torch_dtype = MODEL_MAPPING[self.model_type].get('torch_dtype')
                if model_torch_dtype is not None:
                    self.dtype = dtype_mapping[model_torch_dtype]
                elif isinstance(self, SftArguments):
                    self.dtype = 'bf16'
                else:
                    return None, False, False

        torch_dtype = dtype_mapping_reversed[self.dtype]

        assert torch_dtype in {torch.float16, torch.bfloat16, torch.float32}
        if torch_dtype == torch.float16:
            if isinstance(self, SftArguments) and self.sft_type == 'full':
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
        self.torch_dtype, self.fp16, self.bf16 = torch_dtype, fp16, bf16

    def select_model_type(self) -> None:
        """model_type may be None, find the right one by `model_id_or_path`"""
        from swift.llm.argument import InferArguments
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
                if (isinstance(self, InferArguments) and 'checkpoint-' in model_id_or_path
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

    def __post_init__(self: Union['SftArguments', 'InferArguments']):
        if self.rope_scaling:
            logger.info(f'rope_scaling is set to {self.rope_scaling}, please remember to set max_length')
        self.prepare_model_extra_args()
        self.prepare_device_map_args()
        self.check_flash_attn()
        self.select_dtype()
        self.select_model_type()

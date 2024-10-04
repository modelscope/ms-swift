# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union

import torch
from transformers.utils import is_torch_bf16_gpu_available, is_torch_cuda_available, is_torch_npu_available
from transformers.utils.versions import require_version

from swift.llm import MODEL_KEYS_MAPPING, MODEL_MAPPING, ConfigReader
from swift.utils import get_dist_setting, get_logger, use_hf_hub

logger = get_logger()

dtype_mapping = {torch.float16: 'fp16', torch.bfloat16: 'bf16', torch.float32: 'fp32', None: 'auto'}
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
        from swift.llm import InferArguments, SftArguments
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
    quant_method: Literal['bnb', 'hqq', 'eetq', 'awq', 'gptq', 'aqlm'] = 'bnb'
    quantization_bit: Literal[0, 1, 2, 3, 4, 8] = 0  # bnb: 4,8, hqq: 1,2,3,4,8.
    # hqq
    hqq_axis: Literal[0, 1] = 0
    hqq_dynamic_config_path: Optional[str] = None
    # bnb
    bnb_4bit_comp_dtype: Literal['fp16', 'bf16', 'fp32', 'auto'] = 'auto'
    bnb_4bit_quant_type: Literal['fp4', 'nf4'] = 'nf4'
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_storage: Optional[str] = None

    def select_bnb(self) -> None:
        """Find proper arguments when doing model quantization"""
        if self.bnb_4bit_comp_dtype == 'auto':
            self.bnb_4bit_comp_dtype = self.dtype

        bnb_4bit_compute_dtype = dtype_mapping_reversed[self.bnb_4bit_comp_dtype]
        assert bnb_4bit_compute_dtype in {torch.float16, torch.bfloat16, torch.float32}

        load_in_4bit, load_in_8bit = False, False  # default value
        if self.quant_method == 'bnb':
            if self.quantization_bit == 4:
                require_version('bitsandbytes')
                load_in_4bit, load_in_8bit = True, False
            elif self.quantization_bit == 8:
                require_version('bitsandbytes')
                load_in_4bit, load_in_8bit = False, True
            else:
                logger.warning('bnb only support 4/8 bits quantization, you should assign --quantization_bit 4 or 8, '
                               'Or specify another quantization method; No quantization will be performed here.')
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self.load_in_4bit, self.load_in_8bit = load_in_4bit, load_in_8bit

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
    model_revision: Optional[str] = None

    dtype: Literal['bf16', 'fp16', 'fp32', 'auto'] = 'auto'
    model_kwargs: Optional[str] = None
    # flash_attn: It will automatically convert names based on the model.
    # auto: It will be automatically selected between sdpa and eager.
    attn_impl: Literal['flash_attn', 'sdpa', 'eager', 'auto'] = 'auto'
    # rope-scaling
    rope_scaling: Literal['linear', 'dynamic'] = None

    local_repo_path: Optional[str] = None

    device_map_config: Optional[str] = None
    device_max_memory: List[str] = field(default_factory=list)

    def prepare_model_extra_args(self: 'SftArguments'):
        """Prepare model kwargs and set them to the env"""
        self.parse_to_dict(self, 'model_kwargs')
        for k, v in self.model_kwargs.items():
            k = k.upper()
            os.environ[k] = str(v)

    def prepare_device_map_args(self: 'SftArguments'):
        """Prepare device map args"""
        self.parse_to_dict('device_map_config')
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

    def select_dtype(self) -> None:
        """If dtype is `auto`, find a proper dtype by the sft_type/GPU"""
        # Compatible with --fp16/--bf16
        from .train_args import SftArguments
        if isinstance(self, SftArguments):
            for key in ['fp16', 'bf16']:
                value = getattr(self, key)
                if value:
                    assert self.dtype == 'auto'
                    self.dtype = key
                    break
        # handle dtype == 'auto'
        if self.dtype == 'auto':
            if is_torch_cuda_available() or is_torch_npu_available():
                if is_torch_bf16_gpu_available():
                    model_torch_dtype = MODEL_MAPPING[self.model_type].get('torch_dtype')
                    if model_torch_dtype is not None:
                        self.dtype = dtype_mapping[model_torch_dtype]
                    elif isinstance(self, SftArguments):
                        self.dtype = 'bf16'
                    # else: Keep 'auto'. According to the model's config.json file,
                    # this behavior is executed in the get_model_tokenizer function.
                    # This situation will only occur during inference.
                else:
                    self.dtype = 'fp16'
            else:
                # cpu
                self.dtype = 'fp32'
            logger.info(f'Setting args.dtype: {self.dtype}')
        # Check the validity of dtype
        if is_torch_cuda_available() or is_torch_npu_available():
            if self.dtype == 'fp16':
                if isinstance(self, SftArguments) and self.sft_type == 'full':
                    self.dtype = 'fp32'
                    logger.warning(
                        'Fine-tuning with full parameters does not support fp16, and is prone to NaN. '
                        'We will use the fp32 & AMP approach, which consumes approximately twice the memory of bf16. '
                        f'Setting args.dtype: {self.dtype}')
            elif self.dtype == 'bf16':
                support_bf16 = is_torch_bf16_gpu_available()
                if not support_bf16:
                    logger.warning(f'support_bf16: {support_bf16}')
        else:
            # cpu
            assert self.dtype != 'fp16', 'The CPU does not support matrix multiplication with FP16.'

        self.torch_dtype = dtype_mapping_reversed[self.dtype]
        # Mixed Precision Training
        if isinstance(self, SftArguments):
            if self.dtype in {'fp16', 'fp32'}:
                self.fp16, self.bf16 = True, False
            elif self.dtype == 'bf16':
                self.fp16, self.bf16 = False, True
            else:
                raise ValueError(f'args.dtype: {self.dtype}')

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

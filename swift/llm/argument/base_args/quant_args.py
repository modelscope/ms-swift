# Copyright (c) Alibaba, Inc. and its affiliates.

from dataclasses import dataclass
from typing import Literal, Optional

import torch

from swift.llm import HfConfigFactory
from swift.utils import get_modules_to_not_convert


@dataclass
class QuantizeArguments:
    """A dataclass that holds the configuration for model quantization.

    Args:
        quant_method (Optional[str]): The quantization method to use when loading the model. Can be one of {'bnb',
            'hqq', 'eetq', 'quanto', 'fp8'}. Note: This is not required for QLoRA training on pre-quantized AWQ/GPTQ
            models. Defaults to None.
        quant_bits (Optional[Union[int, str]]): The number of bits for quantization, e.g., {1, 2, 3, 4, 8, 'float8'}.
            Defaults to None.
        hqq_axis (Optional[int]): The quantization axis for HQQ quantization. Defaults to None.
        bnb_4bit_compute_dtype (Optional[str]): The compute data type for 4-bit BNB quantization. Can be one of {
            'float16', 'bfloat16', 'float32'}. Defaults to None, which will use the model's `torch_dtype`.
        bnb_4bit_quant_type (str): The quantization type for 4-bit BNB quantization. Can be one of {'fp4', 'nf4'}.
            Defaults to 'nf4'.
        bnb_4bit_use_double_quant (bool): Whether to use double quantization for 4-bit BNB quantization.
            Defaults to True.
        bnb_4bit_quant_storage (Optional[str]): The storage type for packing quantized 4-bit parameters in BNB.
            Defaults to None.
    """
    # awq, gptq, and aqlm need to be pre-quantized models.
    #   It can be detected automatically, without the need to pass in.
    # while bnb, hqq, and eetq can be quantized during SFT using the original models.
    quant_method: Literal['bnb', 'hqq', 'eetq', 'quanto', 'fp8'] = None
    # bnb: 4,8; hqq: 1,2,3,4,8'; eetq: 8
    # awq: 4; gptq: 2,3,4,8
    quant_bits: Literal[1, 2, 3, 4, 8, 'float8'] = None
    # hqq
    hqq_axis: Optional[int] = None
    # bnb
    bnb_4bit_compute_dtype: Literal['float16', 'bfloat16', 'float32', None] = None
    bnb_4bit_quant_type: Literal['fp4', 'nf4'] = 'nf4'
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_storage: Optional[str] = None

    def get_quantization_config(self):
        if self.quant_method is None or self.quant_method in {'awq', 'gptq', 'gptq_v2'}:
            return None
        assert self.quant_method in {'bnb', 'hqq', 'eetq', 'quanto', 'fp8'}
        if self.quant_method != 'fp8' and self.quant_bits is None:
            raise ValueError(f'Please set the quant_bits. args.quant_bits: {self.quant_bits}')
        if self.quant_method == 'bnb':
            if self.quant_bits == 4:
                load_in_4bit, load_in_8bit = True, False
            elif self.quant_bits == 8:
                load_in_4bit, load_in_8bit = False, True
            else:
                raise ValueError(f'bnb not support quant_bits: {self.quant_bits}')

            from transformers import BitsAndBytesConfig
            llm_int8_skip_modules = self.get_modules_to_not_convert()
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                bnb_4bit_compute_dtype=self.bnb_4bit_compute_dtype,
                bnb_4bit_quant_type=self.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
                bnb_4bit_quant_storage=self.bnb_4bit_quant_storage,
                llm_int8_skip_modules=llm_int8_skip_modules)
        elif self.quant_method == 'fp8':
            if not hasattr(self, 'model_info'):
                return
            from transformers import FineGrainedFP8Config
            from swift.llm import get_model_tokenizer
            with torch.device('meta'):
                hf_model, _ = get_model_tokenizer(self.model_dir, model_type=self.model_type, return_dummy_model=True)
            modules_to_not_convert = get_modules_to_not_convert(hf_model)
            quantization_config = FineGrainedFP8Config(modules_to_not_convert=modules_to_not_convert)
        elif self.quant_method == 'hqq':
            from transformers import HqqConfig
            quantization_config = HqqConfig(nbits=self.quant_bits, axis=self.hqq_axis)
        elif self.quant_method == 'quanto':
            from transformers import QuantoConfig
            if self.quant_bits == 8:
                weights = 'int8'
            elif self.quant_bits == 'float8':
                weights = 'float8'
            elif self.quant_bits == 4:
                weights = 'int4'
            elif self.quant_bits == 2:
                weights = 'int2'
            else:
                raise ValueError('quanto quantization only support quant bits 2/4/8/float8')
            quantization_config = QuantoConfig(weights=weights)
        else:  # 'eetq'
            from transformers import EetqConfig
            quantization_config = EetqConfig(f'int{self.quant_bits}')

        return quantization_config

    def get_modules_to_not_convert(self):
        if not hasattr(self, 'model_meta') or not hasattr(self, 'model_info'):
            return None
        model_arch = self.model_meta.model_arch
        res = []
        if self.model_info.is_moe_model:
            res += ['mlp.gate', 'mlp.shared_expert_gate']
        if model_arch is not None:
            for key in ['vision_tower', 'aligner']:
                value = getattr(model_arch, key, None)
                if value:
                    res += value
        if not res:
            return None
        res.append('lm_head')
        return res

    def __post_init__(self):
        if self.bnb_4bit_compute_dtype is None:
            if self.torch_dtype in {torch.float16, torch.float32}:
                self.bnb_4bit_compute_dtype = torch.float32
            elif self.torch_dtype == torch.bfloat16:
                self.bnb_4bit_compute_dtype = torch.bfloat16
        self.bnb_4bit_compute_dtype: torch.dtype = HfConfigFactory.to_torch_dtype(self.bnb_4bit_compute_dtype)

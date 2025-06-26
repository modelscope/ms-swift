# Copyright (c) Alibaba, Inc. and its affiliates.

from dataclasses import dataclass
from typing import Literal, Optional

import torch

from swift.llm import HfConfigFactory


@dataclass
class QuantizeArguments:
    """
    QuantizeArguments is a dataclass that holds the configuration for model quantization.

    Args:
        quant_method (Literal['bnb', 'hqq', 'eetq', 'quanto', 'fp8']): The quantization method to be used.
        quant_bits (Literal[1, 2, 3, 4, 8]): The number of bits to use for quantization.
        hqq_axis (Optional[int]): The axis for hqq quantization.
        bnb_4bit_compute_dtype (Literal['float16', 'bfloat16', 'float32', None]):
            The compute dtype for bnb 4-bit quantization.
        bnb_4bit_quant_type (Literal['fp4', 'nf4']): The quantization type for bnb 4-bit quantization.
        bnb_4bit_use_double_quant (bool): Whether to use double quantization for bnb 4-bit quantization.
        bnb_4bit_quant_storage (Optional[str]): This sets the storage type to pack the quantized 4-bit params.
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
        if self.quant_method is None or self.quant_method in {'awq', 'gptq'}:
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
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                bnb_4bit_compute_dtype=self.bnb_4bit_compute_dtype,
                bnb_4bit_quant_type=self.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
                bnb_4bit_quant_storage=self.bnb_4bit_quant_storage)
        elif self.quant_method == 'fp8':
            from transformers import FineGrainedFP8Config
            quantization_config = FineGrainedFP8Config()
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

    def __post_init__(self):
        if self.bnb_4bit_compute_dtype is None:
            if self.torch_dtype in {torch.float16, torch.float32}:
                self.bnb_4bit_compute_dtype = torch.float32
            elif self.torch_dtype == torch.bfloat16:
                self.bnb_4bit_compute_dtype = torch.bfloat16
        self.bnb_4bit_compute_dtype: torch.dtype = HfConfigFactory.to_torch_dtype(self.bnb_4bit_compute_dtype)

# Copyright (c) ModelScope Contributors. All rights reserved.
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class QuantizeConfig:
    """Quantization backend, bit-width, and BNB/HQQ options."""

    quant_method: Literal['awq', 'gptq', 'gptq_v2', 'bnb', 'hqq', 'eetq', 'quanto', 'fp8', None] = None
    quant_bits: Literal[1, 2, 3, 4, 8, 'float8', None] = None
    hqq_axis: Optional[int] = None
    bnb_4bit_compute_dtype: Literal['float16', 'bfloat16', 'float32', None] = None
    bnb_4bit_quant_type: Literal['fp4', 'nf4'] = 'nf4'
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_storage: Optional[str] = None

    # Calibration-based methods only; bnb/hqq quantise from the weights alone and ignore these.
    quant_n_samples: int = 256  # calibration rows drawn from the dataset
    quant_batch_size: int = 1  # calibration rows per forward pass
    # Weights sharing one scale; smaller is more accurate and larger on disk, must divide the
    # layer's input dimension.
    group_size: int = 128

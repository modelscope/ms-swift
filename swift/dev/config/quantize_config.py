"""Model quantization method configuration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


# TODO: integrate it
@dataclass
class QuantizeConfig:
    """Quantization backend, bit-width, and BNB/HQQ options."""

    quant_method: Literal['bnb', 'hqq', 'eetq', 'quanto', 'fp8', None] = None
    quant_bits: Literal[1, 2, 3, 4, 8, 'float8', None] = None
    hqq_axis: Optional[int] = None
    bnb_4bit_compute_dtype: Literal['float16', 'bfloat16', 'float32', None] = None
    bnb_4bit_quant_type: Literal['fp4', 'nf4'] = 'nf4'
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_storage: Optional[str] = None

    # === Calibration ===
    # Only the calibration-based methods read these. bnb and hqq quantise from the weights alone and
    # ignore them entirely, which is why they are grouped apart rather than mixed in above.
    #: Calibration rows drawn from the dataset. More gives a better-conditioned result and costs time.
    quant_n_samples: int = 256
    #: Calibration rows per forward pass. Raised only to go faster, at proportional peak memory.
    quant_batch_size: int = 1
    #: Weights sharing one scale. Smaller is more accurate and larger on disk; 128 is the usual
    #: compromise, and the value must divide the layer's input dimension.
    group_size: int = 128

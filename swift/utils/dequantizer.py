import math
import torch
from typing import Tuple


class Fp8Dequantizer:

    def __init__(self, block_size: Tuple[int, int] = (128, 128)):
        self.block_size = block_size

    def convert(
        self,
        quantized: torch.Tensor,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(quantized, torch.Tensor) or not isinstance(scales, torch.Tensor):
            raise TypeError('Fp8Dequantize expects tensors as inputs.')
        if quantized.dtype == torch.uint8:
            quantized = quantized.view(torch.float8_e4m3fn)
        quantized_fp32 = quantized.to(torch.float32)
        rows, cols = quantized_fp32.shape[-2:]
        block_size = self.block_size
        block_m, block_n = block_size
        if rows % block_m != 0 or cols % block_n != 0:
            # Fall back to loop
            dequantized = quantized_fp32.clone()
            for i in range(math.ceil(rows / block_m)):
                for j in range(math.ceil(cols / block_n)):
                    dequantized[i * block_m:(i + 1) * block_m, j * block_n:(j + 1) * block_n] *= scales[i, j]
        else:
            reshaped = quantized_fp32.reshape(-1, rows // block_m, block_m, cols // block_n, block_n)
            expanded_scales = scales.to(torch.float32).reshape(-1, rows // block_m, cols // block_n)
            expanded_scales = expanded_scales.unsqueeze(-1).unsqueeze(2)
            dequantized = reshaped * expanded_scales
            dequantized = dequantized.reshape(quantized_fp32.shape)  # return torch.float32
        return dequantized


class MxFp4Dequantizer:

    def convert(
        self,
        blocks: torch.Tensor,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        from transformers.integrations import convert_moe_packed_tensors
        return convert_moe_packed_tensors(blocks, scales)

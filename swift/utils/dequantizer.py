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
        needs_padding = rows % block_m != 0 or cols % block_n != 0

        input_tensor = quantized_fp32
        if needs_padding:
            pad_rows = (block_m - rows % block_m) % block_m
            pad_cols = (block_n - cols % block_n) % block_n
            input_tensor = torch.nn.functional.pad(input_tensor, (0, pad_cols, 0, pad_rows))

        p_rows, p_cols = input_tensor.shape[-2:]

        reshaped = input_tensor.reshape(-1, p_rows // block_m, block_m, p_cols // block_n, block_n)
        expanded_scales = scales.to(torch.float32).reshape(-1, p_rows // block_m, p_cols // block_n)
        expanded_scales = expanded_scales.unsqueeze(-1).unsqueeze(2)

        dequantized = reshaped * expanded_scales
        dequantized = dequantized.reshape(input_tensor.shape)

        if needs_padding:
            dequantized = dequantized[..., :rows, :cols].contiguous()

        return dequantized


class MxFp4Dequantizer:

    def convert(
        self,
        blocks: torch.Tensor,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        from transformers.integrations import convert_moe_packed_tensors
        return convert_moe_packed_tensors(blocks, scales)

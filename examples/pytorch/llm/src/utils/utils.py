import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from torch import dtype as Dtype
from torch.nn import Module
from transformers import GenerationConfig, TextStreamer

from swift import get_logger
from swift.utils.tb_utils import (TB_COLOR, TB_COLOR_SMOOTH,
                                  read_tensorboard_file, tensorboard_smoothing)

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
logger = get_logger()

DEFAULT_PROMPT = """Here's a conversation between a human and an AI assistant. \
The AI assistant provides detailed, friendly answers for the human.

### Human:
{instruction}

### AI:
"""

DTYPE_MAPPING = {
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
    'fp32': torch.float32
}


def get_dist_setting() -> Tuple[int, int, int, int]:
    """return rank, local_rank, world_size"""
    rank = int(os.getenv('RANK', -1))
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    local_world_size = int(os.getenv('LOCAL_WORLD_SIZE', 1))
    return rank, local_rank, world_size, local_world_size


def is_dist():
    rank = int(os.getenv('RANK', -1))
    return rank >= 0


def show_layers(model: Module, max_lines: Optional[int] = 20) -> None:
    named_p = list(model.named_parameters())
    for i, (n, p) in enumerate(named_p):
        if max_lines is not None and i >= max_lines:
            logger.info('...')
            break
        logger.info(
            f'[{n}]: requires_grad={p.requires_grad}, dtype={p.dtype}, device={p.device}'
        )


def plot_images(images_dir: str,
                tb_dir: str,
                smooth_key: List[str],
                smooth_val: float = 0.9,
                figsize: Tuple[int, int] = (8, 5),
                dpi: int = 100) -> None:
    os.makedirs(images_dir, exist_ok=True)
    fname = [
        fname for fname in os.listdir(tb_dir)
        if os.path.isfile(os.path.join(tb_dir, fname))
    ][0]
    tb_path = os.path.join(tb_dir, fname)
    data = read_tensorboard_file(tb_path)

    for k in data.keys():
        _data = data[k]
        steps = [d['step'] for d in _data]
        values = [d['value'] for d in _data]
        if len(values) == 0:
            continue
        _, ax = plt.subplots(1, 1, squeeze=True, figsize=figsize, dpi=dpi)
        ax.set_title(k)
        if len(values) == 1:
            ax.scatter(steps, values, color=TB_COLOR_SMOOTH)
        elif k in smooth_key:
            ax.plot(steps, values, color=TB_COLOR)
            values_s = tensorboard_smoothing(values, smooth_val)
            ax.plot(steps, values_s, color=TB_COLOR_SMOOTH)
        else:
            ax.plot(steps, values, color=TB_COLOR_SMOOTH)
        fpath = os.path.join(images_dir, k.replace('/', '_'))
        plt.savefig(fpath, dpi=dpi, bbox_inches='tight')


def inference(input_ids: List[int],
              model,
              tokenizer,
              streamer: Optional[TextStreamer] = None,
              generation_config: Optional[GenerationConfig] = None,
              tag: str = '[INFERENCE]') -> str:
    print(f'{tag}{tokenizer.decode(input_ids)}', end='')
    input_ids = torch.tensor(input_ids)[None].cuda()
    attention_mask = torch.ones_like(input_ids)
    generate_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        streamer=streamer,
        generation_config=generation_config)
    output_text = tokenizer.decode(generate_ids[0])
    return output_text


def select_dtype(dtype: str) -> Tuple[Dtype, bool, bool]:
    """
    dtype: Literal['fp16', 'bf16', 'fp32']
    """
    torch_dtype = DTYPE_MAPPING[dtype]

    assert torch_dtype in {torch.float16, torch.bfloat16, torch.float32}
    if torch_dtype == torch.float16:
        fp16, bf16 = True, False
    elif torch_dtype == torch.bfloat16:
        support_bf16 = torch.cuda.is_bf16_supported()
        if not support_bf16:
            logger.warning(f'support_bf16: {support_bf16}')
        fp16, bf16 = False, True
    else:
        fp16, bf16 = False, False
    return torch_dtype, fp16, bf16


def select_bnb(quantization_bit: Optional[int],
               bnb_4bit_compute_dtype: str) -> Tuple[Dtype, bool, bool]:
    bnb_4bit_compute_dtype = DTYPE_MAPPING[bnb_4bit_compute_dtype]
    assert bnb_4bit_compute_dtype in {
        torch.float16, torch.bfloat16, torch.float32
    }
    if quantization_bit == 4:
        load_in_4bit, load_in_8bit = True, False
    elif quantization_bit == 8:
        load_in_4bit, load_in_8bit = False, True
    else:
        load_in_4bit, load_in_8bit = False, False

    return bnb_4bit_compute_dtype, load_in_4bit, load_in_8bit


def broadcast_string(string: Optional[str], buffer_size: int = 100) -> str:
    """
    string: main rank: str
        other rank: None
    return: all rank: str
    """
    assert dist.is_initialized()
    rank, local_rank, _, _ = get_dist_setting()
    assert rank >= 0
    if rank == 0:
        assert string is not None
        tensor = torch.tensor(
            [ord(c) for c in string] + [0] * (buffer_size - len(string)),
            dtype=torch.int64,
            device=local_rank)
    else:
        tensor = torch.zeros(buffer_size, dtype=torch.int64, device=local_rank)
    dist.broadcast(tensor, 0)
    first_zero = (tensor == 0).nonzero()[0].item()
    res = tensor.tolist()[:first_zero]
    return ''.join([chr(x) for x in res])

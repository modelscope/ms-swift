import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from torch import dtype as Dtype
from transformers import GenerationConfig, TextStreamer

from swift import get_logger
from swift.utils.tb_utils import (TB_COLOR, TB_COLOR_SMOOTH,
                                  read_tensorboard_file, tensorboard_smoothing)
from .dataset import DATASET_MAPPING, get_dataset, process_dataset
from .models import MODEL_MAPPING, get_model_tokenizer

logger = get_logger()
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

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
            ax.plot(steps, values, color=TB_COLOR)
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


def select_dtype(dtype: str, model_type: str) -> Tuple[Dtype, bool, bool]:
    """
    dtype: Literal['fp16', 'bf16', 'fp32']
    """
    if dtype is None:
        torch_dtype = MODEL_MAPPING[model_type].get('torch_dtype',
                                                    torch.float16)
    else:
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

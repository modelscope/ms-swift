import base64
import hashlib
import os
import re
from copy import deepcopy
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Union

from torch.nn import Module
from transformers import GenerationConfig

from swift.utils import upper_bound

History = List[Union[Tuple[str, str], List[str]]]
Messages = List[Dict[str, Union[str, List[Dict]]]]


from typing import Any, Mapping, Sequence

import torch


def _decode_prompt(prompt: str, tmp_dir: str = 'tmp') -> str:
    pattern = r'<(?:img|audio|video)>(.+?)</(?:img|audio|video)>'
    match_iter = re.finditer(pattern, prompt)
    new_content = ''
    idx = 0
    for m in match_iter:
        span = m.span(1)
        img_base64 = m.group(1)
        img_path = _from_base64(img_base64, tmp_dir)
        new_content += prompt[idx:span[0]] + img_path
        idx = span[1]
    new_content += prompt[idx:]
    return new_content


def _to_base64(img_path: Union[str, 'PIL.Image.Image', bytes]) -> str:
    if isinstance(img_path, str) and not os.path.isfile(img_path):
        # base64
        return img_path
    if isinstance(img_path, str):
        # local_path
        with open(img_path, 'rb') as f:
            _bytes = f.read()
    elif not isinstance(img_path, bytes):  # PIL.Image.Image
        bytes_io = BytesIO()
        img_path.save(bytes_io, format='png')
        _bytes = bytes_io.getvalue()
    else:
        _bytes = img_path
    img_base64: str = base64.b64encode(_bytes).decode('utf-8')
    return img_base64


def _from_base64(img_base64: Union[str, 'PIL.Image.Image'], tmp_dir: str = 'tmp') -> str:
    from PIL import Image
    if not isinstance(img_base64, str):  # PIL.Image.Image
        img_base64 = _to_base64(img_base64)
    if os.path.isfile(img_base64) or img_base64.startswith('http'):
        return img_base64
    sha256_hash = hashlib.sha256(img_base64.encode('utf-8')).hexdigest()
    img_path = os.path.join(tmp_dir, f'{sha256_hash}.png')
    image = Image.open(BytesIO(base64.b64decode(img_base64)))
    if not os.path.exists(img_path):
        image.save(img_path)
    return img_path


def decode_base64(*,
                  messages: Optional[Messages] = None,
                  prompt: Optional[str] = None,
                  images: Optional[List[str]] = None,
                  tmp_dir: str = 'tmp') -> Dict[str, Any]:
    # base64 -> local_path
    os.makedirs(tmp_dir, exist_ok=True)
    res = {}
    if messages is not None:
        res_messages = []
        for m in messages:
            m_new = deepcopy(m)
            m_new['content'] = _decode_prompt(m_new['content'], tmp_dir)
            res_messages.append(m_new)
        res['messages'] = res_messages
    if prompt is not None:
        prompt = _decode_prompt(prompt, tmp_dir)
        res['prompt'] = prompt
    if images is not None:
        res_images = []
        for image in images:
            image = _from_base64(image, tmp_dir)
            res_images.append(image)
        res['images'] = res_images
    return res


def to_device(inputs: Any, device: torch.device) -> Any:
    """Move inputs to a device"""
    if callable(getattr(inputs, 'to', None)):
        return inputs.to(device=device)

    if isinstance(inputs, Mapping):
        res = {}
        for k, v in inputs.items():
            res[k] = to_device(v, device)
    elif isinstance(inputs, Sequence) and not isinstance(inputs, str):
        res = []
        for b in inputs:
            res.append(to_device(b, device))
    else:
        res = inputs
    return res


def limit_history_length(template: 'Template', query: str, history: Optional[History],
                         max_length: Optional[int]) -> Tuple[History, History]:
    """binary search"""
    if history is None:
        history = []
    if max_length is None:
        return [], history

    def compute_token_length(history_length: int) -> int:
        assert history_length != 0
        example = {'query': query, 'history': history[-history_length:]}
        input_ids = template.encode(example)[0]['input_ids']
        return len(input_ids)

    history_length = upper_bound(0, len(history), lambda mid: compute_token_length(mid) <= max_length)
    old_history = history[:len(history) - history_length]
    history = history[len(history) - history_length:]
    return old_history, history


Messages = List[Dict[str, Union[str, List[Dict]]]]


def set_generation_config(model: Module, generation_config: GenerationConfig) -> None:
    old_generation_config = getattr(model, 'generation_config', None)
    old_generation_priority_config = ['no_repeat_ngram_size', 'num_beams']
    if old_generation_config is not None:
        for k, old_v in old_generation_config.__dict__.items():
            if k.startswith('_'):
                continue
            v = getattr(generation_config, k, None)
            if k in old_generation_priority_config or old_v is not None and v is None:
                setattr(generation_config, k, old_v)
    model.generation_config = generation_config


def calculate_max_steps(args):
    pass

import base64
import hashlib
import os
import re
from copy import deepcopy
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

from .utils import Messages


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


def _encode_prompt(prompt: str) -> str:
    pattern = r'<(?:img|audio|video)>(.+?)</(?:img|audio|video)>'
    match_iter = re.finditer(pattern, prompt)
    new_prompt = ''
    idx = 0
    for m in match_iter:
        span = m.span(1)
        path = m.group(1)
        img_base64 = _to_base64(path)
        new_prompt += prompt[idx:span[0]] + img_base64
        idx = span[1]
    new_prompt += prompt[idx:]
    return new_prompt


def convert_to_base64(*,
                      messages: Optional[Messages] = None,
                      prompt: Optional[str] = None,
                      images: Optional[List[str]] = None) -> Dict[str, Any]:
    """local_path -> base64"""
    res = {}
    if messages is not None:
        res_messages = []
        for m in messages:
            m_new = deepcopy(m)
            m_new['content'] = _encode_prompt(m_new['content'])
            res_messages.append(m_new)
        res['messages'] = res_messages
    if prompt is not None:
        prompt = _encode_prompt(prompt)
        res['prompt'] = prompt
    if images is not None:
        res_images = []
        for image in images:
            res_images.append(_to_base64(image))
        res['images'] = res_images
    return res

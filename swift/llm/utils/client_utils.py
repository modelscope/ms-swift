import base64
import hashlib
import os
import re
from copy import deepcopy
from io import BytesIO
from typing import Any, Dict, Iterator, List, Optional, Union

import json
import requests
from dacite import from_dict
from requests.exceptions import HTTPError

from .model import get_default_template_type
from .protocol import (ChatCompletionResponse, ChatCompletionStreamResponse, CompletionResponse,
                       CompletionStreamResponse, ModelList, XRequestConfig)
from .template import History
from .utils import Messages, history_to_messages


def get_model_list_client(host: str = '127.0.0.1', port: str = '8000') -> ModelList:
    url = f'http://{host}:{port}/v1/models'
    resp_obj = requests.get(url).json()
    return from_dict(ModelList, resp_obj)


def _parse_stream_data(data: bytes) -> Optional[str]:
    data = data.decode(encoding='utf-8')
    data = data.strip()
    if len(data) == 0:
        return
    assert data.startswith('data:')
    return data[5:].strip()


def convert_to_base64(*,
                      messages: Optional[Messages] = None,
                      prompt: Optional[str] = None,
                      images: Optional[List[str]] = None) -> Dict[str, Any]:
    """local_path -> base64"""
    res = {}
    if messages is not None:
        pattern = r'<(?:img|audio)>(.+?)</(?:img|audio)>'
        res_messages = []
        for m in messages:
            m_new = deepcopy(m)
            content = m_new['content']
            match_iter = re.finditer(pattern, content)
            new_content = ''
            idx = 0
            for m in match_iter:
                span = m.span(1)
                path = m.group(1)
                if not os.path.isfile(path):
                    continue
                with open(path, 'rb') as f:
                    img_base64 = base64.b64encode(f.read())
                new_content += content[idx:span[0]] + img_base64.decode('utf-8')
                idx = span[1]
            new_content += content[idx:]
            m_new['content'] = new_content
            res_messages.append(m_new)
        res['messages'] = res_messages
    return res


def decode_base64(*,
                  messages: Optional[Messages] = None,
                  prompt: Optional[str] = None,
                  images: Optional[List[str]] = None,
                  tmp_dir: str = 'tmp') -> Dict[str, Any]:
    from PIL import Image
    os.makedirs(tmp_dir, exist_ok=True)
    res = {}
    if messages is not None:
        pattern = r'<(?:img|audio)>(.+?)</(?:img|audio)>'
        res_messages = []
        for m in messages:
            m_new = deepcopy(m)
            content = m_new['content']
            match_iter = re.finditer(pattern, content)
            new_content = ''
            idx = 0
            for m in match_iter:
                span = m.span(1)
                img_base64 = m.group(1)
                if os.path.isfile(img_base64) or img_base64.startswith('http'):
                    continue
                img_base64 = img_base64.encode('utf-8')
                sha256_hash = hashlib.sha256(img_base64).hexdigest()
                img_path = os.path.join(tmp_dir, f'{sha256_hash}.png')
                image = Image.open(BytesIO(base64.b64decode(img_base64)))
                image.save(img_path)
                new_content += content[idx:span[0]] + img_path
                idx = span[1]
            new_content += content[idx:]
            m_new['content'] = new_content
            res_messages.append(m_new)
        res['messages'] = res_messages
    return res


def inference_client(
    model_type: str,
    query: str,
    history: Optional[History] = None,
    system: Optional[str] = None,
    *,
    request_config: Optional[XRequestConfig] = None,
    host: str = '127.0.0.1',
    port: str = '8000',
    is_chat_request: Optional[bool] = None,
) -> Union[ChatCompletionResponse, CompletionResponse, Iterator[ChatCompletionStreamResponse],
           Iterator[CompletionStreamResponse]]:
    if request_config is None:
        request_config = XRequestConfig()
    model_list = get_model_list_client(host, port)
    for model in model_list.data:
        if model_type == model.id:
            _is_chat = model.is_chat
            is_multimodal = model.is_multimodal
            break
    else:
        raise ValueError(f'model_type: {model_type}, model_list: {[model.id for model in model_list.data]}')

    if is_chat_request is None:
        is_chat_request = _is_chat
    data = {k: v for k, v in request_config.__dict__.items() if not k.startswith('__')}
    data['model'] = model_type
    if is_chat_request:
        messages = history_to_messages(history, query, system)
        if is_multimodal:
            messages = convert_to_base64(messages=messages)['messages']
        data['messages'] = messages
        url = f'http://{host}:{port}/v1/chat/completions'
    else:
        assert system is None and history is None, (
            'The chat template for text generation does not support system and history.')
        if is_multimodal:
            prompt = convert_to_base64(prompt=query)['prompt']
        data['prompt'] = prompt
        url = f'http://{host}:{port}/v1/completions'
    if request_config.stream:
        if is_chat_request:
            ret_cls = ChatCompletionStreamResponse
        else:
            ret_cls = CompletionStreamResponse
        resp = requests.post(url, json=data, stream=True)

        def _gen_stream() -> Union[Iterator[ChatCompletionStreamResponse], Iterator[CompletionStreamResponse]]:
            for data in resp.iter_lines():
                data = _parse_stream_data(data)
                if data == '[DONE]':
                    break
                if data is not None:
                    resp_obj = json.loads(data)
                    if resp_obj['object'] == 'error':
                        raise HTTPError(resp_obj['message'])
                    yield from_dict(ret_cls, resp_obj)

        return _gen_stream()
    else:
        resp_obj = requests.post(url, json=data).json()
        if is_chat_request:
            ret_cls = ChatCompletionResponse
        else:
            ret_cls = CompletionResponse
        if resp_obj['object'] == 'error':
            raise HTTPError(resp_obj['message'])
        return from_dict(ret_cls, resp_obj)

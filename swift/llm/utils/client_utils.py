import base64
import hashlib
import os
import re
from copy import deepcopy
from io import BytesIO
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple, Union

import aiohttp
import json
import requests
from dacite import from_dict
from requests.exceptions import HTTPError

from .protocol import (ChatCompletionResponse, ChatCompletionStreamResponse, CompletionResponse,
                       CompletionStreamResponse, ModelList, XRequestConfig)
from .template import History
from .utils import Messages, history_to_messages


def get_model_list_client(host: str = '127.0.0.1', port: str = '8000', **kwargs) -> ModelList:
    url = kwargs.pop('url', None)
    if url is None:
        url = f'http://{host}:{port}/v1'
    url = url.rstrip('/')
    url = f'{url}/models'
    resp_obj = requests.get(url).json()
    return from_dict(ModelList, resp_obj)


def _parse_stream_data(data: bytes) -> Optional[str]:
    data = data.decode(encoding='utf-8')
    data = data.strip()
    if len(data) == 0:
        return
    assert data.startswith('data:'), f'data: {data}'
    return data[5:].strip()


def _to_base64(img_path: str) -> str:
    if not os.path.isfile(img_path):
        return img_path
    with open(img_path, 'rb') as f:
        img_base64: str = base64.b64encode(f.read()).decode('utf-8')
    return img_base64


def _encode_prompt(prompt: str) -> str:
    pattern = r'<(?:img|audio)>(.+?)</(?:img|audio)>'
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


def _from_base64(img_base64: str, tmp_dir: str) -> str:
    from PIL import Image
    if os.path.isfile(img_base64) or img_base64.startswith('http'):
        return img_base64
    img_base64: bytes = img_base64.encode('utf-8')
    sha256_hash = hashlib.sha256(img_base64).hexdigest()
    img_path = os.path.join(tmp_dir, f'{sha256_hash}.png')
    image = Image.open(BytesIO(base64.b64decode(img_base64)))
    image.save(img_path)
    return img_path


def _decode_prompt(prompt: str, tmp_dir: str) -> str:
    pattern = r'<(?:img|audio)>(.+?)</(?:img|audio)>'
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


def decode_base64(*,
                  messages: Optional[Messages] = None,
                  prompt: Optional[str] = None,
                  images: Optional[List[str]] = None,
                  tmp_dir: str = 'tmp') -> Dict[str, Any]:
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


def _pre_inference_client(model_type: str,
                          query: str,
                          history: Optional[History] = None,
                          system: Optional[str] = None,
                          images: Optional[List[str]] = None,
                          *,
                          is_chat_request: Optional[bool] = None,
                          request_config: Optional[XRequestConfig] = None,
                          host: str = '127.0.0.1',
                          port: str = '8000',
                          **kwargs) -> Tuple[str, Dict[str, Any], bool]:
    if images is None:
        images = []
    model_list = get_model_list_client(host, port, **kwargs)
    for model in model_list.data:
        if model_type == model.id:
            _is_chat = model.is_chat
            is_multimodal = model.is_multimodal
            break
    else:
        raise ValueError(f'model_type: {model_type}, model_list: {[model.id for model in model_list.data]}')

    if is_chat_request is None:
        is_chat_request = _is_chat
    assert is_chat_request is not None, (
        'Please set the `is_chat_request` parameter to indicate whether the model is a chat model.')
    data = {k: v for k, v in request_config.__dict__.items() if not k.startswith('__')}
    url = kwargs.pop('url', None)
    if url is None:
        url = f'http://{host}:{port}/v1'
    url = url.rstrip('/')
    if is_chat_request:
        messages = history_to_messages(history, query, system, kwargs.get('roles'))
        if is_multimodal:
            messages = convert_to_base64(messages=messages)['messages']
            images = convert_to_base64(images=images)['images']
        data['messages'] = messages
        url = f'{url}/chat/completions'
    else:
        assert system is None and history is None, (
            'The chat template for text generation does not support system and history.')
        if is_multimodal:
            query = convert_to_base64(prompt=query)['prompt']
            images = convert_to_base64(images=images)['images']
        data['prompt'] = query
        url = f'{url}/completions'
    data['model'] = model_type
    if len(images) > 0:
        data['images'] = images

    return url, data, is_chat_request


def inference_client(
    model_type: str,
    query: str,
    history: Optional[History] = None,
    system: Optional[str] = None,
    images: Optional[List[str]] = None,
    *,
    is_chat_request: Optional[bool] = None,
    request_config: Optional[XRequestConfig] = None,
    host: str = '127.0.0.1',
    port: str = '8000',
    **kwargs
) -> Union[ChatCompletionResponse, CompletionResponse, Iterator[ChatCompletionStreamResponse],
           Iterator[CompletionStreamResponse]]:
    if request_config is None:
        request_config = XRequestConfig()
    url, data, is_chat_request = _pre_inference_client(
        model_type,
        query,
        history,
        system,
        images,
        is_chat_request=is_chat_request,
        request_config=request_config,
        host=host,
        port=port,
        **kwargs)

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


async def inference_client_async(
    model_type: str,
    query: str,
    history: Optional[History] = None,
    system: Optional[str] = None,
    images: Optional[List[str]] = None,
    *,
    is_chat_request: Optional[bool] = None,
    request_config: Optional[XRequestConfig] = None,
    host: str = '127.0.0.1',
    port: str = '8000',
    **kwargs
) -> Union[ChatCompletionResponse, CompletionResponse, AsyncIterator[ChatCompletionStreamResponse],
           AsyncIterator[CompletionStreamResponse]]:
    if request_config is None:
        request_config = XRequestConfig()
    url, data, is_chat_request = _pre_inference_client(
        model_type,
        query,
        history,
        system,
        images,
        is_chat_request=is_chat_request,
        request_config=request_config,
        host=host,
        port=port,
        **kwargs)

    if request_config.stream:
        if is_chat_request:
            ret_cls = ChatCompletionStreamResponse
        else:
            ret_cls = CompletionStreamResponse

        async def _gen_stream(
        ) -> Union[AsyncIterator[ChatCompletionStreamResponse], AsyncIterator[CompletionStreamResponse]]:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as resp:
                    async for _data in resp.content:
                        _data = _parse_stream_data(_data)
                        if _data == '[DONE]':
                            break
                        if _data is not None:
                            resp_obj = json.loads(_data)
                            if resp_obj['object'] == 'error':
                                raise HTTPError(resp_obj['message'])
                            yield from_dict(ret_cls, resp_obj)

        return _gen_stream()
    else:
        if is_chat_request:
            ret_cls = ChatCompletionResponse
        else:
            ret_cls = CompletionResponse
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as resp:
                resp_obj = await resp.json()
                if resp_obj['object'] == 'error':
                    raise HTTPError(resp_obj['message'])
                return from_dict(ret_cls, resp_obj)

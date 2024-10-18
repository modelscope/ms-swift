import os
import re
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple, Union

import aiohttp
import json
import requests
from dacite import from_dict
from requests.exceptions import HTTPError

from swift.llm.infer.protocol import (ChatCompletionResponse, ChatCompletionStreamResponse, CompletionResponse,
                                      CompletionStreamResponse, ModelList, XRequestConfig)
from swift.llm import Messages, History


def _get_request_kwargs(api_key: Optional[str] = None) -> Dict[str, Any]:
    timeout = float(os.getenv('TIMEOUT', '300'))
    request_kwargs = {}
    if timeout > 0:
        request_kwargs['timeout'] = timeout
    if api_key is not None:
        request_kwargs['headers'] = {'Authorization': f'Bearer {api_key}'}
    return request_kwargs


def get_model_list_client(host: str = '127.0.0.1', port: str = '8000', api_key: str = 'EMPTY', **kwargs) -> ModelList:
    url = kwargs.pop('url', None)
    if url is None:
        url = f'http://{host}:{port}/v1'
    url = url.rstrip('/')
    url = f'{url}/models'
    resp_obj = requests.get(url, **_get_request_kwargs(api_key)).json()
    return from_dict(ModelList, resp_obj)


async def get_model_list_client_async(host: str = '127.0.0.1',
                                      port: str = '8000',
                                      api_key: str = 'EMPTY',
                                      **kwargs) -> ModelList:
    url = kwargs.pop('url', None)
    if url is None:
        url = f'http://{host}:{port}/v1'
    url = url.rstrip('/')
    url = f'{url}/models'
    async with aiohttp.ClientSession() as session:
        async with session.get(url, **_get_request_kwargs(api_key)) as resp:
            resp_obj = await resp.json()
    return from_dict(ModelList, resp_obj)


def _parse_stream_data(data: bytes) -> Optional[str]:
    data = data.decode(encoding='utf-8')
    data = data.strip()
    if len(data) == 0:
        return
    assert data.startswith('data:'), f'data: {data}'
    return data[5:].strip()


def compat_openai(messages: Messages, request) -> None:
    for message in messages:
        content = message['content']
        if isinstance(content, list):
            text = ''
            for line in content:
                _type = line['type']
                value = line[_type]
                if _type == 'text':
                    text += value
                elif _type in {'image_url', 'audio_url', 'video_url'}:
                    value = value['url']
                    if value.startswith('data:'):
                        match_ = re.match(r'data:(.+?);base64,(.+)', value)
                        assert match_ is not None
                        value = match_.group(2)
                    if _type == 'image_url':
                        text += '<image>'
                        request.images.append(value)
                    elif _type == 'audio_url':
                        text += '<audio>'
                        request.audios.append(value)
                    else:
                        text += '<video>'
                        request.videos.append(value)
                else:
                    raise ValueError(f'line: {line}')
            message['content'] = text


def _pre_inference_client(model_type: str,
                          query: str,
                          messages: Optional[Messages] = None,
                          system: Optional[str] = None,
                          images: Optional[List[str]] = None,
                          tools: Optional[List[Dict[str, Union[str, Dict]]]] = None,
                          tool_choice: Optional[Union[str, Dict]] = 'auto',
                          *,
                          model_list: Optional[ModelList] = None,
                          is_chat_request: Optional[bool] = None,
                          is_multimodal: Optional[bool] = None,
                          request_config: Optional[XRequestConfig] = None,
                          host: str = '127.0.0.1',
                          port: str = '8000',
                          **kwargs) -> Tuple[str, Dict[str, Any], bool]:
    if model_list is not None:
        for model in model_list.data:
            if model_type == model.id:
                if is_chat_request is None:
                    is_chat_request = model.is_chat
                if is_multimodal is None:
                    is_multimodal = model.is_multimodal
                break
        else:
            raise ValueError(f'model_type: {model_type}, model_list: {[model.id for model in model_list.data]}')
    assert is_chat_request is not None and is_multimodal is not None
    data = {}
    request_config_origin = XRequestConfig()
    for k, v in request_config.__dict__.items():
        v_origin = getattr(request_config_origin, k)
        if v != v_origin:
            data[k] = v
    url = kwargs.pop('url', None)
    if url is None:
        url = f'http://{host}:{port}/v1'
    url = url.rstrip('/')
    if is_chat_request:
        if is_multimodal:
            messages = convert_to_base64(messages=messages)['messages']
        data['messages'] = messages
        url = f'{url}/chat/completions'
    else:
        assert system is None and history is None, (
            'The chat template for text generation does not support system and history.')
        if is_multimodal:
            query = convert_to_base64(prompt=query)['prompt']
        data['prompt'] = query
        url = f'{url}/completions'
    data['model'] = model_type
    for media_key, medias in zip(['images', 'audios', 'videos'], [images, kwargs.get('audios'), kwargs.get('videos')]):
        if medias:
            medias = convert_to_base64(images=medias)['images']
            data[media_key] = medias
    if tools:
        data['tools'] = tools
    if tool_choice and tool_choice != 'auto':
        data['tool_choice'] = tool_choice
    return url, data, is_chat_request


def inference_client(
    model_type: str,
    query: str,
    history: Optional[History] = None,
    system: Optional[str] = None,
    images: Optional[List[str]] = None,
    tools: Optional[List[Dict[str, Union[str, Dict]]]] = None,
    tool_choice: Optional[Union[str, Dict]] = 'auto',
    *,
    is_chat_request: Optional[bool] = None,
    is_multimodal: Optional[bool] = None,
    request_config: Optional[XRequestConfig] = None,
    host: str = '127.0.0.1',
    port: str = '8000',
    api_key: str = 'EMPTY',
    **kwargs
) -> Union[ChatCompletionResponse, CompletionResponse, Iterator[ChatCompletionStreamResponse],
           Iterator[CompletionStreamResponse]]:
    if request_config is None:
        request_config = XRequestConfig()
    model_list = None
    is_chat_request = is_chat_request or kwargs.get('is_chat')
    if is_chat_request is None or is_multimodal is None:
        model_list = get_model_list_client(host, port, api_key=api_key, **kwargs)

    url, data, is_chat_request = _pre_inference_client(
        model_type,
        query,
        history,
        system,
        images,
        tools,
        tool_choice,
        model_list=model_list,
        is_chat_request=is_chat_request,
        is_multimodal=is_multimodal,
        request_config=request_config,
        host=host,
        port=port,
        **kwargs)

    if request_config.stream:
        if is_chat_request:
            ret_cls = ChatCompletionStreamResponse
        else:
            ret_cls = CompletionStreamResponse
        resp = requests.post(url, json=data, stream=True, **_get_request_kwargs(api_key))

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
        resp_obj = requests.post(url, json=data, **_get_request_kwargs(api_key)).json()
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
    tools: Optional[List[Dict[str, Union[str, Dict]]]] = None,
    tool_choice: Optional[Union[str, Dict]] = 'auto',
    *,
    is_chat_request: Optional[bool] = None,
    is_multimodal: Optional[bool] = None,
    request_config: Optional[XRequestConfig] = None,
    host: str = '127.0.0.1',
    port: str = '8000',
    api_key: str = 'EMPTY',
    **kwargs
) -> Union[ChatCompletionResponse, CompletionResponse, AsyncIterator[ChatCompletionStreamResponse],
           AsyncIterator[CompletionStreamResponse]]:
    if request_config is None:
        request_config = XRequestConfig()
    model_list = None
    is_chat_request = is_chat_request or kwargs.get('is_chat')
    if is_chat_request is None or is_multimodal is None:
        model_list = await get_model_list_client_async(host, port, api_key=api_key, **kwargs)

    url, data, is_chat_request = _pre_inference_client(
        model_type,
        query,
        history,
        system,
        images,
        tools,
        tool_choice,
        model_list=model_list,
        is_chat_request=is_chat_request,
        is_multimodal=is_multimodal,
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
                async with session.post(url, json=data, **_get_request_kwargs(api_key)) as resp:
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
            async with session.post(url, json=data, **_get_request_kwargs(api_key)) as resp:
                resp_obj = await resp.json()
                if resp_obj['object'] == 'error':
                    raise HTTPError(resp_obj['message'])
                return from_dict(ret_cls, resp_obj)

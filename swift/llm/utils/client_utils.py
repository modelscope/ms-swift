from typing import Iterator, Optional, Union

import json
import requests
from dacite import from_dict
from requests.exceptions import HTTPError

from .model import get_default_template_type
from .protocol import (ChatCompletionResponse, ChatCompletionStreamResponse, CompletionResponse,
                       CompletionStreamResponse, ModelList, XRequestConfig)
from .template import History
from .utils import history_to_messages


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


def inference_client(
    model_type: str,
    query: str,
    history: Optional[History] = None,
    system: Optional[str] = None,
    *,
    request_config: Optional[XRequestConfig] = None,
    host: str = '127.0.0.1',
    port: str = '8000',
    adapter_name: str = None,
    is_chat_request: Optional[bool] = None,
) -> Union[ChatCompletionResponse, CompletionResponse, Iterator[ChatCompletionStreamResponse],
           Iterator[CompletionStreamResponse]]:
    if request_config is None:
        request_config = XRequestConfig()
    if is_chat_request is None:
        template_type = get_default_template_type(model_type)
        is_chat_request = 'generation' not in template_type
    data = {k: v for k, v in request_config.__dict__.items() if not k.startswith('__')}
    data['model'] = adapter_name or model_type
    if is_chat_request:
        data['messages'] = history_to_messages(history, query, system)
        url = f'http://{host}:{port}/v1/chat/completions'
    else:
        assert system is None and history is None, (
            'The chat template for text generation does not support system and history.')
        data['prompt'] = query
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

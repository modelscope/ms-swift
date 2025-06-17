# Copyright (c) Alibaba, Inc. and its affiliates.
import base64
import io
import os
import time
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import json
from PIL import Image
from pydantic import BaseModel

from ..template import InferRequest
from ..utils import Messages, Tool


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


@dataclass
class Model:
    id: str  # model_type

    object: str = 'model'
    created: int = field(default_factory=lambda: int(time.time()))
    owned_by: str = 'ms-swift'


@dataclass
class ModelList:
    data: List[Model]
    object: str = 'list'


@dataclass
class RequestConfig:
    """NOTE: The following behavior is inconsistent with the OpenAI API.
    Default values for OpenAI:
        temperature = 1.
        top_k = -1
        top_p = 1.
        repetition_penalty = 1.
    """
    max_tokens: Optional[int] = None  # None: max_model_len - num_tokens
    # None: use deploy_args
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    num_beams: int = 1
    stop: Optional[List[str]] = field(default_factory=list)

    seed: Optional[int] = None
    stream: bool = False
    logprobs: bool = False
    top_logprobs: Optional[int] = None

    n: int = 1
    best_of: Optional[int] = None
    presence_penalty: float = 0.
    frequency_penalty: float = 0.
    length_penalty: float = 1.

    def __post_init__(self):
        if self.stop is None:
            self.stop = []


@dataclass
class CompletionRequestMixin:
    model: str
    prompt: str


@dataclass
class ChatCompletionRequestMixin:
    model: str
    messages: Messages
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, Dict]] = None

    def __post_init__(self):
        if self.tool_choice is None:
            self.tool_choice = 'none' if self.tools is None else 'auto'

        if self.tools:
            if self.tool_choice == 'none':
                self.tools = None
            elif isinstance(self.tool_choice, dict):
                name = self.tool_choice['function']['name']
                tool = next(tool for tool in self.tools if tool['function']['name'] == name)
                if tool is None:
                    raise ValueError(f"Tool choice '{name}' not found in tools.")
                self.tools = [tool]


@dataclass
class MultiModalRequestMixin:
    images: List[str] = field(default_factory=list)
    audios: List[str] = field(default_factory=list)
    videos: List[str] = field(default_factory=list)
    objects: Dict[str, List[Any]] = field(default_factory=dict)

    @staticmethod
    def to_base64(mm_data: Union[str, Image.Image, bytes]) -> str:
        if isinstance(mm_data, dict) and 'bytes' in mm_data:
            mm_data = mm_data['bytes'] or mm_data['path']
        if isinstance(mm_data, str) and not os.path.isfile(mm_data):
            # base64 or url
            return mm_data
        if isinstance(mm_data, str):
            # local_path
            with open(mm_data, 'rb') as f:
                bytes_ = f.read()
        elif isinstance(mm_data, Image.Image):
            bytes_io = io.BytesIO()
            mm_data.save(bytes_io, format='png')
            bytes_ = bytes_io.getvalue()
        else:
            bytes_ = mm_data
        img_base64: str = base64.b64encode(bytes_).decode('utf-8')
        return img_base64

    def __post_init__(self):
        for key in ['images', 'audios', 'videos']:
            values = getattr(self, key)
            if isinstance(values, str):
                values = [values]
                setattr(self, key, values)
            for i, val in enumerate(values):
                values[i] = self.to_base64(val)


@dataclass
class CompletionRequest(RequestConfig, MultiModalRequestMixin, CompletionRequestMixin):

    def __post_init__(self):
        RequestConfig.__post_init__(self)
        MultiModalRequestMixin.__post_init__(self)


@dataclass
class ChatCompletionRequest(RequestConfig, MultiModalRequestMixin, ChatCompletionRequestMixin):

    def __post_init__(self):
        RequestConfig.__post_init__(self)
        MultiModalRequestMixin.__post_init__(self)
        ChatCompletionRequestMixin.__post_init__(self)
        self.convert_to_base64()

    def convert_to_base64(self):
        for message in self.messages:
            content = message['content']
            if isinstance(content, str):
                continue
            for item in content:
                key: str = item['type']
                if key == 'text':
                    continue

                key_origin = key
                value = item[key]
                if key.endswith('_url'):
                    key = key[:-len('_url')]
                is_dict = False
                if isinstance(value, dict):
                    is_dict = True
                    value = value['url']
                if isinstance(value, str) and (value.startswith('data:') or value.startswith('http')
                                               or len(value) > 200):
                    continue

                # local_path / PIL.Image
                if isinstance(value, str) and os.path.isfile(value):
                    suffix = os.path.splitext(value)[1][1:].lower()
                elif isinstance(value, Image.Image):
                    suffix = 'jpeg'
                else:
                    raise ValueError(f'value: {value}')
                mm_data_base64 = self.to_base64(value)
                new_value = f'data:{key}/{suffix};base64,{mm_data_base64}'
                if is_dict:
                    new_value = {'url': new_value}
                item[key_origin] = new_value

    def parse(self) -> Tuple['InferRequest', 'RequestConfig']:
        data = asdict(self)
        res = []
        for cls_type in [InferRequest, RequestConfig]:
            parameters = set(f.name for f in fields(cls_type))
            _data = {k: v for k, v in data.items() if k in parameters}
            res.append(cls_type(**_data))
        return tuple(res)

    @classmethod
    def from_cmpl_request(cls, cmpl_request: CompletionRequest) -> 'ChatCompletionRequest':
        cmpl_request = asdict(cmpl_request)
        prompt = cmpl_request.pop('prompt')
        cmpl_request['messages'] = [{'role': 'user', 'content': prompt}]
        return cls(**cmpl_request)


@dataclass
class UsageInfo:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class Function:
    name: str
    arguments: Optional[str]

    def __post_init__(self):
        if not isinstance(self.arguments, str):
            self.arguments = json.dumps(self.arguments)
        self.name = self.name.strip()
        self.arguments = self.arguments.strip()


@dataclass
class ChatCompletionMessageToolCall:
    function: Function
    type: str = 'function'
    id: str = field(default_factory=lambda: f'toolcall-{random_uuid()}')


@dataclass
class ChatMessage:
    role: Literal['system', 'user', 'assistant']
    content: Union[str, List[Dict[str, Any]], int, float]
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None


@dataclass
class ChatCompletionResponseChoice:
    index: int
    message: ChatMessage
    finish_reason: Literal['stop', 'length', None]
    logprobs: Optional[Dict[str, List[Dict[str, Any]]]] = None

    def to_cmpl_choice(self) -> 'CompletionResponseChoice':
        self = deepcopy(self)
        assert not self.message.tool_calls, f'message: {self.message}'
        return CompletionResponseChoice(self.index, self.message.content, self.finish_reason, self.logprobs)


@dataclass
class RolloutResponseChoice(ChatCompletionResponseChoice):
    messages: Optional[Messages] = None


@dataclass
class CompletionResponseChoice:
    index: int
    text: str
    finish_reason: Literal['stop', 'length', None]
    logprobs: Optional[Dict[str, List[Dict[str, Any]]]] = None


@dataclass
class ChatCompletionResponse:
    model: str
    choices: List[Union[ChatCompletionResponseChoice, RolloutResponseChoice]]
    usage: UsageInfo
    id: str = field(default_factory=lambda: f'chatcmpl-{random_uuid()}')
    object: str = 'chat.completion'
    created: int = field(default_factory=lambda: int(time.time()))

    def to_cmpl_response(self) -> 'CompletionResponse':
        self = deepcopy(self)
        choices = [choice.to_cmpl_choice() for choice in self.choices]
        id_ = f'cmpl{self.id[len("chatcmpl"):]}'
        return CompletionResponse(self.model, choices, self.usage, id_, created=self.created)


@dataclass
class CompletionResponse:
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo
    id: str = field(default_factory=lambda: f'cmpl-{random_uuid()}')
    object: str = 'text_completion'
    created: int = field(default_factory=lambda: int(time.time()))


@dataclass
class DeltaMessage:
    role: Literal['system', 'user', 'assistant', None] = None
    content: Optional[str] = None
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None


@dataclass
class ChatCompletionResponseStreamChoice:
    index: int
    delta: DeltaMessage
    finish_reason: Literal['stop', 'length', None]
    logprobs: Optional[Dict[str, List[Dict[str, Any]]]] = None

    def to_cmpl_choice(self) -> 'CompletionResponseStreamChoice':
        self = deepcopy(self)
        assert not self.delta.tool_calls
        return CompletionResponseStreamChoice(self.index, self.delta.content, self.finish_reason, self.logprobs)


@dataclass
class CompletionResponseStreamChoice:
    index: int
    text: str
    finish_reason: Literal['stop', 'length', None]
    logprobs: Optional[Dict[str, List[Dict[str, Any]]]] = None


@dataclass
class ChatCompletionStreamResponse:
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = None
    id: str = field(default_factory=lambda: f'chatcmpl-{random_uuid()}')
    object: str = 'chat.completion.chunk'
    created: int = field(default_factory=lambda: int(time.time()))

    def to_cmpl_response(self) -> 'CompletionStreamResponse':
        self = deepcopy(self)
        choices = [choice.to_cmpl_choice() for choice in self.choices]
        id_ = f'cmpl{self.id[len("chatcmpl"):]}'
        return CompletionStreamResponse(self.model, choices, self.usage, id_, created=self.created)


@dataclass
class CompletionStreamResponse:
    model: str
    choices: List[CompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = None
    id: str = field(default_factory=lambda: f'cmpl-{random_uuid()}')
    object: str = 'text_completion.chunk'
    created: int = field(default_factory=lambda: int(time.time()))


class InitCommunicatorRequest(BaseModel):
    host: str
    port: int
    world_size: int


class UpdateWeightsRequest(BaseModel):
    name: str
    dtype: str
    shape: list[int]

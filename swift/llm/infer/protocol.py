# Copyright (c) Alibaba, Inc. and its affiliates.
import time
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from http import HTTPStatus
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import json
from fastapi.responses import JSONResponse
from PIL import Image

from swift.llm import TemplateInputs
from swift.llm.template import Messages, Tool


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
    stop: List[str] = field(default_factory=list)

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
class CompletionRequest(RequestConfig, CompletionRequestMixin):
    pass


@dataclass
class ChatCompletionRequest(RequestConfig, ChatCompletionRequestMixin):

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
    arguments: str


@dataclass
class ChatCompletionMessageToolCall:
    function: Function
    type: str = 'function'
    id: str = field(default_factory=lambda: f'toolcall-{random_uuid()}')


@dataclass
class ChatMessage:
    role: Literal['system', 'user', 'assistant']
    content: str
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None


@dataclass
class ChatCompletionResponseChoice:
    index: int
    message: ChatMessage
    finish_reason: Literal['stop', 'length', None]  # None: for infer_backend='pt'
    logprobs: Optional[Dict[str, List[Dict[str, Any]]]] = None

    def to_cmpl_choice(self) -> 'CompletionResponseChoice':
        assert not self.message.tool_calls
        return CompletionResponseChoice(self.index, self.message.content, self.finish_reason, deepcopy(self.logprobs))


@dataclass
class CompletionResponseChoice:
    index: int
    text: str
    finish_reason: Literal['stop', 'length', None]  # None: for infer_backend='pt'
    logprobs: Optional[Dict[str, List[Dict[str, Any]]]] = None


@dataclass
class ChatCompletionResponse:
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo
    id: str = field(default_factory=lambda: f'chatcmpl-{random_uuid()}')
    object: str = 'chat.completion'
    created: int = field(default_factory=lambda: int(time.time()))

    def to_cmpl_response(self) -> 'CompletionResponse':
        choices = [choice.to_cmpl_choice() for choice in self.choices]
        id_ = f'cmpl{self.id[len("chatcmpl"):]}'
        return CompletionResponse(self.model, choices, deepcopy(self.usage), id_, created=self.created)


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
    role: Literal['system', 'user', 'assistant']
    content: str
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None


@dataclass
class ChatCompletionResponseStreamChoice:
    index: int
    delta: DeltaMessage
    finish_reason: Literal['stop', 'length', None]
    logprobs: Optional[Dict[str, List[Dict[str, Any]]]] = None

    def to_cmpl_choice(self) -> 'CompletionResponseStreamChoice':
        assert not self.delta.tool_calls
        return CompletionResponseStreamChoice(self.index, self.delta.content, self.finish_reason,
                                              deepcopy(self.logprobs))


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
    usage: UsageInfo
    id: str = field(default_factory=lambda: f'chatcmpl-{random_uuid()}')
    object: str = 'chat.completion.chunk'
    created: int = field(default_factory=lambda: int(time.time()))

    def to_cmpl_response(self) -> 'CompletionStreamResponse':
        choices = [choice.to_cmpl_choice() for choice in self.choices]
        id_ = f'cmpl{self.id[len("chatcmpl"):]}'
        return CompletionStreamResponse(self.model, choices, deepcopy(self.usage), id_, created=self.created)


@dataclass
class CompletionStreamResponse:
    model: str
    choices: List[CompletionResponseStreamChoice]
    usage: UsageInfo
    id: str = field(default_factory=lambda: f'cmpl-{random_uuid()}')
    object: str = 'text_completion.chunk'
    created: int = field(default_factory=lambda: int(time.time()))


@dataclass
class InferRequest(TemplateInputs):

    def remove_response(self):
        last_role = self.messages[-1]['role']
        if last_role == 'assistant':
            self.messages.pop()

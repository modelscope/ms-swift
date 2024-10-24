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

from swift.llm import Messages

Tool = Dict[str, Union[str, Dict]]


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
    top_p: Optional[float] = None

    n: int = 1
    seed: Optional[int] = None
    stop: List[str] = field(default_factory=list)
    stream: bool = False
    logprobs: bool = False
    top_logprobs: Optional[int] = None

    best_of: Optional[int] = None
    presence_penalty: float = 0.
    frequency_penalty: float = 0.
    length_penalty: float = 1.

    # additional
    # None: use deploy_args
    num_beams: Optional[int] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None

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

    def to_chat_request(self) -> 'ChatCompletionRequest':
        cmpl_request = asdict(self)
        prompt = cmpl_request.pop('prompt')
        cmpl_request['messages'] = [{'role': 'user', 'content': prompt}]
        return ChatCompletionRequest(**cmpl_request)


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
class InferRequest:
    """
    messages: Input in messages format.
        Examples: [{
            "role": "user",  # or assistant/system/role
            "content": [  # str or List[Dict[str, Any]]
                {
                    "type": "image",  # or audio/video
                    # This content is usually written in the `images` field (recommended).
                    "image": "<url/path/base64/PIL.Image>",
                },
                {"type": "text", "text": "<text>"},
            ],
        }]
    objects: Used for grounding tasks in a general format.
    tools: Organize tools into the format of tools_prompt for system. for example, 'react_en'.
        Specifying this parameter will override system.
    """
    messages: Messages

    images: List[Union[str, Image.Image]] = field(default_factory=list)
    audios: List[str] = field(default_factory=list)
    videos: List[str] = field(default_factory=list)

    objects: Union[str, None, List[Dict[str, Any]]] = None  # List[Dict[str, Any]]
    tools: Optional[List[Tool]] = None

    def __post_init__(self):
        # Format objects(groundings/refs) to json
        if isinstance(self.objects, str):
            # reload grounding from str
            self.objects = json.loads(self.objects)
        elif self.objects is None:
            self.objects = []

    def copy(self):
        return InferRequest(
            deepcopy(self.messages), self.images.copy(), self.audios.copy(), self.videos.copy(), deepcopy(self.objects),
            deepcopy(self.tools))

    @staticmethod
    def remove_messages_media(messages: Messages) -> Dict[str, Any]:
        res = {'images': [], 'audios': [], 'videos': []}
        for message in messages:
            content = message['content']
            if isinstance(content, str):
                continue
            # List[Dict[str, Any]]
            new_content = ''
            for item in content:
                key: str = item['type']
                value = item[key]
                if key == 'text':
                    new_content += value
                    continue
                # image/audio/video
                # image_url/audio_url/video_url
                if key.endswith('_url'):
                    key = key[:-len('_url')]
                new_content += f'<{key}>'
                res[f'{key}s'].append(value)
            message['content'] = new_content
        return res

    def to_template_inputs(self, *, tools_prompt: str = 'react_en') -> 'TemplateInputs':
        from swift.llm.template import get_tools_prompt, TemplateInputs
        request = self.copy()
        messages = request.messages
        tools = request.tools

        assert len(messages) >= 1

        if messages[0]['role'] == 'system':
            message = messages.pop(0)
            system = message['content']
        else:
            system = None

        if tools is not None:
            assert system is None
            if isinstance(tools, str):
                tools = json.loads(tools)
            system = get_tools_prompt(tools, tools_prompt)

        media_kwargs = InferRequest.remove_messages_media(messages)
        for k in list(media_kwargs.keys()):
            mm_data = media_kwargs[k]
            self_mm_data = getattr(self, k)
            if mm_data:
                assert not self_mm_data, f'self.{k}: {self_mm_data}'
            else:
                media_kwargs[k] = self_mm_data

        InferRequest.messages_join_observation(messages)
        inputs = TemplateInputs(messages, system, **media_kwargs, objects=request.objects)
        return inputs

    @staticmethod
    def messages_join_observation(messages: Messages) -> None:
        """
            Joins observations from 'tool' message into the 'assistant' response.

            Example:
            ---------
            Original messages:
            messages = [
                {'role': 'user', 'content': "What's the weather today in Hangzhou?"},
                {'role': 'assistant', 'content': 'Action: get_weather\nAction Input:\
                      [{"location": "Hangzhou"}]\nObservations:'},
                {'role': 'tool', 'content': 'It is 26 degrees Celsius and sunny in Hangzhou today.'}
            ]

            Transformed messages:
            messages = [
                {'role': 'user', 'content': "What's the weather today in Hangzhou?"},
                {'role': 'assistant', 'content': 'Action: get_weather\nAction Input:\
                      [{"location": "Hangzhou"}]\nObservations: It is 26 degrees Celsius and sunny in Hangzhou today.'}
            ]
        """
        if len(messages) < 2:
            return
        i = 1
        while i < len(messages):
            pre_message, message = messages[i - 1], messages[i]
            pre_role, pre_content = pre_message['role'], pre_message['content']
            role, content = message['role'], message['content']
            if pre_role == 'assistant' and role == 'tool' and isinstance(content,
                                                                         str) and content.endswith('Observation:'):
                assert isinstance(pre_content, str)
                pre_message['content'] = pre_content + content  # assistant
                messages.pop(i)  # remove tool
            else:
                i += 1

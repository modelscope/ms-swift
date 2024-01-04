# Copyright (c) Alibaba, Inc. and its affiliates.
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


@dataclass
class Model:
    id: str  # model_type


@dataclass
class ModelList:
    data: List[Model]
    object: str = 'list'


@dataclass
class CompletionRequest:
    prompt: str
    #
    n: int = 1
    max_tokens: Optional[int] = None
    temperature: float = 1.
    top_k: int = -1
    top_p: float = 1.
    repetition_penalty: float = 1.
    num_beams: int = 1
    #
    seed: Optional[int] = None
    stop: List[str] = field(default_factory=list)
    stream: bool = False
    #
    best_of: Optional[int] = None
    presence_penalty: float = 0.
    frequency_penalty: float = 0.


@dataclass
class ChatCompletionRequest:
    model: str
    messages: List[Dict[str, str]]
    #
    n: int = 1
    max_tokens: Optional[int] = None
    temperature: float = 1.
    top_k: int = -1
    top_p: float = 1.
    repetition_penalty: float = 1.
    num_beams: int = 1
    #
    seed: Optional[int] = None
    stop: List[str] = field(default_factory=list)
    stream: bool = False
    best_of: Optional[int] = None
    presence_penalty: float = 0.
    frequency_penalty: float = 0.


@dataclass
class UsageInfo:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ChatMessage:
    role: Literal['system', 'user', 'assistant']
    content: str


@dataclass
class ChatCompletionResponseChoice:
    index: int
    message: ChatMessage
    finish_reason: Literal['stop', 'length']


@dataclass
class CompletionResponseChoice:
    index: int
    text: str
    finish_reason: Literal['stop', 'length']


@dataclass
class ChatCompletionResponse:
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo
    id: str = field(default_factory=lambda: f'chatcmpl-{random_uuid()}')
    object: str = 'chat.completion'
    created: int = field(default_factory=lambda: int(time.time()))


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


@dataclass
class ChatCompletionResponseStreamChoice:
    index: int
    delta: DeltaMessage
    finish_reason: Literal['stop', 'length']


@dataclass
class ChatCompletionStreamResponse:
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
    usage: UsageInfo
    id: str = field(default_factory=lambda: f'chatcmpl-{random_uuid()}')
    object: str = 'chat.completion.chunk'
    created: int = field(default_factory=lambda: int(time.time()))


@dataclass
class CompletionResponseStreamChoice:
    index: int
    text: str
    finish_reason: Literal['stop', 'length']


@dataclass
class CompletionStreamResponse:
    model: str
    choices: List[CompletionResponseStreamChoice]
    usage: UsageInfo
    id: str = field(default_factory=lambda: f'cmpl-{random_uuid()}')
    object: str = 'text_completion.chunk'
    created: int = field(default_factory=lambda: int(time.time()))

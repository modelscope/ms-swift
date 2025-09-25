# Copyright (c) Alibaba, Inc. and its affiliates.
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Union

import json
from PIL import Image

from swift.utils import get_logger
from ..utils import Messages, Tool, messages_to_history

logger = get_logger()


@dataclass
class InferRequest:
    """
    Data structure for inference requests.

    Attributes:
        messages (Messages):
            The input conversation in messages format. Each message is a dict containing at least
            a "role" field (e.g., "user", "assistant", "system") and a "content" field.
            Example:
                [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",  # can also be audio/video
                            "image": "<url/path/base64/PIL.Image>",
                        },
                        {"type": "text", "text": "Please describe the picture."},
                    ],
                }]
            The above is equivalent to:
                [{"role": "user", "content": "<image>Please describe the picture."}]
            with an additional argument:
                images = ["<url/path/base64/PIL.Image>"]

        images (List[Union[str, Image.Image]]):
            Optional, a list of images associated with the request.
            Each image can be a URL, local path, base64 string, or PIL.Image object.

        audios (List[str]):
            Optional, a list of audio resources associated with the request.

        videos (List[str]):
            Optional, a list of video resources associated with the request.

        tools (Optional[List[Tool]]):
            An optional list of tools. These should be organized in the agent_template format for
            tools requested by the system, for example 'react_en'.

        objects (Dict[str, List[Any]]):
            Container for additional multimodal objects, grouped by type (key).
    """
    messages: Messages

    images: List[Union[str, Image.Image]] = field(default_factory=list)
    audios: List[str] = field(default_factory=list)
    videos: List[str] = field(default_factory=list)

    tools: Optional[List[Tool]] = None
    objects: Dict[str, List[Any]] = field(default_factory=dict)

    def __post_init__(self):
        for key in ['images', 'audios', 'videos']:
            val = getattr(self, key)
            if isinstance(val, str):
                setattr(self, key, [val])
        assert isinstance(self.messages, list), f'messages: {self.messages}'

    @staticmethod
    def remove_response(messages) -> Optional[str]:
        last_role = messages[-1]['role'] if messages else None
        if last_role == 'assistant':
            return messages.pop()['content']

    @staticmethod
    def _to_printable(obj, key: Optional[str] = None):
        if isinstance(obj, str) and key not in {'content', 'text'} and len(obj) >= 1000:
            return f'<<<base64:{obj[:50]}..>>>'
        elif isinstance(obj, list):
            res = []
            for item in obj:
                res.append(InferRequest._to_printable(item))
            return res
        elif isinstance(obj, dict):
            res = {}
            for k, v in obj.items():
                res[k] = InferRequest._to_printable(v, key=k)
            return res
        return obj

    def to_printable(self):
        return InferRequest._to_printable(asdict(self))


@dataclass
class RolloutInferRequest(InferRequest):
    """
    An inference request class for rollout scenarios.

    This class extends `InferRequest` and specifically overrides the `images` attribute
    to be a list of strings for compatibility with POST requests. Each string may
    represent an image URL or a Base64-encoded image.

    Inherits all fields from `InferRequest`:
        messages (Messages):
            Input conversation messages, supporting multimodal content.
        audios (List[str]):
            List of audio resources associated with the request.
        videos (List[str]):
            List of video resources associated with the request.
        tools (Optional[List[Tool]]):
            List of tools, organized by the agent template (e.g. 'react_en').
        objects (Dict[str, List[Any]]):
            Optional container for additional multimodal objects.

    Additional / Overridden fields:
        images (List[str]):
            List of image resources, each as a string (URL or base64).
        data_dict (Dict):
            Optional dictionary for extra request data.
        uuid (Optional[str]):
            Optional unique identifier for this request instance.
    """
    images: List[str] = field(default_factory=list)
    data_dict: Dict = field(default_factory=dict)
    uuid: Optional[str] = None


@dataclass
class StdTemplateInputs:
    # only user/tool/assistant
    messages: List[Dict[str, str]]
    # None: use default system; '': not use system
    system: Optional[str] = None
    tools: Optional[List[Tool]] = None

    label: Optional[int] = None
    channel: Optional[str] = None

    images: List[Union[str, Image.Image]] = field(default_factory=list)
    videos: List[str] = field(default_factory=list)
    audios: List[str] = field(default_factory=list)
    objects: Dict[str, List[Any]] = field(default_factory=dict)

    margin: Optional[float] = None  # for reward modeling
    mm_processor_kwargs: Dict[str, Any] = field(default_factory=dict)
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)
    # compat
    rejected_response: Optional[List[str]] = None

    def __post_init__(self):
        self.image_idx = 0
        self.audio_idx = 0
        self.video_idx = 0
        self.ref_idx = 0
        self.bbox_idx = 0
        if self.images and not isinstance(self.images, (list, tuple)):
            self.images = [self.images]
        if self.videos and not isinstance(self.videos, (list, tuple)):
            self.videos = [self.videos]
        if self.audios and not isinstance(self.audios, (list, tuple)):
            self.audios = [self.audios]
        if self.rejected_response:
            assert isinstance(self.rejected_response, list) and all(
                isinstance(item, str) for item in self.rejected_response)

    def to_history(self):
        if not self.messages:
            return None
        return messages_to_history(self.messages)

    @property
    def is_multimodal(self):
        return bool(self.images or self.audios or self.videos or self.objects)

    @classmethod
    def from_dict(cls, inputs: Dict[str, Any]) -> 'StdTemplateInputs':
        inputs = deepcopy(inputs)
        kwargs = {}
        for key in ['label', 'channel', 'margin', 'rejected_response']:
            if key in inputs:
                kwargs[key] = inputs[key]
        messages = inputs['messages']
        tools = inputs.get('tools')
        objects = inputs.get('objects') or {}

        if messages and messages[0]['role'] == 'system':
            message = messages.pop(0)
            system = message['content']
        else:
            system = None

        for message in messages:
            if message['role'] == 'tool_response':
                message['role'] = 'tool'
            if message['role'] in {'tool_call', 'tool'} and not isinstance(message['content'], str):
                message['content'] = json.dumps(message['content'], ensure_ascii=False)

        media_kwargs = StdTemplateInputs.remove_messages_media(messages)
        for k in list(media_kwargs.keys()):
            mm_data = media_kwargs[k]

            inputs_mm_data = inputs.get(k)
            if isinstance(inputs_mm_data, str):
                inputs_mm_data = [inputs_mm_data]
            inputs_mm_data = (inputs_mm_data or []).copy()
            if mm_data:
                assert not inputs_mm_data, f'self.{k}: {inputs_mm_data}'
            else:
                media_kwargs[k] = inputs_mm_data

        all_keys = set(f.name for f in fields(StdTemplateInputs))
        extra_kwargs = {k: v for k, v in inputs.items() if k not in all_keys}
        return cls(
            messages=messages,
            system=system,
            tools=tools,
            objects=objects,
            extra_kwargs=extra_kwargs,
            **kwargs,
            **media_kwargs)

    @staticmethod
    def remove_messages_media(messages: Messages) -> Dict[str, Any]:
        res = {'images': [], 'audios': [], 'videos': []}
        for message in messages:
            content = message['content']
            if isinstance(content, str):
                continue
            elif (isinstance(content, list) and content
                  and isinstance(content[0], int)) or (isinstance(content, dict) and 'token_ids' in content):
                continue
            # List[Dict[str, Any]]
            new_content = ''
            for item in content:
                key: str = item['type']
                value = item.get(key)
                if key == 'text':
                    new_content += value
                    continue
                # image/audio/video
                # image_url/audio_url/video_url
                if key.endswith('_url'):
                    key = key[:-len('_url')]
                new_content += f'<{key}>'
                if isinstance(value, dict):
                    value = value['url']
                if value:
                    res[f'{key}s'].append(value)
            message['content'] = new_content
        return res


@dataclass
class TemplateInputs:
    chosen: StdTemplateInputs  # or Dict[str, Any]
    rejected: Optional[StdTemplateInputs] = None
    positive: List[StdTemplateInputs] = field(default_factory=list)  # or Dict[str, Any]
    negative: List[StdTemplateInputs] = field(default_factory=list)

    def __post_init__(self):
        all_keys = set(f.name for f in fields(StdTemplateInputs))
        for key in ['chosen', 'rejected', 'positive', 'negative']:
            value_dict = getattr(self, key, None)
            if not isinstance(value_dict, dict):
                continue
            if key in {'chosen', 'rejected'}:
                setattr(self, key, StdTemplateInputs.from_dict(value_dict))
            else:
                res = []
                for i in range(len(value_dict['messages'])):
                    kwargs = {}
                    for k in all_keys:
                        val = value_dict.get(k)
                        if val is None:
                            continue
                        kwargs[k] = val[i]
                    res.append(StdTemplateInputs.from_dict(kwargs))
                setattr(self, key, res)

    @staticmethod
    def _compat_rejected_response(inputs: Dict[str, Any]):
        if 'rejected_response' not in inputs:
            return
        # Find the first round's 'assistant'.
        messages = inputs['messages']
        assert len(messages) > 0, f'messages: {messages}'
        for idx in range(len(messages), 0, -1):
            message = messages[idx - 1]
            if message['role'] in {'user', 'tool', 'tool_response'}:
                break

        rejected_response = inputs.pop('rejected_response')
        if isinstance(rejected_response, list) and rejected_response and isinstance(rejected_response[0], str):
            inputs['rejected_response'] = rejected_response
            return
        assert isinstance(rejected_response, str), f'rejected_response: {rejected_response}'
        # Check that the response is different from the rejected_response.
        if isinstance(rejected_response, str):
            if len(messages[idx:]) == 1:
                response = messages[idx]['content']
                assert rejected_response != response, f'rejected_response: {rejected_response}, response: {response}'
            rejected_response = [{'role': 'assistant', 'content': rejected_response}]
        inputs['rejected_messages'] = deepcopy(messages[:idx]) + rejected_response

    @classmethod
    def from_dict(cls, inputs: Dict[str, Any]) -> 'TemplateInputs':
        inputs = deepcopy(inputs)

        has_rejected_messages = inputs.get('rejected_messages') is not None
        cls._compat_rejected_response(inputs)
        rejected_response = inputs.pop('rejected_response', None)
        kwargs = {}
        non_chosen_keys = ['rejected', 'positive', 'negative']
        for prefix in ['chosen'] + non_chosen_keys:
            if prefix == 'chosen':
                std_inputs = {
                    k: v
                    for k, v in inputs.items() if not any(k.startswith(f'{p}_') for p in non_chosen_keys)
                }
            else:
                std_inputs = {k[len(f'{prefix}_'):]: v for k, v in inputs.items() if k.startswith(f'{prefix}_')}
            if std_inputs:
                kwargs[prefix] = std_inputs

        if not has_rejected_messages and kwargs.get('rejected') is not None:
            chosen = kwargs['chosen']
            rejected = kwargs['rejected']
            # Supplement additional key-value pairs
            for k, chosen_v in chosen.items():
                rejected_v = rejected.get(k)
                if chosen_v is not None and rejected_v is None:
                    rejected[k] = chosen_v
        if rejected_response and 'chosen' in kwargs:
            kwargs['chosen']['rejected_response'] = rejected_response

        return cls(**kwargs)

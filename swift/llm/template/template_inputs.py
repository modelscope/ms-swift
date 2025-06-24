# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple, Union

import json
from PIL import Image

from swift.utils import get_logger
from ..utils import Messages, Tool, messages_to_history

logger = get_logger()


@dataclass
class InferRequest:
    """
    messages: Input in messages format.
        Examples: [{
            "role": "user",  # or assistant/system/role
            "content": [  # str or List[Dict[str, Any]]
                {
                    "type": "image",  # or audio/video
                    "image": "<url/path/base64/PIL.Image>",
                },
                {"type": "text", "text": "Please describe the picture."},
            ],
        }]
        The above content is equivalent to:
        [{"role": "user", "content": "<image>Please describe the picture."}]
        and additionally passing in images: ["<url/path/base64/PIL.Image>"].
    tools: Organize tools into the format of agent_template for system. for example, 'react_en'.
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
    A request class that modifies the 'images' attribute
    to be a list of strings for compatibility with POST requests.
    The strings can represent image URLs or Base64 encoded images.
    """
    images: List[str] = field(default_factory=list)
    data_dict: Dict = field(default_factory=dict)


@dataclass
class TemplateInputs(InferRequest):
    """The training functionality has been added on top of the InferRequest.

    objects: Used for grounding tasks in a general format.
    """
    rejected_response: Optional[str] = None
    label: Optional[bool] = None


@dataclass
class StdTemplateInputs:
    # only user/tool/assistant
    messages: List[Dict[str, str]]
    # None: use default system; '': not use system
    system: Optional[str] = None
    tools: Optional[List[Tool]] = None

    rejected_response: Optional[str] = None
    label: Optional[int] = None
    channel: Optional[str] = None

    images: List[Union[str, Image.Image]] = field(default_factory=list)
    audios: List[str] = field(default_factory=list)
    videos: List[str] = field(default_factory=list)
    objects: Dict[str, List[Any]] = field(default_factory=dict)

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

    def to_history(self):
        if not self.messages:
            return None
        return messages_to_history(self.messages)

    @property
    def is_multimodal(self):
        return bool(self.images or self.audios or self.videos or self.objects)

    @classmethod
    def from_dict(cls, inputs: Dict[str, Any]) -> Tuple['StdTemplateInputs', Dict[str, Any]]:
        kwargs = {}
        for key in ['rejected_response', 'label', 'channel']:
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
            messages=messages, system=system, tools=tools, objects=objects, **kwargs, **media_kwargs), extra_kwargs

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

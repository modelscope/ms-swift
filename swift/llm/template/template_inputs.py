# Copyright (c) Alibaba, Inc. and its affiliates.
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

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
    tools: Organize tools into the format of tools_prompt for system. for example, 'react_en'.
        Specifying this parameter will override system.
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

    rejected_response: Optional[str] = None
    label: Optional[int] = None

    images: List[Union[str, Image.Image]] = field(default_factory=list)
    audios: List[str] = field(default_factory=list)
    videos: List[str] = field(default_factory=list)
    objects: Dict[str, List[Any]] = field(default_factory=dict)

    agent_keyword: Optional[Dict[str, str]] = None

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
        if self.agent_keyword is None:
            self.agent_keyword = {}

    def to_history(self):
        if not self.messages:
            return None
        return messages_to_history(self.messages)

    @property
    def is_multimodal(self):
        return bool(self.images or self.audios or self.videos or self.objects)

    @classmethod
    def from_dict(cls, inputs: Dict[str, Any], *, tools_prompt: str = 'react_en') -> 'StdTemplateInputs':
        from swift.plugin import get_tools_prompt, get_tools_keyword
        kwargs = {}
        for key in ['rejected_response', 'label']:
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

        keyword = None
        if tools is not None:
            if system is not None:
                logger.warning_once(
                    'You have tools prompt but you also have a system field, so the system field will be ignored')
            if isinstance(tools, str):
                tools = json.loads(tools)
            system = get_tools_prompt(tools, tools_prompt)
            keyword = get_tools_keyword(tools_prompt)

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

        StdTemplateInputs.messages_join_observation(messages, tools_prompt)
        return cls(messages=messages, system=system, objects=objects, agent_keyword=keyword, **kwargs, **media_kwargs)

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

    @staticmethod
    def messages_join_observation(messages: Messages, tools_prompt='react_en') -> None:
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
        from swift.plugin import get_tools_keyword
        keyword = get_tools_keyword(tools_prompt)
        while i < len(messages):
            pre_message, message = messages[i - 1], messages[i]
            pre_role, pre_content = pre_message['role'], pre_message['content']
            role, content = message['role'], message['content']
            if (pre_role == 'assistant' and role == 'tool' and isinstance(pre_content, str)
                    and pre_content.endswith(keyword.get('observation'))):
                assert isinstance(pre_content, str)
                pre_message['content'] = pre_content + content  # assistant
                messages.pop(i)  # remove tool
            elif (pre_role == 'assistant' and role == 'assistant' and isinstance(pre_content, str)
                  and isinstance(content, str)):
                # Consecutive messages from the assistant role need to be merged to prevent errors.
                pre_message['content'] = pre_content + content
                messages.pop(i)
            else:
                i += 1

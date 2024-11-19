from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

import json
from PIL import Image

from swift.utils import get_logger
from ..utils import messages_to_history

logger = get_logger()

Tool = Dict[str, Union[str, Dict]]
Message = Dict[str, Union[str, List[Dict[str, Any]]]]
Messages = List[Message]


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
    tools: Organize tools into the format of tools_prompt for system. for example, 'react_en'.
        Specifying this parameter will override system.
    """
    messages: Messages

    images: List[Union[str, Image.Image]] = field(default_factory=list)
    audios: List[str] = field(default_factory=list)
    videos: List[str] = field(default_factory=list)

    tools: Optional[List[Tool]] = None

    def __post_init__(self):
        for key in ['images', 'audios', 'videos']:
            val = getattr(self, key)
            if isinstance(val, str):
                setattr(self, key, [val])

        self._remove_response()

    def _remove_response(self):
        last_role = self.messages[-1]['role']
        if last_role == 'assistant':
            self.messages.pop()

    def copy(self):
        return self.__class__(
            messages=deepcopy(self.messages),
            images=self.images.copy(),
            audios=self.audios.copy(),
            videos=self.videos.copy(),
            tools=deepcopy(self.tools))


@dataclass
class TemplateInputs(InferRequest):
    """
    objects: Used for grounding tasks in a general format.
    """
    rejected_response: Optional[str] = None
    label: Optional[bool] = None
    objects: Union[str, None, List[Dict[str, Any]]] = None  # List[Dict[str, Any]]

    def __post_init__(self):
        # Format objects(groundings/refs) to json
        if isinstance(self.objects, str):
            # reload grounding from str
            self.objects = json.loads(self.objects)
        elif self.objects is None:
            self.objects = []

    def copy(self):
        res = super().copy()
        res.rejected_response = self.rejected_response
        res.label = self.label
        res.objects = deepcopy(self.objects)
        return res


@dataclass
class StdTemplateInputs:
    # only user/tool/assistant
    messages: List[Dict[str, str]]
    # None: use default system; '': not use system
    system: Optional[str] = None

    rejected_response: Optional[str] = None
    label: Optional[bool] = None

    images: List[Union[str, Image.Image]] = field(default_factory=list)
    audios: List[str] = field(default_factory=list)
    videos: List[str] = field(default_factory=list)
    objects: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        self.image_idx = 0
        self.audio_idx = 0
        self.video_idx = 0
        self.object_idx = 0
        self.box_idx = 0
        if self.images and not isinstance(self.images, list):
            self.images = [self.images]
        if self.videos and not isinstance(self.videos, list):
            self.videos = [self.videos]
        if self.audios and not isinstance(self.audios, list):
            self.audios = [self.audios]

    def copy(self):
        return self.__class__(
            messages=deepcopy(self.messages),
            system=self.system,
            rejected_response=self.rejected_response,
            label=self.label,
            images=self.images.copy(),
            audios=self.audios.copy(),
            videos=self.videos.copy(),
            objects=self.objects.copy())

    def to_history(self):
        if not self.messages:
            return None
        return messages_to_history(self.messages)

    @property
    def is_multimodal(self):
        return bool(self.images or self.audios or self.videos or self.objects)

    @classmethod
    def from_dict(cls, inputs: Dict[str, Any], *, tools_prompt: str = 'react_en') -> 'StdTemplateInputs':
        from .agent import get_tools_prompt
        messages = deepcopy(inputs['messages'])
        rejected_response = inputs.get('rejected_response')
        tools = deepcopy(inputs.get('tools'))
        objects = deepcopy(inputs.get('objects') or [])

        assert len(messages) >= 1

        if messages[0]['role'] == 'system':
            message = messages.pop(0)
            system = message['content']
        else:
            system = None

        if tools is not None:
            if system is not None:
                logger.warning_once('You have tools prompt but you also have a system field, which will be ignored')
            if isinstance(tools, str):
                tools = json.loads(tools)
            system = get_tools_prompt(tools, tools_prompt)

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

        StdTemplateInputs.messages_join_observation(messages)
        return cls(messages, rejected_response, system, **media_kwargs, objects=objects)

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
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from swift.template import Messages, Tool


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

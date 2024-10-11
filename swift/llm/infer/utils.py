from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from swift.llm import Messages


@dataclass
class InferRequest:
    """
    messages: Input in messages format.
        Examples: [{
            "role": "user",  # or assistant/system/role
            "content": [  # str or List[Dict[str, Any]]
                {
                    "type": "image",  # or audio/video
                    # This content can also be written in the `images` field
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

    images: Optional[List[Union[Image.Image, str]]] = None
    audios: Optional[List[str]] = None
    videos: Optional[List[str]] = None

    objects: Union[str, None, List[Dict[str, Any]]] = None
    tools: Optional[List[Dict[str, Union[str, Dict]]]] = None

# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass
from typing import Optional

from swift.llm import InferArguments


@dataclass
class WebUIArguments(InferArguments):
    """
    WebUIArguments is a dataclass that inherits from InferArguments.

    Args:
        host (str): The hostname or IP address to bind the web UI server to. Default is '0.0.0.0'.
        port (int): The port number to bind the web UI server to. Default is 7860.
        share (bool): A flag indicating whether to share the web UI publicly. Default is False.
        lang (str): The language setting for the web UI. Default is 'zh'.
        studio_title(str): The title of the chat studio when specify `--model` or `--adapters`.
    """
    host: str = '0.0.0.0'
    port: int = 7860
    share: bool = False
    lang: str = 'zh'
    studio_title: Optional[str] = None

    def __post_init__(self):
        if self.model or self.adapters or self.ckpt_dir:
            super().__post_init__()
            if self.studio_title is None:
                self.studio_title = self.model_suffix

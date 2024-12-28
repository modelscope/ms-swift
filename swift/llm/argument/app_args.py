# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass
from typing import Literal, Optional

from swift.utils import get_logger
from .deploy_args import DeployArguments

logger = get_logger()


@dataclass
class AppArguments(DeployArguments):
    base_url: Optional[str] = None
    studio_title: Optional[str] = None

    server_name: str = '0.0.0.0'
    server_port: int = 7860
    share: bool = False
    lang: Literal['en', 'zh'] = 'zh'

    def _init_torch_dtype(self) -> None:
        if self.base_url:
            return
        super()._init_torch_dtype()

    def __post_init__(self):
        super().__post_init__()
        if self.studio_title is None:
            self.studio_title = self.model_suffix

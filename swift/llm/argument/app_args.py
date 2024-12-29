# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass
from typing import Literal, Optional

from swift.utils import find_free_port, get_logger
from ..template import get_template_meta
from .deploy_args import DeployArguments
from .webui_args import WebUIArguments

logger = get_logger()


@dataclass
class AppArguments(WebUIArguments, DeployArguments):
    base_url: Optional[str] = None
    studio_title: Optional[str] = None

    lang: Literal['en', 'zh'] = 'en'

    def _init_torch_dtype(self) -> None:
        if self.base_url:
            return
        super()._init_torch_dtype()

    def __post_init__(self):
        super().__post_init__()
        self.server_port = find_free_port(self.server_port)
        if self.studio_title is None:
            self.studio_title = self.model_suffix
        if self.system is None:
            self.system = get_template_meta(self.model_meta.template).default_system

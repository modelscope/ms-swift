# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass
from typing import Literal, Optional

from swift.utils import find_free_port, get_logger
from ..model import get_matched_model_meta
from ..template import get_template_meta
from .deploy_args import DeployArguments
from .webui_args import WebUIArguments

logger = get_logger()


@dataclass
class AppArguments(WebUIArguments, DeployArguments):
    base_url: Optional[str] = None
    studio_title: Optional[str] = None
    is_multimodal: Optional[bool] = None

    lang: Literal['en', 'zh'] = 'en'
    verbose: bool = False

    def _init_torch_dtype(self) -> None:
        if self.base_url:
            self.model_meta = get_matched_model_meta(self.model)
            return
        super()._init_torch_dtype()

    def __post_init__(self):
        super().__post_init__()
        self.server_port = find_free_port(self.server_port)
        if self.model_meta:
            if self.system is None:
                self.system = get_template_meta(self.model_meta.template).default_system
            if self.is_multimodal is None:
                self.is_multimodal = self.model_meta.is_multimodal
        if self.is_multimodal is None:
            self.is_multimodal = False

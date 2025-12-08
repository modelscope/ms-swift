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
    """Arguments for configuring the Web UI inference.

    This dataclass inherits from WebUIArguments and DeployArguments, combining their settings to configure the user
    interface for model inference.

    Args:
        base_url (Optional[str]): The base URL for the model deployment API, e.g., `http://localhost:8000/v1`. If set
            to `None`, a local deployment will be used instead. Defaults to None.
        studio_title (Optional[str]): The title for the Web UI studio. If set to `None`, the title will default to the
            model's name. Defaults to None.
        is_multimodal (Optional[bool]): Whether to launch the multimodal version of the application. If `None`, the
            app will attempt to auto-detect this setting based on the model. If auto-detection is not possible, it
            defaults to `False`. Defaults to None.
        lang (str): Overrides the language setting for the Web UI. Defaults to 'en'.
        verbose (bool): Whether to log detailed request information. Defaults to False.
        stream (bool): Whether to enable streaming output for model responses. Defaults to True.
    """
    base_url: Optional[str] = None
    studio_title: Optional[str] = None
    is_multimodal: Optional[bool] = None

    lang: Literal['en', 'zh'] = 'en'
    verbose: bool = False
    stream: bool = True

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

# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass
from typing import Optional

from swift.utils import get_logger
from .deploy_args import DeployArguments

logger = get_logger()


@dataclass
class AppArguments(DeployArguments):
    api_url: Optional[str] = None
    studio_title: Optional[str] = None


    def _init_torch_dtype(self) -> None:
        if self.api_url:
            return
        super()._init_torch_dtype()

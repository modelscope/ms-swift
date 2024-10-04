# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass
from typing import Optional

from .infer_args import InferArguments


@dataclass
class WebuiArguments:
    share: bool = False
    lang: str = 'zh'
    host: str = '127.0.0.1'
    port: Optional[int] = None


@dataclass
class AppUIArguments(InferArguments):
    host: str = '127.0.0.1'
    port: int = 7860
    share: bool = False
    # compatibility. (Deprecated)
    server_name: Optional[str] = None
    server_port: Optional[int] = None

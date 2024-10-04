# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass
from typing import Optional

from .infer_args import InferArguments


@dataclass
class WebuiArguments:
    host: str = '127.0.0.1'
    port: Optional[int] = None
    share: bool = False
    lang: str = 'zh'

@dataclass
class AppUIArguments(InferArguments):
    host: str = '127.0.0.1'
    port: int = 7860
    share: bool = False

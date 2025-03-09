# Copyright (c) Alibaba, Inc. and its affiliates.
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

from swift.llm import BaseArguments
from .megatron_args import MegatronArguments


@dataclass
class MegatronTrainArguments(MegatronArguments, BaseArguments):

    def __post_init__(self):
        BaseArguments.__post_init__(self)
        MegatronArguments.__post_init__(self)

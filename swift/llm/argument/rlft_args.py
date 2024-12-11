# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass

from swift.llm import RLHFArguments


@dataclass
class RLFTArguments(RLHFArguments):

    reward_type: str = 'agent'

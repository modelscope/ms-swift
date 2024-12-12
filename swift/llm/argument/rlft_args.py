# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass

from swift.llm import RLHFArguments


@dataclass
class RLFTArguments(RLHFArguments):

    reward_type: str = 'agent'

    def __post_init__(self):
        self.rlhf_type = 'ppo'
        super().__post_init__()
        self.rlhf_type = 'rlft'

# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass

from swift.llm import RLHFArguments


@dataclass
class RLFTArguments(RLHFArguments):

    reward_model: str = "skywork/Skywork-Reward-Models"
    gt_ratio: float = 0.0

    def __post_init__(self):
        self.rlhf_type = 'dpo'
        super().__post_init__()
        self.rlhf_type = 'rlft'
        self.training_args.max_new_tokens = self.max_new_tokens

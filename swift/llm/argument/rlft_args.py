# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass
from typing import Optional, Literal

from swift.llm import RLHFArguments


@dataclass
class RLFTArguments(RLHFArguments):

    rlft_type: Literal['causal_lm', 'dpo'] = 'dpo'

    prm_model: str = "AI-ModelScope/GRM_Llama3.1_8B_rewardmodel-ft"
    orm_model: Optional[str] = None

    # sample/mcts/dvts/xxx
    sampler_type: str = 'sample'
    sampler_output: str = 'rollout_output'
    num_return_sequences: int = 10

    num_rollout_iters: int = 50
    num_rollout_batches: int = 300

    temperature: float = 0.4
    end_temperature: float = 1.2
    start_threshold: float = 0.0
    end_threshold: float = -5.0

    def __post_init__(self):
        self.rlhf_type = self.rlft_type
        self.padding_side = 'left'
        super().__post_init__()
        self.training_args.max_new_tokens = self.max_new_tokens

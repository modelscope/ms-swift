# Copyright (c) Alibaba, Inc. and its affiliates.
import dataclasses
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Literal, Optional

import json

from swift.llm import BaseArguments
from swift.utils import get_logger

logger = get_logger()


@dataclass
class SamplingArguments(BaseArguments):
    # rm models
    prm_model: Optional[str] = None
    orm_model: Optional[str] = None

    # sampler settings
    # sample/mcts/dvts/xxx
    sampler_type: Literal['sample', 'mcts', 'distill'] = 'sample'
    sampler_engine: Literal['pt', 'lmdeploy', 'vllm', 'no', 'client'] = 'pt'
    output_dir: str = 'sample_output'
    output_file: Optional[str] = None
    resume: bool = False
    override_exist_file: bool = False
    num_return_sequences: int = 64
    num_sampling_per_gpu_batch_size: int = 1
    num_sampling_per_gpu_batches: Optional[int] = None
    n_best_to_keep: int = 5
    data_range: List[int] = dataclasses.field(default_factory=list)

    # generate settings
    temperature: float = 1.0
    prm_threshold: float = 0.0
    easy_query_threshold: Optional[float] = None

    # engine settings
    engine_kwargs: Optional[str] = None

    # Vanilla
    cache_files: List[str] = dataclasses.field(default_factory=list)

    # MCTS
    rollout_depth: int = 5
    rollout_start_depth: int = 3
    max_iterations: int = 100
    process_reward_rate: float = 0.0
    exploration_rate: float = 0.5
    api_key: str = 'EMPTY'
    base_url: str = 'https://dashscope.aliyuncs.com/compatible-mode/v1'

    def _init_model_info(self):
        if self.sampler_engine != 'client':
            return super()._init_model_info()
        self.task_type = 'causal_lm'
        return

    def __post_init__(self):
        if self.output_file is None:
            now = datetime.now()
            formatted_time = now.strftime('%Y-%m-%d-%H-%M-%S')
            self.output_file = formatted_time + '.jsonl'
            logger.info(f'Setting output_file to {self.output_file}')
        else:
            if '/' in self.output_file or '\\' in self.output_file:
                raise ValueError(f'Please use a string prefix without directory to '
                                 f'`--output_file` but now is: {self.output_file}')
        self.padding_side = 'left'
        if self.engine_kwargs is not None:
            print(self.engine_kwargs)
            self.engine_kwargs = json.loads(self.engine_kwargs)
        else:
            self.engine_kwargs = {}

        super().__post_init__()

        if self.system is not None:
            self.system_message = [{
                'role': 'system',
                'content': self.system,
            }]
        else:
            self.system_message = []

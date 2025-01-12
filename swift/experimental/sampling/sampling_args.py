# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass
import dataclasses
from typing import Optional, Literal
from datetime import datetime
from swift.llm import BaseArguments
from swift.utils import get_logger
import json
from typing import List
logger = get_logger()


@dataclass
class SamplingArguments(BaseArguments):
    # rm models
    prm_model: str = "AI-ModelScope/GRM-llama3.2-3B-rewardmodel-ft"
    orm_model: Optional[str] = None

    # sampler settings
    # sample/mcts/dvts/xxx
    sampler_type: str = 'sample'
    sampler_engine: Literal['pt', 'lmdeploy', 'vllm'] = 'pt'
    output_dir: str = 'sample_output'
    file_prefix: Optional[str] = None
    override_exist_file: bool = False
    num_return_sequences: int = 64
    num_sampling_per_gpu_batch_size: int = 2
    num_sampling_per_gpu_batches: Optional[int] = None
    n_best_to_keep: int = 5
    data_range: List[int] = dataclasses.field(default_factory=list)

    # generate settings
    temperature: float = 1.0
    prm_threshold: float = 0.0
    easy_query_threshold: Optional[float] = None

    # engine settings
    engine_kwargs: Optional[str] = None

    def __post_init__(self):
        if self.file_prefix is None:
            now = datetime.now()
            formatted_time = now.strftime("%Y-%m-%d-%H-%M-%S")
            self.file_prefix = formatted_time
            logger.info(f'Setting file_prefix to {self.file_prefix}')
        else:
            if '/' in self.file_prefix or '\\' in self.file_prefix:
                raise ValueError(f'Please use a string prefix without directory to '
                                 f'`--file_prefix` but now is: {self.file_prefix}')
        self.padding_side = 'left'
        if self.engine_kwargs is not None:
            print(self.engine_kwargs)
            self.engine_kwargs = json.loads(self.engine_kwargs)
        else:
            self.engine_kwargs = {}
        super().__post_init__()

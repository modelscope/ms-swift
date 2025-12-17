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
    """A dataclass for configuring sampling parameters.

    Args:
        prm_model (Optional[str]): The type of the Process Reward Model (PRM). Can be a model ID (loaded via 'pt'
            engine) or a PRM key defined in a plugin for custom inference. Defaults to None.
        orm_model (Optional[str]): The type of the Outcome Reward Model (ORM). Typically a wildcard or test case,
            usually defined in a plugin. Defaults to None.
        sampler_type (Literal['sample', 'distill']): The type of sampling to perform. Supported types are 'sample' and
            'distill'. Defaults to 'sample'.
        sampler_engine (Literal['pt', 'lmdeploy', 'vllm', 'no', 'client']): The inference engine for the sampling
            model. Supported options are 'pt', 'lmdeploy', 'vllm', 'client', and 'no'. Defaults to 'pt'.
        output_dir (str): The directory to save the output files. Defaults to 'sample_output'.
        output_file (Optional[str]): The name of the output file. If None, a timestamp will be used as the filename.
            The path should not be included, only the filename. Only the '.jsonl' format is supported. Defaults to
            None.
        resume (bool): Whether to resume file. Defaults to False.
        override_exist_file (bool): Whether to override the output file if it already exists. This is only effective
            when `output_file` is specified. Defaults to False.
        num_return_sequences (int): The number of raw sequences to return from sampling. Effective for the 'sample'
            `sampler_type`. Defaults to 64.
        num_sampling_batch_size (int): The batch size for each sampling iteration. Defaults to 1.
        num_sampling_batches (Optional[int]): The total number of batches to sample. Defaults to None.
        n_best_to_keep (int): The number of best sequences to keep after evaluation. Defaults to 5.
        data_range (List[int]): Specifies the data shard to process. A list of two integers `[shard_index,
            num_shards]`. For example, `[1, 3]` means the dataset is split into 3 shards and this process handles the
            second shard (0-indexed). Defaults to [].
        temperature (float): The temperature for sampling. Defaults to 1.0.
        prm_threshold (float): The threshold for the Process Reward Model (PRM). Results with a score below this
            threshold will be filtered out. Defaults to 0.0.
        easy_query_threshold (Optional[float]): For a single query, if the proportion of correctly sampled sequences
            (as evaluated by the ORM) is greater than this threshold, the query will be discarded. This prevents overly
            simple queries from appearing in the final results. Defaults to None, which disables this filter.
        engine_kwargs (Optional[str]): Additional arguments to pass to the `sampler_engine`, provided as a JSON string.
            For example: '{"cache_max_entry_count":0.7}'. Defaults to None.
        cache_files (List[str]): A list of cache files for a two-step sampling process to avoid OOM errors.
            Step 1: Set `prm_model`, and `orm_model` to None. All generated sequences are saved to a file.
            Step 2: Set `sampler_engine` to 'no' and provide the output file from Step 1 to `cache_files`.
            This run will perform PRM and ORM evaluation on the cached results.
            Note: The `--dataset` argument must still be provided, as IDs in the cache files are MD5 hashes of the
            original data and need to be linked.
    """
    # rm models
    prm_model: Optional[str] = None
    orm_model: Optional[str] = None

    # sampler settings
    sampler_type: Literal['sample', 'distill'] = 'sample'
    sampler_engine: Literal['pt', 'lmdeploy', 'vllm', 'no', 'client'] = 'pt'
    output_dir: str = 'sample_output'
    output_file: Optional[str] = None
    resume: bool = False
    override_exist_file: bool = False
    num_return_sequences: int = 64
    num_sampling_batch_size: int = 1
    num_sampling_batches: Optional[int] = None
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

    def _init_model_info(self):
        if self.sampler_engine != 'client':
            return super()._init_model_info()
        else:
            self.model_info = None
            self.model_meta = None
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

import math
import os
from dataclasses import dataclass

from swift.llm import BaseArguments
from swift.utils import get_logger
from ..model import get_megatron_model_meta
from ..utils import convert_hf_config
from .megatron_args import MegatronArguments

logger = get_logger()


@dataclass
class MegatronBaseArguments(MegatronArguments, BaseArguments):

    def __post_init__(self):
        os.environ['SWIFT_USE_MEGATRON'] = '1'
        self.sequence_parallel_size = self.context_parallel_size
        if self.packing:
            self.padding_free = True
        BaseArguments.__post_init__(self)
        self.megatron_model_meta = get_megatron_model_meta(self.model_type)
        self.seq_length = self.seq_length or self.packing_length or self.max_length
        if self.streaming:
            self.dataloader_type = 'external'
            if self.num_workers > 1:
                self.num_workers = 1
                logger.info('Using streaming dataset, setting args.num_workers to 1.')

    def init_model_args(self, tokenizer, config):
        if self.task_type == 'seq_cls':
            self.problem_type = self.problem_type or getattr(config, 'problem_type', None)
            logger.info(f'args.problem_type: {self.problem_type}')
        kwargs = convert_hf_config(config)
        if self.new_special_tokens and kwargs['padded_vocab_size'] < len(tokenizer):
            kwargs['padded_vocab_size'] = math.ceil(len(tokenizer) / 128) * 128
            self.initialize_embedding = True
        logger.info(f'megatron_config: {kwargs}')
        for k, v in kwargs.items():
            if getattr(self, k) is None:
                setattr(self, k, v)
        MegatronArguments.__post_init__(self)
        self.extra_args = self.parse_to_megatron()
        self.extra_args['model_info'] = self.model_info
        self.extra_args['model_meta'] = self.model_meta
        self.extra_args['megatron_model_meta'] = self.megatron_model_meta

import math
from dataclasses import dataclass, fields

from swift.llm import BaseArguments
from swift.utils import get_logger
from ..model import get_megatron_model_meta
from ..utils import convert_hf_config
from .megatron_args import MegatronArguments

logger = get_logger()


@dataclass
class MegatronBaseArguments(MegatronArguments, BaseArguments):

    def __post_init__(self):
        self.sequence_parallel_size = self.context_parallel_size
        if self.packing:
            self.padding_free = True
        BaseArguments.__post_init__(self)
        self.megatron_model_meta = get_megatron_model_meta(self.model_type)
        assert self.megatron_model_meta is not None, f'Model: {self.model} is not supported.'
        self.seq_length = self.seq_length or self.packing_length or self.max_length
        if self.streaming:
            self.dataloader_type = 'external'
            if self.num_workers > 1:
                self.num_workers = 1
                logger.info('Using streaming dataset, setting args.num_workers to 1.')

    @staticmethod
    def _check_megatron_kwargs(kwargs):
        # Make sure that the keys in kwargs have default values of None in MegatronArguments.
        default_mapping = {field.name: field.default for field in fields(MegatronArguments)}
        for k in kwargs.keys():
            assert default_mapping[k] is None

    def init_model_args(self, tokenizer, config):
        if self.task_type == 'seq_cls':
            self.problem_type = self.problem_type or getattr(config, 'problem_type', None)
            logger.info(f'args.problem_type: {self.problem_type}')
        kwargs = convert_hf_config(config)
        self._check_megatron_kwargs(kwargs)
        if tokenizer is not None and self.new_special_tokens and kwargs['padded_vocab_size'] < len(tokenizer):
            kwargs['padded_vocab_size'] = math.ceil(len(tokenizer) / 128) * 128
            self.initialize_embedding = True
        if self.task_type == 'seq_cls':
            self.initialize_embedding = True
        logger.info(f'megatron_config: {kwargs}')
        for k, v in kwargs.items():
            if getattr(self, k) is None:
                setattr(self, k, v)
        MegatronArguments.__post_init__(self)
        self.extra_args = self.parse_to_megatron()

import math
from dataclasses import dataclass, fields

from swift.arguments import BaseArguments
from swift.megatron.model import get_megatron_model_meta
from swift.megatron.utils import convert_hf_config
from swift.utils import get_logger
from .megatron_args import MegatronArguments

logger = get_logger()


@dataclass
class MegatronBaseArguments(MegatronArguments, BaseArguments):

    def __post_init__(self):
        self.sequence_parallel_size = self.context_parallel_size
        if self.packing:
            self.padding_free = True
        BaseArguments.__post_init__(self)
        MegatronArguments.__post_init__(self)
        if self.streaming:
            self.dataloader_type = 'external'
            if self.num_workers > 1:
                self.num_workers = 1
                logger.info('Using streaming dataset, setting args.num_workers to 1.')

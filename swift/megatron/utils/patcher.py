from functools import wraps

import torch
import torch.distributed as dist
from megatron.training import get_args, global_vars, initialize, training

from swift.utils import get_dist_setting, get_logger

logger = get_logger()


def patch_megatron(tokenizer):

    def build_tokenizer(args):
        args.extra_vocab_size = args.padded_vocab_size - tokenizer.vocab_size
        return tokenizer

    global_vars.build_tokenizer = build_tokenizer

    _old_initialize_distributed = initialize._initialize_distributed

    @wraps(_old_initialize_distributed)
    def _initialize_distributed(*_args, **kwargs):
        args = get_args()
        if dist.is_initialized():
            args.rank, args.local_rank, args.world_size, args.local_world_size = get_dist_setting()
            torch.cuda.set_device(args.local_rank)
        return _old_initialize_distributed(*_args, **kwargs)

    initialize._initialize_distributed = _initialize_distributed

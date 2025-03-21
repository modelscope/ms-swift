# Copyright (c) Alibaba, Inc. and its affiliates.
from megatron.core.dist_checkpointing.strategies.torch import TorchDistSaveShardedStrategy
from megatron.training import get_args, global_vars, initialize, training

from swift.utils import get_logger

logger = get_logger()


def patch_megatron_tokenizer(tokenizer):

    def build_tokenizer(args):
        args.extra_vocab_size = args.padded_vocab_size - tokenizer.vocab_size
        return tokenizer

    global_vars.build_tokenizer = build_tokenizer


def patch_torch_dist_shard(thread_count):
    __init__ = TorchDistSaveShardedStrategy.__init__

    def __new_init__(*args, **kwargs):
        kwargs['thread_count'] = thread_count
        return __init__(*args, **kwargs)

    TorchDistSaveShardedStrategy.__init__ = __new_init__

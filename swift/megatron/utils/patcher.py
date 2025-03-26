# Copyright (c) Alibaba, Inc. and its affiliates.
from megatron.core.dist_checkpointing.strategies.torch import TorchDistSaveShardedStrategy
from megatron.training import get_args, global_vars, initialize, training

from swift.utils import get_logger

logger = get_logger()


def patch_megatron(tokenizer):
    # patch tokenizer
    def build_tokenizer(args):
        args.extra_vocab_size = args.padded_vocab_size - tokenizer.vocab_size
        return tokenizer

    global_vars.build_tokenizer = build_tokenizer

    patch_torch_dist_shard()


def patch_torch_dist_shard():
    __init__ = TorchDistSaveShardedStrategy.__init__

    def __new_init__(*_args, **kwargs):
        args = get_args()
        if args.thread_count is not None:
            kwargs['thread_count'] = args.thread_count
        return __init__(*_args, **kwargs)

    TorchDistSaveShardedStrategy.__init__ = __new_init__

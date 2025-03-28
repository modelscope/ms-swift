# Copyright (c) Alibaba, Inc. and its affiliates.
from megatron.core.dist_checkpointing.strategies.torch import TorchDistSaveShardedStrategy
from megatron.training import get_args, global_vars, initialize, training

from swift.utils import get_logger

logger = get_logger()


def patch_megatron(tokenizer):
    # patch tokenizer
    patch_tokenizer(tokenizer)
    patch_torch_dist_shard()
    patch_cyclic_iter()

def patch_tokenizer(tokenizer):
    def build_tokenizer(args):
        args.extra_vocab_size = args.padded_vocab_size - tokenizer.vocab_size
        return tokenizer

    global_vars.build_tokenizer = build_tokenizer


def patch_torch_dist_shard():
    __init__ = TorchDistSaveShardedStrategy.__init__

    def __new_init__(*_args, **kwargs):
        args = get_args()
        if args.thread_count is not None:
            kwargs['thread_count'] = args.thread_count
        return __init__(*_args, **kwargs)

    TorchDistSaveShardedStrategy.__init__ = __new_init__


def patch_cyclic_iter():
    from megatron.training import training
    def cyclic_iter(iter):
        args = get_args()
        n_epoch = 0
        while True:
            for x in iter:
                yield x
            logger.info(f'Epoch {n_epoch} has ended.')
            n_epoch += 1
            if args.max_epochs is not None and n_epoch >= args.max_epochs:
                break
    training.cyclic_iter = cyclic_iter

# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.distributed as dist
from megatron.training import get_args, global_vars, initialize, training

from swift.utils import get_dist_setting, get_logger, set_device

logger = get_logger()


def patch_megatron(tokenizer):

    def build_tokenizer(args):
        args.extra_vocab_size = args.padded_vocab_size - tokenizer.vocab_size
        return tokenizer

    global_vars.build_tokenizer = build_tokenizer

# Copyright (c) Alibaba, Inc. and its affiliates.
from contextlib import contextmanager

from megatron.training import get_args, global_vars, initialize, training


@contextmanager
def patch_megatron_data_collator(data_collator):
    origin_build_pretraining_data_loader = training.build_pretraining_data_loader

    def build_pretraining_data_loader(*_args, **kwargs):
        args = get_args()
        res = origin_build_pretraining_data_loader(*_args, **kwargs)
        if res is not None and args.dataloader_type != 'external':
            res.collate_fn = data_collator
        return res

    training.build_pretraining_data_loader = build_pretraining_data_loader
    try:
        yield
    finally:
        training.build_pretraining_data_loader = origin_build_pretraining_data_loader

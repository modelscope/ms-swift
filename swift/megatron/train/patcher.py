# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from contextlib import contextmanager
from functools import wraps

import torch
from megatron.training import get_args, global_vars, initialize, training

from swift.utils import JsonlWriter, is_master


@contextmanager
def patch_training_log():
    jsonl_writer = None
    origin_training_log = training.training_log

    @wraps(origin_training_log)
    def training_log(loss_dict, total_loss_dict, learning_rate, decoupled_learning_rate, iteration, loss_scale,
                     report_memory_flag, skipped_iter, grad_norm, params_norm, num_zeros_in_grad, *_args, **kwargs):
        nonlocal jsonl_writer
        args = get_args()
        if is_master() and iteration % args.log_interval == 0:
            logging_path = os.path.join(args.save, 'logging.jsonl')
            logs = {}
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                logs[k] = round(v, 8)
            logs['grad_norm'] = round(grad_norm, 8)
            logs['params_norm'] = round(params_norm, 8)
            logs['learning_rate'] = round(learning_rate, 8)
            logs['consumed_samples'] = args.consumed_train_samples
            logs['global_step/max_steps'] = f'{iteration}/{args.train_iters}'
            if jsonl_writer is None:
                jsonl_writer = JsonlWriter(logging_path, enable_async=True)
            jsonl_writer.append(logs)
        return origin_training_log(loss_dict, total_loss_dict, learning_rate, decoupled_learning_rate, iteration,
                                   loss_scale, report_memory_flag, skipped_iter, grad_norm, params_norm,
                                   num_zeros_in_grad, *_args, **kwargs)

    training.training_log = training_log
    try:
        yield
    finally:
        training.training_log = origin_training_log


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

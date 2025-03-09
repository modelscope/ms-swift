# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from functools import wraps

import torch
from megatron.training import get_args, global_vars, initialize, training

from swift.utils import append_to_jsonl, is_master


def patch_training_log():
    _old_training_log = training.training_log

    @wraps(_old_training_log)
    def training_log(loss_dict, total_loss_dict, learning_rate, decoupled_learning_rate, iteration, loss_scale,
                     report_memory_flag, skipped_iter, grad_norm, params_norm, num_zeros_in_grad, *_args, **kwargs):
        args = get_args()
        if is_master() and iteration % args.log_interval == 0:
            logging_path = os.path.join(args.save, 'logging.jsonl')
            logs = {}
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                logs[k] = round(v, 8)
            logs['grad_norm'] = round(grad_norm, 8)
            logs['learning_rate'] = round(learning_rate, 8)
            logs['consumed_samples'] = args.consumed_train_samples
            logs['global_step/max_steps'] = f'{iteration}/{args.train_iters}'

            append_to_jsonl(logging_path, logs)
        return _old_training_log(loss_dict, total_loss_dict, learning_rate, decoupled_learning_rate, iteration,
                                 loss_scale, report_memory_flag, skipped_iter, grad_norm, params_norm,
                                 num_zeros_in_grad, *_args, **kwargs)

    training.training_log = training_log

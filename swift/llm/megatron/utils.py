# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import sys
from functools import partial, wraps
from typing import Any, Dict, List, Mapping, Optional

import torch
import torch.distributed as dist

from swift.llm import LazyLLMDataset, Template, git_clone_github, is_megatron_available
from swift.utils import append_to_jsonl, get_dist_setting, get_logger, is_master, subprocess_run

logger = get_logger()


def init_megatron_env() -> None:
    if 'MEGATRON_LM_PATH' not in os.environ:
        megatron_path = git_clone_github(
            'https://github.com/NVIDIA/Megatron-LM', commit_hash='6dbe4cf699880038b1e5cd90b23ee71053c7f2ee')
    else:
        megatron_path = os.environ['MEGATRON_LM_PATH']
    if not is_megatron_available():
        subprocess_run(['pip', 'install', '-e', megatron_path])
    sys.path.append(megatron_path)

    if 'PAI_MEGATRON_PATCH_PATH' not in os.environ:
        megatron_patch_path = git_clone_github(
            'https://github.com/alibaba/Pai-Megatron-Patch', commit_hash='6fd5d050b240fd959f0ba69f1e9cd9a053e5a81d')
    else:
        megatron_patch_path = os.environ['PAI_MEGATRON_PATCH_PATH']
    sys.path.append(megatron_patch_path)

    # rename qwen1.5->qwen1_5 files
    qwen1_5_folders = ['toolkits/model_checkpoints_convertor/qwen']
    for folder in qwen1_5_folders:
        dir_path = os.path.join(megatron_patch_path, folder)
        for fname in os.listdir(dir_path):
            old_path = os.path.join(dir_path, fname)
            new_path = os.path.join(dir_path, fname.replace('qwen1.', 'qwen1_'))
            if old_path != new_path:
                try:
                    shutil.move(old_path, new_path)
                except FileNotFoundError:
                    pass


def patch_megatron(tokenizer):

    def build_tokenizer(args):
        args.extra_vocab_size = args.padded_vocab_size - tokenizer.vocab_size
        return tokenizer

    from megatron.training import get_args, training, initialize, global_vars
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

    _old_load_state_dict = torch.nn.Module.load_state_dict

    @wraps(_old_load_state_dict)
    def _load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, *args, **kwargs):
        if strict:
            keys = self.state_dict().keys() ^ state_dict.keys()
            new_keys = [k for k in keys if not k.endswith('_extra_state')]
            if keys and not new_keys:
                strict = False
        return _old_load_state_dict(self, state_dict, strict, *args, **kwargs)

    torch.nn.Module.load_state_dict = _load_state_dict

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


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function. copy from Pai-Megatron-Patch

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
    """
    from megatron.training import get_args
    from megatron.core import mpu
    from megatron.training.utils import average_losses_across_data_parallel_group
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    if args.context_parallel_size > 1:
        loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), loss_mask.sum().view(1)])
        dist.all_reduce(loss, group=mpu.get_context_parallel_group())
        loss = loss[0] / loss[1]
    else:
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = dist.get_rank()
        assert not loss.isnan(), (f'Rank {global_rank}: found NaN in local forward loss calculation. '
                                  f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}')

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss * args.context_parallel_size, {'loss': averaged_loss[0]}


def get_batch_on_this_tp_rank(data_iterator):
    # copy from Megatron-LM and made some changes.
    from megatron.training import get_args
    from megatron.core import mpu
    args = get_args()

    def _broadcast(item):
        if item is not None:
            dist.broadcast(item, mpu.get_tensor_model_parallel_src_rank(), group=mpu.get_tensor_model_parallel_group())

    if mpu.get_tensor_model_parallel_rank() == 0:

        if data_iterator is not None:
            data = next(data_iterator)
        else:
            data = None
        args.seq_length = data['tokens'].shape[1]
        _broadcast(torch.tensor(args.seq_length).cuda(non_blocking=True))
        batch = {
            'tokens': data['tokens'].cuda(non_blocking=True),
            'labels': data['labels'].cuda(non_blocking=True),
            'loss_mask': data['loss_mask'].cuda(non_blocking=True),
            'attention_mask': None if 'attention_mask' not in data else data['attention_mask'].cuda(non_blocking=True),
            'position_ids': data['position_ids'].cuda(non_blocking=True)
        }

        if args.pipeline_model_parallel_size == 1:
            _broadcast(batch['tokens'])
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['position_ids'])

        elif mpu.is_pipeline_first_stage():
            _broadcast(batch['tokens'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['position_ids'])

        elif mpu.is_pipeline_last_stage():
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])

    else:
        seq_length = torch.empty((), dtype=torch.int64, device=torch.cuda.current_device())
        _broadcast(seq_length)
        args.seq_length = seq_length.item()
        tokens = torch.empty((args.micro_batch_size, args.seq_length),
                             dtype=torch.int64,
                             device=torch.cuda.current_device())
        labels = torch.empty((args.micro_batch_size, args.seq_length),
                             dtype=torch.int64,
                             device=torch.cuda.current_device())
        loss_mask = torch.empty((args.micro_batch_size, args.seq_length),
                                dtype=torch.float32,
                                device=torch.cuda.current_device())
        if args.create_attention_mask_in_dataloader:
            attention_mask = torch.empty((args.micro_batch_size, 1, args.seq_length, args.seq_length),
                                         dtype=torch.bool,
                                         device=torch.cuda.current_device())
        else:
            attention_mask = None
        position_ids = torch.empty((args.micro_batch_size, args.seq_length),
                                   dtype=torch.int64,
                                   device=torch.cuda.current_device())

        if args.pipeline_model_parallel_size == 1:
            _broadcast(tokens)
            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)
            _broadcast(position_ids)

        elif mpu.is_pipeline_first_stage():
            labels = None
            loss_mask = None

            _broadcast(tokens)
            _broadcast(attention_mask)
            _broadcast(position_ids)

        elif mpu.is_pipeline_last_stage():
            tokens = None
            position_ids = None

            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)

        batch = {
            'tokens': tokens,
            'labels': labels,
            'loss_mask': loss_mask,
            'attention_mask': attention_mask,
            'position_ids': position_ids
        }

    return batch


def forward_step(data_iterator, model):
    from megatron.training.utils import get_batch_on_this_cp_rank
    batch = get_batch_on_this_tp_rank(data_iterator)
    batch = get_batch_on_this_cp_rank(batch)
    tokens, labels, loss_mask, attention_mask, position_ids = batch.values()
    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples, train_dataset: LazyLLMDataset,
                                       val_dataset: LazyLLMDataset, template: Template):
    # train_val_test_num_samples: ignored
    from megatron.training import training
    from megatron.training.utils import get_ltor_masks_and_position_ids

    assert not hasattr(training, '_old_build_pretraining_data_loader')
    _old_build_pretraining_data_loader = training.build_pretraining_data_loader

    def data_collator(batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = template.data_collator(batch, padding_to)
        labels = res['labels']
        new_labels = torch.zeros_like(labels)
        new_labels[:, :-1] = labels[:, 1:]
        new_labels[:, -1] = -100
        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(new_labels, -100, False, False, True)
        return {
            'tokens': res['input_ids'],
            'labels': new_labels,
            'attention_mask': attention_mask,
            'loss_mask': loss_mask,
            'position_ids': position_ids
        }

    @wraps(_old_build_pretraining_data_loader)
    def build_pretraining_data_loader(*args, **kwargs):
        res = _old_build_pretraining_data_loader(*args, **kwargs)
        if res is not None:
            res.collate_fn = data_collator
        return res

    training.build_pretraining_data_loader = build_pretraining_data_loader
    training._old_build_pretraining_data_loader = _old_build_pretraining_data_loader
    return train_dataset, val_dataset, None

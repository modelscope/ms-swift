from typing import Any

import datasets
import torch
import torch.distributed as dist
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers.trainer_utils import seed_worker


def assert_xtuner_runtime_condition():
    from swift.llm.utils.utils import is_xtuner_available
    assert is_xtuner_available(), \
        ('Please install XTuner first to pack dataset to `max_length`.'
         '`pip install -U \'xtuner[deepspeed]\'`')
    assert dist.is_initialized(), 'pack_to_max_length is only available with distributed training.'


def pack_dataset_xtuner(dataset: Dataset, args: Any) -> Any:
    assert_xtuner_runtime_condition()
    if dist.get_rank() == 0:
        ds = [i[0] for i in dataset.data]
        train_dataset = Dataset.from_list(ds)
        from xtuner.dataset.huggingface import pack_dataset
        train_dataset = pack_dataset(
            train_dataset, max_length=args.max_length, use_varlen_attn=False, shuffle_before_pack=True, map_num_proc=16)
        objects = [train_dataset]
        train_dataset.save_to_disk('alpaca_pack')
    else:
        objects = [None]
    dist.broadcast_object_list(objects, src=0)
    train_dataset = objects[0]
    return train_dataset


def init_sequence_parallel_xtuner(sequence_parallel_size: int):
    assert_xtuner_runtime_condition()
    from xtuner.parallel.sequence import init_sequence_parallel
    init_sequence_parallel(sequence_parallel_size)


def dispatch_module_xtuner(module):
    assert_xtuner_runtime_condition()
    from xtuner.model.modules.dispatch import dispatch_modules
    dispatch_modules(module)


def pad_and_split_for_sequence_parallel(tokenizer, input_ids, labels, position_ids, attention_mask, loss_scale):
    assert_xtuner_runtime_condition()
    from xtuner.parallel.sequence import (pad_for_sequence_parallel, split_for_sequence_parallel,
                                          get_sequence_parallel_group)
    input_ids = pad_for_sequence_parallel(input_ids, padding_value=tokenizer.pad_token_id, dim=-1)
    labels = pad_for_sequence_parallel(labels, padding_value=-100, dim=-1)
    position_ids = pad_for_sequence_parallel(position_ids, padding_value=0, dim=-1)
    attention_mask = pad_for_sequence_parallel(attention_mask, padding_value=0, dim=-1)

    sp_group = get_sequence_parallel_group()
    input_ids = split_for_sequence_parallel(input_ids, dim=1, sp_group=sp_group)
    labels = split_for_sequence_parallel(labels, dim=1, sp_group=sp_group)
    position_ids = split_for_sequence_parallel(position_ids, dim=1, sp_group=sp_group)
    attention_mask = split_for_sequence_parallel(attention_mask, dim=-1, sp_group=sp_group)
    if loss_scale is not None:
        loss_scale = pad_for_sequence_parallel(loss_scale, padding_value=0., dim=-1)
        loss_scale = split_for_sequence_parallel(loss_scale, dim=1, sp_group=sp_group)

    return input_ids, labels, position_ids, attention_mask, loss_scale


def get_xtuner_sequence_parallel_world_size():
    assert_xtuner_runtime_condition()
    from xtuner.parallel.sequence import get_sequence_parallel_world_size
    return get_sequence_parallel_world_size()


def reduce_xtuner_sequence_parallel_loss(loss, labels):
    from xtuner.parallel.sequence import (reduce_sequence_parallel_loss, get_sequence_parallel_group)
    # reduce loss for logging correctly
    num_tokens = (labels != -100).sum()
    return reduce_sequence_parallel_loss(loss, num_tokens, get_sequence_parallel_group())


def get_xtuner_train_dataloader(trainer):
    # modified from HFTrainer.get_train_dataloader
    # RandomSampler -> SequenceParallelSampler
    assert_xtuner_runtime_condition()
    if trainer.train_dataset is None:
        raise ValueError('Trainer: training requires a train_dataset.')

    train_dataset = trainer.train_dataset
    data_collator = trainer.data_collator
    if isinstance(train_dataset, datasets.Dataset):
        train_dataset = trainer._remove_unused_columns(train_dataset, description='training')
    else:
        data_collator = trainer._get_collator_with_removed_columns(data_collator, description='training')

    dataloader_params = {
        'batch_size': trainer._train_batch_size,
        'collate_fn': data_collator,
        'num_workers': trainer.args.dataloader_num_workers,
        'pin_memory': trainer.args.dataloader_pin_memory,
        'persistent_workers': trainer.args.dataloader_persistent_workers,
    }

    if not isinstance(train_dataset, torch.utils.data.IterableDataset):
        from xtuner.parallel import SequenceParallelSampler
        dataloader_params['sampler'] = SequenceParallelSampler(train_dataset, seed=1024)
        dataloader_params['drop_last'] = trainer.args.dataloader_drop_last
        dataloader_params['worker_init_fn'] = seed_worker

    return DataLoader(train_dataset, **dataloader_params)

# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys
from collections import OrderedDict
from typing import List

import safetensors
import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, trainer
from transformers.modeling_utils import unwrap_model

from swift.utils import get_logger

logger = get_logger()


# DataLoader
def get_bucket_sizes(max_length: int) -> List[int]:
    return [max_length // 4 * (i + 1) for i in range(4)]


def _get_closet_bucket(bucket_sizes, data_length):
    """Select the one from bucket_sizes that is closest in distance to
    data_length. This is required for TorchAcc.
    """
    cloest_length = sys.maxsize
    for b in bucket_sizes:
        if b == data_length or ((b < cloest_length) and (b > data_length)):
            cloest_length = b

    if cloest_length == sys.maxsize:
        bucket_sizes.append(data_length)
        cloest_length = data_length

    return cloest_length


def pad_and_split_batch(padding_to, input_ids, attention_mask, labels, loss_scale, max_length, tokenizer, rank,
                        world_size):
    if padding_to is None:
        longest_len = input_ids.shape[-1]
        bucket_sizes = get_bucket_sizes(max_length)
        bucket_data_length = _get_closet_bucket(bucket_sizes, longest_len)
        padding_length = bucket_data_length - input_ids.shape[1]
        input_ids = F.pad(input_ids, (0, padding_length), 'constant', tokenizer.pad_token_id)
        attention_mask = F.pad(attention_mask, (0, padding_length), 'constant', 0)
        if loss_scale:
            loss_scale = F.pad(loss_scale, (0, padding_length), 'constant', 0.)
        labels = F.pad(labels, (0, padding_length), 'constant', -100)

    # manully split the batch to different DP rank.
    batch_size = input_ids.shape[0] // world_size
    if batch_size > 0:
        start = rank * batch_size
        end = (rank + 1) * batch_size
        input_ids = input_ids[start:end, :]
        attention_mask = attention_mask[start:end, :]
        labels = labels[start:end, :]
        if loss_scale:
            loss_scale = loss_scale[start:end, :]
    return input_ids, attention_mask, labels, loss_scale


def ta_train_dataloader(train_dataset, data_collator, sampler, args, batch_size):
    # patch skip_first_batches for customized dataloader.
    def acc_skip_first_batches(dataloader, num_batches=0):
        from accelerate.data_loader import SkipBatchSampler
        batch_sampler = SkipBatchSampler(dataloader._loader.batch_sampler, skip_batches=num_batches)
        dataset = dataloader.dataset
        dataloader_params = {
            'collate_fn': data_collator,
            'num_workers': args.dataloader_num_workers,
            'pin_memory': args.dataloader_pin_memory,
            'persistent_workers': args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params['batch_sampler'] = batch_sampler
            dataloader_params['worker_init_fn'] = trainer.seed_worker

        return ta.AsyncLoader(DataLoader(dataset, **dataloader_params), args.device)

    trainer.skip_first_batches = acc_skip_first_batches

    # dataloader for TorchAcc.
    import torchacc as ta

    dataloader_params = {
        'batch_size': batch_size,
        'collate_fn': data_collator,
        'num_workers': args.dataloader_num_workers,
        'pin_memory': args.dataloader_pin_memory,
        'persistent_workers': args.dataloader_persistent_workers,
    }

    if not isinstance(train_dataset, torch.utils.data.IterableDataset):
        dataloader_params['sampler'] = sampler
        dataloader_params['drop_last'] = args.dataloader_drop_last
        dataloader_params['worker_init_fn'] = trainer.seed_worker

    return ta.AsyncLoader(DataLoader(train_dataset, **dataloader_params), args.device)


def ta_eval_dataloader(eval_dataset, data_collator, sampler, args):
    import torchacc as ta

    dataloader_params = {
        'batch_size': args.eval_batch_size,
        'collate_fn': data_collator,
        'num_workers': args.dataloader_num_workers,
        'pin_memory': args.dataloader_pin_memory,
        'persistent_workers': args.dataloader_persistent_workers,
    }

    if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
        dataloader_params['sampler'] = sampler
        dataloader_params['drop_last'] = args.dataloader_drop_last

    return ta.AsyncLoader(DataLoader(eval_dataset, **dataloader_params), args.device)


def ta_test_dataloader(test_dataset, data_collator, sampler, args):
    import torchacc as ta

    dataloader_params = {
        'batch_size': args.eval_batch_size,
        'collate_fn': data_collator,
        'num_workers': args.dataloader_num_workers,
        'pin_memory': args.dataloader_pin_memory,
        'persistent_workers': args.dataloader_persistent_workers,
    }

    if not isinstance(test_dataset, torch.utils.data.IterableDataset):
        dataloader_params['sampler'] = sampler
        dataloader_params['drop_last'] = args.dataloader_drop_last

    # We use the same batch_size as for eval.
    return ta.AsyncLoader(DataLoader(test_dataset, **dataloader_params), args.device)


# Save/load checkpoint
def consolidate_checkpoint(resume_from_checkpoint, model_name='adapter_model'):
    """ Consolidate the sharded TorchAcc checkpoints into a single model checkpoint.
    """
    import torch_xla.core.xla_model as xm
    from torch_xla.distributed.fsdp import consolidate_sharded_state_dicts

    if model_name not in ('adapter_model', 'model'):
        logger.error('Only support PeftModel and PreTrainedModel.')
        return

    model_dir = os.path.join(resume_from_checkpoint, '0')
    is_pretrained_model = False
    if os.path.exists(os.path.join(model_dir, f'{model_name}.safetensors')):
        use_safetensors = True
    elif os.path.exists(os.path.join(model_dir, f'{model_name}.bin')):
        use_safetensors = False
    elif os.path.exists(os.path.join(model_dir, 'pytorch_model.bin')):
        # PreTrainedModel use 'pytorch_model.bin' and 'model.safetensors'
        use_safetensors = False
        is_pretrained_model = True
    else:
        logger.error('Cannot find checkpoint.')

    state_dict_list = []
    if xm.is_master_ordinal(local=False) and use_safetensors:
        from safetensors.torch import load_file, save_file
        for rank in range(xm.xrt_world_size()):
            shard_dir = os.path.join(resume_from_checkpoint, f'{rank}')
            filename = os.path.join(shard_dir, f'{model_name}.safetensors')
            state_dict = load_file(filename, device='cpu')
            state_dict = OrderedDict(('_fsdp_wrapped_module.' + k, v) for k, v in state_dict.items())
            state_dict_list.append(state_dict)
        shard_metadata = torch.load(os.path.join(model_dir, 'shard_meta.pth'), map_location='cpu')
    elif xm.is_master_ordinal(local=False):
        for rank in range(xm.xrt_world_size()):
            shard_dir = os.path.join(resume_from_checkpoint, f'{rank}')
            if not is_pretrained_model:
                filename = os.path.join(shard_dir, f'{model_name}.bin')
            else:
                filename = os.path.join(shard_dir, 'pytorch_model.bin')
            state_dict = torch.load(filename, map_location='cpu')
            state_dict = OrderedDict(('_fsdp_wrapped_module.' + k, v) for k, v in state_dict.items())
            state_dict_list.append(state_dict)
        shard_metadata = torch.load(os.path.join(model_dir, 'shard_meta.pth'), map_location='cpu')

    if xm.is_master_ordinal(local=False):
        full_state_dict = consolidate_sharded_state_dicts(state_dict_list, shard_metadata)
        # peft will prepend "default." prefix automatically, so we remove the
        # "default." prefix to prevent the duplication of the prefix.
        full_state_dict = OrderedDict((k.replace('default.', ''), v) for k, v in full_state_dict.items())
        torch.save(full_state_dict, os.path.join(resume_from_checkpoint, f'{model_name}.bin'))
        if model_name == 'adapter_model':
            config_path = os.path.join(resume_from_checkpoint, 'adapter_config.json')
            old_config_path = os.path.join(model_dir, 'adapter_config.json')
            os.system(f'cp {old_config_path} {config_path}')
    xm.rendezvous('ckpt_consolidation')


def ta_save_optimizer_and_scheduler(optimizer, lr_scheduler, output_dir):
    import torch_xla.core.xla_model as xm
    xm.rendezvous('saving_optimizer_states')
    torch.save(optimizer.state_dict(), os.path.join(output_dir, f'optimizer_{xm.get_ordinal()}.pt'))
    torch.save(lr_scheduler.state_dict(), os.path.join(output_dir, f'scheduler_{xm.get_ordinal()}.pt'))


def ta_load_optimizer_and_scheduler(optimizer, lr_scheduler, checkpoint, device):
    import torch_xla.core.xla_model as xm
    optimizer_state = torch.load(os.path.join(checkpoint, f'optimizer_{xm.get_ordinal()}.pt'), map_location='cpu')
    lr_scheduler_state = torch.load(os.path.join(checkpoint, f'scheduler_{xm.get_ordinal()}.pt'), map_location='cpu')
    xm.send_cpu_data_to_device(optimizer_state, device)
    xm.send_cpu_data_to_device(lr_scheduler_state, device)

    optimizer.load_state_dict(optimizer_state)
    lr_scheduler.load_state_dict(lr_scheduler_state)
    return optimizer, lr_scheduler


def save_ta_checkpoint(self_model, tokenizer, args, output_dir):
    import torch_xla.core.xla_model as xm

    if xm.is_master_ordinal(local=False):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(args, os.path.join(output_dir, 'training_args.bin'))

    model = self_model._get_underlay_model().module.module

    supported_classes = (PreTrainedModel, PeftModel)
    save_safetensors = args.save_safetensors
    # Save a trained model and configuration using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    xm.rendezvous('saving_checkpoint')
    out_dir = os.path.join(output_dir, f'{xm.get_ordinal()}')
    if not isinstance(model, supported_classes):
        state_dict = model.state_dict()
        _unwrap_model = unwrap_model(model)
        if isinstance(_unwrap_model, supported_classes):
            _unwrap_model.save_pretrained(out_dir, safe_serialization=save_safetensors)
        else:
            logger.info('Trainer.model is not a `PreTrainedModel`, only saving its state dict.')
            if save_safetensors:
                safetensors.torch.save_file(state_dict, os.path.join(out_dir, 'model.safetensors'))
            else:
                torch.save(state_dict, os.path.join(out_dir, 'pytorch_model.bin'))
    else:
        model.save_pretrained(out_dir, safe_serialization=save_safetensors)
    # save shard_metadata for consolidation.
    shard_meta = self_model._get_underlay_model().get_shard_metadata()
    xm.save(shard_meta, os.path.join(out_dir, 'shard_meta.pth'))
    xm.rendezvous('saving_checkpoint_done')

    if tokenizer is not None and args.should_save:
        tokenizer.save_pretrained(output_dir, is_main_process=xm.is_master_ordinal(local=False), save_function=xm.save)

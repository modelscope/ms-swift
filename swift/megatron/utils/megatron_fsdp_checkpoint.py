# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import os
import torch
import torch.distributed.checkpoint as torch_dist_checkpoint
from megatron.core import mpu
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter, default_planner
from transformers.utils import is_torch_npu_available

from swift.utils import get_logger

logger = get_logger()

__all__ = [
    'build_rng_state', 'is_checkpoint', 'load_checkpoint', 'load_common_state_dict', 'save_checkpoint',
    'select_rng_state'
]


def build_rng_state(rng_state, data_parallel_random_init: bool = False):
    if (data_parallel_random_init and torch.distributed.is_initialized() and mpu.get_data_parallel_world_size() > 1):
        rng_state_list = [None for _ in range(mpu.get_data_parallel_world_size())]
        torch.distributed.all_gather_object(
            rng_state_list,
            rng_state,
            group=mpu.get_data_parallel_group(),
        )
    else:
        rng_state_list = [rng_state]
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    return {f'({pp_rank}, {tp_rank})': rng_state_list}


def select_rng_state(rng_state, data_parallel_random_init: bool):
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    rng_key = f'({pp_rank}, {tp_rank})'
    if rng_key in rng_state:
        rng_state_list = rng_state[rng_key]
    else:
        logger.warning('RNG state not found for current TP/PP rank; falling back to the first saved RNG state.')
        rng_state_list = next(iter(rng_state.values()))
    if data_parallel_random_init:
        return rng_state_list[mpu.get_data_parallel_rank()]
    return rng_state_list[0]


def _preprocess_state_dict(args, state_dict, model):
    from megatron.training.checkpointing import preprocess_fsdp_dtensor_state_dict

    preprocess_args = copy.copy(args)
    config = getattr(model, 'config', None)
    preprocess_args.swiglu = getattr(args, 'swiglu', getattr(config, 'swiglu', False))
    preprocess_args.num_experts = getattr(args, 'num_experts', getattr(config, 'num_moe_experts', None))
    return preprocess_fsdp_dtensor_state_dict(preprocess_args, state_dict, model)


def _validate_optimizer_state(state_dict):
    optimizer_state_dict = state_dict.get('optimizer')
    if optimizer_state_dict is None:
        return
    if not (isinstance(optimizer_state_dict, dict) and 'state' in optimizer_state_dict
            and 'param_to_group_meta' in optimizer_state_dict):
        raise NotImplementedError(
            'Megatron-FSDP fsdp_dtensor checkpointing currently supports exactly one distributed optimizer. '
            'ChainedOptimizer and nested optimizer checkpoint structures are not supported yet.')


def _prepare_state_dict(args, state_dict, model, preserve_raw_state: bool = False):
    _validate_optimizer_state(state_dict)
    if is_torch_npu_available():
        from swift.model.npu_patch.mindspeed import complete_mindspeed_fsdp_dtensor_optimizer_state
        complete_mindspeed_fsdp_dtensor_optimizer_state(state_dict, model)

    raw_state_dict = {}
    if preserve_raw_state:
        raw_state_dict.update({key: value.copy() for key, value in state_dict.items() if key.startswith('model')})
        if 'optimizer' in state_dict:
            raw_state_dict['optimizer'] = state_dict['optimizer'].copy()

    state_dict = _preprocess_state_dict(args, state_dict, model)
    return state_dict, raw_state_dict


def save_checkpoint(args, state_dict, model, checkpoint_dir):
    state_dict, _ = _prepare_state_dict(args, state_dict, model)
    torch_dist_checkpoint.save(
        state_dict=state_dict,
        storage_writer=FileSystemWriter(checkpoint_dir),
    )


def load_common_state_dict(checkpoint_dir):
    state_dict = {'args': None, 'iteration': None}
    torch_dist_checkpoint.load(state_dict=state_dict, checkpoint_id=checkpoint_dir)
    return state_dict


def is_checkpoint(checkpoint_dir):
    metadata_path = os.path.join(checkpoint_dir, '.metadata')
    if not os.path.isfile(metadata_path) or os.path.exists(os.path.join(checkpoint_dir, 'common.pt')):
        return False

    metadata = FileSystemReader(checkpoint_dir).read_metadata()
    keys = metadata.state_dict_metadata
    return 'args' in keys and 'iteration' in keys and any(key == 'model' or key.startswith('model.') for key in keys)


def _get_load_planner(args):
    allow_partial_load = not getattr(args, 'strict_fsdp_dtensor_load', False)
    return default_planner.DefaultLoadPlanner(allow_partial_load=allow_partial_load)


def load_checkpoint(args, state_dict, model, checkpoint_dir):
    state_dict, raw_state_dict = _prepare_state_dict(
        args,
        state_dict,
        model,
        preserve_raw_state=True,
    )
    torch_dist_checkpoint.load_state_dict(
        state_dict=state_dict,
        storage_reader=FileSystemReader(checkpoint_dir),
        planner=_get_load_planner(args),
    )
    state_dict.update(raw_state_dict)
    return state_dict

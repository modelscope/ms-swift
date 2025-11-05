# Copyright (c) Alibaba, Inc. and its affiliates.
from contextlib import contextmanager

import torch
from megatron.core.dist_checkpointing.mapping import ShardedTensorFactory
from megatron.core.dist_checkpointing.strategies.torch import TorchDistSaveShardedStrategy
from megatron.training import checkpointing

from swift.utils import get_logger

logger = get_logger()


def patch_torch_dist_shard(thread_count):
    __init__ = TorchDistSaveShardedStrategy.__init__

    def __new_init__(*args, **kwargs):
        kwargs['thread_count'] = thread_count
        return __init__(*args, **kwargs)

    TorchDistSaveShardedStrategy.__init__ = __new_init__


def patch_merge_fn(state_dict_model):
    # https://github.com/NVIDIA/Megatron-LM/issues/1380

    def sh_ten_merge_fn(sub_state_dict):
        with torch.no_grad():
            shared_storage = sub_state_dict[0].untyped_storage()
            if all(shared_storage.data_ptr() == tensor.untyped_storage().data_ptr() for tensor in sub_state_dict):
                element_size = sub_state_dict[0].element_size()
                total_numel = sum(tensor.numel() for tensor in sub_state_dict)
                if shared_storage.nbytes() == total_numel * element_size:
                    dim_0 = sum(tensor.shape[0] for tensor in sub_state_dict)
                    shape = (dim_0, ) + sub_state_dict[0].shape[1:]
                    combined_tensor = torch.empty(
                        shape, dtype=sub_state_dict[0].dtype,
                        device=sub_state_dict[0].device).set_(shared_storage, 0, shape)
                    return combined_tensor
            return torch.cat(sub_state_dict)

    for v in state_dict_model.values():
        if isinstance(v, ShardedTensorFactory) and 'apply_swiglu_sharded_factory' in v.merge_fn.__qualname__:
            v.merge_fn = sh_ten_merge_fn


@contextmanager
def patch_load_base_checkpoint():
    origin__load_base_checkpoint = checkpointing._load_base_checkpoint

    def _load_base_checkpoint(*_args, **kwargs):
        sharded_state_dict = kwargs.get('sharded_state_dict')
        if sharded_state_dict is None:
            return origin__load_base_checkpoint(*_args, **kwargs)
        model_keys = [k for k in sharded_state_dict.keys() if k.startswith('model')]  # compat vpp
        for k in model_keys:
            patch_merge_fn(sharded_state_dict[k])
        return origin__load_base_checkpoint(*_args, **kwargs)

    checkpointing._load_base_checkpoint = _load_base_checkpoint
    try:
        yield
    finally:
        checkpointing._load_base_checkpoint = origin__load_base_checkpoint

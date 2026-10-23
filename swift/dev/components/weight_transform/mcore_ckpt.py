# Copyright (c) ModelScope Contributors. All rights reserved.
"""mcore ``torch_dist`` checkpoint kernels for the dev weight-transform conversions.

Ported from legacy ``swift.megatron.utils.megatron_lm_utils`` (``save_mcore_checkpoint`` /
``load_mcore_checkpoint`` / ``_filter_adapter_state_dict`` / ``_load_iteration``) and
``swift.megatron.utils.patcher`` (``patch_merge_fn``), narrowed to the conversion contract:
a format migration writes no optimizer/RNG state and resumes no training, so those halves of
the legacy kernels are absent by construction rather than gated behind flags.

What replaces legacy's ``MegatronArguments`` carrier is ``parallel_meta``: a small JSON-safe
dict describing the parallel layout a checkpoint was written with. It is stored under
``state_dict['args']`` (as a :class:`argparse.Namespace`, the form mcore's consistency check
and legacy readers expect) and mirrored into ``args.json`` by the caller, so both legacy
``load_mcore_checkpoint`` and ``MegatronArguments.load_args_config`` still find every key they
read (tp/pp/cp/ep/etp sizes, no_save_optim/no_save_rng, consumed_train_samples, ...).
"""
from __future__ import annotations

import os
from argparse import Namespace
from typing import Any, Dict, List, Optional

import torch
from packaging import version

from swift.dev.utils.logger import get_logger

logger = get_logger()


def _megatron_dist_checkpointing():
    import megatron.core
    from megatron.core import dist_checkpointing

    global _mcore_017
    if _mcore_017 is None:
        _mcore_017 = version.parse(megatron.core.__version__) >= version.parse('0.17.0rc0')
    return dist_checkpointing


_mcore_017: Optional[bool] = None


def _unwrap_models(models: List[Any]) -> List[Any]:
    """Strip DDP/Float16Module wrappers (mcore-bridge's ``unwrap_model``), never the model itself."""
    from mcore_bridge import unwrap_model

    return unwrap_model(models)


def _dp_group_with_cp():
    from megatron.core import mpu

    return mpu.get_data_parallel_group(with_context_parallel=True)


def _sharded_sd_metadata() -> Dict[str, Any]:
    """The metadata every ShardedTensor and the checkpoint's common state carry.

    The conversion equivalent of legacy ``get_sharded_sd_metadata``: no optimizer state is ever
    written, so ``distrib_optim_sharding_type`` only needs the plain-DDP value, and the FSDP /
    pre-0.14 reshard modes legacy selected from training flags cannot occur here.
    """
    return {
        'singleton_local_shards': False,
        'chained_optim_avoid_prefix': True,
        'distrib_optim_sharding_type': 'dp_reshardable',
    }


def _preprocess_common_before_consistancy_check(common_state_dict):
    """Namespace -> dict and drop per-rank keys, so the cross-rank common-state check passes."""
    import copy

    preprocessed = copy.deepcopy(common_state_dict)
    preprocessed['args'] = vars(preprocessed['args'])
    preprocessed['args'].pop('local_rank', None)
    preprocessed['args'].pop('rank', None)
    return preprocessed


def patch_merge_fn(state_dict_model: Dict[str, Any]) -> None:
    """Replace mcore's ``apply_swiglu_sharded_factory`` merge with a shared-storage-aware one.

    Upstream concatenates the SwiGLU gate/up shards even when they already share one storage,
    which both copies needlessly and breaks TP>1 resharding (NVIDIA/Megatron-LM#1380).
    Ported verbatim from legacy ``swift.megatron.utils.patcher.patch_merge_fn``.
    """
    from megatron.core.dist_checkpointing.mapping import ShardedTensorFactory

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


def _filter_adapter_state_dict(state_dict: Dict[str, Any], peft_format: bool, adapter_name: str = 'default') -> None:
    """Keep only the LoRA keys (``peft_format=True``) or only the base keys (``False``), in place.

    Ported verbatim from legacy. The ``False`` half also strips ``base_layer.`` /
    ``modules_to_save.{adapter}.`` path segments so a merged model loads as a plain one.
    """
    if 'model' in state_dict:
        n_models = 1
    else:
        n_models = 0
        while f'model{n_models}' in state_dict:
            n_models += 1

    for i in range(n_models):
        model_key = 'model' if i == 0 and n_models == 1 else f'model{i}'
        new_state_dict = {}
        state_dict_model = state_dict[model_key]
        for k, v in state_dict_model.items():
            if peft_format:
                if '.lora_A.' in k or '.lora_B.' in k or '.modules_to_save.' in k:
                    new_state_dict[k] = v
            else:
                if '.lora_A.' in k or '.lora_B.' in k or 'original_module.' in k:
                    continue
                k = k.replace('base_layer.', '')
                k = k.replace(f'modules_to_save.{adapter_name}.', '')
                v.key = v.key.replace('base_layer.', '')
                v.key = v.key.replace(f'modules_to_save.{adapter_name}.', '')
                new_state_dict[k] = v
        state_dict[model_key] = new_state_dict


def _model_state_keys(n_models: int) -> List[str]:
    return ['model'] if n_models == 1 else [f'model{i}' for i in range(n_models)]


def save_mcore_dist_checkpoint(models: List[Any], output_dir: str, parallel_meta: Dict[str, Any],
                               iteration: int = 1) -> None:
    """Write bare mcore models as a ``torch_dist`` checkpoint under ``output_dir``.

    Mirrors legacy ``save_mcore_checkpoint`` with the optimizer/RNG/async halves dropped:
    ``parallel_meta['no_save_optim']``/``['no_save_rng']`` are constants of a conversion. The
    on-disk layout is unchanged -- ``iter_{iteration:07d}/`` plus the
    ``latest_checkpointed_iteration.txt`` tracker written by the master rank.
    """
    dist_checkpointing = _megatron_dist_checkpointing()
    from megatron.core.dist_checkpointing.strategies.fully_parallel import FullyParallelSaveStrategyWrapper
    from megatron.core.dist_checkpointing.strategies.torch import TorchDistSaveShardedStrategy

    models = _unwrap_models(models)
    checkpoint_dir = os.path.join(output_dir, f'iter_{iteration:07d}')
    metadata = _sharded_sd_metadata()
    os.makedirs(checkpoint_dir, exist_ok=True)

    state_dict: Dict[str, Any] = {
        'args': Namespace(**dict(parallel_meta)),
        'checkpoint_version': 3.0,
        'iteration': iteration,
    }
    for key, model in zip(_model_state_keys(len(models)), models):
        state_dict[key] = model.sharded_state_dict(metadata=metadata)

    if _mcore_017:
        save_strategy = TorchDistSaveShardedStrategy()
    else:
        from megatron.core.dist_checkpointing.serialization import get_default_save_sharded_strategy
        save_strategy = get_default_save_sharded_strategy()
    save_strategy = FullyParallelSaveStrategyWrapper(save_strategy, _dp_group_with_cp())
    dist_checkpointing.save(
        state_dict,
        checkpoint_dir,
        save_strategy,
        async_sharded_save=False,
        validate_access_integrity=True,
        preprocess_common_before_consistancy_check=_preprocess_common_before_consistancy_check,
        content_metadata=metadata)

    # Everyone must be done before the tracker is published (legacy keeps the same order).
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    from swift.dev.utils import is_master

    if is_master():
        tracker_path = os.path.join(output_dir, 'latest_checkpointed_iteration.txt')
        with open(tracker_path, 'w') as f:
            f.write(str(iteration))
        logger.info(f'Successfully saved Megatron model weights in `{output_dir}`.')


def _load_iteration(tracker_path: str) -> int:
    """Read the tracker, then take the MAX across ranks so a partial mount degrades, not crashes."""
    from twinkle import Platform

    iteration = 0
    if os.path.exists(tracker_path):
        with open(tracker_path) as f:
            iteration = int(f.read().strip())
    if torch.distributed.is_initialized():
        iters_cuda = torch.tensor([iteration], dtype=torch.long, device=Platform.get_local_device())
        torch.distributed.all_reduce(iters_cuda, op=torch.distributed.ReduceOp.MAX)
        iteration = int(iters_cuda[0].item())
    return iteration


def load_mcore_dist_checkpoint(models: List[Any],
                               load_dir: str,
                               parallel_meta: Dict[str, Any],
                               *,
                               peft_format: bool = False,
                               adapter_name: str = 'default') -> int:
    """Load a ``torch_dist`` checkpoint into bare mcore models; returns the loaded iteration.

    Mirrors legacy ``load_mcore_checkpoint`` with the optimizer/RNG/resume halves dropped. TP/PP
    resharding is native to ``dist_checkpointing.load`` -- the mismatch log below is the only
    legacy behavior kept (RNG is never restored by a conversion, ``finetune`` is always True).
    """
    dist_checkpointing = _megatron_dist_checkpointing()
    from megatron.core.dist_checkpointing.strategies.fully_parallel import FullyParallelLoadStrategyWrapper
    from megatron.core.dist_checkpointing.strategies.torch import TorchDistLoadShardedStrategy

    models = _unwrap_models(models)
    tracker_path = os.path.join(load_dir, 'latest_checkpointed_iteration.txt')
    iteration = _load_iteration(tracker_path)
    checkpoint_dir = os.path.join(load_dir, f'iter_{iteration:07d}')
    state_dict = dist_checkpointing.load_common_state_dict(checkpoint_dir)

    ckpt_tp_pp = (
        state_dict['args'].tensor_model_parallel_size,
        state_dict['args'].pipeline_model_parallel_size,
    )
    run_tp_pp = (
        parallel_meta['tensor_model_parallel_size'],
        parallel_meta['pipeline_model_parallel_size'],
    )
    if ckpt_tp_pp != run_tp_pp:
        logger.info(f'(TP, PP) mismatch after resume ({run_tp_pp} vs {ckpt_tp_pp} from checkpoint): '
                    'RNG state will be ignored')

    # The load template must carry the checkpoint's own content_metadata so ShardedTensor
    # replica/shard expectations line up with how the checkpoint was written.
    metadata = state_dict.get('content_metadata')
    sharded_state_dict: Dict[str, Any] = {
        'args': Namespace(**dict(parallel_meta)),
        'checkpoint_version': 3.0,
        'iteration': iteration,
    }
    for key, model in zip(_model_state_keys(len(models)), models):
        sharded_state_dict[key] = model.sharded_state_dict(metadata=metadata)
    _filter_adapter_state_dict(sharded_state_dict, peft_format, adapter_name=adapter_name)
    for key in [k for k in sharded_state_dict if k.startswith('model')]:  # compat vpp
        patch_merge_fn(sharded_state_dict[key])

    if _mcore_017:
        load_strategy = TorchDistLoadShardedStrategy()
    else:
        from megatron.core.dist_checkpointing.serialization import get_default_load_sharded_strategy
        load_strategy = get_default_load_sharded_strategy(checkpoint_dir)
    load_strategy = FullyParallelLoadStrategyWrapper(load_strategy, _dp_group_with_cp())
    state_dict = dist_checkpointing.load(sharded_state_dict, checkpoint_dir, load_strategy)

    if len(models) == 1:
        # strict=False mirrors legacy: the sharded load template carries exactly the keys the
        # checkpoint holds, so strictness would only reject checkpoint/model drift twice.
        models[0].load_state_dict(state_dict['model'], strict=False)
    else:
        for i, model in enumerate(models):
            if f'model{i}' in state_dict:
                model.load_state_dict(state_dict[f'model{i}'])

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    logger.info(f'Successfully loaded Megatron model weights from: {load_dir}')
    return iteration


#: Mirror of ``fields(MegatronTunerMixin)`` (legacy megatron_args.py) plus the four non-tuner
#: keys legacy ``MegatronArguments.load_args_config`` reads -- the args.json surface a source
#: mcore checkpoint may carry that a conversion consumes.
_SOURCE_TUNER_KEYS = (
    'tuner_type',
    'freeze_llm',
    'freeze_vit',
    'freeze_aligner',
    'freeze_parameters',
    'freeze_parameters_regex',
    'freeze_parameters_ratio',
    'trainable_parameters',
    'trainable_parameters_regex',
    'target_modules',
    'target_regex',
    'modules_to_save',
    'lora_rank',
    'lora_alpha',
    'lora_dropout',
    'lora_bias',
    'lora_dtype',
    'use_rslora',
)


def load_source_args(ckpt_dir: Optional[str]) -> Dict[str, Any]:
    """Read the tuner/task settings a source mcore checkpoint was trained with, from its args.json.

    Port of legacy ``MegatronArguments.load_args_config``: only these keys are consumed, so only
    these keys are picked out (an mcore checkpoint written by full training carries the whole
    MegatronArguments surface, most of which is meaningless for a conversion). ``mcore_model``
    is dropped for a full-parameter checkpoint -- legacy does the same, so a full checkpoint's
    stale self-reference never pins a later conversion's source.
    """
    res: Dict[str, Any] = {}
    if ckpt_dir is None:
        return res
    args_path = os.path.join(ckpt_dir, 'args.json')
    if not os.path.exists(args_path):
        return res
    import json

    with open(args_path, 'r', encoding='utf-8') as f:
        old_args = json.load(f)
    keys = list(_SOURCE_TUNER_KEYS) + ['mcore_model', 'task_type', 'num_labels', 'bridge_backend']
    for key in keys:
        old_value = old_args.get(key)
        if old_value is not None:
            res[key] = old_value
    if res.get('tuner_type') == 'full':
        res.pop('mcore_model', None)
    return res

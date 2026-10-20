# Copyright (c) ModelScope Contributors. All rights reserved.
"""Re-quantize a BF16 HF export of a packed checkpoint back to its source format.

A full-parameter RL run trains BF16 copies of a packed source (block-FP8 dense
weights, MXFP4 routed experts). ``MegatronModel.save`` writes a plain BF16
checkpoint, which is the truthful training artifact but not loadable by the
packed rollout engine: vLLM's DeepSeek-V4 FlashMLA ``_o_proj`` reads
``wo_a.weight_scale_inv`` unconditionally, and ``expert_dtype=fp4`` would make
the loader interpret a ``(rows, cols)`` BF16 expert as packed bytes.

This module restores the source format after the save:

* every exported tensor that has a sibling ``.scale`` in the source checkpoint is
  re-quantized with the checkpoint's own geometry and dtype family
  (128x128 block-FP8 for dense/shared weights, 32-column MXFP4 for routed experts);
* the new scales replace the source scales, so they describe the trained weights;
* tensors without a source scale (embed/head/norm/router/hc_head) stay BF16;
* the source ``config.json`` is copied back so the packed markers, legacy
  fields the config class consumes (``compress_ratios``/``num_hash_layers``)
  and the flat YaRN ``rope_scaling`` are exactly what the source engine reads.

The BF16 shards are replaced by one packed ``model.safetensors`` only after the
packed file and its index are written, so an interrupted run leaves the BF16
checkpoint intact.
"""
import contextlib
import json
import os
import shutil
from typing import Any, Dict, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from swift.dev.utils.logger import get_logger

logger = get_logger()

#: Config keys the trained export may update; everything else comes from the source.
_TRAINING_CONFIG_KEYS = ('vocab_size', 'num_nextn_predict_layers')


def _weight_map(model_dir: str) -> Dict[str, str]:
    """Return ``name -> shard path`` for a single-file or indexed checkpoint."""
    index = os.path.join(model_dir, 'model.safetensors.index.json')
    if os.path.exists(index):
        weight_map = json.load(open(index))['weight_map']
        return {name: os.path.join(model_dir, shard) for name, shard in weight_map.items()}
    single = os.path.join(model_dir, 'model.safetensors')
    if not os.path.exists(single):
        raise FileNotFoundError(f'No model.safetensors or index found under {model_dir!r}')
    with safe_open(single, framework='pt') as f:
        return {name: single for name in f.keys()}


def _read_json(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _device(name: Optional[str]) -> torch.device:
    name = name or os.environ.get('SWIFT_REQUANTIZE_DEVICE')
    if name:
        return torch.device(name)
    # The save runs in the driver, which usually holds no GPU in Ray mode. CPU is
    # slower but portable; callers on a free GPU pass device='cuda'.
    return torch.device('cpu')


def requantize_checkpoint(ckpt_dir: str,
                          source_dir: str,
                          *,
                          device: Optional[str] = None,
                          audit: bool = False) -> Optional[Dict[str, int]]:
    """Rewrite ``ckpt_dir`` from its BF16 export into the source's packed format.

    Returns ``None`` when ``source_dir`` is not a packed checkpoint (no ``.scale``
    tensors), otherwise the quantized-tensor counters.  ``audit=True`` samples a
    few tensors against the source weights (useful for frozen-policy checks).
    """
    from .fp8_kernel import scaled_fp8_blockwise
    from .mxfp4_kernel import quantize_mxfp4_e2m1

    if not os.path.exists(os.path.join(source_dir, 'config.json')):
        logger.info(f'[requantize] skipped: {source_dir!r} is not a local checkpoint')
        return None
    source_cfg = _read_json(os.path.join(source_dir, 'config.json'))
    saved_cfg = _read_json(os.path.join(ckpt_dir, 'config.json'))
    source_map = _weight_map(source_dir)
    scale_keys = {name for name in source_map if name.endswith('.scale')}
    if not scale_keys:
        return None

    saved_map = _weight_map(ckpt_dir)
    dev = _device(device)
    previous_threads = torch.get_num_threads()
    if dev.type == 'cpu':
        # The driver may have pinned torch to a single thread for the training actors;
        # the CPU quantizers here are embarrassingly parallel.
        torch.set_num_threads(max(1, min(32, os.cpu_count() or 1)))
    expert_dtype = source_cfg.get('expert_dtype')
    packed: Dict[str, torch.Tensor] = {}
    n_fp8 = n_fp4 = n_keep = 0
    audited = exact = 0

    with contextlib.ExitStack() as source_stack:
        source_handles: Dict[str, Any] = {}

        def source_tensor(name: str) -> torch.Tensor:
            path = source_map[name]
            handle = source_handles.get(path)
            if handle is None:
                handle = source_stack.enter_context(safe_open(path, framework='pt'))
                source_handles[path] = handle
            return handle.get_tensor(name)

        for shard in sorted(set(saved_map.values())):
            with safe_open(shard, framework='pt') as saved:
                for name in saved.keys():
                    if name.endswith('.scale'):
                        continue
                    tensor = saved.get_tensor(name)
                    scale_name = name[:-len('.weight')] + '.scale' if name.endswith('.weight') else None
                    if scale_name not in scale_keys:
                        packed[name] = tensor
                        n_keep += 1
                        continue

                    source_scale = source_tensor(scale_name)
                    weight = tensor.to(dev)
                    if '.ffn.experts.' in name and expert_dtype == 'fp4':
                        values, scale = quantize_mxfp4_e2m1(weight, source_scale.to(dev), name=name)
                        packed[name] = values.cpu()
                        packed[scale_name] = scale.cpu()
                        n_fp4 += 1
                    else:
                        if weight.ndim != 2:
                            raise RuntimeError(f'Cannot re-quantize {name!r} with a source scale: '
                                               f'expected a 2-D weight, got {tuple(weight.shape)}')
                        block_m = weight.shape[0] // source_scale.shape[0]
                        block_n = weight.shape[1] // source_scale.shape[1]
                        if (weight.shape[0] % source_scale.shape[0] or weight.shape[1] % source_scale.shape[1]):
                            raise RuntimeError(f'Source scale {tuple(source_scale.shape)} does not tile '
                                               f'{name!r} {tuple(weight.shape)}')
                        values, scale = scaled_fp8_blockwise(weight, [block_m, block_n], use_ue8m0=True)
                        packed[name] = values.cpu()
                        packed[scale_name] = scale.to(torch.float8_e8m0fnu).cpu()
                        n_fp8 += 1

                    if (n_fp8 + n_fp4) % 200 == 0:
                        logger.info(f'[requantize] quantized {n_fp8 + n_fp4} tensors...')
                    if audit and (n_fp8 + n_fp4) % 200 == 0:
                        reference = source_tensor(name)
                        same = (packed[name].view(torch.uint8) == reference.view(torch.uint8)).float().mean().item()
                        same_scale = (packed[scale_name].view(torch.uint8) == source_scale.view(torch.uint8)).float().mean(
                        ).item()
                        audited += 1
                        exact += int(same == 1.0)
                        logger.info(f'[requantize] audit {name}: weight_bytes_same={same:.4f} '
                                    f'scale_bytes_same={same_scale:.4f}')
                    del weight

    # Write the packed file first; only then remove the BF16 shards and their index.
    packed_path = os.path.join(ckpt_dir, 'model.safetensors')
    save_file(packed, packed_path, metadata={'format': 'pt'})
    for path in sorted(set(saved_map.values())):
        if os.path.abspath(path) != os.path.abspath(packed_path):
            os.remove(path)
    old_index = os.path.join(ckpt_dir, 'model.safetensors.index.json')
    if os.path.exists(old_index):
        os.remove(old_index)
    total = sum(t.numel() * t.element_size() for t in packed.values())
    with open(old_index, 'w') as f:
        json.dump({'metadata': {'total_size': total}, 'weight_map': {name: 'model.safetensors'
                                                                     for name in packed}}, f, indent=2)
    del packed

    # The source config is the known-good packed config (quantization_config,
    # expert_dtype=fp4, compress_ratios/num_hash_layers and the flat rope_scaling).
    # Keep the fields training can legitimately change.
    for key in _TRAINING_CONFIG_KEYS:
        if key in saved_cfg:
            source_cfg[key] = saved_cfg[key]
    with open(os.path.join(ckpt_dir, 'config.json'), 'w') as f:
        json.dump(source_cfg, f, indent=2)
    for extra in ('tokenizer.json', 'tokenizer_config.json', 'generation_config.json', 'configuration.json'):
        if not os.path.exists(os.path.join(ckpt_dir, extra)):
            candidate = os.path.join(source_dir, extra)
            if os.path.exists(candidate):
                shutil.copy(candidate, os.path.join(ckpt_dir, extra))

    if dev.type == 'cpu':
        torch.set_num_threads(previous_threads)
    stats = {'fp8': n_fp8, 'mxfp4': n_fp4, 'kept': n_keep, 'size_bytes': total}
    logger.info(f'[requantize] {ckpt_dir}: fp8={n_fp8} mxfp4={n_fp4} kept={n_keep} '
                f'size={total / 1024 ** 3:.2f} GiB' + (f' audit={exact}/{audited}' if audit else ''))
    return stats

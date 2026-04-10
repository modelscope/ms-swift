# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared inference utilities for MegatronWorker and HybridWorker.

Provides common logic for:
  - Creating frozen Megatron models for inference (ref / teacher)
  - Batch preparation for pipeline/context parallelism
  - Log-probability computation from model output
  - Batch slicing for micro-batching
"""
import torch
from typing import Any, Dict, List, Optional, Tuple

from swift.utils import get_logger

logger = get_logger()


def slice_batch(batch: Dict[str, Any], start: int, end: int) -> Dict[str, Any]:
    """Slice a batch along dim 0 for micro-batching."""
    result = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] >= end:
            result[k] = v[start:end]
        elif isinstance(v, list) and len(v) >= end:
            result[k] = v[start:end]
        else:
            result[k] = v
    return result


def create_inference_model(
    pipeline,
    model_id_override: Optional[str] = None,
):
    """Create a frozen Megatron model for inference (ref or teacher).

    Returns a list of Float16Module-wrapped models.
    """
    from megatron.core.transformer.module import Float16Module

    from swift.megatron.model import get_mcore_model
    from swift.utils import safe_snapshot_download

    args = pipeline.args
    if args.train_iters is None:
        args.train_iters = 1

    models = get_mcore_model(args, pipeline.template.config)
    bridge = models[0].config.bridge

    if model_id_override:
        model_id = model_id_override
    else:
        model_id = getattr(args, 'ref_model', None) or args.model

    model_dir = safe_snapshot_download(model_id, use_hf=args.use_hf, hub_token=args.hub_token)
    bridge.load_weights(models, model_dir)

    for m in models:
        m.requires_grad_(False)
        m.eval()

    if args.fp16 or args.bf16:
        models = [Float16Module(m.config, m) for m in models]

    torch.cuda.empty_cache()
    return models


def prepare_batch_inference(args, batch):
    """Prepare a batch for PP/CP-aware inference."""
    from swift.megatron.trainers.utils import (get_batch_on_this_cp_rank, get_batch_on_this_pp_rank,
                                               get_packed_seq_params)

    data = get_batch_on_this_pp_rank(args, batch)
    num_samples = data.pop('num_samples', None)
    text_position_ids = data.pop('text_position_ids', None)
    data.pop('attention_mask_2d', None)
    if text_position_ids is None:
        text_position_ids = data.get('position_ids')
    if args.padding_free and text_position_ids is not None:
        data['packed_seq_params'] = get_packed_seq_params(text_position_ids)
        if num_samples is not None:
            data['packed_seq_params'].num_samples = num_samples
    data = get_batch_on_this_cp_rank(args, data)
    return data


def compute_logps(args, output_tensor, labels, packed_seq_params, num_samples):
    """Compute per-sample log probabilities from model output."""
    from megatron.core import mpu
    from torch.distributed.nn import all_reduce

    per_token_logps = -output_tensor
    loss_mask = labels != -100
    per_token_logps = per_token_logps * loss_mask
    if args.padding_free:
        cu_seqlens = (packed_seq_params.cu_seqlens_q[:num_samples + 1] // args.context_parallel_size)
        all_logps = per_token_logps.new_zeros((num_samples, ))
        for i in range(num_samples):
            s, e = cu_seqlens[i], cu_seqlens[i + 1]
            all_logps[i] = per_token_logps[:, s:e].sum()
    else:
        all_logps = per_token_logps.sum(-1)
    if args.context_parallel_size > 1:
        all_logps = all_reduce(all_logps, group=mpu.get_context_parallel_group())
    return all_logps


def inference_forward(args, models, raw_batch):
    """Run inference forward on frozen models, returning log-probs.

    Works for both ref and teacher models with any rlhf_type.

    Args:
        args: Megatron arguments.
        models: List of Megatron model modules.
        raw_batch: Raw batch dict from driver.

    Returns:
        dict with 'logps' tensor, or None for non-last PP ranks.
    """
    from swift.megatron.utils import forward_step_helper
    from swift.utils import to_device

    batch = to_device(raw_batch, 'cuda', non_blocking=True)
    batch = prepare_batch_inference(args, batch)

    labels = batch.get('labels')
    packed_seq_params = batch.get('packed_seq_params')
    batch.pop('loss_scale', None)

    model = models[0]

    with torch.no_grad():
        output = forward_step_helper(args, model, batch)

    if output is None or labels is None:
        return None

    num_samples = labels.shape[0]
    if packed_seq_params is not None:
        num_samples = getattr(packed_seq_params, 'num_samples', num_samples)

    logps = compute_logps(args, output, labels, packed_seq_params, num_samples)
    return {'logps': logps.cpu()}

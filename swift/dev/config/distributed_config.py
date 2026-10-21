# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union


@dataclass
class DistributedConfig:
    """Distributed training and parallelism settings."""

    # === DeepSpeed ===
    deepspeed: Optional[str] = None
    zero_hpz_partition_size: Optional[int] = None
    deepspeed_autotp_size: Optional[int] = None

    # === FSDP ===
    fsdp: Optional[str] = None
    # FSDP's own settings (wrapping policy, sharding, offload) as a JSON object/file.
    fsdp_config: Optional[Dict[str, Any]] = None

    # === Launch ===
    mode: Literal['ray', 'local'] = 'local'
    nproc_per_node: Optional[int] = None
    # Run name handed to twinkle.initialize(name=...); only read in mode='ray'.
    ray_exp_name: Optional[str] = None

    # === DDP ===
    # Collective timeout (transformers backend); a long first step can trip the default.
    ddp_timeout: Optional[int] = None
    ddp_find_unused_parameters: Optional[bool] = None
    # Broadcast buffers from rank 0 each forward; off is a throughput win when there are none.
    ddp_broadcast_buffers: Optional[bool] = None
    # All-reduce bucket size; larger overlaps communication better, delays the first reduction.
    ddp_bucket_cap_mb: Optional[int] = None
    # Fixed reduction order; faster but wrong when the active parameter set varies per step.
    ddp_static_graph: Optional[bool] = None
    # This process's rank, supplied by the launcher (-1 = not launched distributed); set by torchrun.
    local_rank: int = -1

    # === Backend ===
    # 'unsloth' rides transformers construction (LoRA/full-param); None/'hf' is plain transformers.
    backend: Optional[Literal['megatron', 'hf', 'unsloth']] = None
    # Megatron-only: which bridge converts mcore checkpoints (training load + export).
    bridge_backend: Literal['mcore-bridge', 'megatron-bridge'] = 'mcore-bridge'

    # === Parallel layout (backend-agnostic) ===
    # 'dp4tp2pp2' via DeviceMesh.from_spec (dp/fsdp/tp/pp/cp/ep/etp/epfsdp/vpp + 'sp'); both read it.
    parallel_spec: Optional[str] = None

    # === Megatron ===
    # Distributed subset of legacy MegatronArguments; None keeps Megatron's own default.
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    # Megatron TP-SP (activations split along the sequence inside TP regions); distinct from
    # TemplateConfig.sequence_parallel_size (transformers Ulysses). Only effective when tp > 1.
    sequence_parallel: bool = False
    use_distributed_optimizer: bool = True
    # Shard parameters with Megatron-FSDP (NOT the transformers backend's ``fsdp`` above);
    # requires use_distributed_optimizer, incompatible with context_parallel_size > 1 (validate).
    use_megatron_fsdp: bool = False
    recompute_granularity: Optional[Literal['selective', 'full', 'none']] = None
    recompute_method: Optional[Literal['uniform', 'block']] = None
    recompute_num_layers: Optional[int] = None
    # Which submodules are recomputed ('core_attn'/'mlp'/'moe'); finer than recompute_granularity.
    recompute_modules: List[str] = field(default_factory=lambda: ['core_attn'])

    # === Megatron: expert & context parallel detail ===
    # TP width inside expert layers, which may differ from the dense tp size.
    expert_tensor_parallel_size: int = 1
    # How CP ranks exchange keys/values ('p2p'/'a2a'/'allgather', or a per-layer list).
    cp_comm_type: Optional[Union[str, List[str]]] = None
    # 'zigzag' interleaves so every rank gets early and late positions (even causal-attention
    # work); 'contiguous' gives each rank one block and leaves the first rank less to do.
    cp_partition_mode: Literal['zigzag', 'contiguous'] = 'zigzag'
    # What the DP optimizer shards; 'optim_grads_params' is ZeRO-3-equivalent.
    data_parallel_sharding_strategy: Literal['no_shard', 'optim', 'optim_grads', 'optim_grads_params'] = (
        'optim_grads_params')

    # === Megatron: pipeline layout ===
    # Interleaved (virtual) stages per rank: shrinks the bubble, costs an activation set per stage.
    virtual_pipeline_model_parallel_size: Optional[int] = None
    # Explicit layer-to-stage assignment overriding the even split.
    pipeline_model_parallel_layout: Optional[str] = None
    # First/last stage layer counts (those two stages also carry the embedding and the loss).
    decoder_first_pipeline_num_layers: Optional[int] = None
    decoder_last_pipeline_num_layers: Optional[int] = None
    # Count the embedding / the loss as a layer when splitting.
    account_for_embedding_in_pipeline_split: bool = False
    account_for_loss_in_pipeline_split: bool = False

    # === Megatron: communication overlap ===
    # Refused by plain 'muon' (whole parameters); 'dist_muon' accepts them.
    overlap_grad_reduce: bool = False
    overlap_param_gather: bool = False
    # Overlap the parameter gather with the optimizer step itself; requires overlap_param_gather.
    overlap_param_gather_with_optimizer_step: bool = False
    overlap_p2p_comm: bool = True
    # Batch the pipeline's send+receive into one call; None takes Megatron's default.
    batch_p2p_comm: Optional[bool] = None
    # Pad buckets so every rank reduces/gathers at the same offsets; without it the overlaps
    # stop overlapping.
    align_grad_reduce: bool = True
    align_param_gather: bool = True
    # TP communication/GEMM overlap; needs Transformer Engine and a userbuffer configuration.
    tp_comm_overlap: bool = False
    # Warm-up collective on every NCCL communicator so the first real step pays no setup cost.
    nccl_comm_warmup: bool = False

"""Distributed training and parallelism configuration."""
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
    #: FSDP's own settings -- wrapping policy, sharding strategy, offload. The CLI accepts a JSON
    #: object or JSON file. Nested because the accepted keys are FSDP's and change with its version.
    fsdp_config: Optional[Dict[str, Any]] = None

    # === Launch ===
    mode: Literal['ray', 'local'] = 'local'
    nproc_per_node: Optional[int] = None

    # === DDP ===
    ddp_find_unused_parameters: Optional[bool] = None
    #: Broadcast buffers (e.g. batchnorm statistics) from rank 0 each forward. None takes torch's
    #: default of True; turning it off is a throughput win for models with no such buffers.
    ddp_broadcast_buffers: Optional[bool] = None
    #: Gradient bucket size for the all-reduce. Larger buckets overlap communication better and delay
    #: the first reduction. None takes torch's default.
    ddp_bucket_cap_mb: Optional[int] = None
    #: Declare the graph unchanging so DDP can reuse its reduction order. Faster, but wrong for a model
    #: whose active parameters vary per step -- e.g. an MoE with a changing set of routed experts.
    ddp_static_graph: Optional[bool] = None
    #: This process's rank, supplied by the launcher. -1 means not launched distributed. Set by
    #: torchrun rather than by a user.
    local_rank: int = -1

    # === Megatron ===
    # This contains the distributed subset of legacy MegatronArguments. Model, optimizer, scheduler,
    # checkpoint and MoE fields live in their owning Config classes; every field below is either
    # forwarded to Twinkle/Megatron or rejected explicitly when the active backend cannot consume it.
    # A None value means "keep Twinkle/Megatron's own default".
    backend: Optional[Literal['megatron', 'hf']] = None
    bridge_backend: Literal['mcore-bridge', 'megatron-bridge'] = 'mcore-bridge'
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    # sequence_parallel: Megatron tensor-parallel sequence parallelism (splits activations along
    # the sequence dim inside TP regions). Distinct from TemplateConfig.sequence_parallel_size
    # (Ulysses/CP-style SP for the transformers path). Only effective when tp > 1.
    sequence_parallel: bool = False
    use_distributed_optimizer: bool = True
    #: Shard the parameters with Megatron-FSDP instead of replicating them under Megatron-DDP.
    #: NOT the same knob as ``fsdp`` above: that one is the transformers backend's (torch FSDP via
    #: accelerate) and has no effect on the megatron path. Named after the legacy Megatron CLI flag
    #: so args_to_configs picks it up by same-name copy.
    #: Requires ``use_distributed_optimizer`` (sharded parameters need matching master-weight shards)
    #: and is incompatible with ``context_parallel_size > 1``; both are checked in validate_configs.
    #: megatron's other implementation, Torch FSDP2, is deliberately not exposed: it additionally
    #: demands untied embeddings and a second checkpoint sharding format, neither of which the legacy
    #: surface supports either.
    use_megatron_fsdp: bool = False
    recompute_granularity: Optional[Literal['selective', 'full', 'none']] = None
    recompute_method: Optional[Literal['uniform', 'block']] = None
    recompute_num_layers: Optional[int] = None
    #: Which submodules are recomputed, e.g. 'core_attn', 'mlp', 'moe'. Finer than
    #: ``recompute_granularity``, which only chooses between whole layers and selected ops.
    recompute_modules: List[str] = field(default_factory=lambda: ['core_attn'])

    # === Megatron: expert & context parallel detail ===
    #: Tensor-parallel width used inside expert layers, which may differ from the dense
    #: ``tensor_model_parallel_size`` because an expert matrix is a different shape.
    expert_tensor_parallel_size: int = 1
    #: How context-parallel ranks exchange keys and values, e.g. 'p2p', 'a2a', 'allgather', or a
    #: per-layer list. None takes Megatron's default.
    cp_comm_type: Optional[Union[str, List[str]]] = None
    #: How the sequence is split across context-parallel ranks. 'zigzag' interleaves so every rank gets
    #: both early and late positions, which is what keeps causal-attention work even; 'contiguous'
    #: gives each rank one block and so leaves the first rank with much less to do.
    cp_partition_mode: Literal['zigzag', 'contiguous'] = 'zigzag'
    #: What the data-parallel optimizer shards. Increasing it trades communication for memory, ending at
    #: 'optim_grads_params', which is ZeRO-3-equivalent.
    data_parallel_sharding_strategy: Literal['no_shard', 'optim', 'optim_grads', 'optim_grads_params'] = (
        'optim_grads_params')

    # === Megatron: pipeline layout ===
    #: Interleaved (virtual) pipeline stages per rank. Shrinks the pipeline bubble and costs one extra
    #: activation set per virtual stage. None means non-interleaved.
    virtual_pipeline_model_parallel_size: Optional[int] = None
    #: An explicit layer-to-stage assignment, overriding the even split. The way to compensate when the
    #: first and last stages also carry the embedding and the loss.
    pipeline_model_parallel_layout: Optional[str] = None
    #: Layer counts for the first and last stages, as a lighter alternative to a full layout: the even
    #: split is otherwise unbalanced precisely because those two stages do more.
    decoder_first_pipeline_num_layers: Optional[int] = None
    decoder_last_pipeline_num_layers: Optional[int] = None
    #: Count the embedding / the loss as a layer when splitting, so the balancing arithmetic accounts
    #: for them instead of leaving those stages overloaded.
    account_for_embedding_in_pipeline_split: bool = False
    account_for_loss_in_pipeline_split: bool = False

    # === Megatron: communication overlap ===
    # Each overlaps a collective with compute. They are off or on following Megatron's own defaults;
    # note that the two gradient/parameter overlaps are refused by the plain 'muon' optimizer
    # (TrainConfig.optimizer), which needs whole parameters -- 'dist_muon' accepts them.
    overlap_grad_reduce: bool = False
    overlap_param_gather: bool = False
    #: Overlap the parameter gather with the optimizer step itself. Requires ``overlap_param_gather``.
    overlap_param_gather_with_optimizer_step: bool = False
    overlap_p2p_comm: bool = True
    #: Batch the pipeline's send and receive into one call. None takes Megatron's default, which depends
    #: on whether p2p is being overlapped.
    batch_p2p_comm: Optional[bool] = None
    #: Pad buckets so every rank reduces or gathers at the same offsets. On by default: without it the
    #: ranks issue differently-shaped collectives and the overlap above stops overlapping.
    align_grad_reduce: bool = True
    align_param_gather: bool = True
    #: Overlap tensor-parallel communication with compute. Needs Transformer Engine and a userbuffer
    #: configuration, which is why it is not on by default.
    tp_comm_overlap: bool = False
    #: Run a warm-up collective on every NCCL communicator at startup, so the first real step does not
    #: pay the connection setup and skew the initial timings.
    nccl_comm_warmup: bool = False

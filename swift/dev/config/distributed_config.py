"""Distributed training and parallelism configuration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class DistributedConfig:
    """DeepSpeed, FSDP, DDP, and Ray distributed settings."""

    # === DeepSpeed ===
    deepspeed: Optional[str] = None
    zero_hpz_partition_size: Optional[int] = None
    deepspeed_autotp_size: Optional[int] = None

    # === FSDP ===
    fsdp: Optional[str] = None

    # === Ray ===
    use_ray: bool = False
    ray_exp_name: Optional[str] = None
    device_groups: Optional[str] = None

    # === DDP ===
    ddp_timeout: int = 18000000
    ddp_backend: Optional[str] = None
    ddp_find_unused_parameters: Optional[bool] = None

    # === Megatron ===
    # NOTE: this is a MINIMAL subset of legacy MegatronArguments (megatron_args.py has 200+
    # fields). Only the parallelism sizes + the few high-frequency knobs below are wired into
    # dev's build_model path today; the rest (fusion/fp8/mtp/muon/precision-aware-optimizer/
    # vpp/... ) are intentionally deferred. Full alignment is a separate follow-up task.
    # A None value means "keep twinkle MegatronModel's own default" (so the bit-exact SFT
    # baseline is unchanged unless the user explicitly sets the knob).
    backend: Optional[Literal['megatron', 'hf']] = None
    bridge_backend: Literal['mcore-bridge', 'megatron-bridge'] = 'mcore-bridge'
    # NB: this is the twinkle launch model, NOT legacy swift's.
    mode: Literal['ray', 'local'] = 'local'
    nproc_per_node: Optional[int] = None
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

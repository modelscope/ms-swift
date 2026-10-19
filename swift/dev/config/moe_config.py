# Copyright (c) ModelScope Contributors. All rights reserved.
from dataclasses import dataclass
from typing import List, Literal, Optional, Union


@dataclass
class MoEConfig:
    """MoE routing/dispatch/capacity knobs (Megatron backend only; sizes live in DistributedConfig)."""

    # === Auxiliary losses ===
    # Load-balancing loss weight (list = per-layer); 0 risks router collapse onto few experts.
    moe_aux_loss_coeff: Union[float, List[float]] = 0.
    # Z-loss keeps router logits small and the routing numerically stable; None disables.
    moe_z_loss_coeff: Optional[float] = None
    # Per-layer balancing strategy ('aux_loss'/'seq_aux_loss'/'sinkhorn'/'none'); None = mcore default.
    moe_router_load_balancing_type: Optional[List[str]] = None
    # fp32 by default: the router decides discrete assignments, where bf16 rounding flips choices.
    moe_router_dtype: Literal['none', 'fp32', 'fp64'] = 'fp32'

    # === Token dispatch ===
    # 'alltoall' scales with expert parallelism; 'allgather' only for small EP; 'flex' picks per case.
    moe_token_dispatcher_type: Literal['allgather', 'alltoall', 'flex'] = 'alltoall'
    moe_token_drop_policy: Literal['probs', 'position'] = 'probs'
    # Expert capacity as a multiple of the even share; None = no cap, memory follows the worst expert.
    moe_expert_capacity_factor: Optional[float] = None
    # Pad each expert's input to capacity for static shapes; wasted compute otherwise.
    moe_pad_expert_input_to_capacity: bool = False

    # === Performance ===
    moe_grouped_gemm: bool = True  # one grouped GEMM instead of per-expert small ones
    moe_permute_fusion: bool = False  # fuse the expert-grouping permutation into dispatch
    # Overlap the shared expert's compute with routed-expert communication (needs a shared expert).
    moe_shared_expert_overlap: bool = False
    # Recompute the whole MoE layer in backward; the unit that matters when experts dominate.
    moe_layer_recompute: bool = False
    moe_enable_deepep: bool = False  # DeepEP is a separate dependency, hence off by default

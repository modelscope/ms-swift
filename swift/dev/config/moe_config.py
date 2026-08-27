"""Mixture-of-experts routing, dispatch, and capacity configuration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Union


@dataclass
class MoEConfig:
    """How tokens reach experts, and what keeps that routing balanced.

    Megatron-backend only: the transformers backend runs whatever routing its modelling code implements
    and reads none of these. The expert parallel *sizes* live in DistributedConfig, because they say how
    the model is placed across devices; this says how it behaves once placed.
    """

    # === Auxiliary losses ===
    #: Weight on the load-balancing loss, which pushes the router towards using experts evenly. A list
    #: gives a per-layer weight, a scalar applies to every layer. 0 disables it, at the risk of the
    #: router collapsing onto a few experts.
    moe_aux_loss_coeff: Union[float, List[float]] = 0.
    #: Weight on the z-loss, which keeps router logits small and so the routing numerically stable.
    #: None disables it.
    moe_z_loss_coeff: Optional[float] = None
    #: Balancing strategy per layer, e.g. 'aux_loss', 'seq_aux_loss', 'sinkhorn', 'none'. None takes
    #: Megatron's default.
    moe_router_load_balancing_type: Optional[List[str]] = None
    #: Precision the router's logits and softmax are computed in. fp32 by default because the router
    #: decides discrete assignments, where bf16 rounding can flip an expert choice.
    moe_router_dtype: Literal['none', 'fp32', 'fp64'] = 'fp32'

    # === Token dispatch ===
    #: How tokens are moved to their experts. 'alltoall' scales with expert parallelism; 'allgather'
    #: sends everything everywhere and is only reasonable for small EP; 'flex' is the newer
    #: implementation that picks per case.
    moe_token_dispatcher_type: Literal['allgather', 'alltoall', 'flex'] = 'alltoall'
    #: Which tokens are dropped once an expert is full -- lowest routing probability, or last position.
    moe_token_drop_policy: Literal['probs', 'position'] = 'probs'
    #: Expert capacity as a multiple of the even share. None means no cap, so nothing is dropped and
    #: memory follows the worst-imbalanced expert.
    moe_expert_capacity_factor: Optional[float] = None
    #: Pad every expert's input up to capacity, making shapes static. Needed by kernels that cannot
    #: take a ragged batch, and wasted compute otherwise.
    moe_pad_expert_input_to_capacity: bool = False

    # === Performance ===
    #: Run all experts as one grouped GEMM rather than a loop of small ones. On by default: the
    #: per-expert matrices are usually too small to saturate a GPU on their own.
    moe_grouped_gemm: bool = True
    #: Fuse the permutation that groups tokens by expert into the dispatch kernel.
    moe_permute_fusion: bool = False
    #: Overlap the shared expert's compute with the routed experts' communication. Only helps a model
    #: that actually has a shared expert.
    moe_shared_expert_overlap: bool = False
    #: Recompute the whole MoE layer in backward instead of keeping its activations. A larger unit than
    #: DistributedConfig's ``recompute_granularity``, and the one that matters when experts dominate.
    moe_layer_recompute: bool = False
    #: Use DeepEP for expert communication. A separate dependency, and the reason it is off by default.
    moe_enable_deepep: bool = False

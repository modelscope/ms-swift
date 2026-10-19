# Copyright (c) ModelScope Contributors. All rights reserved.
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional


@dataclass
class MegatronConfig:
    """Megatron engine knobs: kernel fusions, attention implementation, process initialisation.

    Parallel sizes are DistributedConfig's and MoE routing is MoEConfig's; defaults match Megatron.
    """

    # === Kernel fusion ===
    # Each folds ops into one kernel: less memory traffic, different last-bit numerics.
    apply_rope_fusion: bool = False
    bias_activation_fusion: bool = True
    bias_dropout_fusion: bool = True
    cross_entropy_loss_fusion: bool = True
    # 'te' needs Transformer Engine and is faster where available.
    cross_entropy_fusion_impl: Literal['native', 'te'] = 'native'
    # Folds grad accumulation into the weight-grad GEMM; the one fusion that can fail at import.
    gradient_accumulation_fusion: bool = True
    masked_softmax_fusion: bool = True
    # Fuse the DeepSeek sparse-attention indexer kernels (only for models with that indexer).
    apply_dsa_kernel_fusion: bool = False

    # === Attention ===
    # fp32 softmax: where a long sequence's logits overflow first.
    attention_softmax_in_fp32: bool = True
    # Scale q·k by layer number too (original Megatron fp16 stability); None = Megatron's choice.
    apply_query_key_layer_scaling: Optional[bool] = None
    # Keep q/k/v projections as separate matrices instead of one fused weight.
    linear_decoupled_in_proj: bool = False

    # === DeepSeek sparse attention (DSA) ===
    # Indexer aux-loss weight; 0 leaves the indexer untrained (needs a checkpoint with one).
    dsa_indexer_loss_coeff: float = 0.
    dsa_indexer_use_sparse_loss: bool = False  # loss over the sparse selection only

    # === Multi-head compression (MHC) / cross-sparse attention (CSA) ===
    use_fused_mhc: bool = False
    mhc_recompute_layer_num: Optional[int] = None  # MHC layers recomputed in backward; None = none
    # Run CSA layers densely to measure what the sparsity costs; everything else stays identical.
    csa_dense_mode: bool = False

    # === Initialisation ===
    # Off because dev loads a pretrained checkpoint; on for from-scratch training.
    perform_initialization: bool = False
    # Build initial weights on CPU: slower, keeps a large init off the device.
    use_cpu_initialization: bool = False
    # Per-DP-rank init seed; only meaningful without a checkpoint, wrong with one.
    data_parallel_random_init: Optional[bool] = False
    # Skip Megatron's global init entirely (e.g. a conversion that only needs the model built).
    skip_megatron_init: bool = False
    # Track RNG through Transformer Engine so dropout stays reproducible when TE owns the layers.
    te_rng_tracker: bool = False

    # === Garbage collection ===
    # Fixed-step collection; an unplanned one stalls a rank and blocks the next collective.
    manual_gc: bool = False
    manual_gc_eval: bool = True
    # Steps between manual collections; 0 disables during training, still honours manual_gc_eval.
    manual_gc_steps: int = 0

    # === Sequence handling ===
    # Skip the padding inside the MLP as well as attention; only read when padding_free is on.
    mlp_padding_free: bool = False
    # 'dp_balanced' equalises tokens per DP rank; 'default_dynamic_cp' varies the CP split per
    # batch; None keeps Megatron's static behaviour.
    sequence_packing_scheduler: Optional[Literal['dp_balanced', 'default_dynamic_cp']] = None

    # === Escape hatch ===
    # Passed straight through to Megatron's own argument namespace (JSON object or file).
    megatron_extra_kwargs: Optional[Dict[str, Any]] = None

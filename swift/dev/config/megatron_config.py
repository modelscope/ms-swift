"""Megatron engine knobs: kernel fusions, attention implementation, and process initialisation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Union


@dataclass
class MegatronConfig:
    """Megatron-specific settings with no counterpart on the transformers backend.

    What lands here rather than elsewhere: these change how Megatron builds and runs its kernels and
    its process group, not what is trained. Parallel sizes and communication overlap are
    DistributedConfig's, MoE routing is MoEConfig's, and anything the transformers backend also
    understands stays in the shared configs -- so a field here is one the other backend would have no
    use for.

    Everything defaults to what Megatron itself does today, so leaving this config untouched reproduces
    the current behaviour exactly.
    """

    # === Kernel fusion ===
    # Each folds several ops into one kernel: less memory traffic, and a different numerical result in
    # the last bits. Defaults follow Megatron's, which is why most are already on.
    apply_rope_fusion: bool = False
    bias_activation_fusion: bool = True
    bias_dropout_fusion: bool = True
    cross_entropy_loss_fusion: bool = True
    #: Which fused cross-entropy to use. 'te' needs Transformer Engine and is faster where available.
    cross_entropy_fusion_impl: Literal['native', 'te'] = 'native'
    #: Fold the gradient accumulation into the weight-gradient GEMM. Requires the apex/TE extension, so
    #: it is the one fusion that can fail at import rather than degrade.
    gradient_accumulation_fusion: bool = True
    masked_softmax_fusion: bool = True
    #: Fuse the DeepSeek sparse-attention indexer kernels. Only for models that have that indexer.
    apply_dsa_kernel_fusion: bool = False

    # === Attention ===
    #: Which attention implementation the decoder uses, e.g. 'unfused', 'flash', 'fused'. Distinct from
    #: ModelConfig's ``attn_impl``, which names an HF implementation -- the two backends accept
    #: different sets of names.
    attention_backend: str = 'unfused'
    #: Compute the attention softmax in fp32 regardless of the training dtype. On by default: the
    #: softmax is where a long sequence's logits overflow first.
    attention_softmax_in_fp32: bool = True
    #: Scale q·k by the layer number as well as head dim, as the original Megatron did for fp16
    #: stability. None leaves Megatron's own choice, which depends on the dtype.
    apply_query_key_layer_scaling: Optional[bool] = None
    #: Keep the q/k/v input projections as separate matrices instead of one fused weight.
    linear_decoupled_in_proj: bool = False

    # === DeepSeek sparse attention (DSA) ===
    #: Weight on the indexer's auxiliary loss, which trains it to select the right keys. 0 leaves the
    #: indexer untrained, so it only makes sense with a checkpoint that already has one.
    dsa_indexer_loss_coeff: float = 0.
    #: Compute that loss over the sparse selection only, rather than densely.
    dsa_indexer_use_sparse_loss: bool = False

    # === Multi-head compression (MHC) / cross-sparse attention (CSA) ===
    use_fused_mhc: bool = False
    #: Layers whose MHC block is recomputed in backward. None recomputes none.
    mhc_recompute_layer_num: Optional[int] = None
    #: Run CSA layers densely, i.e. without their sparsity. The way to check what the sparsity costs in
    #: quality, since everything else stays identical.
    csa_dense_mode: bool = False

    # === Initialisation ===
    #: Let Megatron initialise weights itself. Off because dev loads a pretrained checkpoint, where
    #: initialising first only wastes the work -- turn it on to train from scratch.
    perform_initialization: bool = False
    #: Build the initial weights on CPU. Slower, but it keeps a large model's init off the device.
    use_cpu_initialization: bool = False
    #: Give each data-parallel rank a different init seed. Only meaningful without a checkpoint, and
    #: wrong with one, since the ranks would then disagree about the weights.
    data_parallel_random_init: Optional[bool] = False
    #: Skip Megatron's global init entirely. For a process that only needs the model built, e.g. a
    #: conversion, where initialising distributed state would be pointless.
    skip_megatron_init: bool = False
    #: Track RNG state through Transformer Engine rather than Megatron. Needed for dropout to stay
    #: reproducible when TE owns the layers.
    te_rng_tracker: bool = False

    # === Garbage collection ===
    #: Take Python's GC into the training loop's own hands, collecting on a fixed step interval instead
    #: of whenever the allocator decides. It matters at scale because an unplanned collection stalls one
    #: rank and every other rank then waits at the next collective.
    manual_gc: bool = False
    manual_gc_eval: bool = True
    #: Steps between manual collections. 0 disables collecting during training while still honouring
    #: ``manual_gc_eval``.
    manual_gc_steps: int = 0

    # === Sequence handling ===
    #: Skip the padding inside the MLP as well, not just attention. Read only when the run is
    #: padding-free; see TemplateConfig's ``padding_free``.
    mlp_padding_free: bool = False
    #: How packed sequences are balanced across ranks. 'dp_balanced' equalises tokens per data-parallel
    #: rank; 'default_dynamic_cp' varies the context-parallel split per batch. None keeps Megatron's
    #: static behaviour, where one rank can end up with a much longer batch than its peers.
    sequence_packing_scheduler: Optional[Literal['dp_balanced', 'default_dynamic_cp']] = None

    # === Escape hatch ===
    #: Passed straight through to Megatron's own argument namespace. A dict, or a JSON string for
    #: command lines. This exists so a Megatron flag that has no field here is still reachable, and is
    #: the reason this config does not need to mirror all of Megatron's several hundred arguments.
    megatron_extra_kwargs: Optional[Union[dict, str]] = None

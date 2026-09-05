"""Embedding (contrastive) assembly: run_embedding orchestration.

Sibling of ``run_sft``: same atomic-Config composition, same builders, same SFTLoop. Only three
things differ, and all three follow from embeddings being per-SEQUENCE rather than per-token:

1. ``task='embedding'`` is threaded to twinkle's forward. It swaps the lm_head for an identity and
   pools the last valid token, so the loss sees ``outputs['embeddings']`` (``[n_seqs, hidden]``,
   L2-normalized) instead of logits.
2. The template runs legacy's ``task_type='embedding'`` branch, which encodes each row into
   ``anchor_*``/``positive_*``/``negative_*`` keys with ONE label per sequence, then flattens them
   into interleaved rows at collate time. dev's next-token label shift is suppressed for this
   task_type -- see ``DevMixin._NO_SHIFT_TASK_TYPES``.
3. The loss comes from ``configure_embedding_loss`` (infonce / contrastive / ...) and is set on BOTH
   backends, whereas SFT skips it under Megatron (which computes CE internally).
"""
from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        DistributedConfig,
        ModelConfig,
        TemplateConfig,
        TrainConfig,
        TunerConfig,
    )

#: twinkle's task name; selects the embedding pooling patch inside forward.
TASK = 'embedding'


def run_embedding(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    dataset_config: DatasetConfig,
    train_config: TrainConfig,
    distributed_config: DistributedConfig,
    checkpoint_config: CheckpointConfig,
    tuner_config: Optional[TunerConfig] = None,
    *,
    output_dir: str = 'output',
    _save_final: bool = True,
) -> List[dict]:
    """Assemble and run embedding training from atomic Configs. Returns loss/grad_norm history.

    Orchestration mirrors ``run_sft`` step for step (see its docstring for the resume ordering
    contract, which is unchanged here); ``_save_final`` has the same test-oriented meaning.

    The loss is chosen by ``TrainConfig.loss_type`` (default 'infonce') and Matryoshka aggregation by
    ``TrainConfig.mrl_dims``; both are read here rather than passed separately so the same Config
    that drives legacy drives this recipe.
    """
    from swift.dev.recipe.assembly import TrainAssembly

    TrainAssembly.initialize_twinkle(distributed_config)
    return _run_embedding_body(
        model_config,
        template_config,
        dataset_config,
        train_config,
        distributed_config,
        checkpoint_config,
        tuner_config,
        output_dir=output_dir,
        _save_final=_save_final)


def _run_embedding_body(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    dataset_config: DatasetConfig,
    train_config: TrainConfig,
    distributed_config: DistributedConfig,
    checkpoint_config: CheckpointConfig,
    tuner_config: Optional[TunerConfig] = None,
    *,
    output_dir: str = 'output',
    _save_final: bool = True,
) -> List[dict]:
    """The backend-agnostic embedding orchestration body (see run_embedding for the contract)."""
    from swift.dev.loss import configure_embedding_loss
    from swift.dev.recipe.assembly import TrainAssembly

    def embedding_loss(model) -> None:
        # Unlike SFT, this runs on Megatron too: under task='embedding' the Megatron scheduler pools the
        # last stage's output and calls loss_instance explicitly instead of its internal vocab-parallel
        # CE, and its set_loss binds process_group to the DP group so InfonceLoss's in-batch all-gather
        # does not deadlock ranks on earlier pipeline stages.
        configure_embedding_loss(model, loss_type=train_config.loss_type or 'infonce', mrl_dims=train_config.mrl_dims)

    return TrainAssembly(
        'run_embedding',
        model_config,
        template_config,
        dataset_config,
        train_config,
        distributed_config,
        checkpoint_config,
        tuner_config,
        task=TASK,
        output_dir=output_dir,
    ).fit(embedding_loss, save_final=_save_final)

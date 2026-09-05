"""Reranker (cross-encoder) assembly: run_reranker orchestration.

Sibling of ``run_embedding``: same atomic-Config composition, builders, and SFTLoop. A reranker
scores each query-document pair with a single relevance logit, so unlike embedding it reads
``outputs['logits']`` (``[n_seqs, 1]``), not a pooled+normalized vector. Two task_types share this
recipe:

1. ``reranker`` -- rides a ``num_labels=1`` SequenceClassification head. On transformers the head is
   built by ``build_model`` (AutoModelForSequenceClassification) and forward is a no-op patch; on
   Megatron it maps to mcore-bridge's seq_cls head (num_labels=1) and the last valid token is pooled
   in the processor. Loss: pointwise_reranker / listwise_reranker.
2. ``generative_reranker`` -- keeps the CausalLM and scores via the vocab head as
   ``logit('yes') - logit('no')``. On transformers a forward-time lm_head patch does this; on
   Megatron the bridge's generative head does. Same reranker loss.

The template runs legacy's reranker encode/collate (one label per pair), and dev's next-token label
shift is suppressed for these task_types (see ``DevMixin._NO_SHIFT_TASK_TYPES``).
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

#: task_types this recipe accepts; each threads to twinkle's forward as-is.
RERANKER_TASK_TYPES = ('reranker', 'generative_reranker')


def run_reranker(
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
    """Assemble and run reranker training from atomic Configs. Returns loss/grad_norm history.

    Orchestration mirrors ``run_embedding`` step for step (see its docstring for the resume ordering
    contract, which is unchanged here); ``_save_final`` has the same test-oriented meaning. The loss
    is chosen by ``TrainConfig.loss_type`` (default 'pointwise_reranker').
    """
    from swift.dev.recipe.assembly import TrainAssembly

    TrainAssembly.initialize_twinkle(distributed_config)
    return _run_reranker_body(
        model_config,
        template_config,
        dataset_config,
        train_config,
        distributed_config,
        checkpoint_config,
        tuner_config,
        output_dir=output_dir,
        _save_final=_save_final)


def _run_reranker_body(
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
    """The backend-agnostic reranker orchestration body (see run_reranker for the contract)."""
    from swift.dev.loss import configure_reranker_loss
    from swift.dev.recipe.assembly import TrainAssembly

    # Unlike its siblings this recipe serves a SET of task_types, so it resolves and validates its own
    # instead of letting the assembly compare against a single constant; the resolved value is what
    # routes the template (reranker vs generative_reranker encode) and the model head.
    task_type = model_config.task_type or 'reranker'
    if task_type not in RERANKER_TASK_TYPES:
        raise ValueError(f'run_reranker requires ModelConfig.task_type in {list(RERANKER_TASK_TYPES)} (or None), '
                         f'got {task_type!r}.')

    def reranker_loss(model) -> None:
        # Runs on Megatron too: under task='reranker' the Megatron scheduler reduces the per-token head
        # output to the last valid token and calls loss_instance explicitly, and set_loss binds the DP
        # process group.
        configure_reranker_loss(model, loss_type=train_config.loss_type or 'pointwise_reranker')

    return TrainAssembly(
        'run_reranker',
        model_config,
        template_config,
        dataset_config,
        train_config,
        distributed_config,
        checkpoint_config,
        tuner_config,
        task=task_type,
        output_dir=output_dir,
    ).fit(reranker_loss, save_final=_save_final)

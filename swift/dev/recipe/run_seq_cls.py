"""Sequence-classification assembly: run_seq_cls orchestration.

Sibling of ``run_embedding``: same atomic-Config composition, builders, and SFTLoop. seq_cls scores
each sequence against a fixed label set, so it reads ``outputs['logits']`` (``[B, num_labels]``) and
picks its objective by ``problem_type`` (regression -> MSE, single_label -> CE, multi_label -> BCE).

Both backends:

- transformers: ``build_model`` builds a ``num_labels``-wide SequenceClassification head; forward is
  a no-op patch (the head already returns ``[B, num_labels]``).
- Megatron: mcore-bridge builds the seq_cls ``OutputLayerLinear(hidden, num_labels)`` head; the last
  valid token is pooled in the processor before the loss runs.

The template runs legacy's seq_cls encode/collate (one label per sequence), and dev's next-token
label shift is suppressed for this task_type (see ``DevMixin._NO_SHIFT_TASK_TYPES``).
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

#: twinkle's task name; the head produces logits, so forward installs no patch.
TASK = 'seq_cls'


def run_seq_cls(
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
    """Assemble and run seq_cls training from atomic Configs. Returns loss/grad_norm history.

    Orchestration mirrors ``run_embedding`` step for step (see its docstring for the resume ordering
    contract, which is unchanged here); ``_save_final`` has the same test-oriented meaning.
    ``ModelConfig.problem_type`` is REQUIRED -- it selects the training objective and is not inferred.
    """
    from swift.dev.recipe.assembly import TrainAssembly

    TrainAssembly.initialize_twinkle(distributed_config)
    return _run_seq_cls_body(
        model_config,
        template_config,
        dataset_config,
        train_config,
        distributed_config,
        checkpoint_config,
        tuner_config,
        output_dir=output_dir,
        _save_final=_save_final)


def _run_seq_cls_body(
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
    """The backend-agnostic seq_cls orchestration body (see run_seq_cls for the contract)."""
    from swift.dev.loss import configure_seq_cls_loss
    from swift.dev.recipe.assembly import TrainAssembly

    # problem_type and num_labels are required (not inferred): the first selects the loss objective and
    # is recorded on the config for HF/legacy inference parity, the second sizes the head.
    problem_type = model_config.problem_type
    if problem_type is None:
        raise ValueError('run_seq_cls requires ModelConfig.problem_type '
                         '(regression / single_label_classification / multi_label_classification).')
    num_labels = model_config.num_labels
    if num_labels is None:
        raise ValueError('run_seq_cls requires ModelConfig.num_labels.')

    def seq_cls_loss(model) -> None:
        # Runs on Megatron too: under task='seq_cls' the Megatron scheduler pools the last stage's
        # per-token head output to the last valid token and calls loss_instance explicitly.
        configure_seq_cls_loss(model, problem_type=problem_type, num_labels=num_labels)

    return TrainAssembly(
        'run_seq_cls',
        model_config,
        template_config,
        dataset_config,
        train_config,
        distributed_config,
        checkpoint_config,
        tuner_config,
        task=TASK,
        output_dir=output_dir,
    ).fit(seq_cls_loss, save_final=_save_final)

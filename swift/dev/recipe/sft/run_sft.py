# Copyright (c) ModelScope Contributors. All rights reserved.
# The supervised family: the five visible mains plus the family's own training loop.
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from swift.dev.recipe.sft.assembly import build_supervised_objects
from swift.dev.recipe.supervised import run_supervised
from swift.dev.utils.logger import get_logger

if TYPE_CHECKING:
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        DistributedConfig,
        LoggingConfig,
        MegatronConfig,
        ModelConfig,
        MoEConfig,
        QuantizeConfig,
        TemplateConfig,
        TrainConfig,
        TunerConfig,
    )

logger = get_logger()

#: twinkle's task name; selects the embedding pooling patch inside forward.
EMBEDDING_TASK = "embedding"
#: twinkle's task name; the head produces logits, so forward installs no patch.
SEQ_CLS_TASK = "seq_cls"
#: task_types the reranker recipe accepts; each threads to twinkle's forward as-is.
RERANKER_TASK_TYPES = ("reranker", "generative_reranker")


def _supervised_micro_kwargs(task: str) -> Callable[..., Dict[str, Any]]:
    """Build forward kwargs; micro_batch_size counts original examples or packing units."""

    def micro_step_kwargs(unit, *, micro_batch_size: int):
        return {"inputs": unit, "task": task, "gradient_accumulation_steps": 1, "micro_batch_size": micro_batch_size}

    return micro_step_kwargs


def run_sft(
    model_config: "ModelConfig",
    template_config: "TemplateConfig",
    dataset_config: "DatasetConfig",
    train_config: "TrainConfig",
    distributed_config: "DistributedConfig",
    checkpoint_config: "CheckpointConfig",
    tuner_config: Optional["TunerConfig"] = None,
    logging_config: Optional["LoggingConfig"] = None,
    quantize_config: Optional["QuantizeConfig"] = None,
    megatron_config: Optional["MegatronConfig"] = None,
    moe_config: Optional["MoEConfig"] = None,
    *,
    output_dir: str = "output",
    _save_final: bool = True,
) -> List[dict]:
    """Assemble and run causal-LM SFT from atomic Configs; returns the loss/grad_norm history.

    SFT's own contribution is one thing, the objective; the config -> object stages run in
    :func:`build_supervised_objects` and the loop in :func:`run_supervised`. ``_save_final``
    (internal) skips the final checkpoint for the loss-trajectory-only test path.
    """
    from swift.dev.loss import configure_loss, liger_fused_ce_enabled

    # Fused-linear-CE pairs a loss choice with the loop's forward task (one predicate feeds both):
    # configure_loss selects Liger's kernel, loop_task='fused_lm_ce' skips the lm_head GEMM.
    fused_ce = liger_fused_ce_enabled(train_config)

    def sft_loss(model) -> None:
        configure_loss(
            model,
            loss_type=train_config.loss or "cross_entropy",
            enable_channel_loss=train_config.enable_channel_loss,
            dft=train_config.enable_dft_loss,
            fused_linear_cross_entropy=fused_ce,
            label_smoothing=train_config.label_smoothing_factor,
        )

    cluster = build_supervised_objects(
        recipe="run_sft",
        model_config=model_config,
        template_config=template_config,
        dataset_config=dataset_config,
        train_config=train_config,
        distributed_config=distributed_config,
        checkpoint_config=checkpoint_config,
        tuner_config=tuner_config,
        logging_config=logging_config,
        quantize_config=quantize_config,
        megatron_config=megatron_config,
        moe_config=moe_config,
        task=None,
        loop_task=("fused_lm_ce" if fused_ce else None),
        configure_loss=sft_loss,
        configure_metric=None,
        output_dir=output_dir,
    )
    return run_supervised(
        cluster,
        model_config,
        template_config,
        dataset_config,
        train_config,
        distributed_config,
        checkpoint_config,
        tuner_config,
        logging_config,
        quantize_config,
        megatron_config,
        micro_step_fn=_supervised_micro_kwargs(cluster.forward_task),
        output_dir=output_dir,
        _save_final=_save_final,
    )


def run_pt(
    model_config: "ModelConfig",
    template_config: "TemplateConfig",
    dataset_config: "DatasetConfig",
    train_config: "TrainConfig",
    distributed_config: "DistributedConfig",
    checkpoint_config: "CheckpointConfig",
    tuner_config: Optional["TunerConfig"] = None,
    logging_config: Optional["LoggingConfig"] = None,
    quantize_config: Optional["QuantizeConfig"] = None,
    megatron_config: Optional["MegatronConfig"] = None,
    moe_config: Optional["MoEConfig"] = None,
    *,
    output_dir: str = "output",
    _save_final: bool = True,
) -> List[dict]:
    """Run causal-LM pretraining while enforcing the non-chat loss contract (a tightened run_sft)."""
    if model_config.task_type not in (None, "causal_lm"):
        raise ValueError(f'PT requires task_type="causal_lm", got {model_config.task_type!r}.')
    if template_config.use_chat_template not in (None, False):
        raise ValueError("PT requires use_chat_template=false.")
    if template_config.loss_scale not in ("default", "all"):
        raise ValueError('PT requires loss_scale="all".')
    model_config.task_type = "causal_lm"
    template_config.use_chat_template = False
    template_config.loss_scale = "all"
    return run_sft(
        model_config,
        template_config,
        dataset_config,
        train_config,
        distributed_config,
        checkpoint_config,
        tuner_config,
        logging_config,
        quantize_config,
        megatron_config,
        moe_config,
        output_dir=output_dir,
        _save_final=_save_final,
    )


def run_embedding(
    model_config: "ModelConfig",
    template_config: "TemplateConfig",
    dataset_config: "DatasetConfig",
    train_config: "TrainConfig",
    distributed_config: "DistributedConfig",
    checkpoint_config: "CheckpointConfig",
    tuner_config: Optional["TunerConfig"] = None,
    logging_config: Optional["LoggingConfig"] = None,
    quantize_config: Optional["QuantizeConfig"] = None,
    *,
    output_dir: str = "output",
    _save_final: bool = True,
) -> List[dict]:
    """Assemble and run embedding training from atomic Configs; returns the loss/grad_norm history.

    The loss is chosen by ``TrainConfig.loss`` (default 'infonce') and Matryoshka aggregation by
    ``TrainConfig.mrl_dims``; both are read here rather than passed separately so the same Config
    that drives legacy drives this recipe. ``_save_final`` has the same test-oriented meaning.
    """
    from swift.dev.loss import configure_embedding_loss, configure_embedding_metric

    def embedding_loss(model) -> None:
        # Runs on Megatron too: its scheduler pools the last stage's output and calls loss_instance
        # explicitly, with process_group bound to the DP group so the in-batch all-gather is safe.
        configure_embedding_loss(model, loss_type=train_config.loss or "infonce", mrl_dims=train_config.mrl_dims)

    cluster = build_supervised_objects(
        recipe="run_embedding",
        model_config=model_config,
        template_config=template_config,
        dataset_config=dataset_config,
        train_config=train_config,
        distributed_config=distributed_config,
        checkpoint_config=checkpoint_config,
        tuner_config=tuner_config,
        logging_config=logging_config,
        quantize_config=quantize_config,
        megatron_config=None,
        moe_config=None,
        task=EMBEDDING_TASK,
        loop_task=None,
        configure_loss=embedding_loss,
        configure_metric=configure_embedding_metric,
        output_dir=output_dir,
    )
    return run_supervised(
        cluster,
        model_config,
        template_config,
        dataset_config,
        train_config,
        distributed_config,
        checkpoint_config,
        tuner_config,
        logging_config,
        quantize_config,
        None,
        micro_step_fn=_supervised_micro_kwargs(cluster.forward_task),
        output_dir=output_dir,
        _save_final=_save_final,
    )


def run_reranker(
    model_config: "ModelConfig",
    template_config: "TemplateConfig",
    dataset_config: "DatasetConfig",
    train_config: "TrainConfig",
    distributed_config: "DistributedConfig",
    checkpoint_config: "CheckpointConfig",
    tuner_config: Optional["TunerConfig"] = None,
    logging_config: Optional["LoggingConfig"] = None,
    quantize_config: Optional["QuantizeConfig"] = None,
    *,
    output_dir: str = "output",
    _save_final: bool = True,
) -> List[dict]:
    """Assemble and run reranker training from atomic Configs; returns the loss/grad_norm history.

    Serves a SET of task_types (``reranker`` rides a num_labels=1 SequenceClassification head;
    ``generative_reranker`` keeps the CausalLM and scores via the vocab head), so it resolves and
    validates its own rather than comparing against a single constant; the resolved value routes
    the template (reranker vs generative_reranker encode) and the model head.
    """
    from swift.dev.loss import configure_reranker_loss

    task_type = model_config.task_type or "reranker"
    if task_type not in RERANKER_TASK_TYPES:
        raise ValueError(
            f"run_reranker requires ModelConfig.task_type in {list(RERANKER_TASK_TYPES)} (or None), got {task_type!r}."
        )

    def reranker_loss(model) -> None:
        # Runs on Megatron too: under task='reranker' the Megatron scheduler reduces the per-token
        # head output to the last valid token and calls loss_instance explicitly.
        configure_reranker_loss(model, loss_type=train_config.loss or "pointwise_reranker")

    cluster = build_supervised_objects(
        recipe="run_reranker",
        model_config=model_config,
        template_config=template_config,
        dataset_config=dataset_config,
        train_config=train_config,
        distributed_config=distributed_config,
        checkpoint_config=checkpoint_config,
        tuner_config=tuner_config,
        logging_config=logging_config,
        quantize_config=quantize_config,
        megatron_config=None,
        moe_config=None,
        task=task_type,
        loop_task=None,
        configure_loss=reranker_loss,
        configure_metric=None,
        output_dir=output_dir,
    )
    return run_supervised(
        cluster,
        model_config,
        template_config,
        dataset_config,
        train_config,
        distributed_config,
        checkpoint_config,
        tuner_config,
        logging_config,
        quantize_config,
        None,
        micro_step_fn=_supervised_micro_kwargs(cluster.forward_task),
        output_dir=output_dir,
        _save_final=_save_final,
    )


def run_seq_cls(
    model_config: "ModelConfig",
    template_config: "TemplateConfig",
    dataset_config: "DatasetConfig",
    train_config: "TrainConfig",
    distributed_config: "DistributedConfig",
    checkpoint_config: "CheckpointConfig",
    tuner_config: Optional["TunerConfig"] = None,
    logging_config: Optional["LoggingConfig"] = None,
    quantize_config: Optional["QuantizeConfig"] = None,
    *,
    output_dir: str = "output",
    _save_final: bool = True,
) -> List[dict]:
    """Assemble and run seq_cls training from atomic Configs; returns the loss/grad_norm history.

    ``ModelConfig.problem_type`` is REQUIRED -- it selects the training objective (regression -> MSE,
    single_label -> CE, multi_label -> BCE) and is not inferred; ``num_labels`` sizes the head.
    """
    from swift.dev.loss import configure_seq_cls_loss

    problem_type = model_config.problem_type
    if problem_type is None:
        raise ValueError(
            "run_seq_cls requires ModelConfig.problem_type "
            "(regression / single_label_classification / multi_label_classification)."
        )
    num_labels = model_config.num_labels
    if num_labels is None:
        raise ValueError("run_seq_cls requires ModelConfig.num_labels.")

    def seq_cls_loss(model) -> None:
        # Runs on Megatron too: under task='seq_cls' the Megatron scheduler pools the last stage's
        # per-token head output to the last valid token and calls loss_instance explicitly.
        configure_seq_cls_loss(model, problem_type=problem_type, num_labels=num_labels)

    cluster = build_supervised_objects(
        recipe="run_seq_cls",
        model_config=model_config,
        template_config=template_config,
        dataset_config=dataset_config,
        train_config=train_config,
        distributed_config=distributed_config,
        checkpoint_config=checkpoint_config,
        tuner_config=tuner_config,
        logging_config=logging_config,
        quantize_config=quantize_config,
        megatron_config=None,
        moe_config=None,
        task=SEQ_CLS_TASK,
        loop_task=None,
        configure_loss=seq_cls_loss,
        configure_metric=None,
        output_dir=output_dir,
    )
    return run_supervised(
        cluster,
        model_config,
        template_config,
        dataset_config,
        train_config,
        distributed_config,
        checkpoint_config,
        tuner_config,
        logging_config,
        quantize_config,
        None,
        micro_step_fn=_supervised_micro_kwargs(cluster.forward_task),
        output_dir=output_dir,
        _save_final=_save_final,
    )

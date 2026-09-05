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
import logging
import math
import os
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

logger = logging.getLogger(__name__)

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
    from swift.dev.recipe.run_sft import _initialize_twinkle

    _initialize_twinkle(distributed_config)
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
    from swift.dev.adapter import apply_tuner
    from swift.dev.builders import build_dataset, build_model, build_template
    from swift.dev.config import validate_configs
    from swift.dev.loss import configure_reranker_loss
    from swift.dev.optimizer import configure_optimizer, resolve_max_grad_norm
    from swift.dev.processor import InputProcessor
    from swift.dev.recipe.run_sft import _write_ckpt_args_json
    from swift.dev.recipe.train_loop import SFTLoop, num_optimizer_steps
    from swift.model import get_model_processor

    validate_configs(model_config, template_config, dataset_config, train_config, distributed_config, checkpoint_config,
                     tuner_config)

    # task_type routes the template (reranker vs generative_reranker encode) and the model head.
    # Default to plain 'reranker': reaching this recipe already states the intent, and a 'causal_lm'
    # ModelConfig here would encode SFT-shaped rows the reranker loss cannot score.
    task_type = model_config.task_type or 'reranker'
    if task_type not in RERANKER_TASK_TYPES:
        raise ValueError(f'run_reranker requires ModelConfig.task_type in {list(RERANKER_TASK_TYPES)} (or None), '
                         f'got {task_type!r}.')

    ga = train_config.gradient_accumulation_steps

    resume_dir = checkpoint_config.resume_from_checkpoint
    resume_only_model = checkpoint_config.resume_only_model
    redirect_to_ckpt = bool(resume_dir) and (not resume_only_model) and (tuner_config is None)

    # TODO: refactor to get only processor
    _, processor = get_model_processor(model_config.model, model_type=model_config.model_type, load_model=False)
    # task_type reaches the template explicitly: a load_model=False processor never populates
    # model_info.task_type, so it would default to 'causal_lm' and encode single-sequence SFT rows.
    template = build_template(template_config, processor, task_type=task_type)

    dataloader, eval_dataloader = build_dataset(
        dataset_config, template, train_config, distributed_config=distributed_config, template_config=template_config)

    if train_config.max_steps and train_config.max_steps > 0:
        total_opt_steps = train_config.max_steps
    else:
        try:
            micro_per_epoch = len(dataloader)
        except TypeError:
            micro_per_epoch = 0  # IterableDataset: caller must set max_steps
        total_micro = math.ceil(micro_per_epoch * train_config.num_train_epochs)
        total_opt_steps = num_optimizer_steps(total_micro, ga)

    if total_opt_steps <= 0:
        raise ValueError(f'run_reranker: computed {total_opt_steps} optimizer steps -- the dataloader is too small '
                         f'for gradient_accumulation_steps={ga}, or it is a streaming/iterable dataset with no '
                         f'max_steps. Set TrainConfig.max_steps explicitly, or provide enough data.')

    if redirect_to_ckpt:
        import copy as _copy
        model_config_for_build = _copy.copy(model_config)
        model_config_for_build.model = resume_dir
        model = build_model(model_config_for_build, distributed_config, train_config, tuner_config)
    else:
        model = build_model(model_config, distributed_config, train_config, tuner_config)

    if tuner_config is not None:
        apply_tuner(model, tuner_config, gradient_accumulation_steps=ga)

    model.set_processor(InputProcessor, padding_free=template_config.padding_free)
    model.set_template(template)

    # Runs on Megatron too: under a reranker task the Megatron scheduler pools the last stage's
    # per-token head output to the last valid token and calls loss_instance explicitly, and set_loss
    # binds process_group to the DP group so any in-batch gather does not deadlock earlier PP stages.
    configure_reranker_loss(model, loss_type=train_config.loss_type or 'pointwise_reranker')
    configure_optimizer(model, train_config, num_training_steps=total_opt_steps)

    loop = SFTLoop(
        model,
        dataloader,
        max_steps=total_opt_steps,
        num_train_epochs=train_config.num_train_epochs,
        gradient_accumulation_steps=ga,
        max_grad_norm=resolve_max_grad_norm(train_config),
        output_dir=output_dir,
        eval_dataloader=eval_dataloader,
        eval_steps=train_config.eval_steps,
        save_steps=checkpoint_config.save_steps,
        task=task_type,
    )

    if resume_dir:
        resume_kwargs = {'resume_only_model': resume_only_model}
        if tuner_config is not None:
            resume_kwargs['adapter_name'] = 'default'
        state = model.resume_from_checkpoint(resume_dir, **resume_kwargs)
        loop.resume(state)

    history = loop.fit()
    if _save_final:
        loop.save('checkpoint-final')
        ckpt_dir = os.path.join(output_dir, 'checkpoint-final')
        # task_type is a force_load key on the infer side, so writing it makes `swift infer <ckpt>`
        # load the reranker head instead of defaulting to causal_lm.
        _write_ckpt_args_json(ckpt_dir, processor, model_config, template_config, tuner_config)
    return history

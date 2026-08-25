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
    from swift.dev.recipe.run_sft import _initialize_twinkle

    _initialize_twinkle(distributed_config)
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
    from swift.dev.adapter import apply_tuner
    from swift.dev.builders import build_dataset, build_model, build_template
    from swift.dev.config import validate_configs
    from swift.dev.loss import configure_embedding_loss
    from swift.dev.optimizer import configure_optimizer, resolve_max_grad_norm
    from swift.dev.processor import InputProcessor
    from swift.dev.recipe.run_sft import _write_ckpt_args_json
    from swift.dev.recipe.train_loop import SFTLoop, num_optimizer_steps
    from swift.model import get_model_processor

    validate_configs(model_config, template_config, dataset_config, train_config, distributed_config, checkpoint_config,
                     tuner_config)

    # task_type is what routes the template (and, below, the model head) down the embedding path.
    # Default it rather than demand it: reaching this recipe already states the intent, and a
    # 'causal_lm' ModelConfig here would silently encode SFT-shaped rows the contrastive loss cannot
    # group. An explicit conflicting value is a user error worth surfacing.
    task_type = model_config.task_type or TASK
    if task_type != TASK:
        raise ValueError(f'run_embedding requires ModelConfig.task_type={TASK!r} (or None), got {task_type!r}. '
                         'Other task types encode a different row layout and reach a different loss.')

    ga = train_config.gradient_accumulation_steps

    resume_dir = checkpoint_config.resume_from_checkpoint
    resume_only_model = checkpoint_config.resume_only_model
    redirect_to_ckpt = bool(resume_dir) and (not resume_only_model) and (tuner_config is None)

    # TODO: refactor to get only processor
    _, processor = get_model_processor(model_config.model, load_model=False)
    # task_type reaches the template explicitly: it is normally read off model_info.task_type, which
    # a load_model=False processor never populates, so it would default to 'causal_lm' and encode
    # single-sequence SFT rows instead of anchor/positive/negative groups.
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
        raise ValueError(f'run_embedding: computed {total_opt_steps} optimizer steps -- the dataloader is too small '
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

    # Unlike SFT, this runs on Megatron too: under task='embedding' the Megatron scheduler pools the
    # last stage's output and calls loss_instance explicitly instead of its internal vocab-parallel
    # CE, and its set_loss binds process_group to the DP group so InfonceLoss's in-batch all-gather
    # does not deadlock ranks on earlier pipeline stages.
    configure_embedding_loss(model, loss_type=train_config.loss_type or 'infonce', mrl_dims=train_config.mrl_dims)
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
        task=TASK,
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
        # task_type is a force_load key on the infer side, so writing it is what makes
        # `swift infer <ckpt>` load the embedding head instead of defaulting to causal_lm.
        _write_ckpt_args_json(ckpt_dir, processor, model_config, template_config, tuner_config)
    return history

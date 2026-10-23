# Copyright (c) ModelScope Contributors. All rights reserved.
# The supervised family's assembly: the config -> object stages, up to (not including) the loop.
from __future__ import annotations
import copy
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Optional

from swift.dev.recipe.utils import (
    initialize_supervised_twinkle,
    plan_supervised_steps,
    process_and_validate,
    resolve_step_intervals,
    write_ckpt_args_json,
)
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


def _require_task_type(recipe: str, model_config: "ModelConfig", task: str) -> str:
    """Default ``ModelConfig.task_type`` to this recipe's ``task``, and refuse a conflicting one.

    Defaulting rather than demanding: reaching a recipe already states the intent, while (say) a
    ``'causal_lm'`` task_type on the embedding path would silently encode SFT-shaped rows that the
    contrastive loss cannot group. An explicitly different value is a user error.
    """
    task_type = model_config.task_type or task
    if task_type != task:
        raise ValueError(
            f"{recipe} requires ModelConfig.task_type={task!r} (or None), got "
            f"{task_type!r}. Other task types encode a different row layout and reach a "
            "different loss."
        )
    return task_type


class SupervisedCluster(NamedTuple):
    """Every object the supervised loop drives (a plain handle bundle, not a layer)."""

    model: Any
    processor: Any
    template: Any
    dataloader: Any
    eval_dataloader: Any
    #: The optimizer-step budget the LR scheduler was configured against.
    total_opt_steps: int
    #: The task_type the rows were encoded with ('causal_lm' / 'embedding' / 'seq_cls' / ...).
    task_type: Optional[str]
    #: The twinkle task each forward runs under; fused-linear-CE overrides it to 'fused_lm_ce'.
    forward_task: str


def build_supervised_objects(
    *,
    recipe: str,
    model_config: "ModelConfig",
    template_config: "TemplateConfig",
    dataset_config: "DatasetConfig",
    train_config: "TrainConfig",
    distributed_config: "DistributedConfig",
    checkpoint_config: Optional["CheckpointConfig"] = None,
    tuner_config: Optional["TunerConfig"] = None,
    logging_config: Optional["LoggingConfig"] = None,
    quantize_config: Optional["QuantizeConfig"] = None,
    megatron_config: Optional["MegatronConfig"] = None,
    moe_config: Optional["MoEConfig"] = None,
    task: Optional[str] = None,
    loop_task: Optional[str] = None,
    configure_loss: Callable[[Any], None],
    configure_metric: Optional[Callable[[Any], None]] = None,
    output_dir: str = "output",
) -> SupervisedCluster:
    """Run the config -> object stages in the locked order, up to (not including) the training loop.

    ``configure_loss`` is the one thing a training recipe genuinely owns -- which objective this run
    optimises -- so it arrives as a callable taking the built model, invoked between the model and
    the optimizer (which must see the loss's parameters) and after the tuner. The optional
    ``configure_metric`` peer runs AFTER ``configure_optimizer``, once the group's metric list exists.
    """
    from swift.dev.builders import (
        apply_full_param_freeze,
        build_dataset,
        build_dp_mesh,
        build_model,
        build_template,
        load_model_processor,
    )
    from swift.dev.builders.adapter import apply_tuner
    from swift.dev.builders.optimizer import configure_optimizer
    from swift.dev.processor import InputProcessor, processor_kwargs

    initialize_supervised_twinkle(
        distributed_config,
        seed=train_config.seed,
        full_determinism=train_config.full_determinism,
        tf32=train_config.tf32,
    )
    process_and_validate(
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
    )
    sp = template_config.sequence_parallel_size
    sp_mesh = build_dp_mesh(distributed_config, sp) if sp > 1 else None
    task_type = model_config.task_type
    if task is not None:
        task_type = _require_task_type(recipe, model_config, task)
    _, processor = load_model_processor(model_config)
    kwargs = {"task_type": task_type} if task_type else {}
    template = build_template(template_config, processor, **kwargs)
    # Match legacy save_args: keep one run-level args.json that every periodic/final checkpoint
    # copies, so --load_args works for a checkpoint produced before training reaches checkpoint-final.
    write_ckpt_args_json(
        output_dir, processor, model_config, template_config, tuner_config, quantize_config,
        dataset_config=dataset_config, task_type=task_type
    )
    dataloader, eval_dataloader = build_dataset(
        dataset_config, template, train_config, distributed_config, template_config=template_config, device_mesh=sp_mesh
    )
    total_opt_steps = plan_supervised_steps(recipe, train_config, dataloader)
    resolve_step_intervals(train_config, checkpoint_config, total_opt_steps)
    # Full-param resume loads weights from the ckpt dir instead of the original model id, mirroring
    # legacy's ``_init_ckpt_dir``; LoRA keeps the original base and reloads the adapter via resume.
    resume_dir = checkpoint_config.resume_from_checkpoint if checkpoint_config is not None else None
    build_config = model_config
    if resume_dir and tuner_config is None:
        build_config = copy.copy(model_config)
        build_config.model = resume_dir
    model = build_model(
        build_config,
        distributed_config,
        train_config,
        tuner_config,
        device_mesh=sp_mesh,
        quantize_config=quantize_config,
        megatron_config=megatron_config,
        moe_config=moe_config,
    )
    if tuner_config is not None:
        # task_type threads the seq_cls/reranker head into modules_to_save: LoRA trains/saves
        # target_modules only, so without it the head is neither trained nor persisted.
        apply_tuner(
            model,
            tuner_config,
            gradient_accumulation_steps=train_config.gradient_accumulation_steps,
            task_type=task_type,
        )
    else:
        # Full-parameter path: consume the freeze/trainable knobs BEFORE configure_optimizer reads
        # requires_grad to build its param groups; an adapter run selects through target_modules.
        apply_full_param_freeze(model, train_config)
    model.set_processor(
        InputProcessor,
        padding_free=template_config.padding_free,
        # The collator must pad on the same side as the Template (builders/template.py forwards
        # padding_side to it), or a training-time --padding_side left reaches the Template only.
        padding_side=template_config.padding_side,
        cp_partition_mode=distributed_config.cp_partition_mode,
        **processor_kwargs(template),
    )
    model.set_template(template)
    configure_loss(model)
    configure_optimizer(
        model,
        train_config,
        num_training_steps=total_opt_steps,
        distributed_config=distributed_config,
        tuner_config=tuner_config,
        plugin_config=None,
    )
    if configure_metric is not None:
        configure_metric(model)
    return SupervisedCluster(
        model=model,
        processor=processor,
        template=template,
        dataloader=dataloader,
        eval_dataloader=eval_dataloader,
        total_opt_steps=total_opt_steps,
        task_type=task_type,
        forward_task=loop_task or task or "causal_lm",
    )

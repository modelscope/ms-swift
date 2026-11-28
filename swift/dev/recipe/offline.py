# Copyright (c) ModelScope Contributors. All rights reserved.
"""Offline preparation shared by task training and preference optimization."""

from __future__ import annotations
import copy
from typing import TYPE_CHECKING, Callable, Optional, Sequence

from swift.dev.config.adapter_config import has_adapter
from swift.dev.recipe.configs import RecipeConfigs
from swift.dev.recipe.preparation import prepare_template, prepare_training_model
from swift.dev.recipe.resources import OfflineResources
from swift.dev.recipe.utils import (
    initialize_supervised_twinkle,
    plan_supervised_steps,
    process_and_validate,
    resolve_step_intervals,
)

if TYPE_CHECKING:
    from swift.dev.recipe.preparation import TrainingModel


def build_offline_objects(
    configs: RecipeConfigs,
    *,
    recipe: str,
    configure_loss: Callable[[TrainingModel], None],
    configure_metric: Optional[Callable[[TrainingModel], None]] = None,
    output_dir: str = "output",
    encode: bool = True,
    template_mode: Optional[str] = None,
    auxiliary_groups: Sequence[tuple[str, int]] = (),
    forward_task: Optional[str] = None,
) -> OfflineResources:
    """Resolve data and budget before configuring the trainable model."""
    from twinkle.perf import Perf

    from swift.dev.builder import build_dataset, build_dp_mesh
    from swift.dev.builder.remote import configure_remote_loss, configure_remote_metric

    model_config, template_config, train_config = configs.model, configs.template, configs.train
    distributed_config, tuner_config, rl_config = configs.distributed, configs.tuner, configs.rl
    remote = distributed_config.mode == "remote"
    if rl_config is None:
        Perf.initialize({})
    if not remote or rl_config is not None:
        initialize_supervised_twinkle(
            distributed_config,
            seed=train_config.seed,
            full_determinism=train_config.full_determinism,
            tf32=train_config.tf32,
            auxiliary_groups=auxiliary_groups,
        )
    process_and_validate(
        model_config,
        template_config,
        configs.dataset,
        train_config,
        distributed_config,
        configs.checkpoint,
        tuner_config,
        configs.logging,
        configs.quantize,
        configs.megatron,
        configs.moe,
        rl_config=rl_config,
        reward_config=configs.reward,
    )
    task_type = model_config.task_type
    if (
        rl_config is None
        and remote
        and not has_adapter(tuner_config)
        and task_type in ("seq_cls", "reranker", "decision")
    ):
        raise ValueError(
            f"--mode remote with task_type={task_type!r} and no tuner cannot attach the task "
            "readout head: the server-side head rides the adapter channel. Use --tuner lora, "
            "or run mode=local/ray for full-parameter task heads."
        )
    sp = template_config.sequence_parallel_size
    sp_mesh = (
        None
        if (remote and rl_config is not None) or (distributed_config.mode == "ray" and sp <= 1)
        else build_dp_mesh(distributed_config, sp)
    )
    processor, template = prepare_template(
        configs,
        task_type=task_type,
        output_dir=output_dir,
        mode=template_mode,
    )
    dataloader, eval_dataloader = build_dataset(
        configs.dataset,
        template,
        train_config,
        distributed_config,
        template_config=template_config,
        device_mesh=sp_mesh,
        **({"encode": False} if not encode else {}),
    )
    steps = plan_supervised_steps(recipe, train_config, dataloader)
    resolve_step_intervals(train_config, configs.checkpoint, steps)
    build_config = model_config
    resume = configs.checkpoint.resume_from_checkpoint if configs.checkpoint is not None else None
    if resume and not has_adapter(tuner_config) and not remote:
        build_config = copy.copy(model_config)
        build_config.model = resume

    def loss(model: TrainingModel) -> None:
        if remote and rl_config is None:
            configure_remote_loss(model, train_config, task_type=task_type, model_config=model_config)
        else:
            configure_loss(model)

    def metric(model: TrainingModel) -> None:
        if remote and rl_config is None:
            configure_remote_metric(model, task_type)
        elif configure_metric is not None:
            configure_metric(model)

    model = prepare_training_model(
        configs,
        template,
        num_training_steps=steps,
        device_mesh=sp_mesh,
        build_config=build_config,
        configure_loss=loss,
        configure_metric=metric,
    )
    if forward_task is None:
        forward_task = template.task_type
    return OfflineResources(
        model=model,
        processor=processor,
        template=template,
        dataloader=dataloader,
        eval_dataloader=eval_dataloader,
        total_opt_steps=steps,
        task_type=task_type,
        forward_task=forward_task,
    )

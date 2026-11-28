# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared preparation stages over the existing builders and model handles."""

from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Optional

from swift.dev.config.adapter_config import has_adapter

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase, ProcessorMixin
    from twinkle import DeviceMesh

    from swift.dev.builder.remote import RemoteModel
    from swift.dev.config import (
        ModelConfig,
    )
    from swift.dev.model import TrainableModel
    from swift.dev.recipe.configs import RecipeConfigs
    from swift.dev.template import Template

    TrainingModel = TrainableModel | RemoteModel
    ModelProcessor = PreTrainedTokenizerBase | ProcessorMixin


def prepare_template(
    configs: RecipeConfigs,
    *,
    task_type: Optional[str],
    output_dir: str,
    mode: Optional[str] = None,
) -> tuple[ModelProcessor, Template]:
    """Load the encoding side once and persist the run's resolved template metadata."""
    from swift.dev.builder import build_template, load_model_processor
    from swift.dev.recipe.utils import write_ckpt_args_json

    _, processor = load_model_processor(configs.model)
    template = build_template(configs.template, processor, **({"task_type": task_type} if task_type else {}))
    if mode is not None:
        template.set_mode(mode)
    write_ckpt_args_json(
        output_dir,
        processor,
        configs.model,
        configs.template,
        configs.tuner,
        configs.quantize,
        dataset_config=configs.dataset,
        task_type=task_type,
    )
    return processor, template


def prepare_training_model(
    configs: RecipeConfigs,
    template: Template,
    *,
    num_training_steps: int,
    configure_loss: Callable[[TrainingModel], None],
    configure_metric: Optional[Callable[[TrainingModel], None]] = None,
    device_mesh: Optional[DeviceMesh] = None,
    build_config: Optional[ModelConfig] = None,
    parameter_dtype: Optional[str] = None,
) -> TrainingModel:
    """Prepare parameters and loss before the optimizer; metrics attach after it."""
    from swift.dev.builder import apply_full_param_freeze, build_model
    from swift.dev.builder.adapter import apply_tuner
    from swift.dev.builder.optimizer import configure_optimizer
    from swift.dev.builder.remote import configure_remote_processor
    from swift.dev.processor import InputProcessor, input_processor_kwargs

    model_config, template_config = configs.model, configs.template
    train_config, distributed_config, tuner_config = configs.train, configs.distributed, configs.tuner
    remote = distributed_config.mode == "remote"
    build_config = model_config if build_config is None else build_config
    enable_router_replay = configs.rl is not None and configs.rl.router_replay_mode != "disabled"
    model = build_model(
        build_config,
        distributed_config,
        train_config,
        tuner_config,
        device_mesh=device_mesh,
        quantize_config=configs.quantize,
        megatron_config=configs.megatron,
        moe_config=configs.moe,
        **({"torch_dtype": parameter_dtype} if parameter_dtype is not None else {}),
        **({"enable_router_replay": True} if enable_router_replay else {}),
    )
    if has_adapter(tuner_config):
        apply_tuner(
            model,
            tuner_config,
            gradient_accumulation_steps=1,
            task_type=template.task_type,
            **(
                {"model_config": model_config}
                if remote
                or configs.rl is None
                or configs.rl.rl_type not in ("dpo", "kto", "cpo", "orpo", "simpo", "rm")
                else {}
            ),
        )
    else:
        apply_full_param_freeze(model, train_config)
    if remote:
        configure_remote_processor(model, template_config, model_config, distributed_config)
    else:
        options = input_processor_kwargs(template, template_config, distributed_config, model_config)
        if configs.rl is None:
            from twinkle import Platform

            from swift.dev.processor import processor_cp_size

            if (
                distributed_config.fsdp
                and distributed_config.backend != "megatron"
                and (distributed_config.nproc_per_node or Platform.get_world_size()) > 1
                and processor_cp_size(distributed_config) == 1
                and template.multimodal_graph_adapter is not None
            ):
                options["uniform_multimodal_graph"] = True
        model.set_processor(InputProcessor, **options)
        model.set_template(template)
    configure_loss(model)
    configure_optimizer(
        model,
        train_config,
        num_training_steps=num_training_steps,
        distributed_config=distributed_config,
        tuner_config=tuner_config,
        plugin_config=None,
    )
    if configure_metric is not None:
        configure_metric(model)
    return model

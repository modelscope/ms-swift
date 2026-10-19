# Copyright (c) ModelScope Contributors. All rights reserved.
# The supervised family's assembly: the config -> object stages, up to (not including) the loop.
from __future__ import annotations

import copy
import math
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, List, NamedTuple, Optional

from swift.dev.utils.logger import get_logger  # TODO(dev2): bridge until the utils batch

from swift.dev2.recipe.utils import write_ckpt_args_json

if TYPE_CHECKING:
    from swift.dev2.config import (
        CheckpointConfig, DatasetConfig, DistributedConfig, LoggingConfig, MegatronConfig, ModelConfig,
        MoEConfig, QuantizeConfig, TemplateConfig, TrainConfig, TunerConfig)

logger = get_logger()


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


class GenerativeEval(NamedTuple):
    """The resident generative-eval sampler plus the inputs its EvalScope run needs."""

    sampler: Any
    template: Any
    datasets: List[str]
    model_id: Optional[str]
    task_config: Optional[Dict[str, Any]]
    #: The colocate weight-sync + memory schedule bracketing each run (None on transformers-live).
    enter: Optional[Callable[[], None]]
    exit: Optional[Callable[[], None]]


def _initialize_supervised_twinkle(distributed_config: 'DistributedConfig', *, seed: int,
                                   full_determinism: bool, tf32: Optional[bool]) -> None:
    """Initialize twinkle for EVERY backend -- required for a DeviceMesh to reach the model.

    mode='ray' builds the 'model' DeviceGroup the workers hold (build_model places the model there
    via remote_group='model') while the driver orchestrates; everything else initializes 'local'.
    ``seed`` / ``full_determinism`` / ``tf32`` travel as ONE bundle so twinkle re-seeds every ray
    worker from the same values -- without this it falls back to seed=42 and overwrites the config.
    """
    import twinkle
    from twinkle import DeviceGroup

    extra_torch_kwargs = {'seed': seed, 'full_determinism': full_determinism, 'tf32': tf32}
    if distributed_config.mode == 'ray':
        nproc = distributed_config.nproc_per_node
        if nproc is None:
            raise ValueError("DistributedConfig.nproc_per_node is required in mode='ray' (it sizes the Ray "
                             "'model' DeviceGroup). Pass it explicitly -- there is no default.")
        groups = [DeviceGroup(name='model', ranks=list(range(nproc)), device_type='GPU', gpus_per_worker=1)]
        twinkle.initialize(
            mode='ray',
            nproc_per_node=nproc,
            extra_torch_kwargs=extra_torch_kwargs,
            # --ray_exp_name names this Ray run (the cluster/worker-name prefix); a local run has no
            # Ray experiment to name, so it is read only here. None keeps twinkle's default.
            name=distributed_config.ray_exp_name,
            groups=groups)
    else:
        twinkle.initialize(mode='local', extra_torch_kwargs=extra_torch_kwargs)


def _process_and_validate(model_config: 'ModelConfig', template_config: 'TemplateConfig',
                          dataset_config: 'DatasetConfig', train_config: 'TrainConfig',
                          distributed_config: 'DistributedConfig',
                          checkpoint_config: Optional['CheckpointConfig'],
                          tuner_config: Optional['TunerConfig'], logging_config: Optional['LoggingConfig'],
                          quantize_config: Optional['QuantizeConfig'],
                          megatron_config: Optional['MegatronConfig'],
                          moe_config: Optional['MoEConfig']) -> None:
    """Process and validate the Configs before building anything heavy (idempotent per stage)."""
    from swift.dev2.config import process_configs, validate_configs

    process_configs(
        model_config, template_config, dataset_config, train_config, distributed_config,
        checkpoint_config, tuner_config, None,
        rollout_config=None, megatron_config=megatron_config, quantize_config=quantize_config,
        plugin_config=None)
    validate_configs(
        model_config, template_config, dataset_config, train_config, distributed_config,
        checkpoint_config, tuner_config, None,
        reward_config=None, rollout_config=None, logging_config=logging_config,
        quantize_config=quantize_config, megatron_config=megatron_config, moe_config=moe_config)


def _require_task_type(recipe: str, model_config: 'ModelConfig', task: str) -> str:
    """Default ``ModelConfig.task_type`` to this recipe's ``task``, and refuse a conflicting one.

    Defaulting rather than demanding: reaching a recipe already states the intent, while (say) a
    ``'causal_lm'`` task_type on the embedding path would silently encode SFT-shaped rows that the
    contrastive loss cannot group. An explicitly different value is a user error.
    """
    task_type = model_config.task_type or task
    if task_type != task:
        raise ValueError(f'{recipe} requires ModelConfig.task_type={task!r} (or None), got '
                         f'{task_type!r}. Other task types encode a different row layout and reach a '
                         f'different loss.')
    return task_type


def _resolve_step_interval(value: Optional[float], total_steps: int, name: str) -> Optional[int]:
    """Resolve a checkpoint/eval interval after the optimizer-step budget is known.

    Values in (0, 1) are ratios of the total run and round up so a positive ratio can never become
    step zero. Values >= 1 are absolute intervals and must be integral.
    """
    if value is None:
        return None
    if value < 0:
        raise ValueError(f'{name} must be non-negative, got {value}.')
    if 0 < value < 1:
        return max(1, math.ceil(total_steps * value))
    if not float(value).is_integer():
        raise ValueError(f'{name}={value} is not a ratio in (0, 1) or an integer step interval.')
    return int(value)


def _resolve_eval_steps(train_config: 'TrainConfig', total_steps: int) -> Optional[int]:
    """Turn ``eval_strategy`` into the optimizer-step interval the loop counts evals against.

    ``process_configs`` already chose the strategy (defaulting it to ``save_strategy`` and forcing
    'no' when the run has nothing to evaluate); 'epoch' maps to the per-epoch share of the budget,
    'steps' (or a bare ``--eval_steps``) to the configured interval.
    """
    strategy = (train_config.eval_strategy or '').lower()
    if strategy == 'no':
        return None
    if strategy == 'epoch':
        epochs = train_config.num_train_epochs or 1.0
        return max(1, math.ceil(total_steps / epochs))
    return _resolve_step_interval(train_config.eval_steps, total_steps, 'eval_steps')


def _resolve_step_intervals(train_config: 'TrainConfig',
                            checkpoint_config: Optional['CheckpointConfig'], total_steps: int) -> None:
    """Resolve the eval/save intervals into the concrete optimizer-step counts the loop uses."""
    train_config.eval_steps = _resolve_eval_steps(train_config, total_steps)
    if checkpoint_config is not None:
        checkpoint_config.save_steps = _resolve_step_interval(
            checkpoint_config.save_steps, total_steps, 'save_steps')


def _plan_supervised_steps(recipe: str, train_config: 'TrainConfig', dataloader: Any) -> int:
    """The optimizer-step budget, with the zero-step fail-fast.

    ``N <= ga`` micro-batches yield 0 optimizer steps (the GA gate lags one step, so a dataset that
    never fills a full lagged window would run forward/backward but NEVER update the model); rather
    than silently run a no-op training, fail with an actionable message. Runs BEFORE build_model so
    we fail before loading heavy weights.
    """
    # TODO(dev2): bridge until the recipe/common batch migrates num_optimizer_steps
    from swift.dev.recipe.common.loop_utils import num_optimizer_steps

    if train_config.max_steps and train_config.max_steps > 0:
        return train_config.max_steps
    try:
        micro_per_epoch = len(dataloader)  # micro-batches (already /batch_size) per epoch
    except TypeError:
        micro_per_epoch = 0  # IterableDataset: caller must set max_steps
    total_micro = math.ceil(micro_per_epoch * train_config.num_train_epochs)
    total_opt_steps = num_optimizer_steps(total_micro, train_config.gradient_accumulation_steps)
    if total_opt_steps <= 0:
        raise ValueError(
            f'{recipe}: computed {total_opt_steps} optimizer steps -- the dataloader is too small for '
            f'gradient_accumulation_steps={train_config.gradient_accumulation_steps}, or it is a '
            f'streaming/iterable dataset with no max_steps. Set TrainConfig.max_steps explicitly, or '
            'provide enough data.')
    return total_opt_steps


def build_generative_eval(*, model: Any, template: Any, model_config: 'ModelConfig',
                          train_config: 'TrainConfig', distributed_config: 'DistributedConfig',
                          template_config: 'TemplateConfig',
                          output_dir: str) -> Optional[GenerativeEval]:
    """Build the resident generative-eval sampler and its inputs; None when the run scores val-loss.

    Per ``TrainConfig.eval_sampler_backend``: 'transformers' (default) wraps the live training module
    (no second weight copy, no sync); 'vllm'/'sglang' place a co-resident engine in the trainer's
    'model' DeviceGroup and weight-sync it through a CheckpointEngineManager (ray mode only --
    validate refuses it otherwise), with the colocate hand-over bracketing each eval run.
    """
    if not train_config.predict_with_generate:
        return None
    from swift.dev2.eval import build_task_config, model_name, validate_eval_datasets

    cfg = train_config
    eval_datasets = validate_eval_datasets(cfg.eval_dataset)
    eval_model_id = model_name(model_config)
    # TrainConfig carries no eval_output_dir / eval_num_proc (those belong to ``swift eval``'s
    # EvalConfig): the work dir derives from output_dir, the concurrency is EvalScope's default.
    task_config = build_task_config(
        work_dir=os.path.join(output_dir, 'eval'),
        limit=cfg.eval_limit,
        dataset_args=cfg.eval_dataset_args,
        generation_config=cfg.eval_generation_config,
        extra_eval_args=cfg.extra_eval_args)

    backend = cfg.eval_sampler_backend
    if backend == 'transformers':
        from twinkle.sampler import TransformersSampler

        sampler = TransformersSampler(model=model)
        sampler.set_template(template)
        return GenerativeEval(
            sampler=sampler, template=template, datasets=eval_datasets, model_id=eval_model_id,
            task_config=task_config, enter=None, exit=None)

    from twinkle.checkpoint_engine import CheckpointEngineManager

    from swift.dev2.builders import build_dp_mesh, build_sampler
    # The hand-over sleeps the engine to free the GPU for the trainer, which each backend permits
    # only with its own memory-saver flag on (vLLM: enable_sleep_mode, sglang: enable_memory_saver).
    sleep_arg = 'enable_sleep_mode' if backend == 'vllm' else 'enable_memory_saver'
    sampler = build_sampler(
        model_config,
        backend=backend,
        engine_args={sleep_arg: True},
        template=template,
        # Co-resident in the trainer's 'model' DeviceGroup, carrying the SAME mesh the training
        # model was placed with: an engine slicing inputs on another rank layout would be wrong.
        device_mesh=build_dp_mesh(distributed_config, template_config.sequence_parallel_size),
        remote_group='model')
    manager = CheckpointEngineManager(model=model, sampler=sampler, platform='GPU', mode='colocate')
    # The eval hand-over is ColocateHandover's (the RL rollout's own sequence, so it cannot drift).
    # TODO(dev2): bridge until the recipe/common batch migrates ColocateHandover
    from swift.dev.recipe.common.colocate import ColocateHandover

    handover = ColocateHandover(model, sampler, manager, colocate=True)
    return GenerativeEval(
        sampler=sampler, template=template, datasets=eval_datasets, model_id=eval_model_id,
        task_config=task_config, enter=handover.enter, exit=handover.exit)


def build_supervised_objects(*, recipe: str, model_config: 'ModelConfig',
                             template_config: 'TemplateConfig', dataset_config: 'DatasetConfig',
                             train_config: 'TrainConfig', distributed_config: 'DistributedConfig',
                             checkpoint_config: Optional['CheckpointConfig'] = None,
                             tuner_config: Optional['TunerConfig'] = None,
                             logging_config: Optional['LoggingConfig'] = None,
                             quantize_config: Optional['QuantizeConfig'] = None,
                             megatron_config: Optional['MegatronConfig'] = None,
                             moe_config: Optional['MoEConfig'] = None, task: Optional[str] = None,
                             loop_task: Optional[str] = None, configure_loss: Callable[[Any], None],
                             configure_metric: Optional[Callable[[Any], None]] = None,
                             output_dir: str = 'output') -> SupervisedCluster:
    """Run the config -> object stages in the locked order, up to (not including) the training loop.

    ``configure_loss`` is the one thing a training recipe genuinely owns -- which objective this run
    optimises -- so it arrives as a callable taking the built model, invoked between the model and
    the optimizer (which must see the loss's parameters) and after the tuner. The optional
    ``configure_metric`` peer runs AFTER ``configure_optimizer``, once the group's metric list exists.
    """
    from swift.dev2.builders import (
        apply_full_param_freeze, build_dataset, build_dp_mesh, build_model, build_template,
        load_model_processor)
    from swift.dev2.builders.adapter import apply_tuner
    from swift.dev2.builders.optimizer import configure_optimizer
    from swift.dev2.processor import InputProcessor

    _initialize_supervised_twinkle(
        distributed_config, seed=train_config.seed, full_determinism=train_config.full_determinism,
        tf32=train_config.tf32)
    _process_and_validate(
        model_config, template_config, dataset_config, train_config, distributed_config,
        checkpoint_config, tuner_config, logging_config, quantize_config, megatron_config, moe_config)
    sp = template_config.sequence_parallel_size
    sp_mesh = build_dp_mesh(distributed_config, sp) if sp > 1 else None
    task_type = model_config.task_type
    if task is not None:
        task_type = _require_task_type(recipe, model_config, task)
    _, processor = load_model_processor(model_config)
    kwargs = {'task_type': task_type} if task_type else {}
    template = build_template(template_config, processor, **kwargs)
    # Match legacy save_args: keep one run-level args.json that every periodic/final checkpoint
    # copies, so --load_args works for a checkpoint produced before training reaches checkpoint-final.
    write_ckpt_args_json(
        output_dir, processor, model_config, template_config, tuner_config, quantize_config,
        task_type=task_type)
    dataloader, eval_dataloader = build_dataset(
        dataset_config, template, train_config, distributed_config,
        template_config=template_config, device_mesh=sp_mesh)
    total_opt_steps = _plan_supervised_steps(recipe, train_config, dataloader)
    _resolve_step_intervals(train_config, checkpoint_config, total_opt_steps)
    # Full-param resume loads weights from the ckpt dir instead of the original model id, mirroring
    # legacy's ``_init_ckpt_dir``; LoRA keeps the original base and reloads the adapter via resume.
    resume_dir = checkpoint_config.resume_from_checkpoint if checkpoint_config is not None else None
    build_config = model_config
    if resume_dir and tuner_config is None:
        build_config = copy.copy(model_config)
        build_config.model = resume_dir
    model = build_model(
        build_config, distributed_config, train_config, tuner_config, device_mesh=sp_mesh,
        quantize_config=quantize_config, megatron_config=megatron_config, moe_config=moe_config)
    if tuner_config is not None:
        # task_type threads the seq_cls/reranker head into modules_to_save: LoRA trains/saves
        # target_modules only, so without it the head is neither trained nor persisted.
        apply_tuner(
            model, tuner_config, gradient_accumulation_steps=train_config.gradient_accumulation_steps,
            task_type=task_type)
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
        task_type=task_type)
    model.set_template(template)
    configure_loss(model)
    configure_optimizer(
        model, train_config, num_training_steps=total_opt_steps, distributed_config=distributed_config,
        tuner_config=tuner_config, plugin_config=None)
    if configure_metric is not None:
        configure_metric(model)
    return SupervisedCluster(
        model=model, processor=processor, template=template, dataloader=dataloader,
        eval_dataloader=eval_dataloader, total_opt_steps=total_opt_steps, task_type=task_type,
        forward_task=loop_task or task or 'causal_lm')

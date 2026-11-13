# Copyright (c) ModelScope Contributors. All rights reserved.
# Shared recipe utilities: save / resume / step cadence (all families) + per-family assembly pieces.
from __future__ import annotations
import math
import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

import json

from swift.dev.utils.logger import get_logger
from swift.dev.utils.scoring import collect  # noqa: F401 -- the recipes import it from here

if TYPE_CHECKING:
    from twinkle.perf import TimeRecorder

    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        DistributedConfig,
        LoggingConfig,
        MegatronConfig,
        ModelConfig,
        MoEConfig,
        QuantizeConfig,
        RewardConfig,
        RLConfig,
        RolloutConfig,
        TemplateConfig,
        TrainConfig,
        TunerConfig,
    )
    from swift.dev.model import TrainableModel

logger = get_logger()

MODEL_GROUP = 'model'
SAMPLER_GROUP = 'sampler'
#: A separate frozen teacher (a GRPO RLSD/SDAR or a distillation teacher) is its own Ray DeviceGroup;
#: a disable_adapter / dynamic-self teacher reuses the policy actor and plans no group.
TEACHER_GROUP = 'teacher'
#: A FULL-PARAMETER frozen reference is an independent weight copy: its own DeviceGroup. A LoRA /
#: disable_adapter reference reuses the policy's base and plans no group.
REF_GROUP = 'ref'


#: The console cadence when no LoggingConfig was supplied (legacy logging_steps=1: every step).
_FALLBACK_LOGGING_STEPS = 1


def rollout_metrics(*, prompt_lengths, response_lengths, timings, num_devices,
                    wall_seconds, truncated=(), turns=(), extra_values=None):
    """Batch statistics as a pure function: twinkle's ThroughputMetric plus recipe-free extras."""
    from twinkle.metric import ThroughputMetric

    metric = ThroughputMetric()
    metric.accumulate(prompt_lengths=prompt_lengths, response_lengths=response_lengths,
                      timings=timings, num_devices=num_devices)
    values = metric.calculate()
    values.update(extra_values or {})
    values.update({
        'perf/gpu_seconds': timings['step'] * num_devices,
        'observability/iteration_wall_seconds': wall_seconds,
        'observability/excluded_logging_seconds': max(0.0, wall_seconds - timings['step']),
    })
    if truncated:
        values['response/truncated_ratio'] = sum(truncated) / len(truncated)
    if turns and all(isinstance(turn, (int, float)) for turn in turns):
        values.update({
            'rollout/assistant_turns/mean': sum(turns) / len(turns),
            'rollout/assistant_turns/min': min(turns),
            'rollout/assistant_turns/max': max(turns),
        })
    return values


def rollout_iteration_metrics(state, *, stream_stats=None, resource_metrics=None,
                              terminal=False, publications=0):
    """Back-compat shim: see :meth:`RolloutStepState.fold_iteration_metrics`."""
    return state.fold_iteration_metrics(
        stream_stats=stream_stats, resource_metrics=resource_metrics,
        terminal=terminal, publications=publications)


def init_rollout_metrics(state, device_groups, *, mode='colocate', parameter_sync_step=1,
                         num_generations=1, attributes=None):
    """Back-compat shim: see :meth:`RolloutStepState.setup_rollout_metrics`."""
    state.setup_rollout_metrics(
        device_groups, mode=mode, parameter_sync_step=parameter_sync_step,
        num_generations=num_generations, attributes=attributes)


def begin_rollout_iteration(state):
    """Back-compat shim: see :meth:`RolloutStepState.begin_rollout_iteration`."""
    state.begin_rollout_iteration()


def observe_rollout_batch(state, summary):
    """Back-compat shim: see :meth:`RolloutStepState.observe_rollout_batch`."""
    state.observe_rollout_batch(summary)


def record_rollout_training(state, indices):
    """Back-compat shim: see :meth:`RolloutStepState.record_rollout_training`."""
    state.record_rollout_training(indices)


def rollout_memory_metrics(state):
    """Back-compat shim: see :meth:`RolloutStepState.rollout_memory_metrics`."""
    return state.rollout_memory_metrics()


def finish_rollout_metrics(state, *, stream_stats=None, status='completed'):
    """Back-compat shim: see :meth:`RolloutStepState.finish_rollout_metrics`."""
    state.finish_rollout_metrics(stream_stats=stream_stats, status=status)


class SaveKnobs(NamedTuple):
    """One training run's checkpoint-writing settings (the dissolved CheckpointSaver half)."""

    output_dir: str
    no_save_optim: bool
    no_save_rng: bool
    safe_serialization: bool
    max_shard_size: str
    save_total_limit: Optional[int]
    #: Offloads the staged file I/O to a background writer; the final save is always synchronous.
    async_save: bool
    #: output_dir/initial_model for a base-modifying LoRA init (pissa ...), else None.
    initial_adapter_dir: Optional[str]
    #: Uploads one checkpoint dir to the hub (None when pushing is off); 'end' defers to the final.
    hub_pusher: Optional[Callable[[str], None]]
    hub_strategy: str
    #: mode='remote': the server owns the checkpoint store (save returns a twinkle:// URI), so
    #: the local-dir branches (converted adapter, hub push, async writer) are all off. Derived
    #: from DistributedConfig.mode -- backend identity is read off the config, never the model.
    remote: bool = False


def build_save_knobs(checkpoint_config: Optional['CheckpointConfig'],
                     dataset_config: Optional['DatasetConfig'],
                     distributed_config: 'DistributedConfig',
                     tuner_config: Optional['TunerConfig'], *, output_dir: str) -> SaveKnobs:
    """Collect CheckpointConfig's writer knobs plus the derived adapter / hub faces into one bundle."""
    ckpt = checkpoint_config
    return SaveKnobs(
        output_dir=output_dir,
        no_save_optim=bool(ckpt and (ckpt.no_save_optim or ckpt.save_only_model)),
        no_save_rng=bool(ckpt and (ckpt.no_save_rng or ckpt.save_only_model)),
        safe_serialization=ckpt.safe_serialization if ckpt else True,
        max_shard_size=ckpt.max_shard_size if ckpt else '5GB',
        save_total_limit=ckpt.save_total_limit if ckpt else None,
        async_save=bool(ckpt and ckpt.async_save),
        initial_adapter_dir=initial_adapter_dir(tuner_config, distributed_config, output_dir),
        hub_pusher=build_hub_pusher(ckpt, dataset_config),
        hub_strategy=ckpt.hub_strategy if ckpt else 'every_save',
        remote=distributed_config.mode == 'remote')


def is_main_process() -> bool:
    """True on the rank that owns single-instance side effects (the converted adapter, the hub push).

    Under torchrun every rank drives its own copy of the loop, so an unguarded push would run once
    per rank; under ray the loop runs only on the driver, where ``dist`` is not initialized.
    """
    import torch.distributed as dist

    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def initial_adapter_dir(tuner_config: Optional['TunerConfig'], distributed_config: 'DistributedConfig',
                        output_dir: str) -> Optional[str]:
    """``output_dir/initial_model`` when a base-modifying LoRA init rewrites the base weight.

    A base-modifying init (pissa ...) makes every saved checkpoint need the initial-adapter snapshot
    plus a converted copy; a plain init needs neither, so the knob stays None.
    """
    from swift.dev.builders import is_megatron_backend
    from swift.dev.builders.adapter import is_base_modifying_init

    if tuner_config is None or not is_base_modifying_init(tuner_config.init_weights):
        return None
    if distributed_config.mode == 'remote':
        # The client handle has no save_initial_adapter (the snapshot lives in the server's
        # store); a base-modifying init is refused at validate time rather than half-wired here.
        return None
    if is_megatron_backend(distributed_config):
        logger.warning('base-modifying LoRA init %r on the megatron backend: checkpoint conversion '
                       '(converted/default) is not wired there; only the raw adapter is saved.',
                       tuner_config.init_weights)
        return None
    return os.path.join(output_dir, 'initial_model')


def snapshot_initial_adapter(model: Any, initial_adapter_dir: Optional[str]) -> None:
    """Snapshot a base-modifying LoRA init once, before any training step touches it.

    Idempotent: an existing snapshot (a resumed run already made one) is left as is, because the
    injection re-runs deterministically from the same base checkpoint but re-running it here would
    be a second, needless SVD pass. Main-process only: under torchrun every rank drives this path.
    """
    if initial_adapter_dir is None or not is_main_process():
        return
    marker = os.path.join(initial_adapter_dir, 'adapter_config.json')
    if os.path.exists(marker):
        return
    os.makedirs(initial_adapter_dir, exist_ok=True)
    model.save_initial_adapter(initial_adapter_dir)


def build_hub_pusher(checkpoint_config: Optional['CheckpointConfig'],
                     dataset_config: Optional['DatasetConfig']) -> Optional[Callable[[str], None]]:
    """A callable that uploads one checkpoint dir to the hub, or None when pushing is off.

    Goes through twinkle's ``HubOperation`` rather than ``TrainableModel.upload_to_hub``: the latter
    hardcodes ``private=True`` and never passes a revision, so it would silently ignore
    ``hub_private_repo`` / ``hub_revision``. ``HubOperation`` honors both and dispatches ModelScope
    vs HF off the repo_id prefix (``hf://`` / ``ms://``).
    """
    cfg = checkpoint_config
    if cfg is None or not cfg.push_to_hub:
        return None
    if not cfg.hub_model_id:
        raise ValueError('CheckpointConfig.push_to_hub is set but hub_model_id is empty, so there is no '
                         'target repo to upload to. Pass --hub_model_id <repo_id>.')
    from twinkle.hub import HubOperation

    token = dataset_config.hub_token if dataset_config is not None else None
    repo_id = cfg.hub_model_id
    private = cfg.hub_private_repo
    revision = cfg.hub_revision or 'master'
    # hub_always_push blocks until each upload lands, for a run where a dropped push would go
    # unnoticed; off by default so uploads run on a background thread and do not stall training.
    async_upload = not cfg.hub_always_push

    def push(checkpoint_dir: str) -> None:
        push_kwargs = dict(
            repo_id=repo_id,
            folder_path=checkpoint_dir,
            token=token,
            private=private,
            revision=revision,
            commit_message=f'Upload {os.path.basename(checkpoint_dir.rstrip("/"))} from swift.dev training')
        if async_upload:
            HubOperation.async_push_to_hub(**push_kwargs)
        else:
            HubOperation.push_to_hub(**push_kwargs)

    return push


def write_ckpt_args_json(ckpt_dir: str, processor: Any, model_config: 'ModelConfig',
                         template_config: 'TemplateConfig', tuner_config: Optional['TunerConfig'] = None,
                         quantize_config: Optional['QuantizeConfig'] = None,
                         dataset_config: Optional['DatasetConfig'] = None,
                         *, task_type: Optional[str] = None) -> None:
    """Write the self-describing args.json ``swift infer`` reads back from the checkpoint.

    Legacy dumped the full argument dict; the read side (cli/parser.py) consumes only force-load
    keys (always applied), load keys (applied when the reader's value is empty) and --load_data_args
    fields. Those key sets live once in config/checkpoint_config.py and this writer iterates them,
    so a key the reader expects is written here the day it is declared. ``plugin_config`` is not
    threaded into the recipes (plugins load at the CLI layer), so ``external_plugins`` is the one
    read-side key this writer never emits.
    """
    import json
    import torch.distributed as dist

    # Master-only: the checkpoint dir exists on the master rank alone (twinkle's save_pretrained
    # guards on Platform.is_master()), so an unguarded open() elsewhere raised under multi-GPU DDP.
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return
    os.makedirs(ckpt_dir, exist_ok=True)

    from swift.dev.config.checkpoint_config import (
        CHECKPOINT_DATA_FIELDS,
        CHECKPOINT_FORCE_LOAD_FIELDS,
        CHECKPOINT_LOAD_FIELDS,
    )
    from swift.dev.utils.version import __version__ as swift_version
    model_meta = getattr(processor, 'model_meta', None)
    configs = {
        'model_config': model_config,
        'quantize_config': quantize_config,
        'tuner_config': tuner_config,
        'template_config': template_config,
    }
    args: Dict[str, Any] = {
        # swift_version gates model_type loading in BaseArguments.load_args_from_ckpt
        # (model_type is only honored when swift_version >= 4.0.0.dev); must be present.
        'swift_version': swift_version,
    }
    # force-load keys: the reader applies these regardless of its current value.
    for config_name, names in CHECKPOINT_FORCE_LOAD_FIELDS.items():
        config = configs.get(config_name)
        if config is None:
            continue
        for entry in names:
            key, field_name = entry if isinstance(entry, tuple) else (entry, entry)
            value = getattr(config, field_name, None)
            if value is not None:
                args[key] = value
    # load keys: the reader applies these only when its own value is None/empty.
    for config_name, names in CHECKPOINT_LOAD_FIELDS.items():
        config = configs.get(config_name)
        if config is None:
            continue
        for name in names:
            value = getattr(config, name, None)
            if value is not None:
                args[name] = value
    # --load_data_args fields (restored onto the reader's DatasetConfig verbatim).
    if dataset_config is not None:
        for name in CHECKPOINT_DATA_FIELDS:
            value = getattr(dataset_config, name, None)
            if value is not None:
                args[name] = value
    # Resolved metadata and portable ids win over the CLI spellings captured above: the model id
    # re-resolves on any machine, and the template the run actually built (possibly the meta default
    # rather than the user's) is what reproduces this checkpoint.
    args['model'] = getattr(model_config, '_model_id_or_path', None) or model_config.model
    if getattr(model_meta, 'model_type', None) is not None:
        args['model_type'] = model_meta.model_type
    args['template'] = template_config.template or getattr(model_meta, 'template', None)
    args['task_type'] = model_config.task_type or task_type
    if 'tuner_type' not in args:
        # tuner_config is absent exactly on full-parameter runs, whose checkpoints infer must read
        # back as full too (the reader force-loads tuner_type over its own default 'lora').
        args['tuner_type'] = 'full'
    args = {key: value for key, value in args.items() if value is not None}
    with open(os.path.join(ckpt_dir, 'args.json'), 'w', encoding='utf-8') as f:
        json.dump(args, f, ensure_ascii=False, indent=2)


def save_training_step(model: Any, handler: Any, name: str, knobs: SaveKnobs, *,
                       consumed_train_samples: int = 0,
                       component_states: Optional[Dict[str, dict]] = None,
                       is_final: bool = False,
                       requantize_source: Optional[str] = None) -> str:
    """One training checkpoint: the twinkle-native write, the converted adapter, the hub push.

    ``consumed_train_samples`` / ``component_states`` carry the data position: the offline family
    reads them off its dataloader, an online run passes its step count and no component state.
    Returns the checkpoint dir (twinkle's save result is a deferred handle under ray, so the dir is
    the only portable return).
    """
    if knobs.remote:
        # The server owns the store: model.save returns a twinkle:// URI (the portable handle the
        # loop treats as "the checkpoint dir"), and every local-dir side effect below (converted
        # adapter, hub push, args.json copy) has no directory to act on. save is synchronous on
        # the client (_await_task blocks to the terminal state), so no wait_async_saves either.
        # No component state rides the call: SaveRequest has no field for it, and the data
        # position re-seeds from the server's own progress on resume (resume_from_checkpoint).
        response = model.save(
            name,
            save_optimizer=not knobs.no_save_optim,
            consumed_train_samples=consumed_train_samples)
        return getattr(response, 'twinkle_path', name)

    component_states = dict(component_states or {})
    component_states["training"] = {"micro_step": handler.state.micro_step}
    save_training_checkpoint(
        model, name, output_dir=knobs.output_dir,
        consumed_train_samples=consumed_train_samples,
        component_states=component_states,
        no_save_optim=knobs.no_save_optim,
        no_save_rng=knobs.no_save_rng,
        safe_serialization=knobs.safe_serialization,
        max_shard_size=knobs.max_shard_size,
        save_total_limit=knobs.save_total_limit,
        # dev's CheckpointSaver gate: the final save is synchronous, every earlier one may hand its
        # file I/O to the background writer (async_save x save_total_limit is rejected upstream).
        async_save=knobs.async_save and not is_final,
        # A background writer may still be flushing shards; only the synchronous saves can
        # be re-quantized in place.
        requantize_source=requantize_source if (is_final or not knobs.async_save) else None,
    )
    ckpt_dir = os.path.join(knobs.output_dir, name)
    if knobs.initial_adapter_dir is not None and is_main_process():
        # The portable form of a base-modifying adapter: fold the init factor back out, so a
        # consumer merges the converted copy instead of the residual-base-only raw one.
        model.save_converted_adapter(os.path.join(ckpt_dir, 'converted', 'default'), knobs.initial_adapter_dir)
    handler.on_save(checkpoint_dir=ckpt_dir)
    # 'end' pushes only the final checkpoint; the push-on-save strategies push every one.
    if knobs.hub_pusher is not None and (knobs.hub_strategy != 'end' or is_final) and is_main_process():
        handler.on_push_begin(checkpoint_dir=ckpt_dir)
        knobs.hub_pusher(ckpt_dir)
    return ckpt_dir


def _restore_training(model: Any, checkpoint_config: Optional['CheckpointConfig'],
                      distributed_config: 'DistributedConfig',
                      megatron_config: Optional['MegatronConfig'],
                      tuner_config: Optional['TunerConfig']) -> dict:
    """Restore the trainable model and return twinkle's training-progress state ({} when no resume).

    Weights / optimizer / scheduler / RNG are restored here; the caller then seeds its loop counters
    and the data position from the returned state. Full-parameter checkpoints always provide the
    model weights (build_model redirected the model id); a tuned run keeps the original base and
    reloads its adapter through the ``adapter_name`` path below.
    """
    from swift.dev.builders import is_megatron_backend

    path = checkpoint_config.resume_from_checkpoint if checkpoint_config is not None else None
    if not path:
        return {}
    resume_kwargs = {'resume_only_model': bool(checkpoint_config and checkpoint_config.resume_only_model)}
    if distributed_config.mode == 'remote':
        # A twinkle:// URI names a checkpoint in the SERVER's store; the handle restores weights,
        # optimizer, scheduler and progress state in one call, so the local trainer_state.json
        # probe below (and the megatron kwargs) do not apply.
        if tuner_config is not None:
            resume_kwargs['adapter_name'] = 'default'
        return model.resume_from_checkpoint(path, **resume_kwargs)
    if checkpoint_config is not None and is_megatron_backend(distributed_config):
        resume_kwargs.update(
            no_load_optim=checkpoint_config.no_load_optim,
            no_load_rng=checkpoint_config.no_load_rng,
            data_parallel_random_init=bool(megatron_config and megatron_config.data_parallel_random_init))
    adapter_name = 'default' if tuner_config is not None else None
    if adapter_name is not None:
        resume_kwargs['adapter_name'] = adapter_name

    trainer_state_path = os.path.join(path, 'trainer_state.json')
    if not os.path.isfile(trainer_state_path):
        if not resume_kwargs['resume_only_model']:
            raise FileNotFoundError(
                f'{trainer_state_path} is missing, so this model-only checkpoint cannot resume '
                'optimizer, scheduler, RNG, or progress state. Set --resume_only_model true to start '
                'a new run from its weights, or resume from a checkpoint saved without '
                '--no_save_optim/--save_only_model.')
        # No trainer state means a new run at step 0: only the adapter weights need loading (the
        # full-parameter model was already constructed from ``path`` by build_model).
        if adapter_name:
            model.load(path, adapter_name=adapter_name)
        return {'cur_step': 0, 'consumed_train_samples': 0}
    return model.resume_from_checkpoint(path, **resume_kwargs)


def training_progress(state: dict) -> Tuple[int, int]:
    """Read completed optimizer updates and an optional micro-batch statistic."""
    global_step = int(state.get("cur_step", 0))
    progress = state.get("component_states", {}).get("training", {})
    return global_step, int(progress.get("micro_step", 0))


def _seed_counters(state: dict, *, dataloader: Any, ignore_data_skip: bool) -> Tuple[int, int, int]:
    """Restore update/micro counters and the independent dataloader position."""
    global_step, micro_step = training_progress(state)
    component_states = state.get('component_states', {})
    if not ignore_data_skip and dataloader is not None and 'dataloader' in component_states:
        dataloader.load_state(component_states['dataloader'])
    if ignore_data_skip or dataloader is None:
        return global_step, micro_step, 0
    get_state = getattr(dataloader, 'get_state', None)
    start_epoch = int((get_state() if get_state is not None else {}).get('epoch', 0))
    return global_step, micro_step, start_epoch


def resume_training(model: Any, *, checkpoint_config: Optional['CheckpointConfig'],
                    distributed_config: 'DistributedConfig',
                    megatron_config: Optional['MegatronConfig'],
                    tuner_config: Optional['TunerConfig'],
                    dataloader: Any = None,
                    ignore_data_skip: bool = False) -> Tuple[int, int, int]:
    """Restore the trainable model, then seed the loop counters and the data position.

    Returns ``(global_step, micro_step, start_epoch)``. The offline family passes its dataloader
    (its saved component state restores the read position); a prompt-stream run passes
    ``None`` and always gets ``start_epoch`` 0.
    """
    state = _restore_training(model, checkpoint_config, distributed_config, megatron_config, tuner_config)
    if not state:
        return 0, 0, 0
    return _seed_counters(state, dataloader=dataloader, ignore_data_skip=ignore_data_skip)


def _log_line(record: dict, loss: float) -> str:
    """The default console step line: loss plus grad_norm, lr, the MTP channel and any loss_* channels.

    ``lr`` is read from the TrainMetric key ``learning rate`` (a display string like '5.000000e-05'
    once the tracker has formatted it, or a float before that).
    """
    def _fmt(value, spec='.4f'):
        # A console head must never crash the training loop: coerce, fall back to the value as-is.
        try:
            return format(float(value), spec)
        except (TypeError, ValueError):
            return str(value)

    gn = record.get('grad_norm')
    gn_str = f'  grad_norm={_fmt(gn)}' if gn is not None else ''
    # TrainMetric names single-group lr 'learning rate' and multi-group 'learning rate(param group N)'
    # (LoRA runs have several optimizer groups); the head line shows the first group's lr.
    lr = record.get('learning rate')
    if lr is None:
        for key in sorted(record):
            if key.startswith('learning rate'):
                lr = record[key]
                break
    lr_str = f'  lr={_fmt(lr, ".6e")}' if lr is not None else ''
    mtp = record.get('mtp_loss')
    mtp_str = f'  mtp_loss={_fmt(mtp)}' if mtp is not None else ''
    channel_str = ''.join(f'  {key}={_fmt(value)}' for key, value in record.items() if key.startswith('loss_'))
    # The legacy rollout_correction diagnostics, when the loop computed them: the signed KL is
    # the acceptance number for a freshly synced sampler (target < 1e-2).
    correction_str = ''.join(
        f'  {label}={_fmt(record[key], ".3g")}'
        for label, key in (('rollout_kl', 'rollout_correction/kl'), ('rollout_k3_kl', 'rollout_correction/k3_kl'),
                           ('rollout_logp_absdiff', 'rollout_correction/log_ppl_abs_diff'))
        if isinstance(record.get(key), (int, float)))
    return f"step {record.get('step')}  loss={loss:.4f}{lr_str}{gn_str}{mtp_str}{channel_str}{correction_str}"


def _mtp_metrics(model: Any) -> Dict[str, float]:
    """Per-depth MTP loss (and acceptance rate, on megatron >= 0.19) for this step, or {}.

    Kept out of ``calculate_metric``: megatron accumulates the MTP loss in a module-level tracker of
    its own rather than through twinkle's Metric objects, and it must be drained each step or the
    reported value grows without bound. Absent on transformers and whenever MTP is off.
    """
    pop = getattr(model, 'pop_mtp_metrics', None)
    if pop is None:
        return {}
    return {key: float(value) for key, value in (pop() or {}).items()}


class StepState:
    """The training loop's step cadence, shared by EVERY recipe family (online and offline).

    Owns the global/micro step counters, the per-step metric record (``model.calculate_metric`` +
    the megatron MTP drain, coerced to numeric scalars), the ``RunTracker`` write, the console line,
    the callback fan-out (``on_log`` per record), the CallbackState publication, and the step-end
    tail (the boundary save / eval cadence, ``on_step_end``, the transient control flags).
    """

    def __init__(self, model: Any, train_config: 'TrainConfig', *, gradient_accumulation_steps: int,
                 max_steps: int, max_grad_norm: float, logging_config: Optional['LoggingConfig'],
                 output_dir: str, save_steps: Optional[int] = None, eval_steps: Optional[int] = None,
                 eval_delay: int = 0, manual_gc: bool = False, manual_gc_steps: int = 0,
                 empty_cache_steps: Optional[int] = None,
                 extra_metrics_fn: Optional[Callable[[], Dict[str, Any]]] = None,
                 log_formatter: Optional[Callable[[Dict[str, Any], float], str]] = None,
                 padding_free: bool = False, dp_balance: bool = False):
        # The tracker initializes its reporters as a construction side effect (validation ran above).

        from swift.dev.callbacks import build_callbacks
        from swift.dev.callbacks.handler import CallbackHandler

        self.model = model
        self.dp_balance = dp_balance
        self.padding_free = padding_free
        self.ga = gradient_accumulation_steps
        self.max_steps = max_steps
        self.max_grad_norm = max_grad_norm
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.eval_delay = int(eval_delay or 0)
        self.manual_gc = manual_gc
        self.manual_gc_steps = manual_gc_steps
        self.empty_cache_steps = empty_cache_steps
        self._logging_config = logging_config
        self._extra_metrics_fn = extra_metrics_fn
        self._log_formatter = log_formatter
        self.global_step = 0
        self.micro_step = 0
        # The offline family advances these from its epoch loop / eval cadence; publish_state reads
        # them onto the shared CallbackState.
        self.epoch = 0
        self.best_metric: Optional[float] = None
        self.history: List[dict] = []
        self._output_dir = output_dir
        self.tracker = RunTracker(logging_config, output_dir)
        self.handler = CallbackHandler(build_callbacks(train_config), output_dir=output_dir)

    def reached_max(self) -> bool:
        return self.max_steps > 0 and self.global_step >= self.max_steps

    def publish_state(self, *, loss: Optional[float] = None, metrics: Optional[dict] = None) -> None:
        """Publish the loop position onto the shared CallbackState before an event fires.

        ``loss`` / ``metrics`` are only overwritten when supplied, so a step event keeps the last
        training record while an eval event publishes its own.
        """
        state = self.handler.state
        state.global_step = self.global_step
        state.micro_step = self.micro_step
        state.epoch = self.epoch
        state.best_metric = self.best_metric
        if loss is not None:
            state.loss = loss
        if metrics is not None:
            state.metrics = metrics

    def record(self, *, extra_metrics: Optional[Dict[str, Any]] = None) -> dict:
        """Count one optimizer step: metrics, tracker write, console line, the on_log fan-out.

        Reads the NORMALIZED per-token loss (+ every algorithm metric) via the driver-callable
        ``model.calculate_metric`` (works for transformers + Ray-remote Megatron): the raw
        ``forward_backward`` loss under ``reduction='sum'`` is a token-sum, not comparable across
        steps. The periodic save / eval / ``on_step_end`` tail is :meth:`step_tail`'s, not this's.
        """

        from twinkle.perf import Perf

        self.global_step += 1
        collect_manual_gc(self.manual_gc, self.manual_gc_steps, self.global_step)
        collect_empty_cache(self.model, self.empty_cache_steps, self.global_step)
        # Tick every registered actor's profiler step counter once per optimizer step (non-blocking).
        Perf.advance_step()
        metrics = self.model.calculate_metric(is_training=True)
        loss = float(metrics['loss']) if metrics.get('loss') is not None else float('nan')
        if self._logging_config is not None:
            due = self.tracker.should_log(self.global_step)
        else:
            due = self.global_step % _FALLBACK_LOGGING_STEPS == 0
        should_log = due or self.handler.control.should_log
        # twinkle.metric is the single source of every algorithm metric: merge the WHOLE component
        # output, coerced to numeric scalars; ``loss`` is re-pinned as the normalized float.
        record = {'step': self.global_step}
        record.update(coerce_metric_scalars(metrics))
        record['loss'] = loss
        record.update(_mtp_metrics(self.model))
        if self._extra_metrics_fn is not None:
            record.update(self._extra_metrics_fn())
        if extra_metrics is not None:
            record.update(extra_metrics)
        if should_log:
            # Fold the window's merged per-rank perf summary in only on a logging step: the
            # perf_dump fan-out is blocking, so keeping it off every step leaves it off the hot path.
            record.update(Perf.collect_step_summary())
        from twinkle.perf import default_recorder

        # '_logging' on the shared recorder: the excluded band rollout_iteration_metrics carves
        # out of its phase windows (same name the GRPO recipe's own logging spans use).
        with default_recorder.span('_logging'):
            record = self.tracker.log(record, self.global_step)
            self.history.append(record)
            self.publish_state(loss=loss, metrics=record)
            if should_log:
                # A recipe's own head signal (PPO's value_loss / reward / explained_variance line).
                line = (self._log_formatter(record, loss) if self._log_formatter is not None
                        else _log_line(record, loss))
                logger.info(line)
                self.handler.on_log(logs=record)
                self._write_metrics_jsonl(record)
        return record

    def _write_metrics_jsonl(self, record: Dict[str, Any]) -> None:
        """Append the step record as one JSON line to ``<output_dir>/metrics.jsonl``.

        The machine-readable curve source for monitors (the webui tail-loads it instead of
        parsing the console line); main-process only, best-effort: a metrics write must
        never kill a training step.
        """
        if not is_main_process():
            return
        try:
            with open(os.path.join(self._output_dir, 'metrics.jsonl'), 'a') as f:
                f.write(json.dumps(record, default=str) + '\n')
        except OSError:
            logger.debug('metrics.jsonl write failed', exc_info=True)

    def should_stop(self) -> bool:
        """Whether a callback (early stop / graceful exit) asked the loop to break."""
        return self.handler.control.should_training_stop

    def _eval_due(self) -> bool:
        """Whether the current optimizer step should trigger a periodic evaluation pass."""
        if not self.eval_steps:
            return False
        if self.global_step % self.eval_steps != 0:
            return False
        return self.global_step > self.eval_delay

    def step_tail(self, save: Optional[Callable[[str], Any]] = None,
                  evaluate: Optional[Callable[[], Any]] = None) -> None:
        """The end-of-optimizer-step tail: periodic save, periodic eval, on_step_end, its requests.

        The order is the old loop's save branch followed by its post-step tail: the save_steps
        checkpoint first, then a due eval, then ``on_step_end`` (where EarlyStop / GracefulExit
        raise their flags), then honour + clear the transient ``should_save`` / ``should_evaluate``
        / ``should_log`` flags so a one-shot request does not repeat on every subsequent step.
        """
        if save is not None and self.save_steps and self.global_step % self.save_steps == 0:
            save(f'checkpoint-{self.global_step}')
        if evaluate is not None and self._eval_due():
            evaluate()
        self.handler.on_step_end()
        control = self.handler.control
        if control.should_save and save is not None:
            save(f'checkpoint-{self.global_step}')
        if control.should_evaluate and evaluate is not None:
            evaluate()
        control.should_save = False
        control.should_evaluate = False
        control.should_log = False


class RolloutStepState(StepState):
    """A :class:`StepState` for rollout recipes (GRPO now; every online rollout family later).

    Adds the iteration-level statistics layer on top of the step cadence. The split of labour:
    TIMING lives in twinkle.perf (the dispatch layer auto-spans every remote call on the
    driver's default recorder; ``fold_iteration_metrics`` reads that recorder's window
    reduction), THROUGHPUT formulas live in twinkle.metric.ThroughputMetric, and only the
    recipe-specific facts (sample lengths, turns, visit accounting) live here. The module-level
    ``rollout_*`` functions are back-compat shims onto these methods.

    The rollout fields are always initialised (even before ``setup_rollout_metrics``), so the
    observe/record calls in a loop that never set up metrics are harmless no-ops -- the same
    contract the free functions had via ``getattr(state, ..., None)``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._metric_context: Dict[str, Any] = {}
        self._run_metric: Any = None
        self._completed_batches = 0
        self._batch_wall_ns = 0
        self._rollout_groups = 0
        self._trained_samples = 0
        self._stream_previous: Dict[str, float] = {}
        self._iteration_started_ns: Optional[int] = None
        self._iteration_facts: Optional[Dict[str, Any]] = None
        self._memory_rows: Optional[List[Dict[str, Any]]] = None

    # ------------------------------------------------------------- lifecycle

    def setup_rollout_metrics(self, device_groups, *, mode='colocate', parameter_sync_step=1,
                              num_generations=1, attributes=None) -> None:
        """Enable per-iteration rollout accounting; the shared perf recorder needs no init."""
        from twinkle.metric import ThroughputMetric

        self._metric_context = {
            'device_groups': device_groups,
            'mode': mode,
            'parameter_sync_step': parameter_sync_step,
            'num_generations': num_generations,
            **(attributes or {}),
        }
        self._run_metric = ThroughputMetric()
        self._completed_batches, self._batch_wall_ns = 0, 0
        self._rollout_groups = self._trained_samples = 0
        self._stream_previous = {}
        self._iteration_started_ns = None
        self._iteration_facts = None

    def begin_rollout_iteration(self) -> None:
        """Mark the window start; timing itself is the dispatch layer's auto-spans."""
        from twinkle.perf import default_recorder

        if self._run_metric is not None:
            # The recorder's clock, matching fold_iteration_metrics' end stamp (one clock
            # end to end, or the window would select the wrong spans).
            self._iteration_started_ns = default_recorder.now_ns()
            self._iteration_facts = {'lengths': [], 'truncated': [], 'turns': [],
                                     'trained_indices': set(), 'sample_visits': 0}
            self._memory_rows = None

    def observe_rollout_batch(self, summary: Dict[str, Any]) -> None:
        """Attach this iteration's rollout facts (lengths / truncation / turns), off the light summary.

        The facts are the driver's row-level batch summary -- reading them off the trajectories
        would mean materialising a whole rollout batch for its metadata. """
        from twinkle.perf import default_recorder

        batch = self._iteration_facts
        if batch is None:
            return
        batch['lengths'] = list(zip(summary.get('prompt_lengths') or (),
                                     summary.get('response_lengths') or ()))
        batch['truncated'] = [bool(flag) for flag in (summary.get('truncated') or ())]
        batch['turns'] = list(summary.get('turns') or ())
        # When the batch arrived -- the 'gen' fallback window end when no gen span exists.
        batch['received_ns'] = default_recorder.now_ns()

    def record_rollout_training(self, indices) -> None:
        """Count successful sample visits without duplicating generated tokens on replay."""
        batch = self._iteration_facts
        if batch is not None:
            batch['sample_visits'] += len(indices)
            batch['trained_indices'].update(indices)

    def rollout_memory_metrics(self) -> Dict[str, float]:
        """Aggregate worker memory once per iteration; returns {} when uninstrumented."""
        from twinkle.metric import memory_metrics

        if self._iteration_facts is None:
            return {}
        rows = self._memory_rows
        if rows is None:
            rows = self.model.memory_stats(synchronize=True)
            rows = [dict(rows)] if isinstance(rows, dict) else [dict(row) for row in rows]
            self._memory_rows = rows
        return memory_metrics(rows)

    # ------------------------------------------------------------- accounting

    def fold_iteration_metrics(self, *, stream_stats=None, resource_metrics=None,
                               terminal=False, publications=0) -> Dict[str, Any]:
        """Fold one consumed rollout iteration into the shared metric pipeline.

        Returns the per-iteration values dict (also recorded through the run tracker).
        """
        from twinkle.metric import MetricRecord
        from twinkle.perf import default_recorder

        started_ns = self._iteration_started_ns
        batch = self._iteration_facts
        if started_ns is None or batch is None or not batch['lengths']:
            return {}
        # The recorder's OWN clock (not a second perf_counter_ns site): window bounds and
        # span stamps must come from one clock or the window would select the wrong spans.
        ended_ns = default_recorder.now_ns()

        # Phase timings from the driver recorder: every remote call in the window was
        # auto-spanned by name; fold to union-of-intervals and carve out the logging band.
        # 'step' is the old contract: the window's wall time MINUS that excluded band.
        raw = default_recorder.phase_totals(exclude=('_logging',), window=(started_ns, ended_ns)) \
            if default_recorder.enabled else {}
        timings = {name: seconds for name, seconds in raw.items() if name != 'window'}
        timings['step'] = raw.get('window', (ended_ns - started_ns) / 1e9)

        stats = stream_stats or {}
        for name in ('publish_time', 'publishes'):
            if name in stats:
                delta = stats[name] - self._stream_previous.get(name, 0)
                if name == 'publish_time':
                    timings['update_weights'] = delta
                else:
                    publications = delta
                self._stream_previous[name] = stats[name]

        # The disaggregated branch has no 'gen' span on the driver (generation happens on the
        # sampler behind the driver's pull), so fall back to iteration-start -> samples-received,
        # exactly the old rollout_phase_times contract.
        received_ns = batch.get('received_ns')
        if 'gen' not in timings and received_ns is not None:
            timings['gen'] = max(0.0, (received_ns - started_ns) / 1e9)

        groups = self._metric_context['device_groups']
        num_devices = len({rank for _, ranks in groups for rank in ranks})
        prompt_lengths = [p for p, _ in batch['lengths']]
        response_lengths = [r for _, r in batch['lengths']]
        inputs = {
            'prompt_lengths': prompt_lengths,
            'response_lengths': response_lengths,
            'timings': timings,
            'num_devices': num_devices,
        }
        self._run_metric.accumulate(**inputs)
        values = rollout_metrics(
            **inputs, wall_seconds=(ended_ns - started_ns) / 1e9,
            truncated=batch['truncated'], turns=batch['turns'])
        values.update({
            'training/optimizer_step': self.global_step,
            'training/weight_publications_this_iteration': publications,
            'training/sample_visits': batch['sample_visits'],
            'training/unique_samples_trained': len(batch['trained_indices']),
            'training/samples_not_trained': len(batch['lengths']) - len(batch['trained_indices']),
        })
        values.update(self.rollout_memory_metrics())
        values.update(resource_metrics or {})

        self._completed_batches += 1
        self._rollout_groups += len(batch['lengths']) // self._metric_context['num_generations']
        self._trained_samples += batch['sample_visits']
        self._batch_wall_ns += ended_ns - started_ns
        self.tracker.record(
            MetricRecord(
                stage='rollout',
                optimizer_step=self.global_step,
                values=values,
                attributes={
                    'scope': 'iteration',
                    'step_unit': 'rl_iteration',
                    'iteration': self._completed_batches,
                    'iteration_scope': 'one_consumed_rollout_batch',
                    **self._metric_context,
                    'parameter_sync_step_unit': 'optimizer_update',
                    'terminal': terminal,
                },
            )
        )
        logger.info(
            '[perf] iteration=%s step=%s seconds=%.3f tokens=%s tok/s/GPU=%.3f',
            self._completed_batches, self.global_step, timings['step'],
            values.get('perf/total_num_tokens', 0), values.get('perf/throughput', 0.0),
        )
        self._iteration_started_ns = None
        self._iteration_facts = None
        return values

    def finish_rollout_metrics(self, *, stream_stats=None, status='completed') -> None:
        """Write terminal run totals through the existing metric reporter."""
        from twinkle.metric import MetricRecord

        if self._run_metric is None:
            return
        metrics = self._run_metric.calculate()
        values = {
            'completed_iterations': self._completed_batches,
            'optimizer_step': self.global_step,
            'rollout_groups': self._rollout_groups,
            'rollout_samples': metrics.get('perf/num_samples', 0),
            'trained_samples': self._trained_samples,
            'wall_time_s': self._batch_wall_ns / 1e9,
            'total_num_tokens': metrics.get('perf/total_num_tokens', 0),
            'sum_iteration_seconds': metrics.get('perf/time_per_step', 0),
            'throughput': metrics.get('perf/throughput'),
        }
        values.update(
            {f'stream/{key}': value for key, value in (stream_stats or {}).items() if isinstance(value, (int, float))}
        )
        self.tracker.record(
            MetricRecord(
                stage='run',
                optimizer_step=self.global_step,
                values=values,
                status='failed' if status == 'failed' else 'completed',
                attributes={'step_unit': 'run_summary', 'loop_status': status},
            )
        )


# Optimizer updates each derived rollout batch covers. Colocate pays a sleep/wake + weight-sync
# handover per batch, so a batch covering a single update makes that overhead dominate; a horizon
# in the usual 4-32 range keeps it amortized while the first batch still finishes in minutes.
_STEPS_PER_ROLLOUT_BATCH = 8


def resolve_generation_batch_size(generation_batch_size: Optional[int], *, num_prompts: int,
                                   num_generations: int, num_iterations: int,
                                   train_batch_size: int, gradient_accumulation_steps: int) -> int:
    """Resolve the rollout batch size (in prompts) an online-RL round generates in one go.

    An explicit ``generation_batch_size`` passes through unchanged. The derived default fills
    whole optimizer updates: ``train_batch_size * gradient_accumulation_steps`` samples = that
    many completions = ``global_batch / num_generations`` prompts, times ``num_iterations``
    replays and ``_STEPS_PER_ROLLOUT_BATCH`` updates. The previous fallback -- the whole
    dataset in the first batch -- turned any sizeable dataset into hours of generation before
    the first training log line, indistinguishable from a hang.
    """
    if generation_batch_size is not None:
        return generation_batch_size
    global_batch_size = train_batch_size * max(1, gradient_accumulation_steps)
    prompts_per_update = math.ceil(global_batch_size / max(1, num_generations))
    return max(1, prompts_per_update * _STEPS_PER_ROLLOUT_BATCH * max(1, num_iterations))


def stream_batch_size(*, colocate: bool, generation_batch_size: int, num_generations: int,
                      train_batch_size: int, gradient_accumulation_steps: int) -> int:
    """Rows of one streaming-driver batch: whole groups and whole optimizer updates.

    Disaggregated hands out the smallest such batch, ``lcm(train_batch_size * GA, num_generations)``.
    Colocate pays a sleep / wake / weight sync per batch, so its batch is the generation batch
    (``generation_batch_size`` prompts) rounded up to that quantum, amortizing the hand-over.
    """
    quantum = math.lcm(train_batch_size * max(1, gradient_accumulation_steps), num_generations)
    if not colocate:
        return quantum
    return max(1, math.ceil(generation_batch_size * num_generations / quantum)) * quantum


def rollout_step_budget(*, num_prompts: int, num_generations: int, train_batch_size: int,
                        gradient_accumulation_steps: int, num_train_epochs: float,
                        generation_batch_size: Optional[int] = None,
                        num_iterations: int = 1,
                        stream_batch_rows: Optional[int] = None) -> int:
    """Count full optimizer updates after dropping unusable rollout samples.

    ``train_batch_size`` is B * DP, so GBS = train_batch_size * gradient_accumulation_steps.
    Each generation batch repeats ``num_iterations`` times before its last incomplete update
    is dropped. With ``stream_batch_rows`` (see :func:`stream_batch_size`), the streaming driver
    combines generation batches and drops its own delivery tail before replay.
    """
    # Count the samples actually generated by each batch, including smaller epoch tails.
    prompt_batch_size = min(generation_batch_size or num_prompts, num_prompts)
    batches_per_epoch = math.ceil(num_prompts / prompt_batch_size)
    total_batches = max(1, math.ceil(num_train_epochs * batches_per_epoch))
    sample_counts = []
    for batch_index in range(total_batches):
        batch_in_epoch = batch_index % batches_per_epoch
        remaining_prompts = num_prompts - batch_in_epoch * prompt_batch_size
        actual_prompt_count = min(prompt_batch_size, remaining_prompts)
        sample_counts.append(actual_prompt_count * num_generations)

    global_batch_size = train_batch_size * gradient_accumulation_steps
    # The driver drops before delivery; a round loop instead drops after replaying each batch.
    if stream_batch_rows:
        sample_counts = [(sum(sample_counts) // stream_batch_rows) * stream_batch_rows]

    optimizer_steps = 0
    for num_samples in sample_counts:
        optimizer_steps += (num_samples * max(1, num_iterations)) // global_batch_size

    return optimizer_steps


def resolve_rollout_max_steps(explicit_max_steps: int, budget: int, *, recipe: str) -> int:
    """Explicit ``--max_steps`` wins; a non-positive budget fails loudly instead of a 0-step run."""
    if explicit_max_steps and explicit_max_steps > 0:
        return explicit_max_steps
    if budget <= 0:
        raise ValueError(
            f"{recipe}: derived {budget} training steps -- the rollout data cannot form an optimizer "
            "update under the configured training batch size and gradient accumulation policy. "
            "Enlarge the dataset, generation batch or num_generations, lower the training batch size or GA, or set "
            '--max_steps explicitly.')
    return budget


def plan_device_groups(nproc_per_node: int, sampler_mode: Optional[str], sampler_world_size: int,
                       auxiliary: Sequence[Tuple[str, int]]) -> Tuple[List[Tuple[str, List[int]]], bool]:
    """Plan the twinkle DeviceGroups for an online-RL run (pure: no twinkle import).

    Colocate overlays separate trainer and sampler groups on the same GPUs (independent rank
    spaces); anything else appends a disjoint sampler group, and each auxiliary entry is appended
    last, disjoint, in order. Returns ``(groups, colocate)``.
    """
    if nproc_per_node is None or nproc_per_node < 1:
        raise ValueError(f'nproc_per_node must be >= 1 (the trainer GPU count), got {nproc_per_node!r}.')
    if sampler_world_size < 1:
        raise ValueError(f'sampler_world_size must be >= 1, got {sampler_world_size}.')
    if sampler_mode == 'colocate':
        if sampler_world_size > nproc_per_node:
            raise ValueError(f'colocate needs the sampler ({sampler_world_size} GPUs) to fit within the trainer '
                             f'GPUs ({nproc_per_node}). Use sampler_mode="disaggregated" for its own GPUs.')
        # The overlay has its own worker-rank space; reusing MODEL_GROUP would inherit the trainer's
        # worker count and DeviceMesh, wrong as soon as sampler TP/PP is wider than one GPU.
        groups = [(MODEL_GROUP, list(range(nproc_per_node))), (SAMPLER_GROUP, list(range(sampler_world_size)))]
        colocate, next_rank = True, nproc_per_node
    else:
        total = nproc_per_node + sampler_world_size
        groups = [
            (MODEL_GROUP, list(range(nproc_per_node))),
            (SAMPLER_GROUP, list(range(nproc_per_node, total))),
        ]
        colocate, next_rank = False, total
    reserved = {MODEL_GROUP, SAMPLER_GROUP}
    for name, world_size in auxiliary:
        if world_size <= 0:
            continue
        if name in reserved:
            raise ValueError(f'auxiliary DeviceGroup name {name!r} collides with a reserved group '
                             f'({sorted(reserved)}); each frozen model needs a unique name.')
        groups.append((name, list(range(next_rank, next_rank + world_size))))
        next_rank += world_size
    return groups, colocate


def auxiliary_mesh(group: str, spec: Optional[str], meshes: Optional[Dict[str, Any]]) -> Optional[Any]:
    """The mesh one auxiliary builds with: the planned one, else re-derived from its (rewritten) spec."""
    mesh = (meshes or {}).get(group)
    if mesh is not None or not spec:
        return mesh
    from twinkle import DeviceMesh

    return DeviceMesh.from_spec(spec)


def initialize_rl_devices(*, model_config: 'ModelConfig', template_config: 'TemplateConfig',
                          dataset_config: 'DatasetConfig', train_config: 'TrainConfig',
                          distributed_config: 'DistributedConfig', checkpoint_config: 'CheckpointConfig',
                          rollout_config: 'RolloutConfig', rl_config: 'RLConfig',
                          reward_config: 'RewardConfig', tuner_config: Optional['TunerConfig'],
                          logging_config: Optional['LoggingConfig'] = None,
                          quantize_config: Optional['QuantizeConfig'] = None,
                          megatron_config: Optional['MegatronConfig'] = None,
                          moe_config: Optional['MoEConfig'] = None, output_dir: str = 'output',
                          prompt_rows: Sequence[Dict[str, Any]] = (),
                          group_requests: Callable[[], List[Tuple[str, Optional[str]]]]
                          ) -> Tuple[Any, Any, Dict[str, Any]]:
    """Prepare the configs, build the template, and initialize twinkle + perf for an online-RL run.

    Returns ``(template, resource_plan, auxiliary_meshes)``: the plan is the ClusterAdapter's (already
    APPLIED -- derived parallel sizes written back into the configs before twinkle.initialize reserves
    the GPUs) or ``None``; the meshes map each auxiliary group to the DeviceMesh its model shards with
    (empty under an estimate plan). ``group_requests`` supplies this recipe's own frozen-model groups.
    """
    import twinkle
    from twinkle import DeviceGroup, DeviceMesh, Platform
    from twinkle.perf import Perf
    from twinkle_agentic.async_rl import estimate_plan, resource_device_groups

    from swift.dev.builders import build_dp_mesh, build_template, load_model_processor
    from swift.dev.builders.sandbox import SANDBOX
    from swift.dev.builders.streaming import CONTROL_GROUP, ENV_GROUP, ROLLOUT_GROUP
    from swift.dev.config import process_configs, validate_configs

    process_configs(
        model_config, template_config, dataset_config, train_config, distributed_config,
        checkpoint_config, tuner_config, rl_config, rollout_config=rollout_config,
        megatron_config=megatron_config, quantize_config=quantize_config, plugin_config=None)
    validate_configs(
        model_config, template_config, dataset_config, train_config, distributed_config,
        checkpoint_config, tuner_config, reward_config=reward_config,
        rollout_config=rollout_config, logging_config=logging_config, quantize_config=quantize_config,
        megatron_config=megatron_config, moe_config=moe_config)
    _, processor = load_model_processor(model_config)
    task_type = model_config.task_type or 'causal_lm'
    template = build_template(template_config, processor, task_type=task_type)
    write_ckpt_args_json(
        output_dir, processor, model_config, template_config, tuner_config, quantize_config,
        dataset_config=dataset_config, task_type=task_type)

    backend = rollout_config.rollout_sampler
    if backend not in ('vllm', 'sglang'):
        raise ValueError(f"rollout_sampler={backend!r} cannot back an online-RL rollout: weight sync into the "
                         "sampler needs a CheckpointEngineMixin engine. Use 'vllm' or 'sglang'.")
    requests = group_requests()
    # One mesh per auxiliary, built once so the group's rank count and the model's sharding cannot
    # drift; an estimate plan rewrites the specs, so the dict is dropped and re-derived at the build.
    auxiliary_meshes: Dict[str, Any] = {
        name: (DeviceMesh.from_spec(spec) if spec else None) for name, spec in requests}
    resource_plan = estimate_plan(
        model_config=model_config, distributed_config=distributed_config, rollout_config=rollout_config,
        template=template, prompt_rows=prompt_rows,
        auxiliary_groups=[(name, (mesh.world_size if mesh else 1)) for name, mesh in auxiliary_meshes.items()],
        max_completion_length=rl_config.max_completion_length)
    if backend == 'sglang':
        sampler_width = rollout_config.sglang_tp_size * rollout_config.sglang_pp_size
        sampler_replicas = rollout_config.sglang_dp_size
    else:
        sampler_width = rollout_config.vllm_tensor_parallel_size * rollout_config.vllm_pipeline_parallel_size
        sampler_replicas = rollout_config.vllm_data_parallel_size
    if resource_plan is None:
        groups, colocate = plan_device_groups(
            distributed_config.nproc_per_node, rollout_config.sampler_mode,
            sampler_width * sampler_replicas,
            [(name, (mesh.world_size if mesh else 1)) for name, mesh in auxiliary_meshes.items()])
    else:
        groups = resource_device_groups(resource_plan)
        auxiliary_meshes = {}
        logger.info('RL resource plan:\n%s', resource_plan.describe())
    if distributed_config.mode != 'ray':
        raise ValueError("online RL requires DistributedConfig.mode='ray': the trainer and the rollout sampler "
                         'are separate Ray actors that the driver drives and syncs weights between.')
    allocated = set()
    device_groups = []
    for group_name, ranks in groups:
        rank_set = set(ranks)
        shared = group_name != MODEL_GROUP and bool(rank_set) and rank_set.issubset(allocated)
        # A sampler group's workers each span the engine's parallel width; every other group is one
        # GPU per rank.
        sampler_group = (group_name in (SAMPLER_GROUP, 'shared_sampler') or group_name.startswith('sampler_')
                         or group_name.startswith('shared_sampler_') or group_name.startswith('flex_sampler_'))
        width = sampler_width if sampler_group else 1
        device_groups.append(
            DeviceGroup(name=group_name, ranks=ranks,
                        device_type=Platform.get_platform().__name__, gpus_per_worker=width, shared=shared))
        if not shared:
            allocated.update(rank_set)
    total = max(allocated) + 1 if allocated else 0
    # The throughput denominator is the physical allocation, not the sum of
    # overlapping trainer/sampler worker counts or the currently active replicas.
    template.performance_device_groups = [(name, list(ranks)) for name, ranks in groups]
    # The CPU groups of the stream: rollout replicas, sandbox env replicas, and one rank for the
    # driver / checkpoint manager / scoring workers.
    cpu_groups = [
        DeviceGroup(name=ROLLOUT_GROUP, ranks=rollout_config.rollout_num_workers, device_type='CPU'),
        DeviceGroup(name=CONTROL_GROUP, ranks=1, device_type='CPU'),
    ]
    if SANDBOX in (rollout_config.tools or ()):
        cpu_groups.append(DeviceGroup(name=ENV_GROUP, ranks=rollout_config.sandbox_num_workers, device_type='CPU'))
    twinkle.initialize(
        mode='ray',
        nproc_per_node=total,
        extra_torch_kwargs={'seed': train_config.seed, 'full_determinism': train_config.full_determinism,
                            'tf32': train_config.tf32},
        global_device_mesh=build_dp_mesh(distributed_config, template_config.sequence_parallel_size),
        name=distributed_config.ray_exp_name,
        groups=[*device_groups, *cpu_groups])
    logger.info('twinkle initialized with DeviceGroups: %s', [(g.name, g.ranks) for g in device_groups])
    Perf.initialize({})  # the cluster's actors register themselves once built (see the run_*)
    return template, resource_plan, auxiliary_meshes


def build_rl_sampler(*, model_config: 'ModelConfig', rollout_config: 'RolloutConfig', template: Any,
                     tuner_config: Optional['TunerConfig'], resource_plan: Any = None,
                     colocate: bool = False, enable_sampling_replay: bool = False) -> Any:
    """The weight-syncable rollout sampler: shaped engine args, then the planned or static build.

    The planned layout (estimate/adaptive) sizes the replica pool and stamps the elastic instance id;
    the static shape reads the rollout_config replica knobs (estimate's apply_plan already rewrote
    them, so the two agree). Colocate pins the sleep/save knob the borrow-and-return loop needs.
    """
    from twinkle import DeviceMesh

    from swift.dev.builders import build_engine_args, build_sampler

    backend = rollout_config.rollout_sampler
    sampler_engine_args = build_engine_args(backend, None, rollout_config)
    if backend == 'vllm':
        # RL implements DP with independently dispatched sampler actors. Each actor owns
        # only one TP x PP replica; forwarding the outer DP size would make every vLLM
        # engine launch that many replicas again, outside its assigned device range.
        sampler_engine_args['data_parallel_size'] = 1
        # Every RL weight sync lands through twinkle's vLLM worker extension. Install the
        # quant-aware subclass so an FP8 / MXFP4 engine can absorb the trainer's BF16 stream;
        # it is a pass-through when the engine is not quantized. An explicit engine arg wins.
        from swift.dev.weight_sync import QUANT_SYNC_WORKER_EXTENSION
        sampler_engine_args.setdefault('worker_extension_cls', QUANT_SYNC_WORKER_EXTENSION)
    if colocate:
        sampler_engine_args['enable_sleep_mode' if backend == 'vllm' else 'enable_memory_saver'] = True
    if enable_sampling_replay:
        sampler_engine_args['enable_sampling_replay'] = True
    if resource_plan is not None:
        return build_sampler(
            model_config, backend=backend, engine_args=dict(sampler_engine_args),
            device_mesh=DeviceMesh.from_sizes(dp_size=resource_plan.sampler.replica_count), template=template,
            remote_group=SAMPLER_GROUP, instance_id='adaptive-sampler')
    if backend == 'sglang':
        replicas = rollout_config.sglang_dp_size
    else:
        replicas = rollout_config.vllm_data_parallel_size
    return build_sampler(
        model_config, backend=backend, engine_args=sampler_engine_args,
        device_mesh=DeviceMesh.from_sizes(dp_size=replicas), template=template,
        remote_group=SAMPLER_GROUP)


def resolve_reward_scorers(channel: str, *, reward_config: Any, model_config: 'ModelConfig',
                           template_config: 'TemplateConfig', distributed_config: 'DistributedConfig',
                           rollout_config: 'RolloutConfig', backend: str, sampler: Any,
                           worker_group: Optional[str] = None,
                           quantize_config: Optional['QuantizeConfig'] = None) -> Tuple[List[Any], List[str]]:
    """One reward channel -> ``(scorer workers, metric names)``, one worker per ``--<channel>`` entry, in order.

    ``channel`` is 'orm' or 'prm': the merged ``--<channel>`` list resolves through
    ``resolve_channel``; each rule becomes a ``FunctionWorker`` and each reward MODEL goes through
    ``build_reward_model`` onto its own group (``reward_<i>`` / ``prm_<i>`` -- the SAME names the
    recipe's group plan allocated). Every worker sits on ``worker_group``. A PRM model must resolve
    SCALAR (online RL places only a seq_cls/reranker frozen PRM today).
    """
    from twinkle.reward import FunctionWorker

    from swift.dev.builders import build_reward_model
    from swift.dev.rewards.reward import get_reward_funcs
    from swift.dev.rewards.scorer import ChannelCompanions, resolve_channel

    plan = resolve_channel(
        list(getattr(reward_config, channel)),
        ChannelCompanions(
            adapters=getattr(reward_config, f'{channel}_adapters'),
            model_types=getattr(reward_config, f'{channel}_model_type'),
            task_types=getattr(reward_config, f'{channel}_task_type', None),
            model_revisions=getattr(reward_config, f'{channel}_model_revision'),
            template_names=getattr(reward_config, f'{channel}_template'),
            parallel_spec=getattr(reward_config, f'{channel}_parallel_spec')),
        channel=channel, sampler_model=model_config.model, backend=backend)
    funcs, names = get_reward_funcs(plan.rules, reward_config)
    scorers = [FunctionWorker(func, remote_group=worker_group) for func in funcs]
    names = list(names)
    prefix = 'reward' if channel == 'orm' else channel
    for index, spec in enumerate(plan.specs):
        if channel == 'prm' and spec.kind != 'scalar':
            raise ValueError(f'--prm model {spec.model_id!r} resolves to a {spec.kind!r} reward, but online RL '
                             'places only a SCALAR (seq_cls/reranker) frozen PRM today.')
        scorers.append(build_reward_model(
            spec, model_config=model_config, template_config=template_config,
            distributed_config=distributed_config, rollout_config=rollout_config, backend=backend,
            sampler=sampler, remote_group=f'{prefix}_{index}', worker_group=worker_group,
            judge_template=reward_config.judge_template, quantize_config=quantize_config))
        names.append(spec.display_name)
    return scorers, names


def teacher_group(index: int) -> str:
    """The DeviceGroup name for the ``index``-th frozen teacher when MOPD SPREADS its K apart.

    A spread placement gives each teacher its own group (a parallel ``forward_only``); the packed
    placement builds all K into the single shared :data:`TEACHER_GROUP` and scores them serially.
    """
    return f'teacher_{index}'


def spec_world_size(parallel_spec: Optional[str], default: int = 1) -> int:
    """The DeviceGroup rank count a frozen auxiliary's ``parallel_spec`` needs (``default`` if none).

    The value MUST equal what the group plan allocated, since the frozen build reads its
    ``nproc_per_node`` as the group size.
    """
    if not parallel_spec:
        return default
    from twinkle import DeviceMesh

    return DeviceMesh.from_spec(parallel_spec).world_size


def prompt_rows_from_dataset(dataset_config: 'DatasetConfig') -> Tuple[List[Dict[str, Any]], List[List[dict]], List[Dict[str, Any]]]:
    """Load prompt rows, the prompt message views, and the non-message columns for the rollout.

    Returns ``(rows, prompts, extras)``: ``rows`` is the raw loaded rows (the resource estimator
    reads the prompt lengths from them), ``prompts`` the message lists with any trailing assistant
    turn stripped, ``extras`` everything else (solution/target/teacher_prompt/...) keyed per prompt.
    """
    from swift.dev.builders import load_prompt_only_rows

    rows = load_prompt_only_rows(dataset_config, None, split_dataset_ratio=0.0)
    if not rows:
        raise ValueError('online RL got an empty dataset. Set DatasetConfig.dataset with prompts to roll out on.')
    prompts: List[List[dict]] = []
    extras: List[Dict[str, Any]] = []
    for row in rows:
        messages = row.get('messages') if isinstance(row, dict) else None
        if not messages:
            continue
        if messages[-1].get('role') == 'assistant':
            messages = messages[:-1]
        prompts.append(list(messages))
        extras.append({key: value for key, value in row.items() if key != 'messages'})
    if not prompts:
        raise ValueError('online RL found no prompt messages in the dataset rows (expected a `messages` column).')
    return rows, prompts, extras


# The offline family's assembly pieces; each family's run file holds the loop that drives them.


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


def initialize_supervised_twinkle(distributed_config: 'DistributedConfig', *, seed: int,
                                  full_determinism: bool, tf32: Optional[bool],
                                  auxiliary_groups: Sequence[Tuple[str, int]] = ()) -> None:
    """Initialize twinkle for EVERY offline backend -- required for a DeviceMesh to reach the model.

    mode='ray' builds the 'model' DeviceGroup the workers hold (build_model targets it with
    remote_group='model'); everything else initializes 'local'. ``auxiliary_groups`` is one
    ``(name, world_size)`` per FULL-PARAMETER frozen auxiliary (the RLHF family's reference):
    a disjoint group appended after the 'model' ranks. The seed bundle re-seeds every ray worker.
    """
    import twinkle
    from twinkle import DeviceGroup, Platform

    extra_torch_kwargs = {'seed': seed, 'full_determinism': full_determinism, 'tf32': tf32}
    if distributed_config.mode == 'ray':
        nproc = distributed_config.nproc_per_node
        if nproc is None:
            raise ValueError("DistributedConfig.nproc_per_node is required in mode='ray' (it sizes the Ray "
                             "'model' DeviceGroup). Pass it explicitly -- there is no default.")
        groups = [DeviceGroup(name='model', ranks=list(range(nproc)),
                              device_type=Platform.get_platform().__name__, gpus_per_worker=1)]
        next_rank = nproc
        planned = ['model']
        for name, world_size in auxiliary_groups:
            if world_size < 0:
                raise ValueError(f'auxiliary DeviceGroup {name!r} world_size must be >= 0, got {world_size}.')
            if world_size == 0:
                continue
            if name in planned:
                raise ValueError(f'auxiliary DeviceGroup name {name!r} collides with a reserved or already-planned '
                                 f'group ({planned}); each frozen model needs a unique name.')
            groups.append(DeviceGroup(name=name, ranks=list(range(next_rank, next_rank + world_size)),
                                      device_type=Platform.get_platform().__name__, gpus_per_worker=1))
            planned.append(name)
            next_rank += world_size
        twinkle.initialize(
            mode='ray', nproc_per_node=next_rank, extra_torch_kwargs=extra_torch_kwargs,
            # --ray_exp_name names this Ray run (the cluster/worker-name prefix); a local run has no
            # Ray experiment to name, so it is read only here. None keeps twinkle's default.
            name=distributed_config.ray_exp_name, groups=groups)
    else:
        twinkle.initialize(mode='local', extra_torch_kwargs=extra_torch_kwargs)


def process_and_validate(model_config: 'ModelConfig', template_config: 'TemplateConfig',
                         dataset_config: 'DatasetConfig', train_config: 'TrainConfig',
                         distributed_config: 'DistributedConfig',
                         checkpoint_config: Optional['CheckpointConfig'],
                         tuner_config: Optional['TunerConfig'], logging_config: Optional['LoggingConfig'],
                         quantize_config: Optional['QuantizeConfig'],
                         megatron_config: Optional['MegatronConfig'],
                         moe_config: Optional['MoEConfig'], *,
                         rl_config: Optional['RLConfig'] = None,
                         reward_config: Optional['RewardConfig'] = None) -> None:
    """Process and validate the Configs before building anything heavy (idempotent per stage).

    ``rl_config`` / ``reward_config`` ride along when a family consumes them (the RLHF family).
    """
    from swift.dev.config import process_configs, validate_configs

    process_configs(
        model_config, template_config, dataset_config, train_config, distributed_config,
        checkpoint_config, tuner_config, rl_config,
        rollout_config=None, megatron_config=megatron_config, quantize_config=quantize_config,
        plugin_config=None)
    validate_configs(
        model_config, template_config, dataset_config, train_config, distributed_config,
        checkpoint_config, tuner_config, rl_config,
        reward_config=reward_config, rollout_config=None, logging_config=logging_config,
        quantize_config=quantize_config, megatron_config=megatron_config, moe_config=moe_config)


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


def resolve_step_intervals(train_config: 'TrainConfig',
                           checkpoint_config: Optional['CheckpointConfig'], total_steps: int) -> None:
    """Resolve the eval/save intervals into the concrete optimizer-step counts the loop uses."""
    train_config.eval_steps = _resolve_eval_steps(train_config, total_steps)
    if checkpoint_config is not None:
        checkpoint_config.save_steps = _resolve_step_interval(
            checkpoint_config.save_steps, total_steps, 'save_steps')


def plan_supervised_steps(recipe: str, train_config: 'TrainConfig', dataloader: Any) -> int:
    """The optimizer-step budget, with the zero-step fail-fast.

    ``N <= ga`` micro-batches yield 0 optimizer steps (the GA gate lags one step, so a dataset that
    never fills a full lagged window would run forward/backward but NEVER update the model); rather
    than silently run a no-op training, fail with an actionable message. Runs BEFORE build_model so
    we fail before loading heavy weights.
    """

    if train_config.max_steps and train_config.max_steps > 0:
        return train_config.max_steps
    ga = max(1, train_config.gradient_accumulation_steps)
    try:
        micro_per_epoch = len(dataloader)  # micro-batches (already /batch_size) per epoch
    except (TypeError, NotImplementedError):
        micro_per_epoch = 0  # IterableDataset: caller must set max_steps
    steps_per_epoch = math.ceil(micro_per_epoch / ga)
    total_opt_steps = math.ceil(steps_per_epoch * train_config.num_train_epochs)
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
    from swift.dev.eval import build_task_config, model_name, validate_eval_datasets

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

        sampler = TransformersSampler(template=template, model=model)
        return GenerativeEval(
            sampler=sampler, template=template, datasets=eval_datasets, model_id=eval_model_id,
            task_config=task_config, enter=None, exit=None)

    from twinkle.checkpoint_engine import CheckpointEngineManager

    from swift.dev.builders import build_dp_mesh, build_sampler
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
    # Eval has no RolloutConfig. Sleep at level 1 so the trainer still gets the device back; the
    # manager's colocate sync wakes the weights, pushes them, offloads the trainer and wakes the KV
    # cache (the RL rollout's own sequence, so it cannot drift).
    manager = CheckpointEngineManager(
        model=model, sampler=sampler, mode='colocate', sleep_level=1,
        offload_kwargs={'offload_model': True, 'offload_optimizer': True})

    def exit_eval() -> None:
        """Take the device back for the trainer: sleep the engine, then reload the trainer."""
        sampler.sleep(level=1)
        model.reload_to_gpu()

    return GenerativeEval(
        sampler=sampler, template=template, datasets=eval_datasets, model_id=eval_model_id,
        task_config=task_config, enter=manager.sync_weights, exit=exit_eval)


# The absorbed training infrastructure: the loop cadence helpers, the tracker and the
# single-teacher resolver (was recipe/common).


def filter_perf_actors(candidates: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only the genuine ``@remote_class`` actors from a ``label -> object`` candidate map.

    A dev assembly / loop owns a mix of remote actors and plain in-process objects (a local dataloader, the
    template). ``twinkle.infra`` injects the ``perf_on`` control handle onto every ``@remote_class`` and only
    those, so ``hasattr(handle, 'perf_on')`` is exactly the "is a profileable actor" test -- and
    ``Perf.initialize`` must be handed only actors. Nothing is lost by dropping a plain object: under local
    mode its ``@timed`` spans still land in the one shared driver recorder (labelled ``driver``), and under Ray
    only the actors that own worker ranks have per-rank timings to dump. ``None`` candidates are dropped too, so
    a caller can list every object it might have built without guarding each one.
    """
    return {label: handle for label, handle in candidates.items() if handle is not None and hasattr(handle, 'perf_on')}


def flatten_evalscope_report(summary: Any) -> Dict[str, float]:
    """Flatten EvalScope report rows into a ``{metric_name: score}`` dict.

    ``summarize`` returns one report dict per benchmark (an ``evalscope.report.Report``): its overall
    ``score`` is keyed by the benchmark ``name``, and each entry in ``metrics`` contributes its own
    ``name``/``score``. ``metric_for_best_model`` on the generative path names one of these keys, so
    best-model tracking and ``on_evaluate`` read a flat mapping rather than the nested report.
    """
    flat: Dict[str, float] = {}
    for row in summary or []:
        if not isinstance(row, dict):
            continue
        if row.get('name') is not None and row.get('score') is not None:
            flat[str(row['name'])] = float(row['score'])
        for metric in row.get('metrics') or []:
            if isinstance(metric, dict) and metric.get('name') is not None and metric.get('score') is not None:
                flat[str(metric['name'])] = float(metric['score'])
    return flat


def evaluate_loss(model: Any, eval_dataloader: Any, *, eval_iters: int, manual_gc: bool,
                  manual_gc_eval: bool, micro_step_kwargs: Callable[..., Dict[str, Any]],
                  global_step: int,
                  micro_batch_size: Optional[int] = 1,
                  eval_group_size: int = 1) -> Optional[dict]:
    """Val-loss eval: the normalized per-token loss over eval_dataloader (None if there is none).

    Mirrors the train metric path on the eval side: each val batch runs ``forward_only`` (no grad)
    then ``calculate_loss`` (fills this batch's loss); a final ``calculate_metric(False)`` aggregates
    and resets. Megatron already produced the loss inside forward_only -- its ``calculate_loss``
    is a no-op returning inf; the metric state is already populated.

    ``eval_group_size`` batches N dataloader batches into one forward call (1 = the historic
    one-batch-per-call). N > 1 serves Megatron's VPP scheduler, which rejects a forward whose
    micro-batch count is below PP or not a multiple of it (each dataloader batch is one
    micro-batch here); training groups by GA, so GA is the batch count the schedule already
    proved schedulable. A trailing partial group cannot satisfy that floor and is dropped:
    eval is a sampled estimate, and padding would double-count the padded rows in the
    per-token loss.

    ``micro_batch_size=None`` submits each eval batch as one whole forward -- required for
    tasks the forward-only micro-batcher cannot split at sample level (record-level labels,
    one-record-to-many-sequence expansions) and for packed eval rows (the task set lives in
    ``supervised.py``'s ``_WHOLE_BATCH_EVAL_TASKS``).
    """
    if eval_dataloader is None:
        return None
    if manual_gc and manual_gc_eval:
        import gc
        gc.collect()

    def _forward(chunk: list) -> None:
        model.forward_only(**micro_step_kwargs(chunk, micro_batch_size=micro_batch_size))
        # Megatron's calculate_loss is a no-op returning inf (the loss already came out of
        # forward_only); transformers' populates the metric state. The return value is unused.
        model.calculate_loss()

    group: list = []
    consumed = 0
    for index, batch in enumerate(eval_dataloader):
        if eval_iters > 0 and index >= eval_iters:
            break
        group.extend(batch)
        consumed += 1
        if consumed == eval_group_size:
            _forward(group)
            group, consumed = [], 0
    if group:
        if eval_group_size > 1:
            logger.warning(f'Dropped {len(group)} trailing eval sample(s): a partial group of '
                           f'{consumed} batch(es) cannot fill the {eval_group_size}-micro-batch forward '
                           'that the megatron VPP schedule requires.')
        else:
            _forward(group)
    metrics = model.calculate_metric(is_training=False)
    result = {'step': global_step}
    if metrics.get('loss') is not None:
        result['eval_loss'] = float(metrics['loss'])
    for key, value in metrics.items():
        if key.startswith('loss_'):
            result[f'eval_{key}'] = float(value)
        elif key != "loss":
            # Non-loss metrics (decision's acc/record_acc/ece, or a task head's own component)
            # pass through under an eval_ prefix. Values that are already numeric stay numeric so
            # best-model tracking can compare them; formatted strings ('0.9000', '-') ride along
            # verbatim -- forcing them through float() would crash the eval loop.
            try:
                result[f"eval_{key}"] = float(value)
            except (TypeError, ValueError):
                result[f"eval_{key}"] = value
    if manual_gc and manual_gc_eval:
        import gc
        gc.collect(0)
    return result


def evaluate_generate(generative: Optional[GenerativeEval], *, global_step: int) -> Optional[dict]:
    """Generative eval: run an EvalScope benchmark over the resident sampler (None if unwired).

    Only rank 0 drives EvalScope (a single process owns the report files); the other ranks wait at a
    barrier so the next training step stays collective-safe. ``enter``/``exit`` bracket the run with
    the weight-sync + memory schedule for a co-resident vllm/sglang engine; they run on every rank
    (under ray the driver drives the remote workers, under torchrun they are collective).
    """
    if generative is None:
        return None
    import torch.distributed as dist

    from swift.dev.eval import run_evaluator

    distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    is_main = (not distributed) or dist.get_rank() == 0
    result: Dict[str, Any] = {'step': global_step}
    if generative.enter is not None:
        generative.enter()
    try:
        if is_main:
            summary = run_evaluator(
                generative.sampler,
                generative.template,
                datasets=generative.datasets,
                model_id=generative.model_id,
                task_config=generative.task_config)
            result.update(flatten_evalscope_report(summary))
            result['eval_report'] = summary
        if distributed:
            dist.barrier()
    finally:
        if generative.exit is not None:
            generative.exit()
    return result


def track_best(metrics: dict, best: Tuple[Optional[float], int, Optional[str]], *,
               metric_for_best_model: Optional[str], greater_is_better: bool, global_step: int,
               output_dir: str) -> Tuple[Optional[float], int, Optional[str]]:
    """Update the ``(best_metric, best_step, best_ckpt)`` triple when the watched metric improves."""
    best_metric, best_step, best_ckpt = best
    if not metric_for_best_model:
        return best
    value = metrics.get(metric_for_best_model)
    if value is None:
        logger.warning(f'metric_for_best_model={metric_for_best_model!r} was not produced by this eval '
                       f'(got {sorted(key for key in metrics if key != "step")}); best-model tracking '
                       'skipped for this pass.')
        return best
    value = float(value)
    improved = (best_metric is None
                or (value > best_metric if greater_is_better else value < best_metric))
    if improved:
        return value, global_step, os.path.join(output_dir, f'checkpoint-{global_step}')
    return best


def load_best_model(model: Any, best_ckpt: Optional[str], *, metric_for_best_model: Optional[str],
                    best_metric: Optional[float], best_step: int,
                    adapter_name: Optional[str]) -> None:
    """Reload the best checkpoint at train end (the caller gates on load_best_model_at_end)."""
    if not best_ckpt:
        logger.warning('load_best_model_at_end is set but no eval produced metric_for_best_model; '
                       'keeping the final weights.')
        return
    if not os.path.isdir(best_ckpt):
        raise FileNotFoundError(
            f'load_best_model_at_end wants {best_ckpt} (best {metric_for_best_model}='
            f'{best_metric} at step {best_step}), but it does not exist: save_steps must align '
            'with eval_steps so the best step is checkpointed, and save_total_limit must not have '
            'pruned it.')
    logger.info(f'load_best_model_at_end: reloading best checkpoint {best_ckpt} '
                f'({metric_for_best_model}={best_metric} at step {best_step}).')
    # LoRA checkpoints reload through their adapter group, so load() targets it rather than the
    # full-parameter weights.
    kwargs = {'adapter_name': adapter_name} if adapter_name else {}
    model.load(best_ckpt, **kwargs)


def num_optimizer_steps(num_micro_batches: int, gradient_accumulation_steps: int) -> int:
    """Optimizer-update count for a given number of micro-batches under twinkle GA.

    twinkle's do_grad_sync fires at cur_step = ga+1, 2*ga+1, ... (one micro-step late),
    so over N micro-batches the number of optimizer steps is floor((N-1)/ga) for ga>1,
    and N for ga==1. Used to size the LR scheduler's num_training_steps so it matches the
    actual number of steps taken.

    No production caller today: every recipe runs ga=1 and sizes the scheduler horizon via
    plan_supervised_steps / the rollout step budget instead. The contract only bites at
    MODEL-LEVEL ga>1, and this function stays as that contract's oracle (the alignment tests
    pin configure_optimizer's num_training_steps against it).
    """
    ga = max(1, gradient_accumulation_steps)
    if ga == 1:
        return num_micro_batches
    return max(0, (num_micro_batches - 1) // ga)


def backend_owns_gradient_accumulation(model) -> bool:
    """True when the backend's ``forward_backward`` internalizes gradient accumulation.

    The webui train_step component forks on exactly this axis: most backends (transformers) leave
    the micro-batch schedule to the caller -- one micro-batch per ``forward_backward``, the
    optimizer stepped every ga-th call -- while a backend that owns its own schedule needs the
    caller to instead group the ga batches and call ``forward_backward`` once per optimizer step;
    in dev that is Megatron, whose 1F1B pipeline derives GA from the length of the micro-batch
    LIST it is handed. A future such backend (e.g. torchtitan/veomni running pipeline parallelism)
    is recognized HERE, so the dispatch stays a two-way fork on this capability rather than
    growing a branch per backend name. (The recipe loops moved this fork down into twinkle with
    the micro-batch rework; this copy serves the webui component, which still schedules
    driver-side -- see webui/components/training.py.)
    """
    try:
        from twinkle.model.megatron import MegatronModel
    except Exception:
        return False
    return isinstance(model, MegatronModel)


def is_grad_sync_boundary(micro_step: int, gradient_accumulation_steps: int) -> bool:
    """Whether ``micro_step`` is an optimizer-update boundary, computed driver-side.

    Replicates twinkle's ``OptimizerGroup.do_grad_sync`` (a pure function of ``cur_step`` and the
    accumulation size: boundary when ga==1, else one micro-step late at cur_step = ga+1, 2ga+1,
    ...) using the caller's own ``micro_step`` as ``cur_step``. Module-level because the
    optimizer group lives on the WORKERS: under Ray the driver-side model is a proxy with no such
    attribute, so the webui component's loop must derive the boundary here instead of reaching
    into the proxy.
    """
    ga = gradient_accumulation_steps
    return ga == 1 or ((micro_step - 1) % ga == 0 and micro_step > 1)


def start_manual_gc(enabled: bool) -> Optional[bool]:
    """Disable automatic GC for a training loop and perform one full collection."""
    if not enabled:
        return None
    import gc
    was_enabled = gc.isenabled()
    gc.disable()
    gc.collect()
    return was_enabled


def collect_manual_gc(enabled: bool, interval: int, step: int) -> None:
    """Run the configured periodic full collection after an optimizer step."""
    if enabled and interval and step % interval == 0:
        import gc
        gc.collect()


def collect_empty_cache(model: 'TrainableModel', interval: Optional[int], step: int) -> None:
    """Release the device caching allocator every ``interval`` optimizer steps (torch_empty_cache_steps).

    A remote model call, not a driver-local ``torch.cuda.empty_cache()``: under Ray the loop runs on the
    driver while the tensors live on the workers, so the release has to be dispatched to them (the model
    method is ``dispatch='all'``). None/0 disables it.
    """
    if interval and step % interval == 0:
        model.empty_cache()


def finish_manual_gc(was_enabled: Optional[bool]) -> None:
    """Restore the interpreter's automatic-GC state after a loop exits."""
    if was_enabled:
        import gc
        gc.enable()


def save_training_checkpoint(model: 'TrainableModel', name: str, *, output_dir: str,
                             consumed_train_samples: int = 0, component_states: Optional[Dict[str, dict]] = None,
                             no_save_optim: bool = False, no_save_rng: bool = False,
                             safe_serialization: bool = True, max_shard_size: str = '5GB',
                             save_total_limit: Optional[int] = None,
                             async_save: bool = False,
                             requantize_source: Optional[str] = None) -> str:
    """Save one trainable model, then retain only the newest numbered/final checkpoints."""
    kwargs = {
        'consumed_train_samples': consumed_train_samples,
        'component_states': component_states or {},
        'safe_serialization': safe_serialization,
        'max_shard_size': max_shard_size,
        'save_total_limit': save_total_limit,
        # TransformersModel accepts extra save kwargs; MegatronModel consumes this one.
        'no_save_rng': no_save_rng,
        'async_save': async_save,
    }
    model.save(name, output_dir=output_dir, save_optimizer=not no_save_optim, **kwargs)
    ckpt_dir = os.path.join(output_dir, name)
    if requantize_source is not None:
        # A full-parameter BF16 export of a packed source is not loadable by the packed
        # rollout engine (DeepSeek-V4's FlashMLA o_proj reads FP8 weight_scale_inv).
        # Restore the source's packed format in place, using the trained weights.
        from swift.dev.weight_sync.requantize import requantize_checkpoint
        requantize_checkpoint(ckpt_dir, requantize_source)
    return ckpt_dir


_SUPPORTED_REPORTERS = frozenset({'tensorboard', 'wandb', 'swanlab'})


_SECRET_FIELDS = {'swanlab_token', 'swanlab_secret', 'swanlab_webhook_url'}


def _coerce_scalar(value: Any) -> Optional[float]:
    """Coerce one metric value to a finite number for scalar logging, or None when it is not numeric.

    twinkle.metric components format many of their outputs as fixed-precision strings -- LossMetric's
    ``loss``/``grad_norm``, DPOMetric's ``rewards/*`` and ``logps/*``, EmbeddingMetric's ``pos_sim`` -- so a
    plain ``isinstance(value, (int, float))`` filter would drop every one of them from the scalar backends.
    This reuses twinkle's own numeric parser (it already handles int/float/numeric-string/``tensor(...)``)
    and strips a trailing ``%`` so DPOMetric's ``rewards/accuracies`` (``'85.0%'``) logs as ``85.0``.
    """
    from twinkle.metric.reporting import _finite_number
    if isinstance(value, str):
        text = value.strip()
        if text.endswith('%'):
            text = text[:-1]
        return _finite_number(text)
    return _finite_number(value)


def coerce_metric_scalars(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Reduce a raw component-metric dict to its finite numeric scalars, dropping non-numeric entries.

    Applied when a loop assembles its per-step record so every value the human-readable step line formats
    (``f'{value:.4f}'``) and every scalar the tracker writes is already a number, regardless of whether the
    twinkle.metric component that produced it returned a float or a display string.
    """
    result: Dict[str, float] = {}
    for key, value in metrics.items():
        number = _coerce_scalar(value)
        if number is not None:
            result[key] = number
    return result


def _raw_metric_value(value: Any) -> Any:
    """Adapt legacy percentages; leave all other normalization to Twinkle."""
    if isinstance(value, str) and value.strip().endswith('%'):
        number = _coerce_scalar(value)
        if number is not None:
            return number
    return value


class RunTracker:
    """Own tracker setup, metric writes, and teardown for one training loop."""

    def __init__(self, config: Optional['LoggingConfig'], output_dir: str):
        self.config = config
        self.output_dir = output_dir
        self._writers: Dict[str, Any] = {}
        self._loss_total = 0.0
        self._loss_count = 0
        self._logged_prompts = set()
        import uuid

        from twinkle.metric import create_metrics_reporter
        self.run_id = uuid.uuid4().hex
        self._metric_reporter = (create_metrics_reporter({"jsonl": {
            "path": os.path.join(output_dir, "metrics_raw.jsonl"),
            "summary_path": os.path.join(output_dir, "summary.json"),
            "append": True, "batch_size": 1,
        }}, run_id=self.run_id) if _is_main_process() else None)
        if config is None or not _is_main_process():
            return
        reporters = {name.lower() for name in config.report_to}
        reporters.discard('none')
        unknown = reporters - _SUPPORTED_REPORTERS
        if unknown:
            raise ValueError(f'Unsupported report_to values: {sorted(unknown)}. Supported: '
                             f'{sorted(_SUPPORTED_REPORTERS)} or "none".')
        for reporter in sorted(reporters):
            getattr(self, f'_setup_{reporter}')()

    def should_log(self, step: int, *, epoch_end: bool = False) -> bool:
        if self.config is None or self.config.logging_strategy == 'no':
            return False
        if self.config.logging_strategy == 'epoch':
            return epoch_end
        return bool((self.config.logging_first_step and step == 1)
                    or (self.config.logging_steps and step % self.config.logging_steps == 0))

    def record(self, record) -> None:
        """Send a typed observation through the existing metric output pipeline."""
        if self._metric_reporter is not None:
            if record.context_key is None:
                record = replace(record, context_key=self.run_id)
            self._metric_reporter.record(record)

    def log(self, metrics: Dict[str, Any], step: int, *, epoch_end: bool = False) -> Dict[str, Any]:
        """Filter and write one metric record; returns the record actually emitted."""
        filtered = self._filter_metrics(metrics)
        if self._metric_reporter is not None:
            from twinkle.metric import MetricRecord
            # Twinkle normalizes scalars and preserves other raw observations in
            # attributes. Only percentage strings need Swift's legacy conversion.
            raw = {key: _raw_metric_value(value) for key, value in metrics.items()}
            self.record(MetricRecord(
                stage='train', optimizer_step=step, values=raw, attributes={'step_unit': 'optimizer_update'}))
        if not self.should_log(step, epoch_end=epoch_end):
            return filtered
        scalars = {}
        for key, value in filtered.items():
            if key == 'step':
                continue
            number = _coerce_scalar(value)
            if number is not None:
                scalars[key] = number
        writer = self._writers.get('tensorboard')
        if writer is not None:
            for key, value in scalars.items():
                writer.add_scalar(key, value, step)
        writer = self._writers.get('wandb')
        if writer is not None:
            writer.log(scalars, step=step)
        writer = self._writers.get('swanlab')
        if writer is not None:
            writer.log(scalars, step=step)
        return filtered

    @property
    def logs_prompts(self) -> bool:
        """Whether :meth:`log_prompts` writes anything (W&B unique-prompt logging is on)."""
        return bool(self.config and self.config.wandb_log_unique_prompts and self._writers.get('wandb') is not None)

    def log_prompts(self, rows: Iterable[Tuple[Any, str]], step: int) -> None:
        """Log each distinct ``(prompt, completion)`` pair once when W&B prompt logging is enabled."""
        if not self.logs_prompts:
            return
        writer = self._writers['wandb']
        table = []
        for prompt, completion in rows:
            prompt_key = json.dumps(prompt, ensure_ascii=False, sort_keys=True, default=str)
            if prompt_key in self._logged_prompts:
                continue
            self._logged_prompts.add(prompt_key)
            table.append([prompt_key, completion])
        if table:
            writer.log({'completions': writer.Table(columns=['prompt', 'completion'], data=table)}, step=step)

    def close(self) -> None:
        if self._metric_reporter is not None:
            self._metric_reporter.close()
        writer = self._writers.get('tensorboard')
        if writer is not None:
            writer.close()
        writer = self._writers.get('wandb')
        if writer is not None:
            writer.finish()
        writer = self._writers.get('swanlab')
        if writer is not None:
            finish = getattr(writer, 'finish', None)
            if finish is not None:
                finish()
        self._writers.clear()

    def _filter_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(metrics)
        loss = result.get('loss')
        if isinstance(loss, (int, float)):
            if math.isfinite(loss):
                self._loss_total += float(loss)
                self._loss_count += 1
            elif self.config is not None and self.config.logging_nan_inf_filter and self._loss_count:
                result['loss'] = self._loss_total / self._loss_count
        return result

    def _setup_tensorboard(self) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise ImportError('report_to includes tensorboard; install tensorboard to enable it.') from exc
        config = self.config
        log_dir = config.tensorboard_dir or config.logging_dir or os.path.join(self.output_dir, 'runs')
        self._writers['tensorboard'] = SummaryWriter(log_dir=log_dir, max_queue=config.tensorboard_queue_size)

    def _setup_wandb(self) -> None:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError('report_to includes wandb; install wandb to enable it.') from exc
        config = self.config
        wandb.init(
            dir=config.logging_dir or self.output_dir,
            name=config.wandb_exp_name or config.run_name,
            project=config.wandb_project,
            config=_public_config(config),
        )
        self._writers['wandb'] = wandb

    def _setup_swanlab(self) -> None:
        try:
            import swanlab
        except ImportError as exc:
            raise ImportError('report_to includes swanlab; install swanlab to enable it.') from exc
        config = self.config
        if config.swanlab_token:
            swanlab.login(config.swanlab_token)
        _register_swanlab_notification(swanlab, config)
        swanlab.init(
            project=config.swanlab_project,
            workspace=config.swanlab_workspace,
            experiment_name=config.swanlab_exp_name or config.run_name,
            logdir=config.logging_dir or self.output_dir,
            mode=config.swanlab_mode,
            config=_public_config(config),
        )
        self._writers['swanlab'] = swanlab


def _is_main_process() -> bool:
    return int(os.environ.get('RANK', '0')) == 0


def _public_config(config: 'LoggingConfig') -> Dict[str, Any]:
    return {key: value for key, value in asdict(config).items() if key not in _SECRET_FIELDS}


def _register_swanlab_notification(swanlab: Any, config: 'LoggingConfig') -> None:
    method = config.swanlab_notification_method
    if method is None:
        return
    from swanlab.plugin.notification import (
        DingTalkCallback,
        DiscordCallback,
        EmailCallback,
        LarkCallback,
        SlackCallback,
        WXWorkCallback,
    )
    callbacks = {
        'dingtalk': DingTalkCallback,
        'discord': DiscordCallback,
        'email': EmailCallback,
        'lark': LarkCallback,
        'slack': SlackCallback,
        'wxwork': WXWorkCallback,
    }
    callback_cls = callbacks.get(method)
    if callback_cls is None:
        raise ValueError(f'Unsupported swanlab_notification_method: {method!r}. Supported: {sorted(callbacks)}.')
    if method == 'email':
        required: Iterable[Optional[Any]] = (
            config.swanlab_sender_email,
            config.swanlab_receiver_email,
            config.swanlab_smtp_server,
            config.swanlab_smtp_port,
        )
        if not all(required):
            raise ValueError('SwanLab email notification requires sender_email, receiver_email, smtp_server, and '
                             'smtp_port.')
        callback = callback_cls(
            sender_email=config.swanlab_sender_email,
            receiver_email=config.swanlab_receiver_email,
            password=config.swanlab_secret,
            smtp_server=config.swanlab_smtp_server,
            port=config.swanlab_smtp_port,
            language=config.swanlab_email_language,
        )
    else:
        callback = callback_cls(webhook_url=config.swanlab_webhook_url, secret=config.swanlab_secret)
    swanlab.register_callbacks([callback])


def single_teacher_id(teacher_model: Optional[List[str]], *, algo: str) -> Optional[str]:
    """Resolve the one teacher id a single-teacher algorithm distils from, or ``None`` for no teacher.

    ``RLConfig.teacher_model`` is a list (shared with MOPD, which distils from K teachers); GKD/OPSD and
    GRPO's RLSD/SDAR take exactly one. ``None``/empty passes through so the caller applies its own
    no-teacher semantics (self-distillation, or its own error); more than one is rejected here rather than
    silently training on the first and ignoring the rest.
    """
    if not teacher_model:
        return None
    if len(teacher_model) != 1:
        raise ValueError(f'{algo} distils from a single teacher but RLConfig.teacher_model has '
                         f'{len(teacher_model)} entries ({list(teacher_model)}); pass exactly one, or use '
                         '--rl_type mopd to distil from multiple teachers.')
    return teacher_model[0]


def optimizer_batch_slices(num_samples: int, train_batch_size: int, ga: int):
    """Yield full GA update windows, dropping samples that cannot fill one GBS."""
    update_batch_size = train_batch_size * ga
    usable = (num_samples // update_batch_size) * update_batch_size
    for start in range(0, usable, update_batch_size):
        yield start, start + update_batch_size


def grouped_training_batches(dataloader: Any, ga: int):
    """Yield concatenated samples and their dataloader batch count, keeping the final shorter group."""
    group, count = [], 0
    for batch in dataloader:
        group.extend(batch)
        count += 1
        if count == ga:
            yield group, count
            group, count = [], 0
    if group:
        # trailing partial group
        yield group, count


def fit_supervised(
    state: StepState,
    dataloader: Any,
    train_config: "TrainConfig",
    *,
    start_epoch: int,
    update_step: Callable[[list], None],
) -> None:
    """Run the shared epoch/update cadence; the model owns every micro-batch schedule."""
    epochs = math.ceil(train_config.num_train_epochs) if state.max_steps <= 0 else 10**9
    for epoch in range(start_epoch, epochs):
        if state.reached_max() or state.should_stop():
            break
        state.epoch = epoch
        # Reshuffle each epoch (twinkle's EpochSampler derives each epoch's order from its
        # number, and only set_epoch advances it); resume replays it the same way.
        if hasattr(dataloader, "set_epoch"):
            dataloader.set_epoch(epoch)
        state.publish_state()
        state.handler.on_epoch_begin()
        updates = 0
        for group, batch_count in grouped_training_batches(dataloader, state.ga):
            state.micro_step += batch_count
            update_step(group)
            updates += 1
            if state.reached_max() or state.should_stop():
                break
        resumed_tail = epoch == start_epoch and state.global_step > 0
        if not updates and not resumed_tail and not state.reached_max() and not state.should_stop():
            raise ValueError(f"Epoch {epoch} yielded no training batches; no optimizer update is possible.")
        state.publish_state()
        state.handler.on_epoch_end()
        if state.history:
            state.tracker.log(state.history[-1], state.global_step, epoch_end=True)
        if state.should_stop():
            break


def save_final_checkpoint(model: Any, save: Callable[..., str], *, enabled: bool) -> Optional[str]:
    """Flush asynchronous writes before the optional synchronous final checkpoint."""
    if enabled:
        # The final save is synchronous, and its args.json must land after every earlier
        # checkpoint's background write finished. The remote handle has no background writer
        # (each client save blocks to the terminal state), so the wait is a no-op there.
        wait_async = getattr(model, "wait_async_saves", None)
        if callable(wait_async):
            wait_async()
        return save("checkpoint-final", is_final=True)
    return None

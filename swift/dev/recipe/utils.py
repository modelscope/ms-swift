# Copyright (c) ModelScope Contributors. All rights reserved.
# Shared recipe utilities: save / resume / step cadence (all families) + per-family assembly pieces.
from __future__ import annotations
import math
import os
import time
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

import json

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
        RewardConfig,
        RLConfig,
        RolloutConfig,
        TemplateConfig,
        TrainConfig,
        TunerConfig,
    )
    from swift.dev.config.rollout_config import WeightSyncStrategyName
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


def collect(result: Any) -> Any:
    """Resolve a maybe-lazy remote result to its value.

    A worker method declared ``lazy_collect=True`` hands the ray driver a ``LazyCollect`` future (collect
    on demand), but the SAME handle can resolve eagerly -- a CPU-payload worker (the reward/scoring actor)
    materialises at the call, so ``grpo_advantages``/``gae`` come back as the plain tuple, not a future.
    Recipes that consume the VALUE route through here so they are correct under both, mirroring the
    resolve-if-lazy seam in :func:`swift.dev.utils.scoring.response_token_scores`. Calling ``()`` blindly
    assumes the lazy form and raises ``'tuple' object is not callable`` on the eager one.
    """
    return result() if callable(result) and getattr(result, '_is_lazy_collect', False) else result


#: The console cadence when no LoggingConfig was supplied (legacy logging_steps=1: every step).
_FALLBACK_LOGGING_STEPS = 1


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
        hub_strategy=ckpt.hub_strategy if ckpt else 'every_save')


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
        CHECKPOINT_DATA_FIELDS, CHECKPOINT_FORCE_LOAD_FIELDS, CHECKPOINT_LOAD_FIELDS)
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

    save_training_checkpoint(
        model, name, output_dir=knobs.output_dir,
        consumed_train_samples=consumed_train_samples,
        component_states=component_states or {},
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
        requantize_source=requantize_source if (is_final or not knobs.async_save) else None)
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


def _seed_counters(state: dict, *, gradient_accumulation_steps: int, dataloader: Any,
                   ignore_data_skip: bool) -> Tuple[int, int, int]:
    """Seed the loop counters + dataloader position from a restored trainer_state.

    ``micro_step`` mirrors twinkle's ``cur_step`` so the grad-sync phase stays aligned,
    ``global_step`` is derived from it (single source), and the dataloader either loads its
    component state or skips to the consumed offset. Returns ``(global_step, micro_step,
    start_epoch)``; a ``dataloader=None`` (prompt-stream) run always seeds ``(n, n, 0)``.
    """

    cur_step = int(state['cur_step'])
    micro_step = cur_step
    global_step = num_optimizer_steps(cur_step, gradient_accumulation_steps)
    if 'component_states' in state:
        component_states = state['component_states']
        if not isinstance(component_states, dict):
            raise TypeError('checkpoint component_states must be a dict.')
        if not ignore_data_skip and dataloader is not None and 'dataloader' in component_states:
            load_state = getattr(dataloader, 'load_state', None)
            if load_state is None:
                raise TypeError('the dataloader does not implement load_state().')
            load_state(component_states['dataloader'])
    elif not ignore_data_skip and dataloader is not None:
        # Compatibility with checkpoints written before component_states existed.
        skip = getattr(dataloader, 'skip_consumed_samples', None)
        if skip is not None:
            skip(int(state.get('consumed_train_samples', 0)))
    if ignore_data_skip or dataloader is None:
        return global_step, micro_step, 0
    get_state = getattr(dataloader, 'get_state', None)
    start_epoch = int((get_state() if get_state is not None else {}).get('epoch', 0))
    return global_step, micro_step, start_epoch


def resume_training(model: Any, *, checkpoint_config: Optional['CheckpointConfig'],
                    distributed_config: 'DistributedConfig',
                    megatron_config: Optional['MegatronConfig'],
                    tuner_config: Optional['TunerConfig'],
                    gradient_accumulation_steps: int, dataloader: Any = None,
                    ignore_data_skip: bool = False) -> Tuple[int, int, int]:
    """Restore the trainable model, then seed the loop counters and the data position.

    Returns ``(global_step, micro_step, start_epoch)``. The offline family passes its dataloader
    (its component state reloads, or it skips to the consumed offset); a prompt-stream run passes
    ``None`` and always gets ``start_epoch`` 0.
    """
    state = _restore_training(model, checkpoint_config, distributed_config, megatron_config, tuner_config)
    if not state:
        return 0, 0, 0
    return _seed_counters(
        state, gradient_accumulation_steps=gradient_accumulation_steps, dataloader=dataloader,
        ignore_data_skip=ignore_data_skip)


def _log_line(record: dict, loss: float) -> str:
    """The default console step line: loss plus grad_norm, the MTP channel and any loss_* channels."""
    gn = record.get('grad_norm')
    gn_str = f'  grad_norm={gn:.4f}' if gn is not None else ''
    mtp = record.get('mtp_loss')
    mtp_str = f'  mtp_loss={mtp:.4f}' if mtp is not None else ''
    channel_str = ''.join(f'  {key}={value:.4f}' for key, value in record.items() if key.startswith('loss_'))
    # The legacy rollout_correction diagnostics, when the loop computed them: the signed KL is
    # the acceptance number for a freshly synced sampler (target < 1e-2).
    correction_str = ''.join(
        f'  {label}={record[key]:.3g}'
        for label, key in (('rollout_kl', 'rollout_correction/kl'), ('rollout_k3_kl', 'rollout_correction/k3_kl'),
                           ('rollout_logp_absdiff', 'rollout_correction/log_ppl_abs_diff'))
        if isinstance(record.get(key), (int, float)))
    return f"step {record.get('step')}  loss={loss:.4f}{gn_str}{mtp_str}{channel_str}{correction_str}"


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
                 log_formatter: Optional[Callable[[Dict[str, Any], float], str]] = None):
        # The tracker initializes its reporters as a construction side effect (validation ran above).

        from swift.dev.callbacks import build_callbacks
        from swift.dev.callbacks.handler import CallbackHandler

        self.model = model
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
        self.tracker = RunTracker(logging_config, output_dir)
        self.handler = CallbackHandler(build_callbacks(train_config), output_dir=output_dir)

    def is_grad_sync_boundary(self) -> bool:
        """Whether the current ``micro_step`` closes an optimizer window."""

        return is_grad_sync_boundary(self.micro_step, self.ga)

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

    def record(self) -> dict:
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
        if should_log:
            # Fold the window's merged per-rank perf summary in only on a logging step: the
            # perf_dump fan-out is blocking, so keeping it off every step leaves it off the hot path.
            record.update(Perf.collect_step_summary())
        record = self.tracker.log(record, self.global_step)
        self.history.append(record)
        self.publish_state(loss=loss, metrics=record)
        if should_log:
            # A recipe's own head signal (PPO's value_loss / reward / explained_variance line).
            line = (self._log_formatter(record, loss) if self._log_formatter is not None
                    else _log_line(record, loss))
            logger.info(line)
            self.handler.on_log(logs=record)
        return record

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


def rollout_step_budget(*, num_prompts: int, num_generations: int, train_batch_size: int,
                        gradient_accumulation_steps: int, num_train_epochs: float,
                        generation_batch_size: Optional[int] = None, num_iterations: int = 1) -> int:
    """Optimizer steps for ``num_train_epochs`` passes over the prompt set (the scheduler horizon).

    Each generation batch rolls out ``batch_size * num_generations`` completions, splits into full
    ``train_batch_size`` mini-batches (an undersized tail is dropped), and replays them
    ``num_iterations`` times. Returns 0 when a generation batch cannot fill even one mini-batch.
    """

    batch_size = min(generation_batch_size or num_prompts, num_prompts)
    batches_per_epoch = math.ceil(num_prompts / batch_size)
    total_batches = max(1, math.ceil(num_train_epochs * batches_per_epoch))
    mini_batches = (batch_size * num_generations) // max(1, train_batch_size)
    micro_per_batch = mini_batches * max(1, num_iterations)
    return num_optimizer_steps(micro_per_batch * total_batches, gradient_accumulation_steps)


def resolve_rollout_max_steps(explicit_max_steps: int, budget: int, *, recipe: str) -> int:
    """Explicit ``--max_steps`` wins; a non-positive budget fails loudly instead of a 0-step run."""
    if explicit_max_steps and explicit_max_steps > 0:
        return explicit_max_steps
    if budget <= 0:
        raise ValueError(
            f'{recipe}: derived {budget} training steps -- a generation batch cannot fill one '
            'train_batch_size (per_device_train_batch_size * dp_size) mini-batch. Enlarge the prompt '
            'dataset or num_generations, lower per_device_train_batch_size / the DP world size, or set '
            '--max_steps explicitly.')
    return budget


def plan_device_groups(nproc_per_node: int, sampler_mode: Optional[str], sampler_world_size: int,
                       auxiliary: Sequence[Tuple[str, int]]) -> Tuple[List[Tuple[str, List[int]]], bool]:
    """Plan the twinkle DeviceGroups for an online-RL run (pure: no twinkle import).

    Colocate overlays trainer + sampler in ONE group over the trainer's GPUs (independent rank
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
    from twinkle_agentic.async_rl import REF_ONLY_REMOTE_GROUP, estimate_plan, resource_device_groups

    from swift.dev.builders import build_dp_mesh, build_template, load_model_processor
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
        groups = resource_device_groups(resource_plan, adaptive=rollout_config.resource_mode == 'adaptive')
        colocate = False
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
    twinkle.initialize(
        mode='ray',
        nproc_per_node=total,
        extra_torch_kwargs={'seed': train_config.seed, 'full_determinism': train_config.full_determinism,
                            'tf32': train_config.tf32},
        global_device_mesh=build_dp_mesh(distributed_config, template_config.sequence_parallel_size),
        name=distributed_config.ray_exp_name,
        groups=[*device_groups, DeviceGroup(name=REF_ONLY_REMOTE_GROUP, ranks=1, device_type='CPU')])
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
        # Every RL weight sync lands through twinkle's vLLM worker extension. Install the
        # quant-aware subclass so an FP8 / MXFP4 engine can absorb the trainer's BF16 stream;
        # it is a pass-through when the engine is not quantized. An explicit engine arg wins.
        from swift.dev.weight_sync import QUANT_SYNC_WORKER_EXTENSION
        sampler_engine_args.setdefault('worker_extension_cls', QUANT_SYNC_WORKER_EXTENSION)
    if colocate:
        sampler_engine_args['enable_sleep_mode' if backend == 'vllm' else 'enable_memory_saver'] = True
    if (backend == 'vllm' and tuner_config is not None and tuner_config.tuner == 'lora'
            and rollout_config.weight_sync_strategy == 'adapter_snapshot'):
        sampler_engine_args.setdefault('enable_lora', True)
        sampler_engine_args.setdefault('max_lora_rank', tuner_config.lora_rank)
        sampler_engine_args.setdefault('max_loras', 2)
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
        device_mesh=(None if colocate else DeviceMesh.from_sizes(dp_size=replicas)), template=template,
        remote_group=(MODEL_GROUP if colocate else SAMPLER_GROUP))


def resolve_reward_scorers(channel: str, *, reward_config: Any, model_config: 'ModelConfig',
                           template_config: 'TemplateConfig', distributed_config: 'DistributedConfig',
                           rollout_config: 'RolloutConfig', backend: str, sampler: Any,
                           quantize_config: Optional['QuantizeConfig'] = None) -> List[Any]:
    """One reward channel's ordered scorer list: the rules resolve locally, each MODEL via the builder.

    ``channel`` is 'orm' or 'prm': the merged ``--<channel>`` list resolves through
    ``resolve_channel``; each reward MODEL goes through ``build_reward_model`` onto its own group
    (``reward_<i>`` / ``prm_<i>`` -- the SAME names the recipe's group plan allocated). A PRM
    model must resolve SCALAR (online RL places only a seq_cls/reranker frozen PRM today).
    """
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
    scorers = list(get_reward_funcs(plan.rules, reward_config)[0])
    prefix = 'reward' if channel == 'orm' else channel
    for index, spec in enumerate(plan.specs):
        if channel == 'prm' and spec.kind != 'scalar':
            raise ValueError(f'--prm model {spec.model_id!r} resolves to a {spec.kind!r} reward, but online RL '
                             'places only a SCALAR (seq_cls/reranker) frozen PRM today.')
        scorers.append(build_reward_model(
            spec, model_config=model_config, template_config=template_config,
            distributed_config=distributed_config, rollout_config=rollout_config, backend=backend,
            sampler=sampler, remote_group=f'{prefix}_{index}',
            judge_template=reward_config.judge_template, quantize_config=quantize_config))
    return scorers


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

        sampler = TransformersSampler(model=model)
        sampler.set_template(template)
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
    manager = CheckpointEngineManager(model=model, sampler=sampler, mode='colocate')
    # The eval hand-over is ColocateHandover's (the RL rollout's own sequence, so it cannot drift).

    handover = ColocateHandover(model, sampler, manager, colocate=True)
    return GenerativeEval(
        sampler=sampler, template=template, datasets=eval_datasets, model_id=eval_model_id,
        task_config=task_config, enter=handover.enter, exit=handover.exit)


# The absorbed training infrastructure: the loop cadence helpers, the tracker, the colocate
# hand-over, the streaming-config guard and the single-teacher resolver (was recipe/common).


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


def backend_owns_gradient_accumulation(model) -> bool:
    """True when the backend's ``forward_backward`` internalizes gradient accumulation.

    fit() forks on exactly one axis: who schedules the GA micro-steps. Most backends (transformers) leave
    it to the loop -- one micro-batch per ``forward_backward``, the optimizer stepped every ga-th call. A
    backend that owns its own micro-batch schedule needs the loop to instead group the ga batches and call
    ``forward_backward`` once per optimizer step; in dev that is Megatron, whose 1F1B pipeline derives GA
    from the length of the micro-batch LIST it is handed. A future such backend (e.g. torchtitan/veomni
    running pipeline parallelism) is recognized HERE, so the dispatch stays a two-way fork on this
    capability rather than growing a branch per backend name.
    """
    try:
        from twinkle.model.megatron import MegatronModel
    except Exception:
        return False
    return isinstance(model, MegatronModel)


def is_grad_sync_boundary(micro_step: int, gradient_accumulation_steps: int) -> bool:
    """Whether ``micro_step`` is an optimizer-update boundary, computed driver-side.

    Replicates twinkle's ``OptimizerGroup.do_grad_sync`` (a pure function of ``cur_step`` and the
    accumulation size: boundary when ga==1, else one micro-step late at cur_step = ga+1, 2ga+1, ...) using
    the loop's own ``micro_step`` as ``cur_step``, which the loops advance once per ``forward_backward`` and
    ``resume()`` keeps aligned. It is a module-level helper rather than a read of ``model.optimizer_group``
    because that group lives on the WORKERS: under Ray the driver-side model is a proxy with no such
    attribute, so both the in-process transformers loop and the Ray-remote loops (SFT/GRPO/RFT) must derive
    the boundary here instead of reaching into the proxy.
    """
    ga = gradient_accumulation_steps
    return ga == 1 or ((micro_step - 1) % ga == 0 and micro_step > 1)


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
                  global_step: int) -> Optional[dict]:
    """Val-loss eval: the normalized per-token loss over eval_dataloader (None if there is none).

    Mirrors the train metric path on the eval side: each val batch runs ``forward_only`` (no grad)
    then ``calculate_loss`` (fills this batch's loss); a final ``calculate_metric(False)`` aggregates
    and resets. Megatron already produced the loss inside forward_only -- its ``calculate_loss``
    raises NotImplementedError, treated as already populated.
    """
    if eval_dataloader is None:
        return None
    if manual_gc and manual_gc_eval:
        import gc
        gc.collect()
    for index, batch in enumerate(eval_dataloader):
        if eval_iters > 0 and index >= eval_iters:
            break
        model.forward_only(**micro_step_kwargs(batch, owns_ga=True))
        try:
            model.calculate_loss()
        except NotImplementedError:
            pass
    metrics = model.calculate_metric(is_training=False)
    result = {'step': global_step}
    if metrics.get('loss') is not None:
        result['eval_loss'] = float(metrics['loss'])
    for key, value in metrics.items():
        if key.startswith('loss_'):
            result[f'eval_{key}'] = float(value)
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
    """
    ga = max(1, gradient_accumulation_steps)
    if ga == 1:
        return num_micro_batches
    return max(0, (num_micro_batches - 1) // ga)


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


class RunTracker:
    """Own tracker setup, metric writes, and teardown for one training loop."""

    def __init__(self, config: Optional['LoggingConfig'], output_dir: str):
        self.config = config
        self.output_dir = output_dir
        self._writers: Dict[str, Any] = {}
        self._loss_total = 0.0
        self._loss_count = 0
        self._logged_prompts = set()
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

    def log(self, metrics: Dict[str, Any], step: int, *, epoch_end: bool = False) -> Dict[str, Any]:
        """Filter and write one metric record; returns the record actually emitted."""
        filtered = self._filter_metrics(metrics)
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

    def log_prompts(self, prompts: Iterable[Any], samples: Iterable[Any], step: int, num_generations: int = 1) -> None:
        """Log each distinct prompt/completion pair once when W&B prompt logging is enabled."""
        if not self.config or not self.config.wandb_log_unique_prompts:
            return
        writer = self._writers.get('wandb')
        if writer is None:
            return
        prompts = list(prompts)
        rows = []
        for index, sample in enumerate(samples):
            if not prompts:
                break
            prompt = prompts[min(index // max(1, num_generations), len(prompts) - 1)]
            prompt_key = json.dumps(prompt, ensure_ascii=False, sort_keys=True, default=str)
            if prompt_key in self._logged_prompts:
                continue
            self._logged_prompts.add(prompt_key)
            rows.append([prompt_key, getattr(sample, 'decoded', str(sample))])
        if rows:
            writer.log({'completions': writer.Table(columns=['prompt', 'completion'], data=rows)}, step=step)

    def close(self) -> None:
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


class ColocateHandover:
    """Wake/sync/offload/reload around a generate phase, for a trainer and sampler sharing GPUs.

    ``enter`` hands the device to the sampler and pushes the trained policy into it; ``exit`` hands the
    device back to the trainer. A disaggregated (``colocate=False``) sampler lives on its own GPUs, so
    there is no device to hand over: ``enter`` reduces to a plain weight sync and ``exit`` is a no-op.

    ``model`` / ``sampler`` / ``manager`` are the twinkle objects the caller already built (the manager
    wires the CUDA-IPC or NCCL path per its ``mode``); ``merge_and_sync`` is forwarded to
    ``manager.sync_weights`` (merged base weights every step, correct for both full and LoRA training).
    """

    def __init__(self,
                 model: Any,
                 sampler: Any,
                 manager: Any,
                 *,
                 colocate: bool,
                 merge_and_sync: bool = True,
                 sleep_level: int = 0):
        self.model = model
        self.sampler = sampler
        self.manager = manager
        self.colocate = colocate
        self.merge_and_sync = merge_and_sync
        self.sleep_level = sleep_level

    def enter(self) -> None:
        """Give the sampler the device and the trained weights, ready to generate.

        Colocate order: wake the sampler's WEIGHTS so the sync has somewhere to write, sync the trained
        policy into them, step the trainer aside (offload), then wake ONLY the KV cache so the sampler can
        generate. The two wakes are tag-disjoint on purpose -- vLLM's ``wake_up`` aborts the whole call on
        the first tag that is not still sleeping, so after waking ``weights`` the second wake may ask only
        for ``kv_cache``: passing ``wake_up()`` (both tags) would hit the already-awake ``weights``, return
        early, and leave the KV cache discarded, so the next generate builds attention metadata on freed
        memory and dies with a CUDA illegal access. (A freshly built sampler is not sleeping yet, so on the
        first round both wakes no-op -- which is why the bug only bites from round two on.)
        """
        if self.colocate:
            self.sampler.wake_up(tags=['weights'])
        self.manager.sync_weights(merge_and_sync=self.merge_and_sync)
        if self.colocate:
            self.model.offload_to_cpu()
            self.sampler.wake_up(tags=['kv_cache'])  # weights already awake; ready to generate

    def exit(self) -> None:
        """Take the device back for the trainer: sleep the sampler, then reload the trainer's weights.

        ``sleep_level`` selects how much the sampler releases: level 1 offloads only the KV cache (weights
        stay resident, so the next ``enter`` re-syncs in place), level 2 also drops the weights. The
        default ``0`` maps to twinkle's own default depth (level 1), keeping the previous behaviour
        byte-for-byte; 1/2 are passed through unchanged.
        """
        if self.colocate:
            self.sampler.sleep(level=self.sleep_level or 1)
            self.model.reload_to_gpu()


class AdaptiveColocateHandover:
    """以 actor 子集暂停生成、同步权重，并在训练与 shared 引擎之间交接 GPU。"""

    serializes = False
    adaptive_shared = True

    def __init__(self,
                 model: Any,
                 sampler: Any,
                 dedicated_manager: Any,
                 *,
                 shared_manager: Any = None,
                 additional_models: Iterable[Any] = (),
                 sleep_level: int = 1):
        if sleep_level not in (0, 1):
            raise ValueError('adaptive shared sampler supports only sleep_level 0 or 1.')
        self.model = model
        self.models = (model, *tuple(additional_models))
        self.sampler = sampler
        self.dedicated_manager = dedicated_manager
        self.shared_manager = shared_manager
        self.controller = None
        self.sleep_level = sleep_level or 1
        self.in_generation = False
        self.enabled = False
        self.borrow_time = 0.0
        self.reclaim_time = 0.0

    @property
    def total_handover_time(self) -> float:
        return self.borrow_time + self.reclaim_time

    @staticmethod
    def _wait(result):
        if callable(result):
            return result()
        return result

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled) and self.shared_manager is not None

    def initialize_shared_workers(self, group, ranks, *, engine_args):
        """构造 shared 引擎前释放训练显存，休眠引擎后恢复所有训练模型。"""
        worker_ids = ()
        initialization_error = None
        can_reload = True
        self.in_generation = True
        try:
            for model in self.models:
                self._wait(model.offload_to_cpu())
            worker_ids = self.sampler.add_workers(group, ranks, shared=True, engine_args=engine_args)
            self.sampler.pause_workers(worker_ids)
            self.sampler.pause_generation(_worker_ids=worker_ids)
            self._wait(self.sampler.sleep(level=self.sleep_level, _worker_ids=worker_ids))
        except BaseException as exc:
            initialization_error = exc
            try:
                if worker_ids and not self.sampler.remove_workers(worker_ids):
                    raise RuntimeError('Shared sampler initialization still has unfinished worker calls.')
            except Exception:
                logger.warning('初始化失败后未能完整关闭 shared sampler actors。', exc_info=True)
            can_reload = not set(worker_ids) & set(self.sampler.worker_ids(shared=True))
            raise
        finally:
            if can_reload:
                try:
                    self._reload_models()
                    self.in_generation = False
                except BaseException:
                    if initialization_error is None:
                        raise
                    logger.warning('初始化失败后未能完整恢复训练模型。', exc_info=True)
        return worker_ids

    def _borrow_shared(self, *, publish: bool):
        shared_ids = self.sampler.worker_ids(shared=True)
        outstanding = any(self.sampler.worker_inflight(shared_ids))
        if not shared_ids or (not self.enabled and not outstanding) or self.in_generation:
            return
        started = time.perf_counter()
        # 即使停止 shared 新准入，也必须让已经提交的请求继续生成并正常接收新 policy。
        self.in_generation = True
        try:
            self._wait(self.sampler.wake_up(tags=['weights'], _worker_ids=shared_ids))
            if publish:
                trainer_ranks = [ranks[0] for ranks in self.sampler.worker_ranks(shared_ids)]
                self.shared_manager.set_receiver_workers(shared_ids, colocate_trainer_ranks=trainer_ranks)
                self.shared_manager.sync_weights(merge_and_sync=True)
            for model in self.models:
                self._wait(model.offload_to_cpu())
            self._wait(self.sampler.wake_up(tags=['kv_cache'], _worker_ids=shared_ids))
            self.sampler.resume_generation(_worker_ids=shared_ids)
            if self.enabled:
                self.sampler.activate_workers(shared_ids)
        except BaseException:
            try:
                self.reclaim()
            except Exception:
                logger.warning('shared 借卡失败后未能完整恢复训练模型。', exc_info=True)
            raise
        self.borrow_time += time.perf_counter() - started

    def borrow(self, *, publish: bool) -> None:
        if not publish:
            self._borrow_shared(publish=False)
            return
        if self.in_generation:
            self.reclaim()
        suspended = self.sampler.worker_ids()
        self.sampler.suspend_admission(suspended)
        dedicated_ids = self.sampler.worker_ids(shared=False, states=('ACTIVE', 'DRAINING'))
        try:
            self.sampler.pause_generation(_worker_ids=dedicated_ids)
            if self.controller is not None:
                dedicated_ids = self.controller.prepare_publish()
            self.dedicated_manager.set_receiver_workers(dedicated_ids)
            self.dedicated_manager.sync_weights(merge_and_sync=True)
            self._borrow_shared(publish=True)
            if self.controller is not None:
                self.controller.finish_publish()
            self.sampler.resume_generation(_worker_ids=dedicated_ids)
            self.sampler.resume_admission(suspended + dedicated_ids)
        except BaseException as error:
            try:
                if self.controller is not None:
                    self.controller.fail_publish(error)
            except Exception:
                logger.warning('发布失败后未能完整清理新增 sampler actors。', exc_info=True)
            try:
                self.reclaim()
            except Exception:
                logger.warning('发布失败后未能完整恢复训练模型。', exc_info=True)
            raise

    def _reload_models(self):
        error = None
        for model in reversed(self.models):
            try:
                self._wait(model.reload_to_gpu())
            except BaseException as exc:
                error = error or exc
        if error is not None:
            raise error

    def reclaim(self) -> None:
        """停止 shared 新准入，等生成暂停后释放显存，最后恢复所有训练模型。"""
        if not self.in_generation:
            return
        started = time.perf_counter()
        shared_ids = self.sampler.worker_ids(shared=True)
        try:
            self.sampler.pause_workers(shared_ids)
            self.sampler.pause_generation(_worker_ids=shared_ids)
            self._wait(self.sampler.sleep(level=self.sleep_level, _worker_ids=shared_ids))
            # pause/sleep 失败时保留交接状态；终止 actors 后再恢复训练显存。
            self._reload_models()
            self.in_generation = False
        finally:
            self.reclaim_time += time.perf_counter() - started

    def close_channels(self):
        try:
            self.dedicated_manager.close()
        finally:
            if self.shared_manager is not None:
                self.shared_manager.close()

    def enter(self) -> None:
        self.borrow(publish=True)

    def exit(self) -> None:
        self.reclaim()

    def drained(self, inflight: int) -> bool:
        return True


def check_streaming_config(*,
                           adapter_name: Optional[str],
                           max_staleness: int,
                           weight_sync_strategy: WeightSyncStrategyName,
                           allow_partial_rollout: bool,
                           parameter_sync_step: int,
                           serialize_generation: bool) -> None:
    """Reject a streaming configuration that cannot be driven soundly (fail-loudly).

    Mirrors ``config.validate`` at construction time (defense in depth: a driver built directly, not
    through a ``run_*`` entry, still refuses an unsound stream). ``max_staleness >= 0`` because this
    builder serves the synchronous regime too: ``0`` is ``async_mode='none'`` (admit one window, drain,
    train, publish -- no overlap), ``1`` the one-step overlap, ``> 1`` the deep buffer. The publication
    mechanism then imposes its own requirement:

    * ``adapter_snapshot`` pins each version as a LoRA adapter path, so it needs an ``adapter_name``
      (a full-parameter policy has no adapter to pin) and must NOT set ``allow_partial_rollout``
      (publishing writes a new path and never overwrites a weight copy an in-flight generation decodes,
      so there is nothing to interrupt and resume -- the flag would be inert).
    * ``in_place`` overwrites the sampler's single live weight copy. Under an OVERLAPPING regime
      (``max_staleness >= 1``) a stream always has trajectories in flight at a publish, so it REQUIRES
      ``allow_partial_rollout``: without the abort-and-resume, the rest of an in-flight generation would
      decode under half-updated weights whose logprobs match no consistent policy version. Under the
      SYNCHRONOUS regime (``max_staleness == 0``) the drain barrier empties the in-flight set before
      every publish, so nothing decodes across the overwrite and the flag is unnecessary.
    * ``parameter_sync_step`` is the publish cadence in consume-reported steps; a non-positive cadence
      would publish never (or divide by zero in the window sizing).
    * ``serialize_generation`` (a colocated sampler sharing one device with the trainer) is sound only at
      ``max_staleness == 0`` (an exclusive device cannot overlap generation with training, so it cannot
      hold a lookahead window) and only under ``in_place`` (the per-cycle device hand-back rides in_place's
      publish ``sync_fn``; ``adapter_snapshot`` publishes with no weight sync, so nothing would re-hand the
      device to the sampler after the first consume's exit).
    """
    if max_staleness < 0:
        raise ValueError(f'max_staleness must be >= 0 (0 = the synchronous regime, which this builder '
                         f'also serves; 1 = one-step overlap; >1 = deep buffer); got {max_staleness}.')
    if serialize_generation and max_staleness != 0:
        raise ValueError(
            f'serialize_generation (a colocated sampler time-sharing one device with the trainer) needs '
            f'max_staleness=0: an exclusive device cannot overlap generation with training, so it cannot '
            f'hold a lookahead window; got max_staleness={max_staleness}. A colocated run is synchronous.')
    if serialize_generation and weight_sync_strategy != 'in_place':
        raise ValueError(
            f'serialize_generation (a colocated sampler time-sharing one device with the trainer) requires '
            f"weight_sync_strategy='in_place': the per-cycle device hand-back (wake+sync the sampler before "
            f'each generation window) rides in_place\'s publish ``sync_fn``, whereas adapter_snapshot '
            f'publishes by writing a new adapter path with no weight sync, so after the first consume\'s '
            f'exit() nothing would hand the device back to the sampler and the next window would submit to a '
            f'sleeping engine. dev derives in_place for a colocated run; got {weight_sync_strategy!r}.')
    if parameter_sync_step < 1:
        raise ValueError(f'parameter_sync_step must be >= 1 (a weight publication every K steps); '
                         f'got {parameter_sync_step}.')
    if weight_sync_strategy == 'adapter_snapshot':
        if not adapter_name:
            raise ValueError(
                "weight_sync_strategy='adapter_snapshot' pins each trained version as a LoRA adapter "
                'snapshot, so it needs a non-empty adapter_name; a full-parameter policy has no adapter '
                'to pin. Configure an adapter-based tuner (e.g. LoRA), or use '
                "weight_sync_strategy='in_place' with --allow_partial_rollout for a full-parameter stream.")
        if allow_partial_rollout:
            raise ValueError(
                '--allow_partial_rollout is inert under weight_sync_strategy=adapter_snapshot: each version '
                'is pinned as its own adapter path, so publishing never overwrites a weight copy an '
                'in-flight generation is decoding and there is nothing to interrupt and resume. Drop it, or '
                "use weight_sync_strategy='in_place' (which overwrites one live copy and so needs it).")
    elif weight_sync_strategy == 'in_place':
        if allow_partial_rollout is False and max_staleness >= 1:
            raise ValueError(
                "weight_sync_strategy='in_place' overwrites the sampler's single live weight copy while the "
                'stream still has trajectories in flight, so an overlapping regime (max_staleness >= 1) needs '
                '--allow_partial_rollout: each publish aborts every in-flight generation and resumes it from '
                'its own tokens on the fresh weights (twinkle PartialRolloutMixin + InPlaceWeightSync '
                'abort-on-publish). Without it the rest of an in-flight generation would decode under '
                'half-updated weights whose logprobs match no consistent policy, which importance sampling '
                'cannot correct. Set --allow_partial_rollout, or use weight_sync_strategy=adapter_snapshot '
                '(per-version LoRA pinning, no interrupt). The synchronous regime (max_staleness=0) drains '
                'before it publishes, so it needs no partial rollout.')
    else:
        raise ValueError(f'unknown weight_sync_strategy: {weight_sync_strategy!r}')


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

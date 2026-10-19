# Copyright (c) ModelScope Contributors. All rights reserved.
# Shared recipe utilities: save / resume / step cadence (all families) + per-family assembly pieces.
from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

from swift.dev.utils.logger import get_logger  # TODO(dev2): bridge until the utils batch

if TYPE_CHECKING:
    from swift.dev2.config import (
        CheckpointConfig, DatasetConfig, DistributedConfig, LoggingConfig, MegatronConfig, ModelConfig,
        MoEConfig, QuantizeConfig, RLConfig, RewardConfig, RolloutConfig, TemplateConfig, TrainConfig,
        TunerConfig)

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
    from swift.dev2.builders import is_megatron_backend
    from swift.dev2.builders.adapter import is_base_modifying_init

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
            commit_message=f'Upload {os.path.basename(checkpoint_dir.rstrip("/"))} from swift.dev2 training')
        if async_upload:
            HubOperation.async_push_to_hub(**push_kwargs)
        else:
            HubOperation.push_to_hub(**push_kwargs)

    return push


def write_ckpt_args_json(ckpt_dir: str, processor: Any, model_config: 'ModelConfig',
                         template_config: 'TemplateConfig', tuner_config: Optional['TunerConfig'] = None,
                         quantize_config: Optional['QuantizeConfig'] = None,
                         *, task_type: Optional[str] = None) -> None:
    """Write the self-describing args.json ``swift infer`` reads back from the checkpoint.

    Legacy dumped the full argument dict; infer's read side consumes only force_load_keys (always
    applied) and load_keys (applied when the reader's value is empty) -- that subset is written.
    """
    import json
    import torch.distributed as dist

    # Master-only: the checkpoint dir exists on the master rank alone (twinkle's save_pretrained
    # guards on Platform.is_master()), so an unguarded open() elsewhere raised under multi-GPU DDP.
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return
    os.makedirs(ckpt_dir, exist_ok=True)

    from swift.dev.utils.version import __version__ as swift_version
    model_meta = getattr(processor, 'model_meta', None)
    args = {
        # swift_version gates model_type loading in BaseArguments.load_args_from_ckpt
        # (model_type is only honored when swift_version >= 4.0.0.dev); must be present.
        'swift_version': swift_version,
        # force_load_keys: infer applies these regardless of its current value.
        'task_type': model_config.task_type or task_type,
        'tuner_type': tuner_config.tuner if tuner_config is not None else 'full',
        'quant_method': quantize_config.quant_method if quantize_config is not None else None,
        'quant_bits': quantize_config.quant_bits if quantize_config is not None else None,
        'bnb_4bit_compute_dtype': (
            quantize_config.bnb_4bit_compute_dtype if quantize_config is not None else None),
        'bnb_4bit_quant_type': quantize_config.bnb_4bit_quant_type if quantize_config is not None else None,
        'bnb_4bit_use_double_quant': (
            quantize_config.bnb_4bit_use_double_quant if quantize_config is not None else None),
        'bnb_4bit_quant_storage': (
            quantize_config.bnb_4bit_quant_storage if quantize_config is not None else None),
        # load_keys: infer applies these only when its own value is None/empty. Record the portable
        # id/path the user passed (legacy saves the id; infer re-resolves it on any machine).
        'model': getattr(model_config, '_model_id_or_path', None) or model_config.model,
        'model_type': getattr(model_meta, 'model_type', None),
        'model_revision': model_config.model_revision,
        'torch_dtype': model_config.torch_dtype,
        'attn_impl': model_config.attn_impl,
        'template': template_config.template or getattr(model_meta, 'template', None),
        'system': template_config.system,
        'truncation_strategy': template_config.truncation_strategy,
        'max_length': template_config.max_length,
    }
    args = {key: value for key, value in args.items() if value is not None}
    with open(os.path.join(ckpt_dir, 'args.json'), 'w', encoding='utf-8') as f:
        json.dump(args, f, ensure_ascii=False, indent=2)


def save_training_step(model: Any, handler: Any, name: str, knobs: SaveKnobs, *,
                       consumed_train_samples: int = 0,
                       component_states: Optional[Dict[str, dict]] = None,
                       is_final: bool = False) -> str:
    """One training checkpoint: the twinkle-native write, the converted adapter, the hub push.

    ``consumed_train_samples`` / ``component_states`` carry the data position: the offline family
    reads them off its dataloader, an online run passes its step count and no component state.
    Returns the checkpoint dir (twinkle's save result is a deferred handle under ray, so the dir is
    the only portable return).
    """
    # TODO(dev2): bridge until the recipe/common batch migrates the checkpoint writer
    from swift.dev.recipe.common.loop_utils import save_training_checkpoint

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
        async_save=knobs.async_save and not is_final)
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
    from swift.dev2.builders import is_megatron_backend

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
    # TODO(dev2): bridge until the recipe/common batch migrates num_optimizer_steps
    from swift.dev.recipe.common.loop_utils import num_optimizer_steps

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
    return f"step {record.get('step')}  loss={loss:.4f}{gn_str}{mtp_str}{channel_str}"


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
        # TODO(dev2): bridge until the recipe/common batch migrates RunTracker
        from swift.dev.recipe.common.tracking import RunTracker

        from swift.dev2.callbacks import build_callbacks
        from swift.dev2.callbacks.handler import CallbackHandler

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
        # TODO(dev2): bridge until the recipe/common batch migrates the cadence helpers
        from swift.dev.recipe.common.loop_utils import is_grad_sync_boundary

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
        # TODO(dev2): bridge until the recipe/common batch migrates the cadence helpers
        from swift.dev.recipe.common.tracking import coerce_metric_scalars
        from swift.dev.recipe.common.loop_utils import collect_empty_cache, collect_manual_gc

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
    # TODO(dev2): bridge until the recipe/common batch migrates num_optimizer_steps
    from swift.dev.recipe.common.loop_utils import num_optimizer_steps

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


def encode_privileged_feature(template: Any, prompt_messages: List[dict], teacher_prompt: Any,
                              response_ids: List[int], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Encode one single-turn privileged teacher view (RLSD / OPSD share the exact contract).

    Replaces the FIRST user turn's content with the privileged ``teacher_prompt``, appends the
    assistant turn carrying the student's response token ids verbatim, then delegates the
    encode-and-align-check tail to ``encode_privileged_view`` -- teacher and student must score
    exactly the same response tokens.
    """
    # TODO(dev2): bridge until the recipe/common batch migrates encode_privileged_view
    from swift.dev.recipe.common.scoring import encode_privileged_view

    messages = [dict(message) for message in prompt_messages]
    if messages and messages[-1].get('role') == 'assistant':
        messages = messages[:-1]
    for message in messages:
        if message.get('role') == 'user':
            message['content'] = teacher_prompt
            break
    else:
        raise ValueError('teacher_prompt requires a user turn in the prompt to carry the privileged context.')
    messages.append({'role': 'assistant', 'content': list(response_ids)})
    return encode_privileged_view(template, messages, response_ids, extra)


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
    from twinkle import DeviceGroup, DeviceMesh
    from twinkle.perf import Perf

    from swift.dev2.config import process_configs, validate_configs
    from swift.dev2.builders import build_dp_mesh, build_template, load_model_processor
    from twinkle_agentic.async_rl import REF_ONLY_REMOTE_GROUP, estimate_plan, resource_device_groups

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
        task_type=task_type)

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
            DeviceGroup(name=group_name, ranks=ranks, device_type='GPU', gpus_per_worker=width, shared=shared))
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
                     colocate: bool = False) -> Any:
    """The weight-syncable rollout sampler: shaped engine args, then the planned or static build.

    The planned layout (estimate/adaptive) sizes the replica pool and stamps the elastic instance id;
    the static shape reads the rollout_config replica knobs (estimate's apply_plan already rewrote
    them, so the two agree). Colocate pins the sleep/save knob the borrow-and-return loop needs.
    """
    from twinkle import DeviceMesh

    from swift.dev2.builders import build_engine_args, build_sampler

    backend = rollout_config.rollout_sampler
    sampler_engine_args = build_engine_args(backend, None, rollout_config)
    if colocate:
        sampler_engine_args['enable_sleep_mode' if backend == 'vllm' else 'enable_memory_saver'] = True
    if (backend == 'vllm' and tuner_config is not None and tuner_config.tuner == 'lora'
            and rollout_config.weight_sync_strategy == 'adapter_snapshot'):
        sampler_engine_args.setdefault('enable_lora', True)
        sampler_engine_args.setdefault('max_lora_rank', tuner_config.lora_rank)
        sampler_engine_args.setdefault('max_loras', 2)
    if rollout_config.enable_sampling_replay:
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
    from swift.dev2.builders import build_reward_model
    from swift.dev2.rewards.reward import get_reward_funcs
    from swift.dev2.rewards.scorer import ChannelCompanions, resolve_channel

    plan = resolve_channel(
        list(getattr(reward_config, channel)),
        ChannelCompanions(
            adapters=getattr(reward_config, f'{channel}_adapters'),
            model_types=getattr(reward_config, f'{channel}_model_type'),
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
    from swift.dev2.builders import load_prompt_only_rows

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


def initialize_supervised_twinkle(distributed_config: 'DistributedConfig', *, seed: int,
                                  full_determinism: bool, tf32: Optional[bool],
                                  auxiliary_groups: Sequence[Tuple[str, int]] = ()) -> None:
    """Initialize twinkle for EVERY offline backend -- required for a DeviceMesh to reach the model.

    mode='ray' builds the 'model' DeviceGroup the workers hold (build_model targets it with
    remote_group='model'); everything else initializes 'local'. ``auxiliary_groups`` is one
    ``(name, world_size)`` per FULL-PARAMETER frozen auxiliary (the preference family's reference):
    a disjoint group appended after the 'model' ranks. The seed bundle re-seeds every ray worker.
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
                                      device_type='GPU', gpus_per_worker=1))
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

    ``rl_config`` / ``reward_config`` ride along when a family consumes them (the preference family).
    """
    from swift.dev2.config import process_configs, validate_configs

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

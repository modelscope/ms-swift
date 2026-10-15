"""The single offline training loop every flat recipe shares: assembly -> components -> epochs -> save.

:func:`run_offline` is the one entry point a flat offline recipe (sft / pt / seq_cls / embedding /
reranker / the preference family) calls after ``TrainAssembly.build_objects`` has produced the model +
dataloaders. It composes the three thin components (:class:`StepRunner`, :class:`Evaluator`,
:class:`CheckpointSaver`) and the shared :class:`CallbackHandler`, resumes, runs the epoch loop, then
writes the final checkpoint. Keeping the identical wiring + epoch / GA mechanics here (rather than
copy-pasting them into each recipe) is the "one thin shared layer" the recipes reuse; the loop stays
visible and the components stay explicit.

The default leaf passes a pre-encoded batch straight through as ``inputs`` plus the forward ``task``; a
recipe whose encoding differs (the preference family) injects its own ``micro_step_kwargs`` and, optionally,
a ``log_formatter`` for the recorder's console line.

The two backends express gradient accumulation differently, so the loop forks on exactly one axis -- who
schedules the GA micro-steps (see ``backend_owns_gradient_accumulation``):

  - transformers: ONE micro-batch per ``forward_backward``; twinkle's grad-sync gate no-ops on non-boundary
    micro-steps and steps the optimizer every ga-th call. The loop mirrors that boundary only to log/count.
  - Megatron: GA is internal to a single ``forward_backward`` -- it splits an inputs LIST into len(inputs)
    microbatches and accumulates their grads before one optimizer step. So the loop groups ga dataloader
    batches into one list and calls ``forward_backward`` ONCE per optimizer step.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from swift.dev.utils.logger import get_logger

if TYPE_CHECKING:
    from swift.dev.recipe.assembly import TrainAssembly
    from swift.dev.recipe.checkpoint import CheckpointSaver
    from swift.dev.recipe.evaluator import Evaluator
    from swift.dev.recipe.step_runner import StepRunner

logger = get_logger()

__all__ = ['run_offline']


def run_offline(assembly: 'TrainAssembly',
                *,
                save_final: bool = True,
                micro_step_kwargs: Optional[Callable[..., Dict[str, Any]]] = None,
                log_formatter: Optional[Callable[[dict, float], str]] = None) -> List[dict]:
    """Drive the shared offline loop over an assembly whose config -> object stages have already run.

    The one entry point a flat offline recipe (sft / pt / seq_cls / embedding / reranker / the preference
    family) calls after :meth:`TrainAssembly.build_objects` has produced the model + dataloaders. It composes
    the three thin components (:class:`StepRunner` / :class:`Evaluator` / :class:`CheckpointSaver`) plus the
    shared :class:`CallbackHandler`, resumes, runs the epochs through the backend's fit loop, then writes the
    final checkpoint. Keeping this wiring + the epoch / GA mechanics here once (rather than copy-pasting into
    each recipe) is the single thin shared layer the recipes reuse.

    By default the dataloader already yields ``list[InputFeature]`` (the encode ran in the assembly), so the
    one leaf-specific piece is a passthrough of ``inputs`` plus the forward ``task``: ``owns_ga=False``
    (transformers) adds ``gradient_accumulation_steps`` so twinkle's grad-sync gate steps the optimizer every
    ga-th micro-batch, while ``owns_ga=True`` (Megatron) omits it because Megatron derives GA from the
    microbatch-list length. A recipe whose batch needs a different encoding (the preference family) passes its
    own ``micro_step_kwargs`` callable, and may inject a ``log_formatter`` for the recorder's console line.

    Resume order is twinkle-locked: ``configure_optimizer`` (in the assembly stages) ran BEFORE
    ``resume_from_checkpoint``, which restores optim/sched/RNG/cur_step. Here ``on_init_end`` fires first,
    then ``assembly.resume_model()`` (only when ``assembly.resume_dir`` is set), then its returned state seeds
    the counters + dataloader skip. ``save_final=False`` is test-oriented: it yields the loss trajectory
    without a final checkpoint.
    """
    from swift.dev.callbacks import CallbackHandler
    from swift.dev.recipe._loop_utils import (backend_owns_gradient_accumulation, finish_manual_gc,
                                              start_manual_gc)
    from swift.dev.recipe.checkpoint import CheckpointSaver
    from swift.dev.recipe.evaluator import Evaluator
    from swift.dev.recipe.step_runner import StepRunner
    from twinkle.perf import Perf

    if micro_step_kwargs is None:
        task = assembly.forward_task
        ga = assembly.ga

        def micro_step_kwargs(unit, *, owns_ga: bool):
            kwargs = {'inputs': unit, 'task': task}
            if not owns_ga:
                kwargs['gradient_accumulation_steps'] = ga
            return kwargs

    component_kwargs = assembly.offline_component_kwargs()
    if log_formatter is not None:
        component_kwargs['recorder']['log_formatter'] = log_formatter
    model = assembly.model
    dataloader = assembly.dataloader
    num_train_epochs = assembly.train_config.num_train_epochs
    resume = assembly.resume_model if assembly.resume_dir else None
    write_args_json = assembly.args_json_writer()

    recorder_kwargs = dict(component_kwargs['recorder'])
    handler = CallbackHandler(recorder_kwargs.pop('callbacks'), output_dir=assembly.output_dir)
    recorder = StepRunner(model, handler, **recorder_kwargs)
    evaluator = Evaluator(model, recorder, **component_kwargs['evaluator'])
    # Link so the recorder publishes the evaluator's best_metric onto the shared callback state.
    recorder.evaluator = evaluator
    saver = CheckpointSaver(model, recorder, dataloader, **component_kwargs['saver'])
    recorder.publish_state()
    handler.on_init_end()
    start_epoch = 0
    if resume is not None:
        start_epoch = saver.resume(resume())

    if Perf.initialize(assembly.perf_actors()):
        logger.info('[perf] profiling enabled for the offline training loop.')
    gc_was_enabled = start_manual_gc(recorder.manual_gc)
    recorder.publish_state()
    handler.on_train_begin()
    try:
        if evaluator.eval_on_start:
            evaluator.evaluate(micro_step_kwargs)
        if backend_owns_gradient_accumulation(recorder.model):
            _fit_megatron(
                recorder,
                evaluator,
                saver,
                dataloader=dataloader,
                micro_step_kwargs=micro_step_kwargs,
                num_train_epochs=num_train_epochs,
                start_epoch=start_epoch)
        else:
            _fit_transformers(
                recorder,
                evaluator,
                saver,
                dataloader=dataloader,
                micro_step_kwargs=micro_step_kwargs,
                num_train_epochs=num_train_epochs,
                start_epoch=start_epoch)
        evaluator.load_best_model()
    finally:
        Perf.shutdown()
        recorder.publish_state()
        handler.on_train_end()
        evaluator.shutdown_eval_sampler()
        finish_manual_gc(gc_was_enabled)
        recorder.tracker.close()
    if save_final:
        saver.save_final(write_args_json=write_args_json)
    return recorder.history


def _epochs(num_train_epochs: float, max_steps: int) -> int:
    """Epoch bound: the ceil of num_train_epochs, or a synthetic 10**9 when a max_steps cap drives the end."""
    return math.ceil(num_train_epochs) if max_steps <= 0 else 10**9


def _guard_nonempty_epoch(recorder: 'StepRunner', epoch: int, batches: int) -> None:
    """Fail loudly if an epoch produced no batches, instead of spinning empty epochs forever.

    When ``max_steps > 0``, :func:`_epochs` returns a synthetic ``10**9`` bound, so the loop only ends once
    a step limit or stop flag trips. A dataloader that yields nothing -- the usual cause is a global batch
    (``per_device_train_batch_size * gradient_accumulation_steps * data_parallel_size``) larger than the
    dataset, whose remainder ``drop_last`` then discards -- never runs a step, so nothing ever trips that
    bound and training would burn a billion empty epochs in silence. Raise instead, naming the knob to
    lower. Skipped once the run is already at its step cap or externally stopped, where an empty final
    epoch is expected.
    """
    if batches or recorder.reached_max() or recorder.should_stop():
        return
    raise ValueError(
        f'Epoch {epoch} yielded no training batches, so no optimizer step can ever be taken. This almost '
        'always means the global batch (per_device_train_batch_size * gradient_accumulation_steps * '
        'data_parallel_size) exceeds the dataset size, and drop_last discards the remainder. Enlarge the '
        'dataset or lower the batch size; otherwise training would loop empty epochs forever.')


def _step_cadence(recorder: 'StepRunner',
                  evaluator: 'Evaluator',
                  saver: 'CheckpointSaver',
                  micro_step_kwargs: Callable[..., Dict[str, Any]]) -> None:
    """The end-of-optimizer-step tail: periodic save, periodic eval, on_step_end, then callback-driven saves.

    Runs right after ``recorder.record_step`` (which owns the metric + log half). The order is the old
    ``TrainLoop._record_step`` save branch followed by ``OfflineLoop._post_step`` verbatim: the save_steps
    checkpoint first, then a due eval, then ``on_step_end`` (where EarlyStop / GracefulExit raise their
    flags), then honour + clear the transient ``should_save`` / ``should_evaluate`` / ``should_log`` flags so
    a one-shot request does not repeat on every subsequent step.
    """
    if saver.save_steps and recorder.global_step % saver.save_steps == 0:
        saver.save(f'checkpoint-{recorder.global_step}')
    if evaluator.due_for_eval():
        evaluator.evaluate(micro_step_kwargs)
    recorder.handler.on_step_end()
    control = recorder.handler.control
    if control.should_save:
        saver.save(f'checkpoint-{recorder.global_step}')
    if control.should_evaluate:
        evaluator.evaluate(micro_step_kwargs)
    control.should_save = False
    control.should_evaluate = False
    control.should_log = False


def _fit_transformers(recorder: 'StepRunner',
                      evaluator: 'Evaluator',
                      saver: 'CheckpointSaver',
                      *,
                      dataloader: Any,
                      micro_step_kwargs: Callable[..., Dict[str, Any]],
                      num_train_epochs: float,
                      start_epoch: int) -> None:
    """transformers GA: one micro-batch per forward_backward, twinkle's grad-sync gate steps the optimizer."""
    ga = recorder.gradient_accumulation_steps
    handler = recorder.handler
    # twinkle's grad-sync gate lags one micro-step, so an optimizer window here spans ga+1 micro-batches;
    # on_step_begin fires at each window's first micro-batch (this flag, re-set after every boundary) rather
    # than on a clean ga multiple.
    at_window_start = True
    for epoch in range(start_epoch, _epochs(num_train_epochs, recorder.max_steps)):
        if recorder.reached_max() or recorder.should_stop():
            break
        recorder.epoch = epoch
        # Reshuffle each epoch (twinkle's EpochSampler derives each epoch's order from its number, and only
        # set_epoch advances it); resume replays a given epoch's order the same way.
        if hasattr(dataloader, 'set_epoch'):
            dataloader.set_epoch(epoch)
        recorder.publish_state()
        handler.on_epoch_begin()
        batches_this_epoch = 0
        for batch in dataloader:
            batches_this_epoch += 1
            if at_window_start:
                recorder.publish_state()
                handler.on_step_begin()
                at_window_start = False
            recorder.micro_step += 1
            # forward_backward / clip_grad_and_step carry @timed at their twinkle definition site, so the span
            # is recorded on the rank that actually runs them (per-rank, no driver-side RPC folded in) -- no
            # driver-side `with perf_span(...)` wrapper here.
            recorder.model.forward_backward(**micro_step_kwargs(batch, owns_ga=False))
            recorder.publish_state()
            handler.on_substep_end()
            # Boundary computed loop-side (backend-agnostic); clip_grad_and_step self-guards via the same
            # predicate on the worker, so calling it every micro-step is safe.
            is_boundary = recorder.is_grad_sync_boundary()
            if is_boundary:
                handler.on_pre_optimizer_step()
            recorder.model.clip_grad_and_step(max_grad_norm=recorder.max_grad_norm,
                                              gradient_accumulation_steps=ga)
            if is_boundary:
                handler.on_optimizer_step()
                recorder.record_step()
                _step_cadence(recorder, evaluator, saver, micro_step_kwargs)
                at_window_start = True
                if recorder.reached_max() or recorder.should_stop():
                    break
        _guard_nonempty_epoch(recorder, epoch, batches_this_epoch)
        recorder.publish_state()
        handler.on_epoch_end()
        if recorder.history:
            recorder.tracker.log(recorder.history[-1], recorder.global_step, epoch_end=True)
        if recorder.should_stop():
            break
    evaluator.final_eval(micro_step_kwargs)


def _fit_megatron(recorder: 'StepRunner',
                  evaluator: 'Evaluator',
                  saver: 'CheckpointSaver',
                  *,
                  dataloader: Any,
                  micro_step_kwargs: Callable[..., Dict[str, Any]],
                  num_train_epochs: float,
                  start_epoch: int) -> None:
    """Megatron GA: group ga dataloader batches into one microbatch-list, one optimizer step per group.

    Each dataloader batch is a ``list[InputFeature]`` of size per_device_train_batch_size; ga of them
    concatenated form one optimizer step's microbatch list. ``forward_backward`` derives num_microbatches
    from that list length and accumulates internally, so exactly ONE optimizer step is taken per group.
    ``gradient_accumulation_steps`` is NOT passed (Megatron reads GA from ``len(inputs)`` and the kwarg would
    collide with its internal metric ``accumulate()``); ``micro_batch_size`` is NOT passed either (each worker
    picks ``min(2, per_rank_len)`` itself). On Megatron the dataloader is built with ``drop_last=True``, so a
    trailing partial group cannot occur there; the short-group tail is still handled for safety.
    """
    ga = recorder.gradient_accumulation_steps
    handler = recorder.handler
    for epoch in range(start_epoch, _epochs(num_train_epochs, recorder.max_steps)):
        if recorder.reached_max() or recorder.should_stop():
            break
        recorder.epoch = epoch
        if hasattr(dataloader, 'set_epoch'):
            dataloader.set_epoch(epoch)
        recorder.publish_state()
        handler.on_epoch_begin()
        group: list = []
        batches_in_group = 0
        batches_this_epoch = 0
        for batch in dataloader:
            recorder.micro_step += 1
            batches_this_epoch += 1
            group.extend(batch)
            batches_in_group += 1
            if batches_in_group < ga:
                continue
            _megatron_step(recorder, evaluator, saver, group, micro_step_kwargs)
            group, batches_in_group = [], 0
            if recorder.reached_max() or recorder.should_stop():
                break
        if group and not recorder.reached_max() and not recorder.should_stop():  # trailing partial group
            _megatron_step(recorder, evaluator, saver, group, micro_step_kwargs)
        _guard_nonempty_epoch(recorder, epoch, batches_this_epoch)
        recorder.publish_state()
        handler.on_epoch_end()
        if recorder.history:
            recorder.tracker.log(recorder.history[-1], recorder.global_step, epoch_end=True)
        if recorder.should_stop():
            break
    evaluator.final_eval(micro_step_kwargs)


def _megatron_step(recorder: 'StepRunner',
                   evaluator: 'Evaluator',
                   saver: 'CheckpointSaver',
                   microbatch_list: list,
                   micro_step_kwargs: Callable[..., Dict[str, Any]]) -> None:
    """One Megatron optimizer step over a microbatch list (GA internal to the single forward_backward).

    No ``on_substep_end`` here: Megatron's GA is internal to the one ``forward_backward`` call, so the whole
    group is one optimizer window (on_step_begin -> pre/optimizer-step -> on_step_end via the cadence tail).
    """
    handler = recorder.handler
    recorder.publish_state()
    handler.on_step_begin()
    recorder.model.forward_backward(**micro_step_kwargs(microbatch_list, owns_ga=True))
    handler.on_pre_optimizer_step()
    recorder.model.clip_grad_and_step(max_grad_norm=recorder.max_grad_norm)
    handler.on_optimizer_step()
    recorder.record_step()
    _step_cadence(recorder, evaluator, saver, micro_step_kwargs)

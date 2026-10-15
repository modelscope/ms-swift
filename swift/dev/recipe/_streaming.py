"""The shared per-sample streaming driver pieces for the flat online-RL recipes (GRPO / PPO / distillation).

A recipe-private module (peer of ``_scoring`` / ``_distill`` / ``rl_auxiliary``) holding the three pieces
every streaming recipe assembles instead of the retired ``StreamingRolloutDriver`` class:

* :class:`PhaseHandover` / :class:`NoHandover` / :class:`ExclusiveDeviceHandover` -- how generation and
  training take turns on the device (a colocated sampler time-shares one DeviceGroup with the trainer; a
  disaggregated sampler owns its own GPUs).
* :func:`check_streaming_config` -- the fail-loudly guard a stream's regime/placement/publication knobs must
  satisfy before construction (run BEFORE the recipe's ``RunTracker`` opens any reporter).
* :func:`build_streaming_driver` -- twinkle's algorithm-agnostic
  :class:`~twinkle_agentic.async_rl.streaming_driver.StreamingDriver` assembled in ITERATOR mode
  (``consume=None``): the dev-side control plane (``RLContext`` / ``RLContextManager``), the publication
  strategy (``adapter_snapshot`` / ``in_place`` via ``build_weight_sync_strategy``), the per-sample
  submit/poll/collect/cancel seams over the rollout, the stream geometry, version-span stamping and the
  exclusive-device hand-over, all re-assembled as closures around the recipe's objects.

The algorithm injects only its own seams -- ``score`` / ``assemble`` / ``assembly_ready`` and the optional
``group_complete`` group barrier -- plus its assembly ``unit_size`` (GRPO/RFT pull whole prompt groups;
PPO/distill pull individual samples). ``max_staleness`` is the single regime knob (0 = synchronous
drain-then-publish, 1 = one-step overlap, > 1 = deep buffer), so sync and async are points on one dial,
never separate code paths.
"""
from __future__ import annotations

import math
import os
import shutil
from typing import Any, Callable, Dict, List, Optional, Tuple

from twinkle_agentic.async_rl.context_manager import RLContextManager
from twinkle_agentic.async_rl.streaming_driver import StreamingDriver
from twinkle_agentic.async_rl.types import RLContext
from twinkle_agentic.async_rl.weight_sync import AdapterSnapshotSync, build_weight_sync_strategy

from swift.dev.config.rollout_config import WeightSyncStrategyName
from swift.dev.recipe.train_loop import PromptStream


# --- exclusive-device hand-over (a colocated sampler time-shares one device with the trainer) ------------


class PhaseHandover:
    """How generation and training take turns on the device, as a strategy the driver composes.

    A colocated sampler shares ONE DeviceGroup with the trainer, so generation and training must never run at
    the same time: the driver hands the device to the sampler (``enter``), waits for every in-flight
    trajectory to land before training (``drained``), and hands it back (``exit``). A disaggregated sampler
    owns its own GPUs, so there is nothing to serialize. Encoding this as a strategy (rather than a
    ``serialize_generation`` bool branched on all over the driver) keeps the enter/exit/drain protocol in one
    place and lets the driver read ``serializes`` to decide whether to bracket its run at all.

    ``enter`` is also the in-place publication's weight-sync hook (see :func:`build_streaming_driver`): a
    publish overwrites the sampler's live weights, which for a colocated run IS the device hand-back-in. So
    even the no-serialization strategy implements ``enter`` as a plain weight sync rather than a bare no-op.
    """

    #: Whether generation and training must take turns (True only for a colocated sampler).
    serializes: bool = False

    def enter(self) -> None:
        """Hand the device to the sampler with the current policy, ready to generate."""
        raise NotImplementedError

    def exit(self) -> None:
        """Hand the device back to the trainer (sleep the sampler, reload the model)."""
        raise NotImplementedError

    def drained(self, inflight: int) -> bool:
        """Whether the assembly rules may pull given ``inflight`` trajectories still generating."""
        raise NotImplementedError

    @property
    def in_generation(self) -> bool:
        """Whether the device is currently handed to the sampler (so a mid-window exit can restore it)."""
        return False


class NoHandover(PhaseHandover):
    """A disaggregated sampler: its own GPUs, so nothing to serialize and no device to hand back.

    ``enter`` still pushes the trained policy in (``rollout.sync_weights``), because it is the in-place
    publish's weight-sync hook and a disaggregated sampler generating from merged weights needs that push;
    for a disaggregated rollout ``sync_weights`` reduces to a plain weight sync (no wake/offload). ``exit``
    is a no-op and ``drained`` never waits.
    """

    serializes = False

    def __init__(self, rollout: Any):
        self.rollout = rollout

    def enter(self) -> None:
        self.rollout.sync_weights()

    def exit(self) -> None:
        return None

    def drained(self, inflight: int) -> bool:
        return True


class ExclusiveDeviceHandover(PhaseHandover):
    """A colocated sampler: bracket generation and training so the two never contend for the one device.

    ``enter`` runs ``rollout.sync_weights`` (which, for a colocated rollout, wakes the sampler's weights,
    syncs the policy in, offloads the trainer and wakes the KV cache -- see
    :class:`~swift.dev.recipe._colocate.ColocateHandover`) and marks the device entered; ``exit`` runs
    ``rollout.finish_generate`` (sleep the sampler, reload the trainer) and clears the mark. ``drained``
    defers a pull while any trajectory is still generating, because the consume's ``exit`` would sleep the
    sampler and kill an in-flight generation. The ``in_generation`` mark lets the recipe's ``finally``
    restore the trainer if the run ended mid-window (a budget hit during generation).
    """

    serializes = True

    def __init__(self, rollout: Any):
        self.rollout = rollout
        self._in_generation = False

    def enter(self) -> None:
        self.rollout.sync_weights()
        self._in_generation = True

    def exit(self) -> None:
        self.rollout.finish_generate()
        self._in_generation = False

    def drained(self, inflight: int) -> bool:
        return inflight == 0

    @property
    def in_generation(self) -> bool:
        return self._in_generation


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


def build_streaming_driver(rollout_config: Any,
                           *,
                           colocate: bool,
                           run_id: str,
                           model: Any,
                           rollout: Any,
                           output_dir: str,
                           prompts: List[Any],
                           prompt_extras: List[Dict[str, Any]],
                           adapter_name: Optional[str],
                           train_batch_size: int,
                           gradient_accumulation_steps: int,
                           num_generations: int,
                           num_train_epochs: float,
                           seed: int,
                           sampling_params: Optional[dict],
                           unit_size: int,
                           score: Optional[Callable[[Any], Any]],
                           assemble: Callable[[List[Any]], Any],
                           assembly_ready: Callable[[List[Any]], Optional[List[Any]]],
                           group_complete: Optional[Callable] = None,
                           max_group_regenerations: int = 0,
                           reached_max: Optional[Callable[[], bool]] = None) -> Tuple[Any, Any]:
    """Build twinkle's per-sample StreamingDriver in iterator mode -> ``(driver, handover)``.

    Every flat recipe (GRPO / PPO / distillation with ``lmbda == 1.0``) runs the SAME twinkle driver; only
    the injected algorithm seams and the assembly unit differ:

    * ``unit_size`` -- rows one optimizer step consumes as one assembly unit: GRPO/RFT pull WHOLE prompt
      groups (``unit_size=num_generations``, group-relative advantage is undefined on a partial group);
      PPO/distill pull individual samples (``unit_size=1``), so the buffer prefix up to a
      ``train_batch_size * ga`` multiple is pullable.
    * ``score`` / ``assemble`` / ``assembly_ready`` -- the algorithm's per-sample score seam, batch
      assemble seam and geometric pull rule (all optional; the driver invokes them).
    * ``group_complete`` -- the group barrier hook; ``None`` under a per-sample unit (no groups form).

    The driver runs in ITERATOR mode (``consume=None``): :meth:`StreamingDriver.__iter__` bridges the
    blocking ``run()`` onto a background thread and yields assembled batches through ``for batch in
    driver``; the recipe body trains each and calls :meth:`StreamingDriver.commit`. Everything the retired
    ``StreamingRolloutDriver`` owned that the twinkle driver does not -- the dev-side control plane
    (``RLContext`` / ``RLContextManager``), the publication strategy (``adapter_snapshot`` / ``in_place``
    via ``build_weight_sync_strategy``), the per-sample submit/poll/collect/cancel seams over the rollout,
    the stream geometry (one publish cycle trains ``parameter_sync_step * unit_size`` rows; the buffer
    holds at least one full pull quantum), version-span stamping and the exclusive-device hand-over -- is
    re-assembled here as closures. :func:`check_streaming_config` (the same static guard) runs at
    construction, before the recipe's recorder opens any reporter. The hand-over bracket (initial ``enter``
    / final ``exit``) is the recipe's; later cycles' ``enter`` rides the in-place publish's ``sync_fn``.
    """
    max_staleness = (0 if rollout_config.async_mode == 'none' else 1
                     if rollout_config.async_mode == 'one_step_off' else rollout_config.max_staleness)
    handover = ExclusiveDeviceHandover(rollout) if colocate else NoHandover(rollout)
    check_streaming_config(
        adapter_name=adapter_name,
        max_staleness=max_staleness,
        weight_sync_strategy=rollout_config.weight_sync_strategy,
        allow_partial_rollout=rollout_config.allow_partial_rollout,
        parameter_sync_step=rollout_config.parameter_sync_step,
        serialize_generation=handover.serializes)
    strategy = rollout_config.weight_sync_strategy
    ctx = RLContext(
        tenant_id='dev',
        training_run_id=run_id,
        base_model_id='policy',
        adapter_name=(adapter_name if strategy == 'adapter_snapshot' else None))
    ctx_mgr = RLContextManager(max_staleness=max_staleness)
    allow_partial_rollout = rollout_config.allow_partial_rollout

    #: Adapter snapshot paths written so far (pruned by pin reference count); None under in_place.
    snapshot_paths: Optional[set] = set() if strategy == 'adapter_snapshot' else None
    async_adapter_dir = os.path.join(output_dir, 'async_adapters')

    def _save_snapshot(name: str) -> str:
        path = model.save(name, output_dir=async_adapter_dir, adapter_name=adapter_name, save_optimizer=False)
        snapshot_paths.add(path)
        return path

    def _save_adapter_snapshot(admission: Any) -> str:
        # Named for the version it is about to become (current + 1): publish runs after the partition is
        # trained but before on_partition_trained bumps the tracked version.
        version = ctx_mgr.get_rollout_policy(ctx).version + 1
        return _save_snapshot(f'v{version}')

    def _prune_snapshots() -> None:
        keep = ctx_mgr.adapter_paths_to_keep()
        for path in list(snapshot_paths):
            if path not in keep:
                shutil.rmtree(path, ignore_errors=True)
                snapshot_paths.discard(path)

    weight_sync = build_weight_sync_strategy(
        strategy,
        save_fn=(_save_adapter_snapshot if strategy == 'adapter_snapshot' else None),
        sync_fn=(handover.enter if strategy == 'in_place' else None),
        abort_fn=(rollout.abort_all_inflight if strategy == 'in_place' else None))

    # --- per-sample data plane (the version travels with the submission) --------------------------------
    def _submit(prompt_idx: Any, trajectory_idx: int, policy: Any) -> Any:
        """Admit ONE trajectory without blocking, pinned to ``policy``'s version.

        The driver eats the dataloader directly: this closure indexes the prompt rows the recipe loaded (no
        index-table lookup -- the retired driver's ``prompt_payload`` seam is gone).
        """
        handle = rollout.submit_sample(
            prompts[prompt_idx],
            trajectory_idx,
            prompt_idx=prompt_idx,
            sampling_params=sampling_params,
            prompt_extras=(prompt_extras[prompt_idx] if prompt_extras else None),
            adapter_name=policy.adapter_name or '',
            adapter_path=policy.adapter_path,
            allow_partial_rollout=allow_partial_rollout,
            policy_version=policy.version)
        return handle

    def _poll(handles: List[Any]) -> List[Any]:
        return rollout.poll_completions(handles)

    def _collect(handle: Any) -> Any:
        sample = rollout.collect_sample(handle)
        if strategy == 'in_place':
            # A trajectory may span versions under in_place (publish aborts and resumes it on fresh
            # weights), so its version span is stamped here, at the same instant the driver reads the same
            # current version.
            current = ctx_mgr.get_rollout_policy(ctx).version
            sample.version_span = max(0, current - sample.policy_version)
        return sample

    def _cancel(handle: Any) -> None:
        rollout.cancel_sample(handle)

    # --- stream geometry (mirrors the retired StreamingRolloutDriver.run) ----------------------------------
    unit_rows = rollout_config.parameter_sync_step * unit_size
    groups_per_partition = max(1, math.ceil(unit_rows / num_generations))
    need_rows = train_batch_size * gradient_accumulation_steps
    # The smallest whole-unit row count that fills whole optimizer steps: lcm(need_rows, unit_size). A
    # group consumer can only pull WHOLE groups, so its quantum is need_rows * num_generations / gcd;
    # a per-sample consumer's unit is one row, so the quantum is need_rows itself. A buffer shallower than
    # one quantum could never assemble a batch while admission is capped at buffer_depth (the deadlock
    # bound the driver's contract requires).
    pull_rows = need_rows * num_generations // math.gcd(need_rows, num_generations)
    buffer_depth = max((max_staleness + 1) * groups_per_partition * num_generations, pull_rows)
    driver = StreamingDriver(
        ctx_mgr=ctx_mgr,
        ctx=ctx,
        weight_sync=weight_sync,
        prompt_stream=PromptStream(len(prompts), num_train_epochs=num_train_epochs, seed=seed),
        max_staleness=max_staleness,
        num_generations=num_generations,
        groups_per_partition=groups_per_partition,
        parameter_sync_step=rollout_config.parameter_sync_step,
        buffer_depth=buffer_depth,
        submit=_submit,
        poll=_poll,
        collect=_collect,
        assembly_ready=assembly_ready,
        cancel=_cancel,
        consume=None,
        score=score,
        assemble=assemble,
        group_complete=group_complete,
        max_group_regenerations=max_group_regenerations,
        prune=(_prune_snapshots if isinstance(weight_sync, AdapterSnapshotSync) else None),
        initial_adapter_path=((lambda: _save_snapshot('v0'))
                              if isinstance(weight_sync, AdapterSnapshotSync) else None),
        reached_max=reached_max)
    return driver, handover

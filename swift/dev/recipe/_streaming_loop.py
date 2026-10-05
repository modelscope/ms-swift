# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared per-sample streaming wiring for dev's online RL loops.

:class:`StreamingLoopMixin` holds everything an online loop needs to drive twinkle's algorithm-agnostic
:class:`~twinkle_agentic.async_rl.streaming_driver.StreamingDriver` over the dev sampler, replacing the
retired batch-granular deep-buffer mixin. Trajectories are
admitted ONE AT A TIME (:meth:`swift.dev.rollout.RolloutEngine.submit_sample`), completions are polled
as-completed into the driver's ready buffer, and the consumer trains whatever the algorithm's assembly
rule says is ready -- GRPO pulls COMPLETE GROUPS (group-relative advantage needs a prompt's whole
``num_generations``), PPO pulls individual samples (per-token GAE has no group contract). Either way a
pull is an exact multiple of ``train_batch_size * gradient_accumulation_steps`` rows, so every consume
runs whole optimizer steps and the exact-global-batch invariant Megatron enforces is preserved; the
remainder stays buffered for the next pull instead of being dropped.

In dev this driver serves EVERY online regime, synchronous included: ``max_staleness`` is the single
regime knob (0 = synchronous ``async_mode='none'``, 1 = one-step overlap ``'one_step_off'``, >1 = deep
buffer ``'fully_async'``). The synchronous regime is the same per-sample control flow at staleness 0 --
admit one window at the current version, drain it, train it, publish, repeat -- with NO overlap. A
colocated sampler (which time-shares one DeviceGroup between generation and training) is supported here
through an exclusive-device serialization the mixin adds on top of the driver: ``serialize_generation``
gates every pull behind a DRAIN BARRIER (no trajectory still generating when training starts) and brackets
the phases with the ``ColocateHandover`` enter/exit (wake+sync+offload before generation, sleep+reload
before training), so generation and training never contend for the GPU. The one thing the per-sample
stream cannot serve is a feature that needs the WHOLE fixed batch up front -- GRPO's ``dynamic_sample``
(regenerate zero-variance groups after seeing rewards) and ``remax`` (a synchronous greedy-baseline pass),
and the distillation family's off-policy dataset rounds (``lmbda < 1``); those configs fall back to the
base loop's ``_run_sync`` at routing time (see the ``run_*`` dispatch), which is why ``_run_sync`` is kept.

It is a MIXIN (composed into a loop that already extends its own ``TrainLoop``/``PPOLoop``), never a
base the loop extends instead -- mirroring how the driver itself is composed, not inherited. The mixin
owns the parts identical across algorithms:

* the config guard matrix (:meth:`_check_streaming_config`), run BEFORE ``super().__init__`` because a
  loop builds its ``RunTracker`` (which opens wandb/swanlab/tensorboard reporters) as a construction
  side effect, so a misconfigured stream must fail before any reporter is opened;
* the control-plane / publication construction (:meth:`_init_streaming`), run AFTER ``super().__init__``
  once ``self.rollout`` / ``self.output_dir`` exist;
* :meth:`_drive`, which composes a :class:`StreamingDriver` and injects the loop's seams;
* the per-sample :meth:`_submit_per_sample` / :meth:`_poll` / :meth:`_collect_per_sample` data-plane
  wiring (NO sync-at-submit and NO ``finish_generate``: a disaggregated sampler publishes at the
  driver's publish step and never performed a colocate device hand-over to reverse);
* both assembly rules (:meth:`_assembly_ready_grpo` / :meth:`_assembly_ready_ppo`) -- each consumes
  only loop attributes both loop families carry (``num_generations`` / ``train_batch_size`` /
  ``gradient_accumulation_steps``), so the mixin never branches on the algorithm; the loop picks its
  rule via :meth:`_assembly_ready`;
* the adapter-snapshot save/prune trio, meaningful only under ``adapter_snapshot``.

Each consuming loop supplies the algorithm-specific half the mixin cannot know:

* ``_consume_async_samples(samples)`` -- the score-and-train the driver hands each pulled batch to.
  GRPO inherits it from :class:`~swift.dev.recipe.grpo.GRPOLoop`; PPO maps its own critic+GAE
  ``_consume_samples`` onto that name.
* ``_streaming_unit_size()`` -- the rows one optimizer step consumes (GRPO: ``num_generations``, one
  group; PPO: ``train_batch_size * ga``), which sizes both the pull quantum and the window/backpressure.
* ``_streaming_step_delta()`` -- how many of the loop's recorded steps one consume advanced (GRPO: the
  ``global_step`` delta, i.e. real optimizer steps; PPO: 1, since PPO records one step per consume).
* ``_prompt_payload(indices)`` -- the prompt-row assembly, inherited from the loop. The batch-path
  ``_finalize_samples(samples, indices)`` hook is NOT a streaming seam: ``rollout.collect_sample`` already
  stamps each per-sample collect's group id / version, and that hook's ``num_prompts * num_generations``
  count invariant cannot hold for a single trajectory, so ``_collect_per_sample`` returns it directly.

Two publication mechanisms are supported, selected by ``weight_sync_strategy`` (the same split the deep
buffer used -- see twinkle ``weight_sync``): ``'adapter_snapshot'`` pins each trained version as its own
LoRA adapter path (needs an ``adapter_name``; ``allow_partial_rollout`` is inert and rejected), and
``'in_place'`` overwrites the sampler's single live copy each publish, which -- because a stream always
has trajectories in flight at a publish -- REQUIRES ``allow_partial_rollout`` (abort-and-resume).
"""
from __future__ import annotations

import math
import os
import shutil
from typing import Any, Dict, List, Optional

from twinkle_agentic.async_rl.context_manager import RLContextManager
from twinkle_agentic.async_rl.streaming_driver import BufferRecord, StreamingDriver
from twinkle_agentic.async_rl.types import PartitionAdmission, RLContext, RolloutPolicy
from twinkle_agentic.async_rl.weight_sync import (AdapterSnapshotSync, WeightSyncStrategy,
                                                  build_weight_sync_strategy)


class StreamingLoopMixin:
    """The algorithm-agnostic per-sample streaming half of an online dev loop (composed, not inherited).

    A consuming loop calls :meth:`_check_streaming_config` before ``super().__init__`` and
    :meth:`_init_streaming` after it, then inherits :meth:`_drive` (which its own ``fit`` reaches
    through the ``_drive`` template hook). The loop provides ``_consume_async_samples`` and
    ``_streaming_unit_size`` / ``_assembly_ready``; the per-sample collect needs no loop hook (the rollout
    layer stamps each sample), so ``_finalize_samples`` is a batch/sync-path seam only, not a streaming one.
    """

    # --- config guard (call BEFORE super().__init__: the RunTracker opens reporters as a side effect) ----

    @staticmethod
    def _check_streaming_config(*,
                                adapter_name: Optional[str],
                                max_staleness: int,
                                weight_sync_strategy: str,
                                allow_partial_rollout: bool,
                                parameter_sync_step: int,
                                serialize_generation: bool = False) -> None:
        """Reject a streaming configuration that cannot be driven soundly (fail-loudly).

        Mirrors ``config.validate`` at construction time (defense in depth: a loop built directly, not
        through a ``run_*`` entry, still refuses an unsound stream). ``max_staleness >= 0`` because this
        driver now serves the synchronous regime too: ``0`` is ``async_mode='none'`` (admit one window,
        drain, train, publish -- no overlap), ``1`` the one-step overlap, ``> 1`` the deep buffer. The
        publication mechanism then imposes its own requirement:

        * ``adapter_snapshot`` pins each version as a LoRA adapter path, so it needs an ``adapter_name``
          (a full-parameter policy has no adapter to pin) and must NOT set ``allow_partial_rollout``
          (publishing writes a new path and never overwrites a weight copy an in-flight generation
          decodes, so there is nothing to interrupt and resume -- the flag would be inert).
        * ``in_place`` overwrites the sampler's single live weight copy. Under an OVERLAPPING regime
          (``max_staleness >= 1``) a stream always has trajectories in flight at a publish, so it REQUIRES
          ``allow_partial_rollout``: without the abort-and-resume, the rest of an in-flight generation
          would decode under half-updated weights whose logprobs match no consistent policy version, which
          importance sampling cannot correct. Under the SYNCHRONOUS regime (``max_staleness == 0``) the
          drain barrier empties the in-flight set before every publish, so nothing decodes across the
          overwrite and ``allow_partial_rollout`` is unnecessary (twinkle ``InPlaceWeightSync`` documents
          the abort as a no-op at a strictly synchronous sync point).
        * ``parameter_sync_step`` is the publish cadence in consume-reported steps; a non-positive
          cadence would publish never (or divide by zero in the window sizing).
        * ``serialize_generation`` (a colocated sampler sharing one device with the trainer) is sound only
          at ``max_staleness == 0`` (an exclusive device cannot overlap generation with training, so it
          cannot hold a lookahead window; ``config.validate`` bounds ``max_staleness`` by placement the same
          way) and only under ``in_place`` (the per-cycle device hand-back rides in_place's publish
          ``sync_fn``; ``adapter_snapshot`` publishes with no weight sync, so nothing would re-hand the
          device to the sampler after the first consume's exit).
        """
        if max_staleness < 0:
            raise ValueError(f'max_staleness must be >= 0 (0 = the synchronous regime, which this driver '
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

    # --- control-plane / publication construction (call AFTER super().__init__) -------------------------

    def _init_streaming(self,
                        *,
                        adapter_name: Optional[str],
                        max_staleness: int,
                        weight_sync_strategy: str,
                        allow_partial_rollout: bool,
                        parameter_sync_step: int,
                        run_id: str,
                        serialize_generation: bool = False) -> None:
        """Build the control plane and publication strategy this loop's :meth:`_drive` runs the stream on.

        ``adapter_name`` is the trained LoRA, used only by ``adapter_snapshot`` (to name the saved
        adapter); under ``in_place`` the sampler serves merged base weights with no per-version adapter,
        so the control plane is registered with ``adapter_name=None`` (``policy_slot='full_param'``,
        ``adapter_path=None`` throughout) regardless of whether training itself used a LoRA. ``run_id``
        namespaces the single in-process control-plane key (``RLContextManager`` does not otherwise read
        it). ``serialize_generation`` (a colocated sampler sharing one device with the trainer) turns on
        the exclusive-device protocol: the drain barrier in the assembly rules plus the enter/exit
        hand-over bracketing generation and training (see :meth:`_drive` / :meth:`_consume_streaming`).
        """
        self._adapter_name = adapter_name
        #: Whether an in-flight generation may be interrupted at a publish and resumed on the fresh
        #: weights. True only under ``in_place`` (the guard rejects it as inert under
        #: ``adapter_snapshot``); threaded onto every ``submit_sample`` so the sampler runs its
        #: partial-rollout loop for that submission.
        self._allow_partial_rollout = allow_partial_rollout
        self._weight_sync_strategy = weight_sync_strategy
        #: The publish cadence: one weight publication per ``parameter_sync_step`` consume-reported
        #: steps (the StreamingDriver's ``parameter_sync_step``).
        self._parameter_sync_step = parameter_sync_step
        #: The staleness bound the stream runs under (mirrored onto the control plane, which owns the
        #: gate, and the driver, which sizes backpressure and runs the stale scan). 0 = synchronous.
        self._max_staleness = max_staleness
        #: Exclusive-device serialization for a colocated sampler: gate every pull behind a full drain and
        #: bracket the phases with the ``ColocateHandover`` enter/exit so generation and training never
        #: contend for the one DeviceGroup. False for a disaggregated sampler (its own GPUs, no hand-over).
        self._serialize_generation = serialize_generation
        #: Trajectories admitted but not yet collected (the mixin's own in-flight tally, mirrored from the
        #: submit/collect/cancel seams). The drain barrier defers assembly while this is non-zero under
        #: ``serialize_generation``: the consume's ``exit()`` would sleep the sampler and kill anything
        #: still generating, so training may only start once the device is idle.
        self._streaming_inflight = 0
        #: Whether the shared device is currently handed to the sampler (entered) -- tracked so the initial
        #: enter and each publish's enter (which routes through ``_enter_generation``) stay balanced against
        #: each consume's exit, and ``_drive``'s finally can restore the trainer if the run ended mid-window.
        self._device_in_generation = False
        #: Versioned adapter snapshots live in their own subdir so the run's real checkpoints (and their
        #: rotation) are untouched; the loop prunes stale versions itself, by policy reference count.
        #: Only used under ``adapter_snapshot`` (``in_place`` keeps no snapshots).
        self._async_adapter_dir = os.path.join(self.output_dir, 'async_adapters')
        # A single trained policy in one process, so the control-plane key just needs to be stable and
        # unique; these fields namespace it. Under in_place the policy is served as merged base weights
        # (no per-version adapter), so adapter_name is None there.
        ctx_adapter_name = adapter_name if weight_sync_strategy == 'adapter_snapshot' else None
        self._ctx = RLContext(
            tenant_id='dev', training_run_id=run_id, base_model_id='policy', adapter_name=ctx_adapter_name)
        self._ctx_mgr = RLContextManager(max_staleness=max_staleness)
        # The strategy object owns the publication mechanism, so _drive never branches on it:
        # adapter_snapshot wraps the save callback and returns a new pinned path per version; in_place
        # wraps the sampler's sync_weights with abort-on-publish (abort_all_inflight before the
        # overwrite) and returns None. sync_fn routes through _enter_generation so a colocated publish's
        # weight sync ALSO marks the device entered (for a disaggregated sampler enter() reduces to a
        # plain sync and the flag is never read, so the overlapping path is byte-for-byte unchanged).
        self._weight_sync: WeightSyncStrategy = build_weight_sync_strategy(
            weight_sync_strategy,
            save_fn=(self._save_adapter_snapshot if weight_sync_strategy == 'adapter_snapshot' else None),
            sync_fn=(self._enter_generation if weight_sync_strategy == 'in_place' else None),
            abort_fn=(self.rollout.abort_all_inflight if weight_sync_strategy == 'in_place' else None))
        #: The set of adapter snapshot paths written so far (pruned by policy reference count); ``None``
        #: under ``in_place``, which keeps no snapshots.
        self._snapshot_paths: Optional[set] = set() if weight_sync_strategy == 'adapter_snapshot' else None
        #: Version-spans of the batch the last consume pulled, read by :meth:`_extra_step_metrics` to fold
        #: partial-rollout rate / span distribution into each optimizer step's logged record.
        self._last_batch_version_spans: List[int] = []
        #: Baseline of the driver's cumulative train/idle timers at the last PULL boundary, so the emitted
        #: ``trainer_idle_ratio`` is a per-pull windowed delta (a lifetime cumulative ratio would be smoothed
        #: flat over a long run and stop being informative). The window is folded ONCE per pull in
        #: :meth:`_consume_streaming`, NOT per optimizer step in :meth:`_extra_step_metrics`: the driver adds
        #: ``train_active_time`` only AFTER ``_train_pull`` returns, so reading it mid-pull sees no advance
        #: for the current batch (``denom`` 0 -> ratio pinned 0.0 on every step but a misleading spike on the
        #: first). Snapshotting at the pull boundary gives each batch a coherent (previous-pull train +
        #: idle-waiting-for-this-batch) window.
        self._stats_snapshot: Dict[str, float] = {'train': 0.0, 'idle': 0.0}
        #: The windowed trainer-bubble ratio for the batch being consumed, computed once per pull in
        #: :meth:`_consume_streaming` and reported on every optimizer step of that batch (like the
        #: version-span aggregate). 0.0 before the first pull.
        self._last_pull_idle_ratio: float = 0.0

    # --- async metrics (folded into each optimizer step's record via the loop's _extra_step_metrics hook) ---

    def _extra_step_metrics(self, metrics: dict) -> dict:
        """Fold the streaming driver's async telemetry into this step's logged record.

        Composes with the algorithm loop's own ``_extra_step_metrics`` (GRPO's entropy / rollout-log-ratio;
        PPO's base is empty) via ``super()`` -- the mixin sits BEFORE the loop in the MRO
        (``StreamingGRPOLoop(StreamingLoopMixin, GRPOLoop)``), so ``super()`` reaches the loop's version and
        no field is dropped. ``RunTracker.log`` then emits every int/float scalar in the record generically,
        so ``tracking.py`` needs no async-specific code.

        Sources, all observable at the driver seam without touching the version-agnostic engine:

        * ``trainer_idle_ratio``: the trainer bubble over the window folded at the last PULL boundary (in
          :meth:`_consume_streaming`) -- of the driver thread's time spent EITHER training the previous batch
          OR starved waiting for this one, the fraction starved. Reported identically on every optimizer step
          of this batch (a per-step delta is impossible: the driver times ``consume`` from the outside, so the
          current batch's train time does not exist yet mid-pull). ``rollouter_idle_ratio`` / generation GPU
          time are deliberately NOT emitted -- a single-threaded control loop over a disaggregated sampler
          cannot see sampler GPU idle, and measuring it would require instrumenting the data plane.
        * ``partial_ratio`` / ``max_partial_span`` / ``version_span_mean``: this batch's version-span
          aggregate (partial-rollout rate and span distribution).
        * ``off_policy_consumed`` / ``dropped_stale`` / ``stream_publishes``: lifetime driver counters
          (rare, monotone events -- the rising curve is the informative form, so emitted cumulative).
        """
        extra = super()._extra_step_metrics(metrics)
        driver = getattr(self, '_streaming_driver', None)
        if driver is None:
            return extra
        stats = driver.stats
        spans = self._last_batch_version_spans or [0]
        extra.update({
            'trainer_idle_ratio': self._last_pull_idle_ratio,
            'partial_ratio': sum(1 for span in spans if span > 0) / len(spans),
            'max_partial_span': max(spans),
            'version_span_mean': sum(spans) / len(spans),
            'off_policy_consumed': stats.off_policy_consumed,
            'dropped_stale': stats.dropped_stale,
            'stream_publishes': stats.publishes,
        })
        return extra

    # --- driver -----------------------------------------------------------------------------------------

    def _drive(self) -> None:
        """Run the per-sample streaming driver (overrides the loop's synchronous fixed-batch ``_drive``).

        The control flow is twinkle's algorithm-agnostic :class:`StreamingDriver`: it owns the streaming
        lifecycle (per-sample admit under the staleness gate + backpressure -> as-completed poll ->
        ready buffer -> assembly-gated consume -> publish every ``parameter_sync_step`` steps -> stale
        scan -> prune -> drain) and the per-trajectory version pin/release, running on the
        ``RLContextManager`` + ``WeightSyncStrategy`` this loop built.

        The window sizing and backpressure cap derive from the loop's own training geometry: one publish
        cycle trains ``parameter_sync_step * _streaming_unit_size()`` rows, which is
        ``groups_per_partition`` prompt groups; the buffer holds up to ``max_staleness + 1`` windows'
        worth of samples, so admission always has room to assemble at least one full pull (the deadlock
        bound the driver's contract requires).

        Under ``serialize_generation`` (a colocated sampler) the driver runs unchanged, but this method
        brackets it with the exclusive-device hand-over: an initial ``enter`` puts the version-0 policy in
        the sampler and hands it the device BEFORE the first admission (each later cycle's enter rides the
        in_place publish's ``sync_fn``), every consume's ``exit`` (in :meth:`_consume_streaming`) hands the
        device back for training, and the ``finally`` restores the trainer if the run ended mid-window (a
        budget hit during generation), so downstream final-save / eval always finds the model on the GPU.
        """
        from swift.dev.recipe.train_loop import PromptStream

        unit_rows = self._parameter_sync_step * self._streaming_unit_size()
        groups_per_partition = max(1, math.ceil(unit_rows / self.num_generations))
        # The cap must hold at least one FULL pull. A group consumer can only pull WHOLE groups whose row
        # count is a multiple of ``train_batch_size * ga``, so the smallest such pull is
        # ``lcm(need_rows, num_generations)`` rows (= need_rows * num_generations / gcd) -- NOT merely
        # need_rows rounded up to a group, which can be smaller than the true quantum (e.g. need_rows=6,
        # num_generations=4: ceil-rounding gives 8 but the smallest whole-group multiple of 6 is 12). A
        # buffer shallower than one quantum could never assemble a batch while admission is capped at
        # buffer_depth -- the deadlock bound the driver's contract requires the consumer to respect. For a
        # sample consumer (PPO) the lcm still bounds its need_rows pull, so this is a safe floor for both.
        need_rows = self.train_batch_size * self.gradient_accumulation_steps
        pull_rows = need_rows * self.num_generations // math.gcd(need_rows, self.num_generations)
        buffer_depth = max((self._max_staleness + 1) * groups_per_partition * self.num_generations, pull_rows)
        seams: dict = dict(
            ctx_mgr=self._ctx_mgr,
            ctx=self._ctx,
            weight_sync=self._weight_sync,
            prompt_stream=PromptStream(
                len(self.prompts),
                num_train_epochs=getattr(self, 'num_train_epochs', 1.0),
                seed=getattr(self, 'seed', 42)),
            max_staleness=self._max_staleness,
            num_generations=self.num_generations,
            groups_per_partition=groups_per_partition,
            parameter_sync_step=self._parameter_sync_step,
            buffer_depth=buffer_depth,
            submit=self._submit_per_sample,
            poll=self._poll,
            collect=self._collect_per_sample,
            assembly_ready=self._assembly_ready,
            consume=self._consume_streaming,
            cancel=self._cancel_per_sample,
            reached_max=self._reached_max)
        if isinstance(self._weight_sync, AdapterSnapshotSync):
            # Version 0: snapshot the initial (pre-training) adapter so the first admissions pin a real
            # path and the sampler generates from the trained LoRA, never the bare base model. Pruning
            # then drops a snapshot only once no live version and no in-flight pin reference it.
            seams['prune'] = self._prune_snapshots
            seams['initial_adapter_path'] = lambda: self._save_snapshot('v0')
        self._streaming_driver = StreamingDriver(**seams)
        if self._serialize_generation:
            # Hand the device to the sampler with the version-0 policy before the first admission (later
            # cycles' enter rides the in_place publish's sync_fn = _enter_generation).
            self._enter_generation()
        try:
            self._streaming_driver.run()
        finally:
            if self._serialize_generation and self._device_in_generation:
                # Ended mid-window (e.g. a budget hit during generation): give the device back to the
                # trainer so the post-run final-save / eval finds the model resident.
                self._exit_generation()

    # --- exclusive-device hand-over (colocate only; no-ops for a disaggregated sampler) --------------------

    def _enter_generation(self) -> None:
        """Hand the shared device to the sampler with the current policy (``ColocateHandover.enter``).

        ``rollout.sync_weights`` IS the hand-over's enter (wake the sampler's weights, sync the trained
        policy in, offload the trainer, wake the KV cache); for a disaggregated sampler it reduces to a
        plain weight sync. Routed through here -- and used as the in_place publish's ``sync_fn`` -- so the
        device state is tracked whether the enter comes from the initial pre-run call or a later publish.
        """
        self.rollout.sync_weights()
        self._device_in_generation = True

    def _exit_generation(self) -> None:
        """Hand the shared device back to the trainer (``ColocateHandover.exit``: sleep sampler, reload).

        Called at the top of a consume, AFTER the drain barrier guarantees nothing is still generating (a
        sleeping sampler would otherwise kill an in-flight trajectory). A no-op for a disaggregated sampler.
        """
        self.rollout.finish_generate()
        self._device_in_generation = False

    def _generation_drained(self) -> bool:
        """Whether the assembly rules may pull: always true unless a colocated run still has work in flight.

        The drain barrier -- an exclusive device cannot train while the sampler generates, so under
        ``serialize_generation`` a pull waits until every admitted trajectory is collected (the consume's
        ``exit`` would sleep the sampler and kill anything still running). A disaggregated sampler has its
        own GPUs, so it never waits.
        """
        return not (self._serialize_generation and self._streaming_inflight > 0)

    def _cancel_per_sample(self, handle: Any) -> None:
        """The driver's teardown cancel seam: drop the in-flight tally, then cancel the submission."""
        self._streaming_inflight = max(0, self._streaming_inflight - 1)
        self.rollout.cancel_sample(handle)

    # --- per-sample data plane (the version travels with the submission) ----------------------------------

    def _submit_per_sample(self, prompt_idx: Any, trajectory_idx: int, policy: RolloutPolicy) -> Any:
        """Admit ONE trajectory WITHOUT blocking, pinned to ``policy``'s version.

        One prompt, one sampler submission, so completions arrive as-completed. Under ``adapter_snapshot`` the
        version is selected by ``adapter_path`` on the submission itself, and under ``in_place`` the
        weights are overwritten at the driver's publish step (not here), so either way the sampler's
        live weights are never rewritten at submit and a concurrent generation is never disturbed.
        ``allow_partial_rollout`` (set only under ``in_place``) makes the submission resumable so the
        publish's abort-and-resume can continue it on the fresh weights.
        """
        prompts, extras = self._prompt_payload([prompt_idx])
        handle = self.rollout.submit_sample(
            prompts[0],
            trajectory_idx,
            prompt_idx=prompt_idx,
            sampling_params=self.sampling_params,
            prompt_extras=extras[0] if extras else None,
            adapter_name=policy.adapter_name or '',
            adapter_path=policy.adapter_path,
            allow_partial_rollout=self._allow_partial_rollout,
            policy_version=policy.version)
        # Tally the admission for the drain barrier (mirrors the driver adding the handle to its in-flight
        # set); only after a successful submit, so a failed submit -- whose pin the driver releases -- does
        # not inflate the count and wedge the barrier.
        self._streaming_inflight += 1
        return handle

    def _poll(self, handles: List[Any]) -> List[Any]:
        """Non-blocking as-completed query over the in-flight per-sample handles."""
        return self.rollout.poll_completions(handles)

    def _collect_per_sample(self, handle: Any) -> Any:
        """Collect one completed trajectory (already stamped by the rollout layer).

        ``rollout.collect_sample`` stamps each sample's group id (``prompt_id``) and admission
        ``policy_version`` itself, so there is no per-sample finalize work left. The batch-path
        ``_finalize_samples`` hook is deliberately NOT called here: it carries a
        ``num_prompts * num_generations`` count invariant that a single-trajectory collect can never
        satisfy (it would raise for any ``num_generations > 1``), and its group-id stamping would only
        duplicate what ``collect_sample`` already did. It stays a batch/sync-path hook (the blocking
        ``_generate``), where the count invariant holds.

        Does NOT call ``rollout.finish_generate`` (the device exit): a per-collect exit would reverse the
        hand-over mid-window while siblings are still generating. Under ``serialize_generation`` the exit
        happens ONCE per pull, at the consume boundary after the drain barrier empties the in-flight set
        (see :meth:`_consume_streaming`); for a disaggregated sampler exit is a no-op anyway. The version
        pin is released by the StreamingDriver (which acquired it at submit), not here, so the
        acquire/release lifecycle stays symmetric in the skeleton.
        """
        # Mirror the driver, which pops the handle from its in-flight set BEFORE calling collect: drop the
        # drain-barrier tally first so a collect that raises still leaves the count honest.
        self._streaming_inflight = max(0, self._streaming_inflight - 1)
        sample = self.rollout.collect_sample(handle)
        if self._weight_sync_strategy == 'in_place':
            # in_place 逐轮采活权重、publish 时 abort+resume，故一条 episode 可跨版本；span = collect 时
            # 的 current_version − 准入版本（collect_sample 已把准入版本 stamp 到 sample.policy_version）。
            # 与 driver collect 处同一时刻读同一 current_version，语义一致。adapter_snapshot 下整条 episode
            # pin 一个 frozen path、逐轮同版本，span 保持默认 0，故不在此分支。同步（staleness=0）下 span
            # 恒 0：drain 后才 publish，collect 时 current 仍等于准入版本。
            current = self._ctx_mgr.get_rollout_policy(self._ctx).version
            sample.version_span = max(0, current - sample.policy_version)
        return sample

    # --- assembly rules (the algorithm-specific pull, expressed over loop attributes both families carry) --

    def _streaming_unit_size(self) -> int:
        """Rows one optimizer step consumes; the pull quantum is ``unit_size * ga`` (loop-provided)."""
        raise NotImplementedError

    def _assembly_ready(self, buffer: List[BufferRecord]) -> Optional[List[BufferRecord]]:
        """The loop's pull rule (GRPO: complete groups; PPO: individual samples)."""
        raise NotImplementedError

    def _assembly_ready_grpo(self, buffer: List[BufferRecord]) -> Optional[List[BufferRecord]]:
        """GRPO/RFT pull rule: whole groups, an exact multiple of ``train_batch_size * ga`` rows.

        A group is pullable only when ALL ``num_generations`` trajectories of one prompt are buffered
        (group-relative advantage is undefined on a partial group); siblings are admitted back-to-back
        under one policy, so a complete group always carries ONE version.

        The pull quantum is the smallest whole-group count whose rows fill whole optimizer steps --
        ``need_rows / gcd(num_generations, need_rows)`` groups, i.e. ``lcm(num_generations, need_rows)``
        rows -- so every pull is an exact multiple of ``need_rows`` (no gradient-accumulation raggedness,
        and ``split_mini_batches`` never has to drop a tail). Complete groups are selected in
        first-completion order up to the largest whole-quantum count; the rest stay buffered (the
        streaming replacement of ``split_mini_batches``' dropped tail -- nothing is discarded, the
        remainder simply waits for the next pull).

        The pulled records are re-emitted GROUP-CONTIGUOUS (all siblings of one group adjacent, groups in
        selection order) because ``compute_advantages`` normalizes each consecutive block of
        ``num_generations`` as one prompt group by POSITION -- the buffer's completion order would
        otherwise silently mix two groups into the same normalization block (still divisible by
        ``num_generations``, so it computes wrong advantages without raising).
        """
        if not self._generation_drained():
            return None  # exclusive-device (colocate) drain barrier: no pull while a trajectory generates
        need_rows = self.train_batch_size * self.gradient_accumulation_steps
        counts: dict = {}
        order: List[str] = []
        for record in buffer:
            key = record.group_key
            if key not in counts:
                counts[key] = 0
                order.append(key)
            counts[key] += 1
        complete = [key for key in order if counts[key] >= self.num_generations]
        groups_per_pull = need_rows // math.gcd(self.num_generations, need_rows)
        n_pull = groups_per_pull * (len(complete) // groups_per_pull)
        if n_pull < 1:
            return None
        selected = complete[:n_pull]
        selected_set = set(selected)
        by_group: dict = {}
        for record in buffer:
            if record.group_key in selected_set:
                by_group.setdefault(record.group_key, []).append(record)
        pulled: List[BufferRecord] = []
        for key in selected:
            pulled.extend(by_group[key])
        return pulled

    def _assembly_ready_ppo(self, buffer: List[BufferRecord]) -> Optional[List[BufferRecord]]:
        """PPO/GKD pull rule: individual samples, an exact multiple of ``train_batch_size * ga`` rows.

        PPO scores each sample independently (per-token GAE against the critic), so there is no group
        to wait for -- the prefix of the buffer (completion order) up to the largest whole-unit
        multiple is pulled, and the remainder stays buffered.
        """
        if not self._generation_drained():
            return None  # exclusive-device (colocate) drain barrier: no pull while a trajectory generates
        need_rows = self.train_batch_size * self.gradient_accumulation_steps
        if len(buffer) < need_rows:
            return None
        return buffer[:(len(buffer) // need_rows) * need_rows]

    # --- consume / step accounting -----------------------------------------------------------------------

    def _consume_streaming(self, records: List[BufferRecord]) -> int:
        """Train one pulled batch through the loop's own consume half; report the steps it advanced.

        The records' SAMPLES (already finalized: counted and group-id stamped at collect) go to
        ``_consume_async_samples`` -- GRPO's score-assemble-train or PPO's critic+GAE consume, never a
        fork of either. The return is the loop's recorded-step delta (``_streaming_step_delta``), which
        is what the driver's publish cadence counts.
        """
        if self._serialize_generation:
            # Exclusive device: the drain barrier (in the assembly rule) guaranteed no trajectory is still
            # generating, so hand the GPU back to the trainer -- sleep the sampler, reload the model --
            # before scoring / training this pull. The next cycle's enter rides the publish's sync_fn.
            self._exit_generation()
        samples = [record.sample for record in records]
        driver = getattr(self, '_streaming_driver', None)
        if driver is not None:
            # Fold the trainer-bubble ratio ONCE per pull, here at the batch boundary, BEFORE
            # ``_consume_async_samples`` runs the optimizer steps (whose ``_extra_step_metrics`` read the
            # stashed value). The driver adds ``train_active_time`` only AFTER ``_train_pull`` returns, so
            # at this instant ``train_active_time`` is the sum over pulls < this one; the delta against the
            # last pull boundary is the PREVIOUS pull's train time, and ``idle_time``'s delta is the bubble
            # spent starved waiting for THIS batch. Window = previous-pull train + idle-waiting-for-this-
            # batch, reported identically on every step of this batch (one pull behind, but non-zero and
            # meaningful -- a per-step delta is impossible because the driver times consume from outside).
            stats = driver.stats
            d_train = stats.train_active_time - self._stats_snapshot['train']
            d_idle = stats.idle_time - self._stats_snapshot['idle']
            self._stats_snapshot = {'train': stats.train_active_time, 'idle': stats.idle_time}
            denom = d_train + d_idle
            self._last_pull_idle_ratio = (d_idle / denom) if denom > 0 else 0.0
        # Stash this batch's version-spans for the per-optimizer-step metric fold (_extra_step_metrics):
        # one consume runs ``ga`` steps, and every step of this batch reports the same batch-level span
        # aggregate (partial-rollout rate / span distribution are per-batch signals, not per-step).
        self._last_batch_version_spans = [int(getattr(sample, 'version_span', 0)) for sample in samples]
        self._consume_async_samples(samples)
        return self._streaming_step_delta()

    def _streaming_step_delta(self) -> int:
        """How many recorded steps the last consume advanced (default 1; GRPO overrides with its delta)."""
        return 1

    # --- adapter snapshot publication / pruning (adapter_snapshot only) ---------------------------------

    def _save_snapshot(self, name: str) -> str:
        """Save the current LoRA adapter to ``async_adapters/<name>`` and track the path for pruning."""
        path = self.model.save(
            name, output_dir=self._async_adapter_dir, adapter_name=self._adapter_name, save_optimizer=False)
        self._snapshot_paths.add(path)
        return path

    def _save_adapter_snapshot(self, admission: PartitionAdmission) -> str:
        """The ``AdapterSnapshotSync`` save callback: snapshot the just-trained policy as the NEXT version.

        Called through ``self._weight_sync.publish(admission)`` AFTER the partition is trained but
        BEFORE ``on_partition_trained`` bumps the tracked version, so the new snapshot is named for the
        version it is about to become (``current + 1``).
        """
        version = self._ctx_mgr.get_rollout_policy(self._ctx).version + 1
        return self._save_snapshot(f'v{version}')

    def _prune_snapshots(self) -> None:
        """Delete adapter snapshots no version references, keeping the current policy and every in-flight pin.

        ``RLContextManager.adapter_paths_to_keep`` is the live set (the current policy's path plus every
        reference-counted pin held by a trajectory still running or buffered), so a version is removed
        only once nothing can submit against it -- never under an in-flight sample.
        """
        keep = self._ctx_mgr.adapter_paths_to_keep()
        for path in list(self._snapshot_paths):
            if path not in keep:
                shutil.rmtree(path, ignore_errors=True)
                self._snapshot_paths.discard(path)

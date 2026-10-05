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

sync and async are one mechanism here: the regime is the ``max_staleness`` knob on the control plane
(0 = synchronous, 1 = one-step overlap, >1 = deep buffer). Phase 1 of the streaming migration wires the
overlapping regimes (``async_mode='one_step_off'``/``'fully_async'``); the synchronous fixed-batch paths
migrate in a later phase.

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
* ``_prompt_payload(indices)`` / ``_finalize_samples(samples, indices)`` -- the prompt-row assembly and
  the post-collect group-id stamping, inherited from the loop.

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
from typing import Any, List, Optional

from twinkle_agentic.async_rl.context_manager import RLContextManager
from twinkle_agentic.async_rl.streaming_driver import BufferRecord, StreamingDriver
from twinkle_agentic.async_rl.types import PartitionAdmission, RLContext, RolloutPolicy
from twinkle_agentic.async_rl.weight_sync import (AdapterSnapshotSync, WeightSyncStrategy,
                                                  build_weight_sync_strategy)


class StreamingLoopMixin:
    """The algorithm-agnostic per-sample streaming half of an online dev loop (composed, not inherited).

    A consuming loop calls :meth:`_check_streaming_config` before ``super().__init__`` and
    :meth:`_init_streaming` after it, then inherits :meth:`_drive` (which its own ``fit`` reaches
    through the ``_drive`` template hook). The loop provides ``_consume_async_samples``,
    ``_streaming_unit_size`` / ``_assembly_ready`` and, if it stamps group ids, ``_finalize_samples``.
    """

    # --- config guard (call BEFORE super().__init__: the RunTracker opens reporters as a side effect) ----

    @staticmethod
    def _check_streaming_config(*,
                                adapter_name: Optional[str],
                                max_staleness: int,
                                weight_sync_strategy: str,
                                allow_partial_rollout: bool,
                                parameter_sync_step: int) -> None:
        """Reject a streaming configuration that cannot be driven soundly (fail-loudly).

        Mirrors ``config.validate`` at construction time (defense in depth: a loop built directly, not
        through a ``run_*`` entry, still refuses an unsound stream). ``max_staleness >= 1`` here because
        Phase 1 wires the OVERLAPPING regimes; the synchronous streaming migration (staleness 0) lands
        with the sync paths. The publication mechanism then imposes its own requirement:

        * ``adapter_snapshot`` pins each version as a LoRA adapter path, so it needs an ``adapter_name``
          (a full-parameter policy has no adapter to pin) and must NOT set ``allow_partial_rollout``
          (publishing writes a new path and never overwrites a weight copy an in-flight generation
          decodes, so there is nothing to interrupt and resume -- the flag would be inert).
        * ``in_place`` overwrites the sampler's single live weight copy while trajectories are in
          flight, so it REQUIRES ``allow_partial_rollout``: without the abort-and-resume, the rest of an
          in-flight generation would decode under half-updated weights whose logprobs match no
          consistent policy version, which importance sampling cannot correct.
        * ``parameter_sync_step`` is the publish cadence in consume-reported steps; a non-positive
          cadence would publish never (or divide by zero in the window sizing).
        """
        if max_staleness < 1:
            raise ValueError(f'max_staleness must be >= 1 for the streaming overlap regimes (0 is the '
                             f'synchronous stream, wired by a later phase of the migration); got {max_staleness}.')
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
            if not allow_partial_rollout:
                raise ValueError(
                    "weight_sync_strategy='in_place' overwrites the sampler's single live weight copy while the "
                    'stream still has trajectories in flight, so it needs --allow_partial_rollout: each publish '
                    'aborts every in-flight generation and resumes it from its own tokens on the fresh weights '
                    '(twinkle PartialRolloutMixin + InPlaceWeightSync abort-on-publish). Without it the rest of '
                    'an in-flight generation would decode under half-updated weights whose logprobs match no '
                    'consistent policy, which importance sampling cannot correct. Set --allow_partial_rollout, '
                    "or use weight_sync_strategy='adapter_snapshot' (per-version LoRA pinning, no interrupt).")
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
                        run_id: str) -> None:
        """Build the control plane and publication strategy this loop's :meth:`_drive` runs the stream on.

        ``adapter_name`` is the trained LoRA, used only by ``adapter_snapshot`` (to name the saved
        adapter); under ``in_place`` the sampler serves merged base weights with no per-version adapter,
        so the control plane is registered with ``adapter_name=None`` (``policy_slot='full_param'``,
        ``adapter_path=None`` throughout) regardless of whether training itself used a LoRA. ``run_id``
        namespaces the single in-process control-plane key (``RLContextManager`` does not otherwise read
        it).
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
        #: gate, and the driver, which sizes backpressure and runs the stale scan).
        self._max_staleness = max_staleness
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
        # overwrite) and returns None.
        self._weight_sync: WeightSyncStrategy = build_weight_sync_strategy(
            weight_sync_strategy,
            save_fn=(self._save_adapter_snapshot if weight_sync_strategy == 'adapter_snapshot' else None),
            sync_fn=(self.rollout.sync_weights if weight_sync_strategy == 'in_place' else None),
            abort_fn=(self.rollout.abort_all_inflight if weight_sync_strategy == 'in_place' else None))
        #: The set of adapter snapshot paths written so far (pruned by policy reference count); ``None``
        #: under ``in_place``, which keeps no snapshots.
        self._snapshot_paths: Optional[set] = set() if weight_sync_strategy == 'adapter_snapshot' else None

    # --- driver -----------------------------------------------------------------------------------------

    def _drive(self) -> None:
        """Run the per-sample streaming driver (overrides the loop's sync/1-batch-lookahead ``_drive``).

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
            cancel=self.rollout.cancel_sample,
            reached_max=self._reached_max)
        if isinstance(self._weight_sync, AdapterSnapshotSync):
            # Version 0: snapshot the initial (pre-training) adapter so the first admissions pin a real
            # path and the sampler generates from the trained LoRA, never the bare base model. Pruning
            # then drops a snapshot only once no live version and no in-flight pin reference it.
            seams['prune'] = self._prune_snapshots
            seams['initial_adapter_path'] = lambda: self._save_snapshot('v0')
        self._streaming_driver = StreamingDriver(**seams)
        self._streaming_driver.run()

    # --- per-sample data plane (the version travels with the submission; no colocate hand-over) ----------

    def _submit_per_sample(self, prompt_idx: Any, trajectory_idx: int, policy: RolloutPolicy) -> Any:
        """Admit ONE trajectory WITHOUT blocking, pinned to ``policy``'s version.

        The streaming counterpart of the deep buffer's ``_submit_pinned``, minus the batch: one prompt,
        one sampler submission, so completions arrive as-completed. Under ``adapter_snapshot`` the
        version is selected by ``adapter_path`` on the submission itself, and under ``in_place`` the
        weights are overwritten at the driver's publish step (not here), so either way the sampler's
        live weights are never rewritten at submit and a concurrent generation is never disturbed.
        ``allow_partial_rollout`` (set only under ``in_place``) makes the submission resumable so the
        publish's abort-and-resume can continue it on the fresh weights.
        """
        prompts, extras = self._prompt_payload([prompt_idx])
        return self.rollout.submit_sample(
            prompts[0],
            trajectory_idx,
            prompt_idx=prompt_idx,
            sampling_params=self.sampling_params,
            prompt_extras=extras[0] if extras else None,
            adapter_name=policy.adapter_name or '',
            adapter_path=policy.adapter_path,
            allow_partial_rollout=self._allow_partial_rollout,
            policy_version=policy.version)

    def _poll(self, handles: List[Any]) -> List[Any]:
        """Non-blocking as-completed query over the in-flight per-sample handles."""
        return self.rollout.poll_completions(handles)

    def _collect_per_sample(self, handle: Any) -> Any:
        """Collect one completed trajectory and finalize it (group-id stamping through the loop's seam).

        Deliberately does NOT call ``rollout.finish_generate`` -- that reverses the colocate device
        hand-over, which a disaggregated sampler (mandatory here) never performed. The version pin is
        released by the StreamingDriver (which acquired it at submit), not here, so the
        acquire/release lifecycle stays symmetric in the skeleton.
        """
        sample = self.rollout.collect_sample(handle)
        return self._finalize_samples([sample], [handle.prompt_idx])[0]

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
        samples = [record.sample for record in records]
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

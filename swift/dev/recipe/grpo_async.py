# Copyright (c) ModelScope Contributors. All rights reserved.
"""Per-sample streaming GRPO driver.

:class:`StreamingGRPOLoop` extends :class:`~swift.dev.recipe.grpo.GRPOLoop` with the unified streaming
regime: trajectories are admitted ONE AT A TIME on a sampler that (when disaggregated) keeps generating
while the trainer advances, completions are polled as-completed into the driver's ready buffer, and
training pulls COMPLETE GROUPS whenever enough are ready -- no fixed batch, no head-of-line blocking on
the slowest episode of a batch. Every regime is the SAME loop, differing only in the staleness knob:
``async_mode='none'`` (``max_staleness=0``) drains before it publishes (synchronous), ``'one_step_off'``
(``max_staleness=1``) overlaps one step, and ``'fully_async'`` runs ``max_staleness`` up to the configured
bound. A colocated sampler (``async_mode='none'`` on a shared device) additionally sets
``serialize_generation`` for the exclusive-device hand-over.

It is a DRIVER swap, not a parallel runtime. The loop overrides only :meth:`~GRPOLoop._drive`, which
delegates the control flow to twinkle's algorithm-agnostic
:class:`~twinkle_agentic.async_rl.streaming_driver.StreamingDriver`; every scored feature -- ORM/PRM
rewards, group-relative advantage, MoE routing replay, sampling replay, CHORD, teacher/RLSD
reweighting, rollout importance sampling -- is inherited unchanged because each pulled batch goes to
:meth:`GRPOLoop._consume_async_samples`, the same score-and-train half the other drivers use. So the
streaming regime never forks the feature stack (RL_PLAN's "one loop, every feature" invariant).

The streaming wiring itself -- the config guard matrix, the control-plane / publication construction,
the :meth:`_drive` that composes the driver, the per-sample submit/poll/collect, both assembly rules
and the adapter-snapshot save/prune -- is shared with the PPO streaming loop through
:class:`~swift.dev.recipe._streaming_loop.StreamingLoopMixin` (composed into this loop, not inherited
instead of :class:`GRPOLoop`). This file therefore keeps only what is GRPO-specific: the constructor
(which runs the guard before ``super().__init__`` and the control-plane build after it) and the three
GRPO answers to the mixin's hooks -- the group assembly rule, the unit size (one group =
``num_generations`` rows per optimizer step) and the step accounting (the ``global_step`` delta, i.e.
real optimizer steps, which is what the publish cadence counts).

Publication is selected by ``weight_sync_strategy`` (see the mixin for the full mechanism split):

* ``'adapter_snapshot'`` (single LoRA): each trained version is saved as its own LoRA adapter path and
  every in-flight trajectory is submitted pinned to the version it was admitted under
  (``submit_sample(adapter_path=...)``), so publishing a new version writes a NEW path and never
  disturbs a running generation.
* ``'in_place'`` (full-parameter OR LoRA): the sampler's single live weight copy is overwritten each
  publish. Because a stream always has trajectories in flight at a publish, this is sound ONLY with
  partial rollout (``allow_partial_rollout``): ``InPlaceWeightSync`` aborts every in-flight generation
  BEFORE the overwrite and each resumes from its own tokens on the fresh weights (twinkle's
  ``PartialRolloutMixin``), so no generation decodes across the update and its recorded logprobs match
  a single consistent policy version that importance sampling can correct. The guard matrix (here and
  in ``config.validate``) refuses it without partial rollout.

Runtime note: this delivers the control plane (staleness gate, per-sample version pinning, FIFO window
publication, as-completed collection, group assembly, drain) over the sampler's per-trajectory
``submit_sample``/``poll_completions``/``collect_sample`` API. The sampler-side per-version LoRA
serving, the in-place abort-and-resume, and the real multi-GPU generation/training overlap are
exercised by the sampler runtime; the authoring-time bar here is AST + real import + the ``validate``
guard matrix + a control-plane harness.
"""
from __future__ import annotations

from typing import List, Optional

from swift.dev.recipe._streaming_loop import StreamingLoopMixin
from swift.dev.recipe.grpo import GRPOLoop


class StreamingGRPOLoop(StreamingLoopMixin, GRPOLoop):
    """GRPO loop streaming per-sample up to ``max_staleness`` versions ahead.

    Serves EVERY regime: ``max_staleness=0`` is the synchronous one (drain before publish, no overlap --
    reached under ``async_mode='none'`` for a colocated or disaggregated sampler), ``1`` the one-step
    overlap, ``> 1`` the deep buffer. A colocated sampler (sharing one device with the trainer) additionally
    sets ``serialize_generation`` for the exclusive-device hand-over.

    Extra constructor args (all keyword-only, on top of :class:`GRPOLoop`'s):

    * ``adapter_name``: the trained LoRA adapter. Required under ``weight_sync_strategy='adapter_snapshot'``
      (each version is pinned by its adapter path); ignored under ``'in_place'``, where the sampler serves
      merged base weights and a full-parameter policy has no adapter to pin.
    * ``max_staleness``: how many versions a trajectory may lag the policy that trains it (the per-sample
      integer version lag; sizes the stream -- at most ``max_staleness + 1`` admission windows live).
      ``0`` is the synchronous regime (drain before publish, no overlap), ``1`` the one-step overlap,
      ``> 1`` the deep buffer. Must be ``>= 0``.
    * ``weight_sync_strategy``: how a trained version is published -- ``'adapter_snapshot'`` (per-version
      LoRA pinning) or ``'in_place'`` (overwrite one live copy, needs partial rollout at ``max_staleness
      >= 1``; at ``0`` the drain leaves nothing in flight so the abort is a no-op). See the mixin.
    * ``allow_partial_rollout``: whether an in-flight generation may be interrupted at a publish and
      resumed on the fresh weights. Mandatory under ``'in_place'`` at ``max_staleness >= 1``; inert
      (rejected) under ``'adapter_snapshot'``.
    * ``parameter_sync_step``: the publish cadence -- one weight publication per K optimizer steps
      (default 1: publish every step, the densest sound cadence).
    * ``serialize_generation``: exclusive-device serialization for a COLOCATED sampler (one DeviceGroup
      shared with the trainer). Turns on the drain barrier in the assembly rules plus the enter/exit
      hand-over bracketing generation and training, so the two never contend for the device. Only legal at
      ``max_staleness=0`` (a colocated sampler cannot overlap). False for a disaggregated sampler (its own
      GPUs, no hand-over) -- the default.

    The base order ``(StreamingLoopMixin, GRPOLoop)`` is load-bearing: the mixin's :meth:`_drive` must
    win over :meth:`GRPOLoop._drive` (the synchronous fixed-batch pick), and ``GRPOLoop`` does not
    inherit the mixin, so there is no duplicate-base C3 conflict. Do not reorder.
    """

    def __init__(self,
                 *args,
                 adapter_name: Optional[str] = None,
                 max_staleness: int = 1,
                 weight_sync_strategy: str = 'adapter_snapshot',
                 allow_partial_rollout: bool = False,
                 parameter_sync_step: int = 1,
                 serialize_generation: bool = False,
                 **kwargs):
        # Config guards run BEFORE super().__init__: GRPOLoop/TrainLoop build the RunTracker (which
        # initialises wandb/swanlab/tensorboard reporters) as a construction side effect, so a
        # misconfigured stream must fail before any reporter is opened.
        self._check_streaming_config(
            adapter_name=adapter_name,
            max_staleness=max_staleness,
            weight_sync_strategy=weight_sync_strategy,
            allow_partial_rollout=allow_partial_rollout,
            parameter_sync_step=parameter_sync_step,
            serialize_generation=serialize_generation)
        super().__init__(*args, **kwargs)
        # Control-plane / publication construction runs AFTER super().__init__ (it reads self.rollout /
        # self.output_dir). The gate lives on ``self._ctx_mgr.max_staleness`` and the mechanism on
        # ``self._weight_sync`` (its type names it), so a reader introspects those rather than a
        # duplicate, drift-prone copy.
        self._init_streaming(
            adapter_name=adapter_name,
            max_staleness=max_staleness,
            weight_sync_strategy=weight_sync_strategy,
            allow_partial_rollout=allow_partial_rollout,
            parameter_sync_step=parameter_sync_step,
            run_id='run_grpo',
            serialize_generation=serialize_generation)

    # --- the GRPO answers to the mixin's hooks ------------------------------------------------------------

    def _assembly_ready(self, buffer) -> Optional[List]:
        """GRPO pulls COMPLETE GROUPS (group-relative advantage needs a prompt's whole generation set)."""
        return self._assembly_ready_grpo(buffer)

    def _streaming_unit_size(self) -> int:
        """One GRPO optimizer step consumes exactly one group (``num_generations`` rows).

        ``train_batch_size * ga`` rows per step is the mixin's pull quantum; expressed in GROUPS that is
        ``train_batch_size * ga / num_generations``, but the driver's window/backpressure sizing only
        needs rows-per-step to be a multiple of the unit, and one group is the indivisible unit -- the
        assembly rule enforces the exact row multiple itself.
        """
        return self.num_generations

    def _streaming_step_delta(self) -> int:
        """GRPO counts REAL optimizer steps (``global_step`` advances per ga mini-batches)."""
        return max(1, self.global_step - self._step_before_consume)

    def _consume_streaming(self, records) -> int:
        """Record the pre-consume ``global_step`` so the delta (the publish-cadence unit) is exact."""
        self._step_before_consume = self.global_step
        return super()._consume_streaming(records)

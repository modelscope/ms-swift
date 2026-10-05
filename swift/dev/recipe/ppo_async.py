# Copyright (c) ModelScope Contributors. All rights reserved.
"""Per-sample streaming PPO driver.

:class:`StreamingPPOLoop` extends :class:`~swift.dev.recipe.run_ppo.PPOLoop` with the unified streaming
regime: trajectories are admitted ONE AT A TIME on a DISAGGREGATED sampler that keeps generating while
the trainer advances, completions are polled as-completed into the driver's ready buffer, and training
pulls individual samples whenever a whole optimizer step's rows are ready -- no fixed batch, no
head-of-line blocking on the slowest episode of a batch. ``async_mode='one_step_off'``
(``max_staleness=1``) and ``'fully_async'`` (``max_staleness`` up to the configured bound) are the SAME
loop; the regime is just the staleness knob.

It is a DRIVER swap, not a parallel runtime. The loop overrides only :meth:`~PPOLoop._drive`, which
delegates the control flow to twinkle's algorithm-agnostic
:class:`~twinkle_agentic.async_rl.streaming_driver.StreamingDriver`; every PPO feature -- per-token GAE
against the learned critic, the KL-to-reference penalty, the reward model(s), the clipped policy
surrogate AND the clipped value loss, the ``num_ppo_epochs`` re-use of one rollout's anchors -- is
inherited unchanged because each pulled batch goes to PPO's OWN :meth:`PPOLoop._consume_samples`
(mapped onto the mixin's ``_consume_async_samples`` seam below), the same plan-and-train half the other
drivers use. So the streaming regime never forks the feature stack and never borrows GRPO's
group-relative consume (RL_PLAN's "one loop, every feature" invariant).

The streaming wiring itself -- the config guard matrix, the control-plane / publication construction,
the :meth:`_drive` that composes the driver, the per-sample submit/poll/collect, both assembly rules
and the adapter-snapshot save/prune -- is shared with the GRPO streaming loop through
:class:`~swift.dev.recipe._streaming_loop.StreamingLoopMixin` (composed into this loop, not inherited
instead of :class:`PPOLoop`). This file therefore keeps only what is PPO-specific: the constructor
(which runs the guard before ``super().__init__`` and the control-plane build after it), the
:meth:`_consume_async_samples` alias onto :meth:`PPOLoop._consume_samples`, and the three PPO answers
to the mixin's hooks -- the per-sample assembly rule (PPO has no group contract), the unit size
(``train_batch_size * ga`` rows per recorded step) and the step accounting (PPO records ONE step per
consume, so the publish cadence counts consumes). The prompt assembly
(:meth:`PPOLoop._prompt_payload`) and post-collect finalize (:meth:`PPOLoop._finalize_samples`, a
plain pass-through) are inherited from ``PPOLoop``.

The CRITIC under a stream: only the POLICY is published to the sampler (it is what generates), so the
``value_model`` is never weight-synced and never versioned; it is trained alongside the policy inside
:meth:`PPOLoop._consume_samples`. A sample's ``old_values`` are captured by the critic version current
at that sample's admission, and the clipped value loss bounds how far the critic may then move --
exactly the anchor-and-clip that makes the 1-batch lookahead sound, widened to ``max_staleness``. No
critic-specific publication or pinning is needed, which is why the mixin (policy-only) drives PPO
unchanged.

Publication is selected by ``weight_sync_strategy`` (see the mixin for the full mechanism split):

* ``'adapter_snapshot'`` (single LoRA): each trained policy version is saved as its own LoRA adapter
  path and every in-flight trajectory is submitted pinned to the version it was admitted under
  (``submit_sample(adapter_path=...)``), so publishing a new version writes a NEW path and never
  disturbs a running generation.
* ``'in_place'`` (full-parameter OR LoRA): the sampler's single live policy copy is overwritten each
  publish. Because a stream always has trajectories in flight at a publish, this is sound ONLY with
  partial rollout (``allow_partial_rollout``): ``InPlaceWeightSync`` aborts every in-flight generation
  BEFORE the overwrite and each resumes from its own tokens on the fresh weights (twinkle's
  ``PartialRolloutMixin``), so no generation decodes across the update and its recorded logprobs match
  a single consistent policy version that PPO's importance ratio can correct. The guard matrix (here
  and in ``config.validate``) refuses it without partial rollout.

Runtime note: this delivers the control plane (staleness gate, per-sample version pinning, FIFO window
publication, as-completed collection, drain) over the sampler's per-trajectory
``submit_sample``/``poll_completions``/``collect_sample`` API. The sampler-side per-version LoRA
serving, the in-place abort-and-resume, and the real multi-GPU generation/training overlap are
exercised by the sampler runtime; the authoring-time bar here is AST + real import + the ``validate``
guard matrix + a control-plane harness.
"""
from __future__ import annotations

from typing import Any, List, Optional

from swift.dev.recipe._streaming_loop import StreamingLoopMixin
from swift.dev.recipe.run_ppo import PPOLoop


class StreamingPPOLoop(StreamingLoopMixin, PPOLoop):
    """PPO loop streaming per-sample over a disaggregated sampler up to ``max_staleness`` versions ahead.

    Extra constructor args (all keyword-only, on top of :class:`PPOLoop`'s):

    * ``adapter_name``: the trained LoRA adapter. Required under ``weight_sync_strategy='adapter_snapshot'``
      (each policy version is pinned by its adapter path); ignored under ``'in_place'``, where the sampler
      serves merged base weights and a full-parameter policy has no adapter to pin. The critic is never
      published, so it is never named here.
    * ``max_staleness``: how many versions a trajectory may lag the policy that trains it (the per-sample
      integer version lag; sizes the stream -- at most ``max_staleness + 1`` admission windows live).
      ``1`` is the one-step overlap, ``> 1`` the deep buffer. Must be ``>= 1``.
    * ``weight_sync_strategy``: how a trained policy version is published -- ``'adapter_snapshot'``
      (per-version LoRA pinning) or ``'in_place'`` (overwrite one live copy, needs partial rollout).
      See the mixin.
    * ``allow_partial_rollout``: whether an in-flight generation may be interrupted at a publish and
      resumed on the fresh weights. Mandatory under ``'in_place'``; inert (rejected) under
      ``'adapter_snapshot'``.
    * ``parameter_sync_step``: the publish cadence -- one weight publication per K recorded steps
      (default 1: publish every step, the densest sound cadence).

    The base order ``(StreamingLoopMixin, PPOLoop)`` is load-bearing: the mixin's :meth:`_drive` must
    win over :meth:`PPOLoop._drive` (the sync/1-batch-lookahead pick), and ``PPOLoop`` does not inherit
    the mixin, so there is no duplicate-base C3 conflict. Do not reorder.
    """

    def __init__(self,
                 *args,
                 adapter_name: Optional[str] = None,
                 max_staleness: int = 1,
                 weight_sync_strategy: str = 'adapter_snapshot',
                 allow_partial_rollout: bool = False,
                 parameter_sync_step: int = 1,
                 **kwargs):
        # Config guards run BEFORE super().__init__: PPOLoop builds the RunTracker (which initialises
        # wandb/swanlab/tensorboard reporters) as a construction side effect, so a misconfigured stream
        # must fail before any reporter is opened.
        self._check_streaming_config(
            adapter_name=adapter_name,
            max_staleness=max_staleness,
            weight_sync_strategy=weight_sync_strategy,
            allow_partial_rollout=allow_partial_rollout,
            parameter_sync_step=parameter_sync_step)
        # The inherited sync/1-batch-lookahead driver is replaced by the mixin's _drive, so
        # async_generate (which selects it in PPOLoop._drive) is forced off to keep the stored flag
        # from misleading a reader.
        kwargs['async_generate'] = False
        super().__init__(*args, **kwargs)
        # Control-plane / publication construction runs AFTER super().__init__ (it reads self.rollout /
        # self.output_dir / self.model). The gate lives on ``self._ctx_mgr.max_staleness`` and the
        # mechanism on ``self._weight_sync`` (its type names it), so a reader introspects those, not a
        # drift-prone copy.
        self._init_streaming(
            adapter_name=adapter_name,
            max_staleness=max_staleness,
            weight_sync_strategy=weight_sync_strategy,
            allow_partial_rollout=allow_partial_rollout,
            parameter_sync_step=parameter_sync_step,
            run_id='run_ppo')

    # --- the PPO answers to the mixin's hooks ---------------------------------------------------------------

    def _consume_async_samples(self, samples: List[Any]) -> None:
        """The streaming consume half: PPO's OWN critic+GAE rollout training, NOT GRPO's score-and-train.

        The mixin's :meth:`~swift.dev.recipe._streaming_loop.StreamingLoopMixin._consume_streaming`
        calls this name. GRPO inherits it from ``GRPOLoop``; PPO's per-rollout training -- plan the GAE
        anchors once, replay them ``num_ppo_epochs``, and step BOTH the clipped policy surrogate and the
        clipped value loss -- lives in :meth:`PPOLoop._consume_samples`. This maps the mixin's seam onto
        it, so the stream drives the identical PPO training half the other drivers use and never reaches
        GRPO's group-relative consume.
        """
        return self._consume_samples(samples)

    def _assembly_ready(self, buffer) -> Optional[List]:
        """PPO pulls INDIVIDUAL SAMPLES (per-token GAE against the critic has no group contract)."""
        return self._assembly_ready_ppo(buffer)

    def _streaming_unit_size(self) -> int:
        """One PPO recorded step consumes ``train_batch_size * ga`` rows (its mini-batch geometry)."""
        return self.train_batch_size * self.gradient_accumulation_steps

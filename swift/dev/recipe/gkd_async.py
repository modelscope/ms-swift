# Copyright (c) ModelScope Contributors. All rights reserved.
"""Per-sample streaming GKD driver.

:class:`StreamingGKDLoop` extends :class:`~swift.dev.recipe.run_gkd.GKDLoop` with the same per-sample
streaming regime GRPO/PPO use: trajectories are admitted ONE AT A TIME on a sampler that (when
disaggregated) keeps generating while the trainer advances, completions are polled as-completed into the
driver's ready buffer, and training pulls individual samples whenever a whole optimizer step's rows are
ready. It is the streaming replacement for GKD's retired batch-granular 1-batch-lookahead overlap, and it
serves the purely on-policy regime (``lmbda==1.0``) at either ``async_mode='none'`` (``max_staleness=0``,
synchronous drain-before-publish) or ``'one_step_off'`` (``max_staleness=1``). An off-policy round
(``lmbda != 1.0``) generates nothing, so there is no trajectory to admit ahead and the router keeps it on
the synchronous :class:`GKDLoop` driver.

It is a DRIVER swap, not a parallel runtime. The loop overrides only :meth:`~GKDLoop._drive`, which
delegates the control flow to twinkle's algorithm-agnostic
:class:`~twinkle_agentic.async_rl.streaming_driver.StreamingDriver`; the distillation itself is inherited
unchanged because each pulled batch goes to :meth:`GKDLoop._consume_async_samples` -> ``_train_distill_rows``
-> the SAME teacher ``forward_only`` scoring and student ``forward_backward`` the synchronous driver runs.
So the streaming regime never forks GKD's teacher signal, and the OPSD/MOPD subclasses -- which reuse this
skeleton and only swap ``_teacher_kwargs``/``_forward_kwargs`` -- stream through it unchanged (see
``opsd_async``/``mopd_async``).

The streaming wiring -- the config guard matrix, the control-plane / publication construction, the
:meth:`_drive` that composes the driver, the per-sample submit/poll/collect, both assembly rules and the
adapter-snapshot save/prune -- is shared with the GRPO/PPO streaming loops through
:class:`~swift.dev.recipe._streaming_loop.StreamingLoopMixin` (composed into this loop, not inherited
instead of :class:`GKDLoop`). This file therefore keeps only what distillation-specific streaming needs:
the constructor (guard before ``super().__init__``, control-plane build after it) and the three distill
answers to the mixin's hooks -- the per-sample assembly rule (GKD scores each sample independently against
the teacher, so it has NO group contract, unlike GRPO), the unit size, and the real-optimizer-step
accounting. The per-sample collect stamps each trajectory's group id and policy version in ``collect_sample``
at the rollout layer, so the batch-path ``_finalize_samples`` seam is never touched here.

Publication is selected by ``weight_sync_strategy`` (see the mixin for the full mechanism split):
``'adapter_snapshot'`` pins each trained version as its own LoRA adapter path, ``'in_place'`` overwrites the
sampler's single live copy each publish and so requires ``allow_partial_rollout`` (abort-and-resume). A
frozen separate teacher (or the adapter-disabled base) is never published -- only the student policy
generates -- so it is version-agnostic to the stream, exactly as PPO's critic is.
"""
from __future__ import annotations

from typing import List, Optional

from swift.dev.recipe._streaming_loop import StreamingLoopMixin
from swift.dev.recipe.run_gkd import GKDLoop


class StreamingGKDLoop(StreamingLoopMixin, GKDLoop):
    """GKD loop streaming per-sample over the purely on-policy regime (``lmbda==1.0``).

    ``max_staleness=0`` is the synchronous regime (``async_mode='none'``, drain before publish) and ``1``
    the one-step overlap (``async_mode='one_step_off'``); an off-policy round (``lmbda != 1.0``) generates
    nothing to admit ahead, so the router keeps it on the synchronous :class:`GKDLoop`. A colocated sampler
    (sharing one device with the trainer) additionally sets ``serialize_generation`` for the
    exclusive-device hand-over.

    Extra constructor args (all keyword-only, on top of :class:`GKDLoop`'s):

    * ``adapter_name``: the trained LoRA adapter. Required under ``weight_sync_strategy='adapter_snapshot'``
      (each version is pinned by its adapter path); ignored under ``'in_place'``, where the sampler serves
      merged base weights and a full-parameter policy has no adapter to pin. The teacher is never published,
      so it is never named here.
    * ``max_staleness``: how many versions a trajectory may lag the policy that trains it. Distillation
      overlaps at most a single step, so the router pins this to ``0`` (``async_mode='none'``) or ``1``
      (``async_mode='one_step_off'``); must be ``>= 0``.
    * ``weight_sync_strategy``: how a trained version is published -- ``'adapter_snapshot'`` (per-version
      LoRA pinning) or ``'in_place'`` (overwrite one live copy, needs partial rollout at ``max_staleness
      >= 1``; at ``0`` the drain leaves nothing in flight so the abort is a no-op). See the mixin.
    * ``allow_partial_rollout``: whether an in-flight generation may be interrupted at a publish and resumed
      on the fresh weights. Mandatory under ``'in_place'`` at ``max_staleness >= 1``; inert (rejected) under
      ``'adapter_snapshot'``.
    * ``parameter_sync_step``: the publish cadence -- one weight publication per K optimizer steps.
    * ``serialize_generation``: exclusive-device serialization for a COLOCATED sampler (one DeviceGroup
      shared with the trainer). Turns on the drain barrier in the assembly rules plus the enter/exit
      hand-over bracketing generation and training. Only legal at ``max_staleness=0``. False for a
      disaggregated sampler -- the default.

    The base order ``(StreamingLoopMixin, GKDLoop)`` is load-bearing: the mixin's :meth:`_drive` must win
    over :meth:`GKDLoop._drive` (the sync/async pick), and ``GKDLoop`` does not inherit the mixin, so there
    is no duplicate-base C3 conflict. Do not reorder. The OPSD/MOPD streaming loops recompose this class
    with their own teacher-signal subclass (``StreamingOPSDLoop(StreamingGKDLoop, OPSDLoop)``), inheriting
    everything below and only overriding ``_RUN_ID``.
    """

    #: Namespaces the single in-process control-plane key (``RLContextManager`` does not otherwise read
    #: it); the OPSD/MOPD subclasses set their own so an introspecting consumer sees the right recipe.
    _RUN_ID = 'run_gkd'

    def __init__(self,
                 *args,
                 adapter_name: Optional[str] = None,
                 max_staleness: int = 1,
                 weight_sync_strategy: str = 'adapter_snapshot',
                 allow_partial_rollout: bool = False,
                 parameter_sync_step: int = 1,
                 serialize_generation: bool = False,
                 **kwargs):
        # Config guards run BEFORE super().__init__: GKDLoop/GRPOLoop build the RunTracker (which
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
        # self.output_dir / self.model).
        self._init_streaming(
            adapter_name=adapter_name,
            max_staleness=max_staleness,
            weight_sync_strategy=weight_sync_strategy,
            allow_partial_rollout=allow_partial_rollout,
            parameter_sync_step=parameter_sync_step,
            run_id=self._RUN_ID,
            serialize_generation=serialize_generation)

    # --- the distill answers to the mixin's hooks -------------------------------------------------------

    def _assembly_ready(self, buffer) -> Optional[List]:
        """GKD pulls INDIVIDUAL SAMPLES: teacher-logit distillation scores each row on its own, so -- like
        PPO, unlike GRPO -- there is no group-relative contract and no ``num_generations`` group to wait
        for. The prefix of the buffer up to the largest whole ``train_batch_size * ga`` multiple is pulled;
        the remainder stays buffered for the next pull.
        """
        return self._assembly_ready_ppo(buffer)

    def _streaming_unit_size(self) -> int:
        """One GKD optimizer step consumes ``train_batch_size * ga`` rows (its mini-batch geometry)."""
        return self.train_batch_size * self.gradient_accumulation_steps

    def _streaming_step_delta(self) -> int:
        """GKD counts REAL optimizer steps (``global_step`` advances per grad-sync boundary, and one consume
        can run several when the buffer held a multiple of the pull quantum). Mirrors GRPO's delta so the
        publish cadence counts the same unit.
        """
        return max(1, self.global_step - self._step_before_consume)

    def _consume_streaming(self, records) -> int:
        """Record the pre-consume ``global_step`` so the delta (the publish-cadence unit) is exact."""
        self._step_before_consume = self.global_step
        return super()._consume_streaming(records)

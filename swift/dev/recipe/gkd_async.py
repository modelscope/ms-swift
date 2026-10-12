# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from typing import List, Optional

from swift.dev.config.rollout_config import WeightSyncStrategyName
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
                 weight_sync_strategy: WeightSyncStrategyName = 'adapter_snapshot',
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
        return self._assembly_ready_no_group(buffer)

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

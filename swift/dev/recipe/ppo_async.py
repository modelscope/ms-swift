# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from typing import Any, List, Optional

from swift.dev.config.rollout_config import WeightSyncStrategyName
from swift.dev.recipe._streaming_loop import StreamingLoopMixin
from swift.dev.recipe.run_ppo import PPOLoop


class StreamingPPOLoop(StreamingLoopMixin, PPOLoop):
    """PPO loop streaming per-sample up to ``max_staleness`` versions ahead.

    Serves EVERY regime: ``max_staleness=0`` is the synchronous one (drain before publish, no overlap --
    reached under ``async_mode='none'`` for a colocated or disaggregated sampler), ``1`` the one-step
    overlap, ``> 1`` the deep buffer. A colocated sampler (sharing one device with the trainer) additionally
    sets ``serialize_generation`` for the exclusive-device hand-over.

    Extra constructor args (all keyword-only, on top of :class:`PPOLoop`'s):

    * ``adapter_name``: the trained LoRA adapter. Required under ``weight_sync_strategy='adapter_snapshot'``
      (each policy version is pinned by its adapter path); ignored under ``'in_place'``, where the sampler
      serves merged base weights and a full-parameter policy has no adapter to pin. The critic is never
      published, so it is never named here.
    * ``max_staleness``: how many versions a trajectory may lag the policy that trains it (the per-sample
      integer version lag; sizes the stream -- at most ``max_staleness + 1`` admission windows live).
      ``0`` is the synchronous regime (drain before publish, no overlap), ``1`` the one-step overlap,
      ``> 1`` the deep buffer. Must be ``>= 0``.
    * ``weight_sync_strategy``: how a trained policy version is published -- ``'adapter_snapshot'``
      (per-version LoRA pinning) or ``'in_place'`` (overwrite one live copy, needs partial rollout at
      ``max_staleness >= 1``; at ``0`` the drain leaves nothing in flight so the abort is a no-op).
      See the mixin.
    * ``allow_partial_rollout``: whether an in-flight generation may be interrupted at a publish and
      resumed on the fresh weights. Mandatory under ``'in_place'`` at ``max_staleness >= 1``; inert
      (rejected) under ``'adapter_snapshot'``.
    * ``parameter_sync_step``: the publish cadence -- one weight publication per K recorded steps
      (default 1: publish every step, the densest sound cadence).
    * ``serialize_generation``: exclusive-device serialization for a COLOCATED sampler (one DeviceGroup
      shared with the trainer). Turns on the drain barrier in the assembly rules plus the enter/exit
      hand-over bracketing generation and training, so the two never contend for the device. Only legal at
      ``max_staleness=0`` (a colocated sampler cannot overlap). False for a disaggregated sampler (its own
      GPUs, no hand-over) -- the default.

    The base order ``(StreamingLoopMixin, PPOLoop)`` is load-bearing: the mixin's :meth:`_drive` must
    win over :meth:`PPOLoop._drive` (the synchronous fixed-batch pick), and ``PPOLoop`` does not inherit
    the mixin, so there is no duplicate-base C3 conflict. Do not reorder.
    """

    def __init__(self,
                 *args,
                 adapter_name: Optional[str] = None,
                 max_staleness: int = 1,
                 weight_sync_strategy: WeightSyncStrategyName = 'adapter_snapshot',
                 allow_partial_rollout: bool = False,
                 parameter_sync_step: int = 1,
                 serialize_generation: bool = False,
                 **kwargs):
        # Config guards run BEFORE super().__init__: PPOLoop builds the RunTracker (which initialises
        # wandb/swanlab/tensorboard reporters) as a construction side effect, so a misconfigured stream
        # must fail before any reporter is opened.
        self._check_streaming_config(
            adapter_name=adapter_name,
            max_staleness=max_staleness,
            weight_sync_strategy=weight_sync_strategy,
            allow_partial_rollout=allow_partial_rollout,
            parameter_sync_step=parameter_sync_step,
            serialize_generation=serialize_generation)
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
            run_id='run_ppo',
            serialize_generation=serialize_generation)

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
        return self._assembly_ready_no_group(buffer)

    def _streaming_unit_size(self) -> int:
        """One PPO recorded step consumes ``train_batch_size * ga`` rows (its mini-batch geometry)."""
        return self.train_batch_size * self.gradient_accumulation_steps

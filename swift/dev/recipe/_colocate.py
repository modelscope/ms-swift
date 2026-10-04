"""The colocate device hand-over around a generate phase (RL rollout or generative eval).

twinkle's ``CheckpointEngineManager`` deliberately leaves the memory schedule to the caller: only the
caller knows where in the loop the shared device is free, and (its docstring warns) waking a sampler
that is already awake is not a no-op in vLLM, so the manager refuses to guess the current state. Two dev
call sites therefore ran the SAME hand-over sequence by hand -- the on-policy rollout
(:class:`~swift.dev.recipe.run_grpo.SyncableRollout`) and generative eval
(:meth:`~swift.dev.recipe.assembly.TrainAssembly._build_eval_sampler`) -- which is exactly the duplication
RL_PLAN 1b-E / Bug#8 targets: the schedule (and its subtle wake-tag ordering) must be fixed in one place,
not two that can drift.

The sequence this factors is the manager's documented colocate order, with one refinement both dev sites
already make: the second wake asks ONLY for ``kv_cache`` (see :meth:`enter`), not the manager docstring's
``wake_up()`` (both tags), because ``weights`` is already awake by then.
"""
from __future__ import annotations
from typing import Any


class ColocateHandover:
    """Wake/sync/offload/reload around a generate phase, for a trainer and sampler sharing GPUs.

    ``enter`` hands the device to the sampler and pushes the trained policy into it; ``exit`` hands the
    device back to the trainer. A disaggregated (``colocate=False``) sampler lives on its own GPUs, so
    there is no device to hand over: ``enter`` reduces to a plain weight sync and ``exit`` is a no-op.

    ``model`` / ``sampler`` / ``manager`` are the twinkle objects the caller already built (the manager
    wires the CUDA-IPC or NCCL path per its ``mode``); ``merge_and_sync`` is forwarded to
    ``manager.sync_weights`` (merged base weights every step, correct for both full and LoRA training).
    """

    def __init__(self,
                 model: Any,
                 sampler: Any,
                 manager: Any,
                 *,
                 colocate: bool,
                 merge_and_sync: bool = True,
                 sleep_level: int = 0):
        self.model = model
        self.sampler = sampler
        self.manager = manager
        self.colocate = colocate
        self.merge_and_sync = merge_and_sync
        self.sleep_level = sleep_level

    def enter(self) -> None:
        """Give the sampler the device and the trained weights, ready to generate.

        Colocate order: wake the sampler's WEIGHTS so the sync has somewhere to write, sync the trained
        policy into them, step the trainer aside (offload), then wake ONLY the KV cache so the sampler can
        generate. The two wakes are tag-disjoint on purpose -- vLLM's ``wake_up`` aborts the whole call on
        the first tag that is not still sleeping, so after waking ``weights`` the second wake may ask only
        for ``kv_cache``: passing ``wake_up()`` (both tags) would hit the already-awake ``weights``, return
        early, and leave the KV cache discarded, so the next generate builds attention metadata on freed
        memory and dies with a CUDA illegal access. (A freshly built sampler is not sleeping yet, so on the
        first round both wakes no-op -- which is why the bug only bites from round two on.)
        """
        if self.colocate:
            self.sampler.wake_up(tags=['weights'])
        self.manager.sync_weights(merge_and_sync=self.merge_and_sync)
        if self.colocate:
            self.model.offload_to_cpu()
            self.sampler.wake_up(tags=['kv_cache'])  # weights already awake; ready to generate

    def exit(self) -> None:
        """Take the device back for the trainer: sleep the sampler, then reload the trainer's weights.

        ``sleep_level`` selects how much the sampler releases: level 1 offloads only the KV cache (weights
        stay resident, so the next ``enter`` re-syncs in place), level 2 also drops the weights. The
        default ``0`` maps to twinkle's own default depth (level 1), keeping the previous behaviour
        byte-for-byte; 1/2 are passed through unchanged.
        """
        if self.colocate:
            self.sampler.sleep(level=self.sleep_level or 1)
            self.model.reload_to_gpu()

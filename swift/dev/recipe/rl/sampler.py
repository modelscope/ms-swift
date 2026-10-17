"""The weight-syncable rollout sampler every online-RL recipe shares.

GRPO / PPO / RFT and the on-policy distillation recipes (GKD / OPSD / MOPD) run the same rollout side:
twinkle is initialized in Ray mode with the planned DeviceGroups (:func:`initialize_twinkle_rl`), the
sampler's size and engine args come from ``RolloutConfig`` (:class:`RolloutSamplerSpec`), and the built
sampler is wrapped in :class:`SyncableRollout` so the trained policy is pushed into it before every rollout.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from swift.dev.recipe.rl.auxiliary import MODEL_GROUP, SAMPLER_GROUP
from swift.dev.recipe.rl.ref_only import REF_ONLY_REMOTE_GROUP
from swift.dev.rollout import RolloutEngine

if TYPE_CHECKING:
    from swift.dev.config import DistributedConfig, GenerationConfig, RLConfig, RolloutConfig, TunerConfig


def initialize_twinkle_rl(distributed_config: DistributedConfig,
                          groups: List[Tuple[str, List[int]]],
                          *,
                          sampler_parallel_width: int = 1,
                          seed: int = 42,
                          full_determinism: bool = False,
                          tf32: Optional[bool] = None,
                          sequence_parallel_size: int = 1) -> None:
    """Initialize twinkle in Ray mode with the planned RL DeviceGroups.

    Online RL is Ray-only: the trainer and sampler are separate Ray actors the driver talks to (the
    recipe loop runs on the driver and calls both), which local/torchrun mode cannot express.

    ``seed`` / ``full_determinism`` are forwarded to ``twinkle.initialize`` (see
    ``TrainAssembly.initialize_twinkle``): twinkle seeds the driver and re-seeds each Ray worker from
    the same values, so the configured TrainConfig.seed steers rollout + training reproducibly
    instead of being overwritten by twinkle's default. ``tf32`` rides the same per-process-global path.
    """
    import twinkle
    from twinkle import DeviceGroup

    from swift.dev.builders import build_dp_mesh

    if distributed_config.mode != 'ray':
        raise ValueError("online RL requires DistributedConfig.mode='ray': the trainer and the rollout sampler are "
                         'separate Ray actors that the driver drives and syncs weights between. mode="local" has no '
                         'way to place two roles.')
    if sampler_parallel_width < 1:
        raise ValueError('sampler_parallel_width must be >= 1.')
    allocated = set()
    device_groups = []
    for group_name, ranks in groups:
        rank_set = set(ranks)
        shared = group_name != MODEL_GROUP and bool(rank_set) and rank_set.issubset(allocated)
        sampler_group = (group_name in (SAMPLER_GROUP, 'shared_sampler') or group_name.startswith('sampler_')
                         or group_name.startswith('shared_sampler_') or group_name.startswith('flex_sampler_'))
        width = sampler_parallel_width if sampler_group else 1
        device_groups.append(
            DeviceGroup(
                name=group_name,
                ranks=ranks,
                device_type='GPU',
                gpus_per_worker=width,
                shared=shared))
        if not shared:
            allocated.update(rank_set)
    total = max(allocated) + 1 if allocated else 0
    twinkle.initialize(
        mode='ray',
        nproc_per_node=total,
        extra_torch_kwargs={'seed': seed, 'full_determinism': full_determinism, 'tf32': tf32},
        global_device_mesh=build_dp_mesh(distributed_config, sequence_parallel_size),
        # --ray_exp_name names this Ray run (cluster/worker-name prefix); online RL is always ray mode.
        name=distributed_config.ray_exp_name,
        groups=[*device_groups, DeviceGroup(name=REF_ONLY_REMOTE_GROUP, ranks=1, device_type='CPU')])


class SyncableRollout(RolloutEngine):
    """A weight-syncable rollout over twinkle's vLLM/SGLang sampler.

    Adds weight sync (+ the colocate memory schedule) to the base :class:`RolloutEngine`, overriding its
    warn-once no-op :meth:`sync_weights` / :meth:`finish_generate`: the on-policy loop calls
    :meth:`sync_weights` once per step, BEFORE the rollout, so the behaviour policy tracks the trained one
    (correct GRPO). Everything else -- ``generate`` and the training-feature assembly with the next-token
    label shift -- is inherited unchanged, so the rollout contract lives in one place. Unlike the base
    engine it is handed an already-built sampler (placed on its own ``remote_group``) plus the trainer
    model, rather than building a sampler from a model id.
    """

    def __init__(self,
                 model: Any,
                 sampler: Any,
                 template: Any,
                 *,
                 colocate: bool,
                 platform: str = 'GPU',
                 sleep_level: int = 0,
                 handover: Any = None):
        from twinkle.checkpoint_engine import CheckpointEngineManager

        from swift.dev.recipe.common.colocate import ColocateHandover

        # NB: deliberately does NOT call RolloutEngine.__init__ (which would build a fresh sampler);
        # the sampler is built and placed by the calling recipe and injected here. It MUST still initialise the
        # sampler-independent streaming state (_episode_futures and friends), which poll/collect/cancel
        # read unconditionally -- _init_streaming_state is the shared initializer RolloutEngine.__init__
        # also calls, so the two cannot drift on which attributes a streaming rollout needs.
        self.model = model
        self.sampler = sampler
        self.template = template
        self._init_streaming_state()
        if handover is not None:
            self.manager = None
            self._handover = handover
            self.phase_handover = handover
            return
        # mode= (not a colocate= flag) is the manager's contract: 'colocate' wires the shared-GPU CUDA
        # IPC hand-over, 'standalone' wires disaggregated Ray actors over NCCL. 'auto' is NOT used here
        # because it resolves Ray actor handles to 'standalone' and deliberately never infers 'colocate'
        # from the driver -- so a colocated run must say so explicitly or it silently takes the NCCL path.
        self.manager = CheckpointEngineManager(
            model=model, sampler=sampler, platform=platform, mode=('colocate' if colocate else 'standalone'))
        # The colocate memory schedule (or, when disaggregated, the plain weight sync) is ColocateHandover's
        # job, shared with generative eval so the two sequences cannot drift. merge_and_sync=True sends
        # merged base weights every step (works for both full and LoRA); the incremental LoRA-only path
        # (merge_and_sync=False) is left to a later optimisation. sleep_level only bites under colocate
        # (a disaggregated sampler owns its GPUs and is never slept); validate warns when it is set there.
        self._handover = ColocateHandover(
            model, sampler, self.manager, colocate=colocate, merge_and_sync=True, sleep_level=sleep_level)

    def sync_weights(self) -> None:
        """Push the trained policy into the sampler. Called once per step, BEFORE the rollout.

        Colocate additionally runs the device hand-over (wake weights -> sync -> offload trainer -> wake KV
        cache); see :class:`~swift.dev.recipe.common.colocate.ColocateHandover` for the sequence and why the two
        wakes are tag-disjoint. :meth:`finish_generate` reverses it after the rollout.
        """
        self._handover.enter()

    def finish_generate(self) -> None:
        """Reverse the colocate hand-over after a rollout, so the trainer can take the GPU back."""
        self._handover.exit()

    def shutdown(self) -> None:
        try:
            self.close()
        finally:
            controller = getattr(self, 'resource_controller', None)
            if controller is not None:
                controller.shutdown()
            else:
                try:
                    self._handover.exit()
                finally:
                    try:
                        self.manager.close()
                    finally:
                        self.sampler.shutdown()


class RolloutSamplerSpec:
    """Size, placement mesh and engine args of the weight-syncable rollout sampler, read from ``RolloutConfig``.

    Every online-RL recipe plans its sampler DeviceGroup and builds the sampler through these static methods,
    so the device plan and the built engine read the backend and its parallel sizes the same way.
    """

    #: The rollout engines that can back an independent-process, weight-synced sampler. The on-policy loop
    #: pushes the trained policy into a co-resident engine through twinkle's CheckpointEngineManager, which
    #: needs a CheckpointEngineMixin sampler; both vLLM and SGLang have one, TransformersSampler deliberately
    #: does not (see :meth:`backend`).
    WEIGHT_SYNCABLE = ('vllm', 'sglang')

    @staticmethod
    def backend(rollout_config: RolloutConfig) -> str:
        """The engine backing the rollout sampler, from ``RolloutConfig.rollout_sampler``.

        Only vLLM and SGLang are weight-syncable, so only they can serve as an RL rollout engine: the loop
        syncs the trained policy into the sampler every step through a CheckpointEngineManager, which needs a
        CheckpointEngineMixin sampler -- TransformersSampler has none by design (it generates on the trainer's
        own weights, so there is nothing to sync into). ``rollout_sampler``'s Literal already excludes
        'transformers'; this is the loud backstop at the seam all the on-policy/distill recipes share, so a
        mis-set value fails here rather than silently running without weight sync.
        """
        backend = rollout_config.rollout_sampler
        if backend not in RolloutSamplerSpec.WEIGHT_SYNCABLE:
            raise ValueError(
                f'rollout_sampler={backend!r} cannot back an online-RL rollout: weight sync into the sampler '
                f'needs a CheckpointEngineMixin engine, which only '
                f'{list(RolloutSamplerSpec.WEIGHT_SYNCABLE)} provide. '
                "Use rollout_sampler='vllm' or 'sglang'.")
        return backend

    @staticmethod
    def parallel_width(rollout_config: RolloutConfig, backend: str) -> int:
        """GPUs owned by one complete sampler engine worker (TP * PP)."""
        if backend == 'sglang':
            return rollout_config.sglang_tp_size * rollout_config.sglang_pp_size
        return rollout_config.vllm_tensor_parallel_size * rollout_config.vllm_pipeline_parallel_size

    @staticmethod
    def replica_count(rollout_config: RolloutConfig, backend: str) -> int:
        """Independent sampler replicas represented by the sampler DeviceMesh DP dimension."""
        return rollout_config.sglang_dp_size if backend == 'sglang' else rollout_config.vllm_data_parallel_size

    @staticmethod
    def world_size(rollout_config: RolloutConfig, backend: str) -> int:
        """Physical GPU count for all complete sampler replicas."""
        return (RolloutSamplerSpec.parallel_width(rollout_config, backend)
                * RolloutSamplerSpec.replica_count(rollout_config, backend))

    @staticmethod
    def device_mesh(rollout_config: RolloutConfig, backend: str) -> Any:
        """Sampler-owned DP mesh; it must never inherit the trainer's global mesh."""
        from twinkle import DeviceMesh
        return DeviceMesh.from_sizes(dp_size=RolloutSamplerSpec.replica_count(rollout_config, backend))

    @staticmethod
    def engine_args(rollout_config: RolloutConfig, engine_args: Optional[Dict[str, Any]], colocate: bool,
                    backend: str = 'vllm', tuner_config: Optional[TunerConfig] = None) -> Dict[str, Any]:
        """Engine args for the rollout sampler, from RolloutConfig (+ caller overrides), for ``backend``.

        The whole backend-prefixed knob set is mapped by :func:`swift.dev.builders.build_engine_args` (it
        strips the ``vllm_``/``sglang_`` prefix off every RolloutConfig field the engine names), so each engine
        knob dev models reaches the sampler instead of only the handful the colocate schedule touches. Caller
        ``engine_args`` win over the config. Colocate then forces the backend's memory-saver flag (vLLM
        ``enable_sleep_mode`` / SGLang ``enable_memory_saver``): the schedule sleeps the sampler to free the GPU
        for the trainer between rollouts, which each engine only permits when built with it on.
        """
        from swift.dev.builders import build_engine_args
        # build_engine_args reads infer_config only on its transformers branch, which ``backend`` rules
        # out, so None is safe here.
        kwargs: Dict[str, Any] = build_engine_args(backend, None, rollout_config)
        kwargs.update(engine_args or {})
        if colocate:
            kwargs['enable_sleep_mode' if backend == 'vllm' else 'enable_memory_saver'] = True
        # A LoRA run under adapter_snapshot loads each published version through vLLM's add_lora, which
        # requires the engine built with LoRA on and budgeted for the training rank (the snapshots do
        # not exist yet at construction, so the rank comes from the config, as in the twinkle cookbook);
        # 2 slots cover a lookahead window (in-flight v while v+1 generates). in_place needs none (it
        # merges into the base and syncs whole weights); sglang registers adapters at request time.
        if (backend == 'vllm' and tuner_config is not None and tuner_config.tuner == 'lora'
                and rollout_config.weight_sync_strategy == 'adapter_snapshot'):
            kwargs.setdefault('enable_lora', True)
            kwargs.setdefault('max_lora_rank', tuner_config.lora_rank)
            kwargs.setdefault('max_loras', 2)
        return kwargs


def rollout_sampling_params(rl_config: RLConfig, generation_config: Optional[GenerationConfig]) -> Dict[str, Any]:
    """The per-rollout SamplingParams dict (max_completion_length + optional generation knobs)."""
    params: Dict[str, Any] = {'max_tokens': rl_config.max_completion_length}
    if generation_config is not None:
        if generation_config.temperature is not None:
            params['temperature'] = generation_config.temperature
        if generation_config.top_p is not None:
            params['top_p'] = generation_config.top_p
        if generation_config.top_k is not None:
            params['top_k'] = generation_config.top_k
    return params

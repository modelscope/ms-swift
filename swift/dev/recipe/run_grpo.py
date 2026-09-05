"""Online GRPO assembly: run_grpo orchestration with real weight synchronization.

Peer of ``run_sft``, for the on-policy RL family (GRPO and, via ``RLHFConfig``, its GSPO/estimator
variants). Unlike the ``grpo.py`` smoke loop -- which keeps the rollout engine on its INITIAL weights
and is therefore not algorithmically-correct GRPO -- this recipe closes the loop: after every
optimizer step the trained policy is pushed into the rollout sampler via twinkle's
``CheckpointEngineManager`` before the next rollout, so the behaviour policy tracks the trained one.

Two placements, one code path (see :func:`plan_rl_device_groups`):

  - heterogeneous (``RolloutConfig.vllm_mode='server'`` / default): trainer and sampler occupy
    DISJOINT GPUs -- a ``model`` DeviceGroup on ranks ``[0, M)`` and a ``sampler`` DeviceGroup on
    ``[M, M+S)``. Weight sync is an NCCL broadcast (``CheckpointEngineManager(colocate=False)``). This
    mirrors ``twinkle/tests/sampler/test_weight_sync.py`` exactly.
  - colocate (``vllm_mode='colocate'``): trainer and sampler SHARE the same GPUs -- a single
    DeviceGroup that both roles are placed in (two ``remote_class`` roles on one DeviceGroup land on
    the same devices with independent rank spaces, so no placement change is needed). NCCL refuses two
    ranks on one device, so weight sync is a per-GPU CUDA IPC handover
    (``CheckpointEngineManager(colocate=True)``), and because the two do not fit at once the recipe
    runs the memory schedule the manager documents: wake the sampler's weights, sync, offload the
    trainer, wake the KV cache, generate, sleep the sampler, reload the trainer.

Rollout backend is twinkle's ``vLLMSampler`` (NOT ``swift.dev.rollout.RolloutEngine``): weight sync
requires a ``CheckpointEngineMixin`` Ray-actor sampler with a ``device_mesh``, which the sampler has
and ``RolloutEngine`` (a bare ``GRPOVllmEngine``) does not. old_logps come from the sampler's
per-token ``sequence.logprobs``; the training feature is rebuilt from the prompt+response token ids
with the SAME next-token label shift ``RolloutEngine`` applies (contract 14/15), so the importance
ratio is not silently off by one.

NOTE ON VERIFICATION: the weight-sync / colocate paths need a Ray + multi-GPU + vLLM environment and
are covered by ``@pytest.mark.slow`` tests, not the normal suite; :func:`plan_rl_device_groups` is a
pure function with its own unit test.
"""
from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from swift.dev.rollout import RolloutEngine

if TYPE_CHECKING:
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        DistributedConfig,
        GenerationConfig,
        ModelConfig,
        RLHFConfig,
        RolloutConfig,
        TemplateConfig,
        TrainConfig,
        TunerConfig,
    )

logger = logging.getLogger(__name__)

#: DeviceGroup names. The trainer is always 'model' (build_model targets it); the sampler is a
#: separate 'sampler' group when disaggregated, or shares 'model' when colocated.
_MODEL_GROUP = 'model'
_SAMPLER_GROUP = 'sampler'


def plan_rl_device_groups(nproc_per_node: int, vllm_mode: Optional[str],
                          sampler_world_size: int) -> Tuple[List[Tuple[str, List[int]]], str, bool]:
    """Plan the twinkle DeviceGroups for an online-RL run. Pure function (no twinkle import).

    Args:
        nproc_per_node: the TRAINER's GPU count (== DistributedConfig.nproc_per_node, which sizes the
            model's data-parallel mesh in build_model).
        vllm_mode: 'colocate' shares the trainer's GPUs; anything else (None/'server') disaggregates.
        sampler_world_size: the sampler's GPU count (vllm_tensor_parallel_size * data_parallel_size).

    Returns:
        ``(groups, sampler_remote_group, colocate)`` where ``groups`` is a list of
        ``(name, ranks)`` to hand twinkle.initialize, ``sampler_remote_group`` is the DeviceGroup the
        sampler is placed in, and ``colocate`` is the flag for CheckpointEngineManager.

    Colocate puts both roles in ONE group over ``nproc_per_node`` GPUs (they co-locate with
    independent rank spaces). Disaggregated appends a second, disjoint ``sampler`` group after the
    trainer's ranks, so total GPUs = nproc_per_node + sampler_world_size.
    """
    if nproc_per_node is None or nproc_per_node < 1:
        raise ValueError(f'nproc_per_node must be >= 1 (the trainer GPU count), got {nproc_per_node!r}.')
    if sampler_world_size < 1:
        raise ValueError(f'sampler_world_size must be >= 1, got {sampler_world_size}.')

    if vllm_mode == 'colocate':
        if sampler_world_size > nproc_per_node:
            raise ValueError(f'colocate needs the sampler ({sampler_world_size} GPUs) to fit within the trainer '
                             f'GPUs ({nproc_per_node}); it shares them. Use vllm_mode="server" to disaggregate.')
        return [(_MODEL_GROUP, list(range(nproc_per_node)))], _MODEL_GROUP, True

    total = nproc_per_node + sampler_world_size
    groups = [
        (_MODEL_GROUP, list(range(nproc_per_node))),
        (_SAMPLER_GROUP, list(range(nproc_per_node, total))),
    ]
    return groups, _SAMPLER_GROUP, False


def _initialize_twinkle_rl(distributed_config: DistributedConfig,
                           groups: List[Tuple[str, List[int]]]) -> None:
    """Initialize twinkle in Ray mode with the planned RL DeviceGroups.

    Online RL is Ray-only: the trainer and sampler are separate Ray actors the driver talks to (the
    GRPO loop runs on the driver and calls both), which local/torchrun mode cannot express.
    """
    import twinkle
    from twinkle import DeviceGroup

    if distributed_config.mode != 'ray':
        raise ValueError("run_grpo requires DistributedConfig.mode='ray': the trainer and the rollout sampler are "
                         'separate Ray actors that the driver drives and syncs weights between. mode="local" has no '
                         'way to place two roles.')
    total = sum(len(ranks) for _, ranks in groups)
    twinkle.initialize(
        mode='ray',
        nproc_per_node=total,
        groups=[DeviceGroup(name=name, ranks=ranks, device_type='GPU', gpus_per_worker=1) for name, ranks in groups])


class SamplerRollout(RolloutEngine):
    """A weight-syncable rollout over twinkle's ``vLLMSampler``.

    Adds weight sync (+ the colocate memory schedule) to the base :class:`RolloutEngine`: the GRPO
    loop calls :meth:`sync_weights` once per step, BEFORE the rollout, so the behaviour policy tracks
    the trained one (correct GRPO). Everything else -- ``generate`` and the training-feature assembly
    with the next-token label shift -- is inherited unchanged, so the rollout contract lives in one
    place. Unlike the base engine it is handed an already-built sampler (placed on its own
    ``remote_group``) plus the trainer model, rather than building a sampler from a model id.
    """

    def __init__(self, model: Any, sampler: Any, *, colocate: bool, platform: str = 'GPU'):
        from twinkle.checkpoint_engine import CheckpointEngineManager

        # NB: deliberately does NOT call RolloutEngine.__init__ (which would build a fresh sampler);
        # the sampler is built and placed by run_grpo and injected here. generate()/shutdown() only
        # touch self.sampler, so they work against the injected one.
        self.model = model
        self.sampler = sampler
        self.colocate = colocate
        self.manager = CheckpointEngineManager(model=model, sampler=sampler, platform=platform, colocate=colocate)
        # merge_and_sync sends merged base weights every step (works for both full and LoRA); the
        # incremental LoRA-only path (merge_and_sync=False) is left to a later optimisation.
        self._merge_and_sync = True

    def sync_weights(self) -> None:
        """Push the trained policy into the sampler. Called once per step, BEFORE the rollout.

        Colocate additionally runs the device hand-over the manager documents: the sampler must hold
        its weights to be written into (wake 'weights'), then the trainer steps aside so the sampler
        can build a KV cache and generate; :meth:`finish_generate` reverses it.
        """
        if self.colocate:
            self.sampler.wake_up(tags=['weights'])
        self.manager.sync_weights(merge_and_sync=self._merge_and_sync)
        if self.colocate:
            self.model.offload_to_cpu()
            self.sampler.wake_up()  # KV cache, ready to generate

    def finish_generate(self) -> None:
        """Reverse the colocate hand-over after a rollout, so the trainer can take the GPU back."""
        if self.colocate:
            self.sampler.sleep()
            self.model.reload_to_gpu()


def run_grpo(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    dataset_config: DatasetConfig,
    train_config: TrainConfig,
    distributed_config: DistributedConfig,
    checkpoint_config: CheckpointConfig,
    rollout_config: RolloutConfig,
    rlhf_config: RLHFConfig,
    tuner_config: Optional[TunerConfig] = None,
    generation_config: Optional[GenerationConfig] = None,
    *,
    engine_args: Optional[Dict[str, Any]] = None,
    output_dir: str = 'output',
    _save_final: bool = True,
) -> List[dict]:
    """Assemble and run online GRPO with weight sync. Returns the loss/metric history.

    Placement is chosen from ``rollout_config.vllm_mode`` (see :func:`plan_rl_device_groups`):
    ``DistributedConfig.nproc_per_node`` is the TRAINER GPU count, and the sampler's GPUs
    (``vllm_tensor_parallel_size * vllm_data_parallel_size``) are placed alongside (disaggregated) or
    shared (colocate).
    """
    from swift.dev.builders import build_sampler
    from swift.dev.loss import configure_rlhf_loss
    from swift.dev.optimizer import configure_optimizer, resolve_max_grad_norm
    from swift.dev.recipe.assembly import TrainAssembly
    from swift.dev.recipe.grpo import GRPOLoop

    assembly = TrainAssembly(
        'run_grpo',
        model_config,
        template_config,
        dataset_config,
        train_config,
        distributed_config,
        checkpoint_config,
        tuner_config,
        rlhf_config=rlhf_config,
        output_dir=output_dir)
    # Also imports the run's plugin files -- the reward names handed to GRPOLoop below are resolved
    # against the registry they write into.
    assembly.prepare()

    sampler_world_size = rollout_config.vllm_tensor_parallel_size * rollout_config.vllm_data_parallel_size
    groups, sampler_remote_group, colocate = plan_rl_device_groups(distributed_config.nproc_per_node,
                                                                   rollout_config.vllm_mode, sampler_world_size)
    # RL initializes twinkle itself rather than through the assembly: it needs two device groups (trainer
    # + sampler), whose placement was just planned.
    _initialize_twinkle_rl(distributed_config, groups)

    assembly.build_template()
    # Trainer: a Ray-actor model in the 'model' group (build_model sets remote_group='model' under
    # mode='ray'), with the tuner applied before loss/optimizer so those target its group.
    assembly.build_model()
    configure_rlhf_loss(assembly.model, rlhf_config)
    # No dataloader to derive a step budget from -- prompts are rolled out, not iterated.
    max_steps = train_config.max_steps or 1
    configure_optimizer(assembly.model, train_config, num_training_steps=max_steps)

    # Sampler: vLLMSampler placed in its group (shared 'model' for colocate, separate 'sampler'
    # otherwise). enable_sleep_mode is required for the colocate device hand-over.
    sampler_engine_args = _sampler_engine_args(rollout_config, engine_args, colocate)
    sampler = build_sampler(
        model_config,
        backend='vllm',
        engine_args=sampler_engine_args,
        template=assembly.template,
        remote_group=sampler_remote_group)
    rollout = SamplerRollout(assembly.model, sampler, colocate=colocate)

    prompts = _prompts_from_dataset(dataset_config)
    loop = GRPOLoop(
        assembly.model,
        rollout,
        prompts,
        num_generations=rlhf_config.num_generations,
        reward_funcs=list(rlhf_config.reward_funcs) or None,
        reward_weights=rlhf_config.reward_weights,
        advantage_estimator=rlhf_config.advantage_estimator,
        scale_rewards=rlhf_config.scale_rewards or 'group',
        rlhf_config=rlhf_config,
        max_steps=max_steps,
        gradient_accumulation_steps=assembly.ga,
        max_grad_norm=resolve_max_grad_norm(train_config),
        sampling_params=_grpo_sampling_params(rlhf_config, generation_config))
    try:
        history = loop.fit()
    finally:
        rollout.shutdown()
    del output_dir, _save_final  # checkpointing of the RL policy is a follow-up; smoke returns history
    return history


def _sampler_engine_args(rollout_config: RolloutConfig, engine_args: Optional[Dict[str, Any]],
                         colocate: bool) -> Dict[str, Any]:
    """vLLM engine args for the rollout sampler, from RolloutConfig (+ caller overrides).

    Colocate forces ``enable_sleep_mode=True``: the memory schedule sleeps the sampler to free the GPU
    for the trainer between rollouts, which vLLM only permits when the engine was built with it.
    """
    kwargs: Dict[str, Any] = dict(engine_args or {})
    kwargs.setdefault('gpu_memory_utilization', rollout_config.vllm_gpu_memory_utilization)
    kwargs.setdefault('tensor_parallel_size', rollout_config.vllm_tensor_parallel_size)
    if rollout_config.vllm_max_model_len is not None:
        kwargs.setdefault('max_model_len', rollout_config.vllm_max_model_len)
    kwargs.setdefault('enforce_eager', rollout_config.vllm_enforce_eager)
    if colocate:
        kwargs['enable_sleep_mode'] = True
    return kwargs


def _grpo_sampling_params(rlhf_config: RLHFConfig, generation_config: Optional[GenerationConfig]) -> Dict[str, Any]:
    """The per-rollout SamplingParams dict (max_completion_length + optional generation knobs)."""
    params: Dict[str, Any] = {'max_tokens': rlhf_config.max_completion_length}
    if generation_config is not None:
        if generation_config.temperature is not None:
            params['temperature'] = generation_config.temperature
        if generation_config.top_p is not None:
            params['top_p'] = generation_config.top_p
        if generation_config.top_k is not None:
            params['top_k'] = generation_config.top_k
    return params


def _prompts_from_dataset(dataset_config: DatasetConfig) -> List[List[dict]]:
    """Load the prompt message-lists for rollout from the (un-encoded) dataset.

    Reuses run_infer's row loader, then keeps only the prompt turns of each row (a trailing assistant
    message, if any, is the reference answer and must not be fed to the policy as context).
    """
    from swift.dev.recipe.run_infer import _load_prompt_rows

    rows = _load_prompt_rows(dataset_config, None, split_dataset_ratio=0.0)
    if not rows:
        raise ValueError('run_grpo got an empty dataset. Set DatasetConfig.dataset with prompts to roll out on.')
    prompts: List[List[dict]] = []
    for row in rows:
        messages = row.get('messages') if isinstance(row, dict) else None
        if not messages:
            continue
        if messages and messages[-1].get('role') == 'assistant':
            messages = messages[:-1]
        prompts.append(list(messages))
    if not prompts:
        raise ValueError('run_grpo found no prompt messages in the dataset rows (expected a `messages` column).')
    return prompts

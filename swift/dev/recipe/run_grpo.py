"""Online GRPO assembly: run_grpo orchestration with real weight synchronization.

Peer of ``run_sft``, for the on-policy RL family (GRPO and, via ``RLHFConfig``, its GSPO/estimator
variants). Unlike the ``grpo.py`` smoke loop -- which keeps the rollout engine on its INITIAL weights
and is therefore not algorithmically-correct GRPO -- this recipe closes the loop: after every
optimizer step the trained policy is pushed into the rollout sampler via twinkle's
``CheckpointEngineManager`` before the next rollout, so the behaviour policy tracks the trained one.

Two placements, one code path (see :func:`plan_rl_device_groups`):

  - disaggregated (``RolloutConfig.vllm_mode='disaggregated'``, or its legacy alias ``'server'``, or the
    ``None`` default): trainer and sampler occupy DISJOINT GPUs -- a ``model`` DeviceGroup on ranks
    ``[0, M)`` and a ``sampler`` DeviceGroup on ``[M, M+S)``. Weight sync is an NCCL broadcast
    (``CheckpointEngineManager(mode='standalone')``). This mirrors
    ``twinkle/tests/sampler/test_weight_sync.py`` exactly.
  - colocate (``vllm_mode='colocate'``): trainer and sampler SHARE the same GPUs -- a single
    DeviceGroup that both roles are placed in (two ``remote_class`` roles on one DeviceGroup land on
    the same devices with independent rank spaces, so no placement change is needed). NCCL refuses two
    ranks on one device, so weight sync is a per-GPU CUDA IPC handover
    (``CheckpointEngineManager(mode='colocate')``), and because the two do not fit at once the recipe
    runs the memory schedule the manager documents: wake the sampler's weights, sync, offload the
    trainer, wake the KV cache, generate, sleep the sampler, reload the trainer.

Rollout backend is twinkle's weight-syncable sampler -- vLLM or SGLang, chosen by
``RolloutConfig.rollout_sampler`` (NOT ``swift.dev.rollout.RolloutEngine``): weight sync requires a
``CheckpointEngineMixin`` Ray-actor sampler with a ``device_mesh``, which those samplers have and
``RolloutEngine`` (a bare ``GRPOVllmEngine``) does not. old_logps come from the sampler's
per-token ``sequence.logprobs``; the training feature is rebuilt from the prompt+response token ids
with the SAME next-token label shift ``RolloutEngine`` applies (contract 14/15), so the importance
ratio is not silently off by one.

NOTE ON VERIFICATION: the weight-sync / colocate paths need a Ray + multi-GPU + vLLM environment and
are covered by ``@pytest.mark.slow`` tests, not the normal suite; :func:`plan_rl_device_groups` is a
pure function with its own unit test.
"""
from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from swift.dev.rollout import RolloutEngine

if TYPE_CHECKING:
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        DistributedConfig,
        GenerationConfig,
        LoggingConfig,
        MegatronConfig,
        ModelConfig,
        MoEConfig,
        QuantizeConfig,
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
#: A separate frozen teacher (GKD/OPSD/MOPD) is its own Ray DeviceGroup, on GPUs apart from the
#: trainer/sampler (plan §3.3); a ``disable_lora``/dynamic-self teacher reuses the student's actor and
#: plans no group. This is the ``remote_group`` the teacher is built with and planned under.
_TEACHER_GROUP = 'teacher'
#: A FULL-PARAMETER frozen auxiliary model -- a full-fine-tuning reference, a seq_cls reward model, or a
#: separate RLSD/SDAR teacher -- is an independent weight copy, so it must be heterogeneous: its OWN Ray
#: DeviceGroup on GPUs apart from the trainer/sampler (plan §3.5). A LoRA/``disable_lora`` auxiliary reuses
#: the policy's already-loaded base (no second copy) and plans NO group. ``plan_rl_device_groups`` appends
#: one disjoint group per ``auxiliary_groups`` entry, under the name the caller then builds with.
_REF_GROUP = 'ref'


def _reward_group(index: int) -> str:
    """The DeviceGroup name for the ``index``-th frozen reward model (each full-parameter RM gets its own)."""
    return f'reward_{index}'


def _prm_group(index: int) -> str:
    """The DeviceGroup name for the ``index``-th frozen PRM model (each full-parameter PRM gets its own).

    Distinct from :func:`_reward_group` so an ``orm`` reward model and a ``prm`` process-reward model that
    happen to be the same checkpoint still land on their own actors -- they are scored at different points
    (outcome vs per-step prefix) and must not share one forward.
    """
    return f'prm_{index}'


def _split_prm_channel(prm_items: Optional[Sequence[Any]]) -> Tuple[List[Any], Optional[str]]:
    """Split ``--prm`` into ``(rule_specs, model_id)``: the rules scored on completions and the one frozen PRM.

    ``prm`` is a merged heterogeneous selector (rule names + at most one reward-model id), the same grammar
    ``orm`` uses, so it is split by :func:`swift.dev.reward.is_registered_reward` -- registered name / class /
    callable is a rule, anything else is a model id. At most one model id is allowed because a PRM channel has
    a single ``prm_parallel_spec`` and a single dedicated DeviceGroup; two model ids would have nowhere
    distinct to live. Both the group PLANNING (run body) and the BUILD (:func:`_build_prm_channel`) call this
    one splitter, so they cannot drift on which item is the model.
    """
    from swift.dev.reward import is_registered_reward
    items = list(prm_items or [])
    rule_specs = [item for item in items if is_registered_reward(item)]
    model_ids = [item for item in items if not is_registered_reward(item)]
    if len(model_ids) > 1:
        raise ValueError(
            f'--prm resolves at most one frozen PRM model alongside any number of rule names, but got '
            f'{len(model_ids)} model ids: {model_ids}. Keep one model id in the prm channel.')
    return rule_specs, (model_ids[0] if model_ids else None)


def teacher_group_world_size(rlhf_config: RLHFConfig) -> int:
    """GPU count for a separate frozen teacher's OWN DeviceGroup: ``teacher_parallel_spec``'s world size, else 1.

    A separate teacher (a GKD/OPSD/MOPD distillation teacher, or a GRPO RLSD/SDAR teacher) is a Ray actor on
    GPUs apart from the trainer/sampler (plan §3.3/§3.5), so it needs its own group. ``teacher_parallel_spec
    = None`` is the card-frugal default -- one teacher rank, serial ``forward_only``; a spec (``dp2``/
    ``fsdp2``/...) sizes the group for a big or concurrency-friendly teacher. ``disable_lora``/dynamic-self
    teachers build no separate actor and need no group (caller passes 0).
    """
    spec = getattr(rlhf_config, 'teacher_parallel_spec', None)
    if not spec:
        return 1
    from twinkle import DeviceMesh
    return DeviceMesh.from_spec(spec).world_size


def plan_rl_device_groups(nproc_per_node: int, vllm_mode: Optional[str], sampler_world_size: int,
                          teacher_world_size: int = 0,
                          auxiliary_groups: Sequence[Tuple[str, int]] = ()) -> Tuple[List[Tuple[str, List[int]]], str,
                                                                                      bool]:
    """Plan the twinkle DeviceGroups for an online-RL run. Pure function (no twinkle import).

    Args:
        nproc_per_node: the TRAINER's GPU count (== DistributedConfig.nproc_per_node, which sizes the
            model's data-parallel mesh in build_model).
        vllm_mode: 'colocate' shares the trainer's GPUs; anything else (None/'disaggregated', or its
            legacy alias 'server') disaggregates.
        sampler_world_size: the sampler's GPU count, computed per backend by :func:`_sampler_world_size`
            (vLLM tp*dp; SGLang tp*pp*dp).
        teacher_world_size: GPU count for a separate frozen teacher's OWN group (distillation, plan §3.3).
            ``0`` (the default, and every GRPO/RFT call) plans no teacher group, so those runs are unchanged;
            a ``disable_lora``/dynamic-self teacher reuses the student's actor and passes 0. Backward-compatible
            sugar for one ``(_TEACHER_GROUP, n)`` entry of ``auxiliary_groups``.
        auxiliary_groups: ``(name, world_size)`` for each FULL-PARAMETER frozen auxiliary model that needs its
            own GPUs -- a full-fine-tuning reference (``_REF_GROUP``), each seq_cls reward model
            (``_reward_group(i)``), a separate RLSD/SDAR teacher (plan §3.5). One disjoint group is appended
            per entry with ``world_size > 0``; ``name`` is the ``remote_group`` the caller builds that model
            with. A LoRA/``disable_lora`` auxiliary reuses the policy actor and is simply omitted here.

    Returns:
        ``(groups, sampler_remote_group, colocate)`` where ``groups`` is a list of
        ``(name, ranks)`` to hand twinkle.initialize, ``sampler_remote_group`` is the DeviceGroup the
        sampler is placed in, and ``colocate`` is the flag for CheckpointEngineManager. The teacher group
        (when planned) and every ``auxiliary_groups`` entry are appended last, disjoint, in argument order.

    Colocate puts trainer+sampler in ONE group over ``nproc_per_node`` GPUs (independent rank spaces);
    disaggregated appends a disjoint ``sampler`` group after the trainer's ranks. Every frozen auxiliary
    group is always disjoint (its resident weights never contend with the trainer/sampler), appended after
    them, so total GPUs = nproc_per_node + (sampler_world_size if disaggregated) + teacher_world_size +
    sum(world_size for each auxiliary group).
    """
    if nproc_per_node is None or nproc_per_node < 1:
        raise ValueError(f'nproc_per_node must be >= 1 (the trainer GPU count), got {nproc_per_node!r}.')
    if sampler_world_size < 1:
        raise ValueError(f'sampler_world_size must be >= 1, got {sampler_world_size}.')
    if teacher_world_size < 0:
        raise ValueError(f'teacher_world_size must be >= 0, got {teacher_world_size}.')

    if vllm_mode == 'colocate':
        if sampler_world_size > nproc_per_node:
            raise ValueError(f'colocate needs the sampler ({sampler_world_size} GPUs) to fit within the trainer '
                             f'GPUs ({nproc_per_node}); it shares them. Use vllm_mode="disaggregated" to '
                             'place the sampler on its own GPUs.')
        groups = [(_MODEL_GROUP, list(range(nproc_per_node)))]
        sampler_remote_group, colocate, next_rank = _MODEL_GROUP, True, nproc_per_node
    else:
        total = nproc_per_node + sampler_world_size
        groups = [
            (_MODEL_GROUP, list(range(nproc_per_node))),
            (_SAMPLER_GROUP, list(range(nproc_per_node, total))),
        ]
        sampler_remote_group, colocate, next_rank = _SAMPLER_GROUP, False, total

    # Append one disjoint group per frozen auxiliary model: the backward-compatible teacher sugar first,
    # then any caller-supplied groups (full-FT reference / seq_cls reward models / RLSD-SDAR teacher). A
    # LoRA/disable_lora auxiliary reuses the policy actor, so it is never listed and plans no group.
    reserved = {_MODEL_GROUP, sampler_remote_group}
    appended: List[str] = []
    teacher_entry = [(_TEACHER_GROUP, teacher_world_size)] if teacher_world_size > 0 else []
    for name, world_size in teacher_entry + list(auxiliary_groups):
        if world_size < 0:
            raise ValueError(f'auxiliary DeviceGroup {name!r} world_size must be >= 0, got {world_size}.')
        if world_size == 0:
            continue
        if name in reserved or name in appended:
            raise ValueError(f'auxiliary DeviceGroup name {name!r} collides with a reserved or already-planned '
                             f'group ({sorted(reserved | set(appended))}); each frozen model needs a unique name.')
        groups.append((name, list(range(next_rank, next_rank + world_size))))
        appended.append(name)
        next_rank += world_size
    return groups, sampler_remote_group, colocate


def _initialize_twinkle_rl(distributed_config: DistributedConfig,
                           groups: List[Tuple[str, List[int]]],
                           *,
                           seed: int = 42,
                           full_determinism: bool = False,
                           sequence_parallel_size: int = 1) -> None:
    """Initialize twinkle in Ray mode with the planned RL DeviceGroups.

    Online RL is Ray-only: the trainer and sampler are separate Ray actors the driver talks to (the
    GRPO loop runs on the driver and calls both), which local/torchrun mode cannot express.

    ``seed`` / ``full_determinism`` are forwarded to ``twinkle.initialize`` (see
    ``TrainAssembly.initialize_twinkle``): twinkle seeds the driver and re-seeds each Ray worker from
    the same values, so the configured TrainConfig.seed steers rollout + training reproducibly
    instead of being overwritten by twinkle's default.
    """
    import twinkle
    from twinkle import DeviceGroup

    from swift.dev.builders import build_ray_dp_mesh

    if distributed_config.mode != 'ray':
        raise ValueError("run_grpo requires DistributedConfig.mode='ray': the trainer and the rollout sampler are "
                         'separate Ray actors that the driver drives and syncs weights between. mode="local" has no '
                         'way to place two roles.')
    total = sum(len(ranks) for _, ranks in groups)
    twinkle.initialize(
        mode='ray',
        nproc_per_node=total,
        seed=seed,
        full_determinism=full_determinism,
        # Ray mode -- unlike local -- does NOT auto-install a global default DeviceMesh (twinkle.initialize
        # builds one only for mode='local'). A Ray object placed with device_mesh=None then falls back to
        # that missing global and is rejected ("Set device_mesh=DeviceMesh(...) to enable ray"). The trainer
        # model dodges this by passing an explicit mesh (build_ray_dp_mesh, via _apply_ray_placement), but
        # the rollout sampler is built with device_mesh=None and relies on the global. Install the SAME
        # mesh the model carries (pure-DP, or dp x ulysses under sequence_parallel_size) as the global
        # default so the colocated sampler inherits it -- which is exactly the mesh build_ray_dp_mesh's
        # contract says a colocated engine must share with the model it syncs weights from. Sized to
        # nproc_per_node (the trainer/colocate group); giving a disaggregated 'sampler' group its own
        # differently-sized mesh is a later-stage concern (rollout sampler universalization), not
        # exercised by the colocate default.
        global_device_mesh=build_ray_dp_mesh(distributed_config, sequence_parallel_size),
        # --ray_exp_name names this Ray run (cluster/worker-name prefix); online RL is always ray mode.
        name=distributed_config.ray_exp_name,
        groups=[DeviceGroup(name=group_name, ranks=ranks, device_type='GPU', gpus_per_worker=1)
                for group_name, ranks in groups])


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
                 sleep_level: int = 0):
        from twinkle.checkpoint_engine import CheckpointEngineManager

        from swift.dev.recipe._colocate import ColocateHandover

        # NB: deliberately does NOT call RolloutEngine.__init__ (which would build a fresh sampler);
        # the sampler is built and placed by run_grpo and injected here. It MUST still initialise the
        # sampler-independent streaming state (_episode_futures and friends), which poll/collect/cancel
        # read unconditionally -- _init_streaming_state is the shared initializer RolloutEngine.__init__
        # also calls, so the two cannot drift on which attributes a streaming rollout needs.
        self.model = model
        self.sampler = sampler
        self.template = template
        self._init_streaming_state()
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
        cache); see :class:`~swift.dev.recipe._colocate.ColocateHandover` for the sequence and why the two
        wakes are tag-disjoint. :meth:`finish_generate` reverses it after the rollout.
        """
        self._handover.enter()

    def finish_generate(self) -> None:
        """Reverse the colocate hand-over after a rollout, so the trainer can take the GPU back."""
        self._handover.exit()


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
    logging_config: Optional[LoggingConfig] = None,
    quantize_config: Optional[QuantizeConfig] = None,
    megatron_config: Optional[MegatronConfig] = None,
    moe_config: Optional[MoEConfig] = None,
    *,
    engine_args: Optional[Dict[str, Any]] = None,
    output_dir: str = 'output',
    _save_final: bool = True,
) -> List[dict]:
    """Assemble and run online GRPO with weight sync. Returns the loss/metric history.

    Placement is chosen from ``rollout_config.vllm_mode`` (see :func:`plan_rl_device_groups`):
    ``DistributedConfig.nproc_per_node`` is the TRAINER GPU count, and the sampler's GPUs
    (:func:`_sampler_world_size`, per backend) are placed alongside (disaggregated) or shared (colocate).
    """
    from swift.dev.builders import build_sampler
    from swift.dev.loss import configure_rlhf_loss, configure_rlhf_metrics
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
        output_dir=output_dir,
        logging_config=logging_config,
        quantize_config=quantize_config,
        megatron_config=megatron_config,
        moe_config=moe_config)
    # Also imports the run's plugin files -- the reward names handed to GRPOLoop below are resolved
    # against the registry they write into.
    assembly.prepare()

    backend = _sampler_backend(rollout_config)
    sampler_world_size = _sampler_world_size(rollout_config, backend)
    # Plan the DeviceGroups for any FULL-PARAMETER frozen auxiliary (RL_PLAN §3.5): the KL reference, an
    # RLSD/SDAR teacher, and each reward model is a separate Ray actor on its own GPUs. A LoRA/disable_lora
    # auxiliary reuses the policy actor and plans NO group, so _frozen_model_needs_group -- the exact
    # predicate _build_frozen_model branches on -- decides each, keeping the plan and the build in lockstep.
    # The names/world sizes planned here are the ones the build calls below pass as remote_group/world_size.
    from swift.dev.recipe.assembly import single_teacher_id
    teacher_model_id = single_teacher_id(rlhf_config.teacher_model, algo='RLSD/SDAR')
    ref_active = bool(rlhf_config.beta and rlhf_config.calculate_KL is not False)
    ref_disable_adapter = tuner_config is not None and not rlhf_config.ref_adapters
    ref_remote_group: Optional[str] = None
    auxiliary_groups: List[Tuple[str, int]] = []
    if ref_active and _frozen_model_needs_group(model_id=rlhf_config.ref_model, disable_adapter=ref_disable_adapter):
        ref_remote_group = _REF_GROUP
        auxiliary_groups.append((_REF_GROUP, 1))
    teacher_needs_group = _frozen_model_needs_group(
        model_id=teacher_model_id, disable_adapter=rlhf_config._teacher_use_disable_adapter)
    teacher_world_size = teacher_group_world_size(rlhf_config) if teacher_needs_group else 0
    teacher_remote_group = _TEACHER_GROUP if teacher_needs_group else None
    teacher_parallel_spec = rlhf_config.teacher_parallel_spec if teacher_needs_group else None
    for reward_index in range(len(rlhf_config.reward_model or [])):
        auxiliary_groups.append((_reward_group(reward_index), 1))
    # A frozen PRM model (a non-rule --prm item) is a full-parameter auxiliary too, so it gets its OWN
    # single-GPU group exactly like a reward model; _split_prm_channel is the one splitter both this
    # planning and _build_prm_channel use, so the planned name/world size and the build cannot drift.
    if _split_prm_channel(rlhf_config.prm)[1] is not None:
        auxiliary_groups.append((_prm_group(0), 1))
    groups, sampler_remote_group, colocate = plan_rl_device_groups(
        distributed_config.nproc_per_node,
        rollout_config.vllm_mode,
        sampler_world_size,
        teacher_world_size=teacher_world_size,
        auxiliary_groups=auxiliary_groups)
    # RL initializes twinkle itself rather than through the assembly: it needs two device groups (trainer
    # + sampler), whose placement was just planned.
    _initialize_twinkle_rl(
        distributed_config,
        groups,
        seed=train_config.seed,
        full_determinism=train_config.full_determinism,
        sequence_parallel_size=template_config.sequence_parallel_size)

    assembly.build_template()
    # Ulysses SP mesh (None unless sequence_parallel_size>1): planned before build_model so the trainable
    # policy is placed on the dp x ulysses mesh, the same one _initialize_twinkle_rl installed globally for
    # the colocated sampler and the one the dp_size batch-width formula below reads.
    assembly.plan_sp_mesh()
    # Trainer: a Ray-actor model in the 'model' group (build_model sets remote_group='model' under
    # mode='ray'), with the tuner applied before loss/optimizer so those target its group.
    assembly.build_model()
    configure_rlhf_loss(assembly.model, rlhf_config)
    # Prompts are rolled out, not iterated by a dataloader, so the step budget is derived from the prompt
    # set the way assembly derives it from a dataloader: one generation batch per rollout, exhausted after
    # num_train_epochs passes (B1). Load the prompts + design-B mini-batch width first, then size max_steps
    # (the LR-scheduler horizon) from them; an explicit --max_steps still overrides.
    prompts, prompt_extras = _prompt_rows_from_dataset(dataset_config)
    # Design-B mini-batch width: per_device_train_batch_size * dp_size, the same global-batch formula SFT
    # uses (builders/dataset._twinkle_loader_layout). Online RL is Ray-only and pure data-parallel over
    # nproc_per_node, so dp_size is read off build_ray_dp_mesh -- the exact mesh the trainer model was
    # placed on and slice_dp splits each forward_backward across, so every rank gets per_device rows.
    from swift.dev.builders import build_ray_dp_mesh
    from swift.dev.recipe.train_loop import resolve_rollout_max_steps, rollout_step_budget
    # data_world_size derives nproc/ulysses from the mesh, so under SP the dp_size (and thus the global
    # train_batch_size) counts SP peers as ONE data rank -- they share a sample and split its sequence.
    dp_size = build_ray_dp_mesh(distributed_config, template_config.sequence_parallel_size).data_world_size
    train_batch_size = train_config.per_device_train_batch_size * dp_size
    chord_features = _load_chord_features(rlhf_config, dataset_config, assembly.template)
    if chord_features and dp_size > 1:
        # CHORD rows are appended to the mini-batch tail and chord_count is one broadcast scalar, so only a
        # single DP rank can split RL vs SFT rows correctly; with dp>1 the tail lands on the last rank alone
        # and the others would mislabel real rollout rows as CHORD. Fail loudly rather than train wrong.
        raise ValueError(
            'CHORD auxiliary SFT (--chord_sft_dataset) is only correct on one DP rank: its rows are appended '
            'to the mini-batch tail while chord_count is broadcast as a single scalar, so with '
            f'dp_size={dp_size} the RL/SFT split would be wrong on all but the last rank. Run CHORD with '
            '--nproc_per_node 1, or drop --chord_sft_dataset.')
    max_steps = resolve_rollout_max_steps(
        train_config.max_steps,
        rollout_step_budget(
            num_prompts=len(prompts),
            num_generations=rlhf_config.num_generations,
            train_batch_size=train_batch_size,
            gradient_accumulation_steps=assembly.ga,
            num_train_epochs=train_config.num_train_epochs,
            generation_batch_size=rollout_config.generation_batch_size,
            num_iterations=rlhf_config.num_iterations),
        recipe='run_grpo')
    assembly.resolve_step_intervals(max_steps)
    configure_optimizer(
        assembly.model, train_config, num_training_steps=max_steps, distributed_config=distributed_config)
    # Register the GRPO-family policy metric (GRPOMetric/GSPOMetric/CISPOMetric by loss_type) AFTER the
    # optimizer group exists (add_metric appends onto it). twinkle then accumulates it inside forward_backward
    # from old_logps/advantages, so its ratio/clip/entropy stats ride calculate_metric into every step record.
    configure_rlhf_metrics(assembly.model, rlhf_config)

    # Sampler: a vLLM/SGLang sampler placed in its group (shared 'model' for colocate, separate 'sampler'
    # otherwise). The backend's memory-saver flag (forced on for colocate) is required for the device
    # hand-over.
    sampler_engine_args = _sampler_engine_args(rollout_config, engine_args, colocate, backend)
    if rlhf_config.enable_sampling_replay:
        # Sampling replay trains against the sampler's per-token support set, so the engine must export each
        # token's sampling mask. This is VLLMEngine's own construction flag (it derives vLLM's
        # enable_return_sampling_mask + processed_logprobs); set it here rather than in the shared
        # _sampler_engine_args because only the GRPO loop consumes sampling masks.
        sampler_engine_args['enable_sampling_replay'] = True
    sampler = build_sampler(
        model_config,
        backend=backend,
        engine_args=sampler_engine_args,
        template=assembly.template,
        remote_group=sampler_remote_group)
    rollout = SyncableRollout(
        assembly.model, sampler, assembly.template, colocate=colocate, sleep_level=rollout_config.sleep_level)
    # Multi-turn is requested by setting max_turns; the engine is twinkle-native (no scheduler, no
    # gym). Per-round length is sampling_params.max_tokens, whole-trajectory length is
    # max_trajectory_tokens. Tools are opt-in via RolloutConfig.tools: when set, a sandbox env pool is
    # built and each episode leases its own env (see swift.dev.rollout.sandbox); harness / followup_fn
    # remain caller-supplied extension points. The env pool is closed by rollout.shutdown().
    if rlhf_config.max_turns is not None:
        from swift.dev.rollout.sandbox import build_tool_sandbox
        env_pool, tool_plugins = build_tool_sandbox(rollout_config)
        rollout.configure_multi_turn(
            max_turns=rlhf_config.max_turns,
            max_trajectory_tokens=rlhf_config.max_trajectory_tokens,
            env_pool=env_pool,
            tool_plugins=tool_plugins)

    reward_model_plugins, reward_model_names = _build_reward_model_scorers(
        model_config, template_config, rlhf_config, distributed_config)
    # Process-reward (PRM) channel: the scorer plugin + its rule/model scorers, built like the outcome
    # reward above (a frozen PRM model gets its OWN prm_0 group, planned in the auxiliary_groups step).
    prm_scorer, prm_funcs, prm_model_plugins = _build_prm_channel(
        model_config, template_config, rlhf_config, distributed_config)
    # async_mode selects the driver, and -- since the per-sample stream now serves EVERY regime -- almost
    # every GRPO run rides it: 'none' runs StreamingGRPOLoop at max_staleness=0 (admit one window, DRAIN it,
    # train, publish -- the same per-sample control flow, no overlap), 'one_step_off' pins staleness to 1,
    # and 'fully_async' uses the configured max_staleness. The ONE exception is a feature that needs the
    # WHOLE fixed batch up front, which a per-sample stream cannot serve: dynamic_sample (regenerate
    # zero-variance groups after seeing rewards) and remax (a synchronous greedy-baseline pass over the
    # assembled batch). Those stay on GRPOLoop's synchronous fixed-batch _run_sync (validate._check_async_mode
    # refuses them under an overlapping mode, so they only ever reach _run_sync under 'none'). A colocated
    # sampler (sharing one device with the trainer) additionally sets serialize_generation for the
    # exclusive-device hand-over; a disaggregated one has its own GPUs and never hands over.
    loop_cls = GRPOLoop
    streaming_kwargs: Dict[str, Any] = {}
    sync_only = bool(rlhf_config.dynamic_sample or rlhf_config.advantage_estimator == 'remax')
    if rollout_config.async_mode != 'none' or not sync_only:
        from swift.dev.recipe.grpo_async import StreamingGRPOLoop
        loop_cls = StreamingGRPOLoop
        streaming_kwargs = {
            'max_staleness': (0 if rollout_config.async_mode == 'none' else 1
                              if rollout_config.async_mode == 'one_step_off' else rollout_config.max_staleness),
            'weight_sync_strategy': rollout_config.weight_sync_strategy,
            # in_place publishes by aborting every in-flight generation and resuming it on the freshly
            # overwritten weights, so an OVERLAPPING regime needs the sampler's partial-rollout loop; at
            # staleness 0 the drain empties the in-flight set first, so the abort is a no-op and the flag is
            # not required. adapter_snapshot pins each version by its own adapter path and never overwrites a
            # live copy, so the flag is inert there (validate enforces exactly this pairing per regime).
            'allow_partial_rollout': rollout_config.allow_partial_rollout,
            # The publish cadence: one weight publication per K optimizer steps (default 1).
            'parameter_sync_step': rollout_config.parameter_sync_step,
            # adapter_snapshot pins each trained version by its LoRA adapter path, so it needs the trained
            # adapter's name: dev applies a single trainable adapter named 'default' (the same name assembly
            # uses to save/load it). A full-parameter run has none -- refused under adapter_snapshot, and
            # served as merged base weights under in_place (which ignores adapter_name), so None is correct there.
            'adapter_name': ('default' if tuner_config is not None else None),
            # Exclusive-device serialization: only a colocated sampler (async_mode='none' on a shared device)
            # hands the one DeviceGroup between generation and training. A disaggregated sampler never does,
            # and an overlapping mode is disaggregated by validation, so this is False everywhere but colocate.
            'serialize_generation': colocate,
        }
    loop = loop_cls(
        assembly.model,
        rollout,
        prompts,
        prompt_extras=prompt_extras,
        reference=_build_frozen_model(
            model_config,
            assembly.template,
            assembly.model,
            distributed_config=distributed_config,
            model_id=rlhf_config.ref_model,
            remote_group=ref_remote_group,
            adapters=rlhf_config.ref_adapters,
            adapter_role='ref',
            disable_adapter=ref_disable_adapter)
        if ref_active else None,
        teacher=_build_frozen_model(
            model_config,
            assembly.template,
            assembly.model,
            distributed_config=distributed_config,
            model_id=teacher_model_id,
            remote_group=teacher_remote_group,
            adapters=rlhf_config.teacher_adapters,
            adapter_role='teacher',
            disable_adapter=rlhf_config._teacher_use_disable_adapter,
            world_size=teacher_world_size or 1,
            parallel_spec=teacher_parallel_spec),
        template=assembly.template,
        chord_features=chord_features,
        num_generations=rlhf_config.num_generations,
        reward_funcs=list(rlhf_config.orm) or None,
        reward_model_plugins=reward_model_plugins,
        reward_model_names=reward_model_names,
        reward_weights=rlhf_config.orm_weights,
        prm_scorer=prm_scorer,
        prm_funcs=prm_funcs,
        prm_model_plugins=prm_model_plugins,
        prm_weights=rlhf_config.prm_weights,
        advantage_estimator=rlhf_config.advantage_estimator,
        scale_rewards=rlhf_config.scale_rewards or 'group',
        rlhf_config=rlhf_config,
        max_steps=max_steps,
        gradient_accumulation_steps=assembly.ga,
        train_batch_size=train_batch_size,
        generation_batch_size=rollout_config.generation_batch_size,
        num_train_epochs=train_config.num_train_epochs,
        seed=train_config.seed,
        max_grad_norm=resolve_max_grad_norm(train_config),
        sampling_params=_grpo_sampling_params(rlhf_config, generation_config),
        logging_config=logging_config,
        output_dir=output_dir,
        save_steps=checkpoint_config.save_steps,
        no_save_optim=checkpoint_config.no_save_optim or checkpoint_config.save_only_model,
        no_save_rng=checkpoint_config.no_save_rng or checkpoint_config.save_only_model,
        safe_serialization=checkpoint_config.safe_serialization,
        max_shard_size=checkpoint_config.max_shard_size,
        save_total_limit=checkpoint_config.save_total_limit,
        manual_gc=bool(megatron_config and megatron_config.manual_gc),
        manual_gc_steps=megatron_config.manual_gc_steps if megatron_config else 0,
        **streaming_kwargs)
    assembly.loop = loop
    if assembly.resume_dir:
        loop.resume(assembly.resume_model())
    try:
        history = loop.fit()
        if _save_final:
            assembly.save_final()
        return history
    finally:
        rollout.shutdown()


#: The rollout engines that can back an independent-process, weight-synced sampler. The on-policy loop
#: pushes the trained policy into a co-resident engine through twinkle's CheckpointEngineManager, which
#: needs a CheckpointEngineMixin sampler; both vLLM and SGLang have one, TransformersSampler deliberately
#: does not (see :func:`_sampler_backend`).
_WEIGHT_SYNCABLE_SAMPLERS = ('vllm', 'sglang')


def _sampler_backend(rollout_config: RolloutConfig) -> str:
    """The engine backing the rollout sampler, from ``RolloutConfig.rollout_sampler``.

    Only vLLM and SGLang are weight-syncable, so only they can serve as an RL rollout engine: the loop
    syncs the trained policy into the sampler every step through a CheckpointEngineManager, which needs a
    CheckpointEngineMixin sampler -- TransformersSampler has none by design (it generates on the trainer's
    own weights, so there is nothing to sync into). ``rollout_sampler``'s Literal already excludes
    'transformers'; this is the loud backstop at the seam all the on-policy/distill recipes share, so a
    mis-set value fails here rather than silently running without weight sync.
    """
    backend = rollout_config.rollout_sampler
    if backend not in _WEIGHT_SYNCABLE_SAMPLERS:
        raise ValueError(
            f'rollout_sampler={backend!r} cannot back an online-RL rollout: weight sync into the sampler '
            f'needs a CheckpointEngineMixin engine, which only {list(_WEIGHT_SYNCABLE_SAMPLERS)} provide. '
            "Use rollout_sampler='vllm' or 'sglang'.")
    return backend


def _sampler_world_size(rollout_config: RolloutConfig, backend: str) -> int:
    """The rollout sampler's GPU count, per backend, for :func:`plan_rl_device_groups` to size its group.

    vLLM spreads a replica over ``tensor_parallel_size * data_parallel_size`` GPUs. SGLang additionally
    shards the model by pipeline stages, so its world size is ``tp * pp * dp`` -- omitting ``pp`` (the B3
    defect) would under-size the group and mis-place the engine.
    """
    if backend == 'sglang':
        return rollout_config.sglang_tp_size * rollout_config.sglang_pp_size * rollout_config.sglang_dp_size
    return rollout_config.vllm_tensor_parallel_size * rollout_config.vllm_data_parallel_size


def _sampler_engine_args(rollout_config: RolloutConfig, engine_args: Optional[Dict[str, Any]], colocate: bool,
                         backend: str = 'vllm') -> Dict[str, Any]:
    """Engine args for the rollout sampler, from RolloutConfig (+ caller overrides), for ``backend``.

    The whole backend-prefixed knob set is mapped by :func:`swift.dev.builders.build_engine_args` (it
    strips the ``vllm_``/``sglang_`` prefix off every RolloutConfig field the engine names), so each engine
    knob dev models reaches the sampler instead of only the handful the colocate schedule touches. Caller
    ``engine_args`` win over the config. Colocate then forces the backend's memory-saver flag (vLLM
    ``enable_sleep_mode`` / SGLang ``enable_memory_saver``): the schedule sleeps the sampler to free the GPU
    for the trainer between rollouts, which each engine only permits when built with it on.
    """
    from swift.dev.builders import build_engine_args
    # build_engine_args reads infer_config only on its transformers branch, which _sampler_backend rules
    # out, so None is safe here.
    kwargs: Dict[str, Any] = build_engine_args(backend, None, rollout_config)
    kwargs.update(engine_args or {})
    if colocate:
        kwargs['enable_sleep_mode' if backend == 'vllm' else 'enable_memory_saver'] = True
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


def _frozen_model_needs_group(*, model_id: Optional[str], disable_adapter: bool) -> bool:
    """Whether :func:`_build_frozen_model` builds a separate frozen actor that needs its own DeviceGroup.

    Mirrors ``_build_frozen_model``'s branch EXACTLY so the group PLANNING (run body) and the BUILD cannot
    drift: an adapter-disabled base reuses the student actor (no group), no ``model_id`` means no scorer at
    all (no group), and only a distinct full-parameter frozen copy is a separate Ray actor on its own GPUs
    (RL_PLAN §3.5: full-parameter auxiliaries are heterogeneous, LoRA ones share the policy).
    """
    return bool(model_id) and not disable_adapter


def _build_frozen_model(model_config: ModelConfig,
                        template: Any,
                        student: Any,
                        *,
                        distributed_config: DistributedConfig,
                        model_id: Optional[str],
                        remote_group: Optional[str],
                        adapters: Optional[List[str]] = None,
                        adapter_role: str = 'frozen',
                        disable_adapter: bool = False,
                        world_size: int = 1,
                        parallel_spec: Optional[str] = None) -> Any:
    """Build a frozen scoring owner, or the student's own adapter-disabled base.

    Returns a :class:`_teacher.Teacher` wrapper: ``DisableAdapterTeacher`` scores on the student actor with
    the LoRA adapter disabled (reusing the policy's own weights -- no separate model, no DeviceGroup),
    ``FrozenModelTeacher`` wraps a freshly built frozen model that is a separate Ray actor on its OWN
    ``remote_group`` DeviceGroup, and ``None`` means "no such scorer" (skipped by the loop). The frozen copy
    inherits the run's ``backend`` and is built ``mode='ray'`` via :func:`frozen_auxiliary_distributed_config`
    (RL_PLAN basic principles 1/2); ``world_size`` MUST equal the rank count ``plan_rl_device_groups``
    allocated to ``remote_group`` (build_model reads it as the group size), and ``parallel_spec`` lays out a
    sharded auxiliary (a distinct full-parameter copy is heterogeneous, plan §3.5).
    """
    from swift.dev.recipe._teacher import DisableAdapterTeacher, FrozenModelTeacher
    if disable_adapter:
        return DisableAdapterTeacher(student)
    if not model_id:
        return None
    from copy import copy

    from swift.dev.builders import build_model, frozen_auxiliary_distributed_config
    from swift.dev.recipe.assembly import configure_frozen_adapter

    config = copy(model_config)
    config.model = model_id
    aux_dist = frozen_auxiliary_distributed_config(distributed_config, world_size, parallel_spec=parallel_spec)
    # device_mesh drives the transformers placement; the megatron path ignores it and derives its own mesh
    # from aux_dist.parallel_spec, so setting both keeps the auxiliary correct on either backend.
    device_mesh = None
    if parallel_spec:
        from twinkle import DeviceMesh
        device_mesh = DeviceMesh.from_spec(parallel_spec)
    frozen = build_model(config, aux_dist, device_mesh=device_mesh, remote_group=remote_group)
    configured = configure_frozen_adapter(frozen, template, adapters or [], role=adapter_role)
    return FrozenModelTeacher(configured)


def _build_reward_model_scorers(model_config: ModelConfig, template_config: TemplateConfig, rlhf_config: RLHFConfig,
                                distributed_config: DistributedConfig) -> Tuple[List[Any], List[str]]:
    """Build each frozen reward model with its own tokenizer/template and adapt it to a batch scorer.

    Every reward model is a FULL-PARAMETER frozen seq_cls scorer, so each is a separate Ray actor on its own
    ``reward_<i>`` DeviceGroup (RL_PLAN §3.5), built with the run's backend via
    :func:`frozen_auxiliary_distributed_config`. The group names and world sizes (1 each) here MUST match
    what the run body planned through ``plan_rl_device_groups(auxiliary_groups=...)``.
    """
    reward_models = list(rlhf_config.reward_model or [])
    if not reward_models:
        return [], []

    from swift.dev.builders import frozen_auxiliary_distributed_config
    from swift.dev.reward import build_frozen_reward_model, build_reward_model_plugins

    count = len(reward_models)

    def _aligned(values, default, field):
        resolved = [default] * count if values is None else list(values)
        if len(resolved) != count:
            raise ValueError(f'{field} must contain exactly one value per reward_model.')
        return resolved

    model_types = _aligned(rlhf_config.reward_model_type, None, 'reward_model_type')
    revisions = _aligned(rlhf_config.reward_model_revision, None, 'reward_model_revision')
    template_names = _aligned(rlhf_config.reward_template, None, 'reward_template')
    adapters = _aligned(rlhf_config.reward_adapters or None, None, 'reward_adapters')

    models = []
    templates = []
    for index, (model_id, model_type, revision, template_name, adapter) in enumerate(
            zip(reward_models, model_types, revisions, template_names, adapters)):
        reward_model, reward_template = build_frozen_reward_model(
            model_id,
            model_config,
            template_config,
            model_type=model_type,
            revision=revision,
            template_name=template_name,
            adapter=adapter,
            distributed_config=frozen_auxiliary_distributed_config(distributed_config, 1),
            remote_group=_reward_group(index))
        models.append(reward_model)
        templates.append(reward_template)

    return build_reward_model_plugins(models, templates)


def _build_prm_channel(model_config: ModelConfig, template_config: TemplateConfig, rlhf_config: RLHFConfig,
                       distributed_config: DistributedConfig) -> Tuple[Any, List[Any], List[Any]]:
    """Build GRPO's process-reward (PRM) channel -> ``(prm_scorer, prm_funcs, prm_model_plugins)``.

    Three pieces, each mirroring an existing outcome-reward path so a PRM is built and placed the same way:

    - ``prm_funcs``: the rule items of ``--prm``, resolved through :func:`get_reward_funcs` exactly like
      ``orm`` rules (a rule scores the step's response prefix as one scalar).
    - ``prm_model_plugins``: the single frozen PRM model id, built by :func:`build_frozen_reward_model` on
      its OWN ``prm_0`` DeviceGroup (a full-parameter frozen seq_cls scorer, heterogeneous per RL_PLAN §3.5)
      -- the group name and world size (1) MUST match what the run body planned through
      ``plan_rl_device_groups(auxiliary_groups=...)``. ``prm_parallel_spec`` is synthesis-only (like
      ``orm_parallel_spec``); GRPO reward/PRM models are always a single-GPU group.
    - ``prm_scorer``: the :class:`~swift.dev.rewards.prm.PRMScorer` plugin that segments a response into
      steps and broadcasts each step's score, resolved from ``rlhf_config.prm_scorer`` (``None`` -> the
      default ``'delimiter'`` scorer). It is returned only when a PRM channel is actually wired, so a
      scorer with no rule/model -- which would score nothing -- is left ``None`` for validate to reject.

    Returns ``(None, [], [])`` when ``--prm`` is empty, so the loop's ``_prm_active`` stays False.
    """
    rule_specs, prm_model_id = _split_prm_channel(rlhf_config.prm)
    if not rule_specs and prm_model_id is None:
        return None, [], []

    from swift.dev.plugin import PluginRegistry
    from swift.dev.reward import build_frozen_reward_model, build_reward_model_plugins, get_reward_funcs
    from swift.dev.rewards import PRM_STEP_SCORER

    prm_funcs, _prm_rule_names = get_reward_funcs(rule_specs, rlhf_config) if rule_specs else ([], [])
    prm_model_plugins: List[Any] = []
    if prm_model_id is not None:
        from swift.dev.builders import frozen_auxiliary_distributed_config
        prm_model, prm_template = build_frozen_reward_model(
            prm_model_id,
            model_config,
            template_config,
            distributed_config=frozen_auxiliary_distributed_config(distributed_config, 1),
            remote_group=_prm_group(0))
        prm_model_plugins, _prm_model_names = build_reward_model_plugins([prm_model], [prm_template])

    prm_scorer = PluginRegistry.resolve(PRM_STEP_SCORER, rlhf_config.prm_scorer or 'delimiter', config=rlhf_config)
    return prm_scorer, prm_funcs, prm_model_plugins


def _load_chord_features(rlhf_config: RLHFConfig, dataset_config: DatasetConfig, template: Any) -> List[dict]:
    """Encode CHORD's expert SFT rows once; the loop cycles over these features."""
    if not rlhf_config.chord_sft_dataset:
        return []
    from copy import copy

    from swift.dev.builders import load_prompt_rows

    chord_config = copy(dataset_config)
    chord_config.dataset = list(rlhf_config.chord_sft_dataset)
    chord_config.val_dataset = []
    rows = load_prompt_rows(chord_config, None, split_dataset_ratio=0.0)
    features = [template.encode(row) for row in rows]
    if not features:
        raise ValueError('chord_sft_dataset produced no encodable rows.')
    return features


def _prompt_rows_from_dataset(dataset_config: DatasetConfig) -> Tuple[List[List[dict]], List[Dict[str, Any]]]:
    """Load prompt messages and preserve all non-message columns for rewards and teacher views."""
    from swift.dev.builders import load_prompt_rows

    rows = load_prompt_rows(dataset_config, None, split_dataset_ratio=0.0)
    if not rows:
        raise ValueError('run_grpo got an empty dataset. Set DatasetConfig.dataset with prompts to roll out on.')
    prompts: List[List[dict]] = []
    extras: List[Dict[str, Any]] = []
    for row in rows:
        messages = row.get('messages') if isinstance(row, dict) else None
        if not messages:
            continue
        if messages[-1].get('role') == 'assistant':
            messages = messages[:-1]
        prompts.append(list(messages))
        extras.append({key: value for key, value in row.items() if key != 'messages'})
    if not prompts:
        raise ValueError('run_grpo found no prompt messages in the dataset rows (expected a `messages` column).')
    return prompts, extras


def _prompts_from_dataset(dataset_config: DatasetConfig) -> List[List[dict]]:
    """Compatibility helper used by PPO/GKD, which only need message lists."""
    return _prompt_rows_from_dataset(dataset_config)[0]

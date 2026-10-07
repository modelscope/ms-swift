# Copyright (c) ModelScope Contributors. All rights reserved.
"""Placement dimension (RL_PLAN §四/⑥): where the trainer, the sampler and any frozen auxiliary actually
live, and how the trained policy reaches the sampler across that placement.

Two placements exist (``plan_rl_device_groups``): ``colocate`` shares ONE DeviceGroup between trainer and
sampler (independent rank spaces over the same GPUs) and hands the policy over with CUDA-IPC;
``disaggregated`` gives the sampler its OWN disjoint GPUs and pushes the policy with an NCCL broadcast of
the merged base weights (``config.validate``: "merged base weights over NCCL when disaggregated, CUDA IPC
when colocated"). The failure this dimension kills is a placement that silently mis-wires: overlapping
device groups (an NCCL collective between two actors on the SAME card conflicts/hangs), the wrong sync
transport for the placement, or an auxiliary model contending with the policy for process-level
``parallel_state``.

PLAN table -- every placement corner case, mapped to the test that owns it (this file implements the ONE
genuine e2e gap; the rest are already anchored elsewhere and are cross-referenced, not rewritten, per
RL_PLAN §二 "只补缺口，不重写"):

| # | case | input | expected (contract) | where covered | failure it kills |
|---|------|-------|---------------------|---------------|------------------|
| 1 | colocate shares one group, CUDA-IPC sync | ``vllm_mode='colocate'`` GRPO | trainer+sampler in ONE ``model`` group; policy handed over via CUDA-IPC | planning: ``test_recipes.py::test_plan_device_groups_colocate_shares_one_group``; e2e IPC delivery: ``test_backends_e2e.py`` megatron GRPO logs "Updated N base weights via IPC"; e2e per-window cadence: ``test_samplers_e2e.py::test_colocate_grpo_publishes_weights_every_step`` | weight-sync mode chosen wrong / colocate run never hands the device over |
| 2 | **disaggregated: disjoint groups, NCCL sync** | ``vllm_mode='disaggregated'`` + ``weight_sync_strategy='in_place'`` GRPO, nproc=1 | sampler on its OWN GPU (disjoint ``sampler`` group), policy pushed by NCCL each window, run trains on-policy | **THIS FILE** ``test_disaggregated_grpo_nccl_sync_disjoint_gpus`` (planning unit: ``test_recipes.py::test_plan_device_groups_heterogeneous_disjoint_ranks``) | overlapping device groups (NCCL same-card conflict) / the cross-GPU NCCL sync silently skipped |
| 3 | colocate that cannot fit the sampler | ``vllm_mode='colocate'``, sampler GPUs > trainer GPUs | ``ValueError`` pointing at ``disaggregated`` | ``test_recipes.py::test_plan_device_groups_validation`` | silent mis-placement instead of a loud refusal |
| 4 | frozen auxiliary owns a disjoint group | separate teacher / value model | the auxiliary is a Ray actor on its OWN group, never contending with the policy's ``parallel_state`` | ``test_algorithms_e2e.py::test_gkd_e2e_separate_teacher`` (separate teacher, accel(2)) and ``::test_ppo_*`` (critic); planning: ``test_recipes.py::test_plan_device_groups_heterogeneous_disjoint_ranks`` | auxiliary shares the policy's process-level ``parallel_state`` and corrupts it |

Why row 2 is the only NEW e2e here: rows 1/3/4 already run end to end (or are pure-function contracts) in
the files cited above -- ``colocate`` is the default every other RL e2e in this suite rides, the oversize
``ValueError`` and the disjoint-rank planning are pure functions with no I/O to drive, and a frozen
teacher/critic on its own group is exactly what the GKD/PPO e2e tests already place. The disaggregated
SAMPLER -- a second GPU owning the rollout engine, reached by an NCCL broadcast instead of an IPC hand-over
-- is exercised by no other test, so it is the gap this file fills.

Reverse-verification: ``test_disaggregated_grpo_nccl_sync_disjoint_gpus`` goes red if the sampler group is
planned onto the trainer's ranks (the NCCL broadcast between two same-card actors conflicts, so the run
crashes instead of training), and red if the per-window publication is skipped (``stream_publishes`` stays
flat at 0 -- a disaggregated run has NO initial ``_enter_generation`` hand-over, so a skipped publish means
the sampler never receives the trained policy at all). Its anti-degeneracy guard additionally goes red if a
future change silently downgrades ``vllm_mode`` to colocate or makes the two device groups overlap, so the
test cannot pass while riding the colocate path it claims not to. Run with ``CUDA_VISIBLE_DEVICES=<two
cards> pytest swift/dev/tests/feature/rl/test_placement_e2e.py -m slow``.
"""
import pytest

from swift.dev.tests.feature.rl.conftest import rl_configs, run_rl


@pytest.mark.slow
@pytest.mark.accel(2)
def test_disaggregated_grpo_nccl_sync_disjoint_gpus(tiny_qwen2_5, rl_data, tmp_path, assert_rl_trained):
    """Row 2: a disaggregated GRPO run puts the sampler on its OWN GPU and NCCL-syncs the policy each window.

    ``vllm_mode='disaggregated'`` makes ``plan_rl_device_groups`` append a ``sampler`` group over ranks
    ``[nproc, nproc + sampler_world_size)`` -- DISJOINT from the trainer's ``model`` group ``[0, nproc)`` --
    so with ``nproc=1`` the trainer owns one GPU and the vLLM sampler the other (this test is ``accel(2)``).
    ``CheckpointEngineManager(mode='standalone')`` then wires the NCCL transport, and
    ``weight_sync_strategy='in_place'`` is what actually DRIVES it: each driver publish calls
    ``SyncableRollout.sync_weights`` -> ``ColocateHandover.enter`` (which, with ``colocate=False``, reduces to
    ``manager.sync_weights(merge_and_sync=True)`` -- a plain NCCL broadcast of the LoRA-merged base weights,
    no device hand-over). ``adapter_snapshot`` would instead pin per-version adapter paths and never touch
    the NCCL broadcast, so ``in_place`` is set explicitly to exercise the transport the plan names for a
    disaggregated run.

    Two oracles, both anchored to the placement rather than to a self-consistent round trip:

    * the run TRAINS on-policy across the disjoint layout -- ``assert_rl_trained`` reads the checkpoint back
      from disk and requires a non-zero ``lora_B`` (a real policy-gradient step), and ``version_span_mean`` is
      ``0`` on every step (``async_mode='none'`` drains each window before publishing, so no sample is trained
      against a newer policy than it was generated under -- the disaggregated run is still strictly
      on-policy). Had the two device groups overlapped, the NCCL collective between two actors on one card
      would conflict and the run would crash rather than reach these assertions.
    * the NCCL sync fires EVERY generation window, not once -- ``stream_publishes`` must reach ``>= 2``. The
      geometry is chosen so the run spans three windows: one GRPO window is exactly one group
      (``groups_per_partition = ceil(parameter_sync_step * num_generations / num_generations) = 1``) and one
      group of ``num_generations=2`` rows trains ``2 / (per_device_train_batch_size * ga * dp) = 2`` optimizer
      steps, so ``max_steps=6`` needs three windows and the driver must re-broadcast the policy twice
      (``[0,0,1,1,2,2]``). This is the disaggregated analogue of ``test_samplers_e2e`` case 3, and it is the
      sharper half of the W14 guard here: a DISAGGREGATED run performs NO initial ``_enter_generation``
      hand-over (that is colocate-only, ``serialize_generation``), so if the per-window publish were skipped
      the sampler would keep its disk-loaded base weights for the WHOLE run -- a flat ``stream_publishes``
      is therefore a stale-behaviour-policy bug, not a benign no-op.
    """
    configs = rl_configs(
        rlhf_type='grpo',
        model=tiny_qwen2_5,
        model_type='qwen2',
        template='qwen2_5',
        dataset=rl_data.prompt_only(n=6),
        out_dir=str(tmp_path / 'out_disagg_grpo'),
        nproc=1,
        max_steps=6,
        rlhf_over={'num_generations': 2, 'orm': [_varied_content_reward]},
        rollout_over={'vllm_mode': 'disaggregated', 'weight_sync_strategy': 'in_place'},
        generation_over={'temperature': 1.0, 'top_p': 1.0, 'max_new_tokens': 16},
    )
    history = run_rl(configs)

    # Anti-degeneracy guard (RL_PLAN DoD "非退化" / testcase-planning step 8): every e2e assertion below
    # (the run trains, version_span_mean==0, stream_publishes>=2) ALSO holds under a colocate run, so on
    # their own they do not prove this run really took the disaggregated (separate-GPU, NCCL) path rather
    # than silently degrading to colocate. Pin the placement from the RESOLVED config -- the same
    # ``plan_rl_device_groups`` run_grpo calls, fed this run's own nproc and its derived sampler world size
    # -- and require the trainer/sampler groups to be DISJOINT with ``colocate=False``. (The real NCCL
    # transport itself is evidenced at runtime by the driver logs: a 2-rank ``ncclCommInitRankConfig`` and
    # ``Rank 0 send weights done`` / ``receive_weights done: rank=1`` per window, absent for a colocate run
    # which logs ``IPC checkpoint engine`` instead.)
    from swift.dev.recipe.run_grpo import _MODEL_GROUP, _sampler_backend, _sampler_world_size
    from swift.dev.recipe.run_grpo import plan_rl_device_groups

    rollout_config = configs['rollout_config']
    assert rollout_config.vllm_mode == 'disaggregated', (
        f'the resolved run downgraded vllm_mode to {rollout_config.vllm_mode!r}: this test must exercise the '
        f'disaggregated placement (sampler on its OWN GPUs, policy pushed by NCCL), not a colocate hand-over')
    sampler_ws = _sampler_world_size(rollout_config, _sampler_backend(rollout_config))
    groups, sampler_group, colocate = plan_rl_device_groups(
        configs['distributed_config'].nproc_per_node, rollout_config.vllm_mode, sampler_ws)
    assert not colocate, 'plan_rl_device_groups resolved colocate=True for a disaggregated run'
    ranks = dict(groups)
    model_ranks, sampler_ranks = set(ranks[_MODEL_GROUP]), set(ranks[sampler_group])
    assert model_ranks.isdisjoint(sampler_ranks), (
        f'trainer {sorted(model_ranks)} and sampler {sorted(sampler_ranks)} device groups OVERLAP: an NCCL '
        f'broadcast between two actors on the SAME card conflicts, so a disjoint layout is required')

    assert_rl_trained(
        history, configs, 'grpo', max_loss=20.0, require_keys=('stream_publishes', 'version_span_mean'))

    spans = [r['version_span_mean'] for r in history]
    assert all(s == 0 for s in spans), (
        f'a disaggregated async_mode="none" run must stay strictly on-policy (every sample trained at the '
        f'version it was generated under), but version_span_mean was non-zero: {spans}')

    publishes = [r['stream_publishes'] for r in history]
    assert all(b >= a for a, b in zip(publishes, publishes[1:])), \
        f'stream_publishes is a monotone lifetime counter but went backwards: {publishes}'
    assert publishes[-1] >= 2, (
        f'stream_publishes only reached {publishes[-1]} across a 3-window disaggregated run -- the policy was '
        f'not NCCL-broadcast into the separate-GPU sampler per window (a skipped publish leaves the sampler on '
        f'its disk-loaded base weights for the whole run, since a disaggregated run has no initial hand-over): '
        f'{publishes}')
    assert publishes[-1] > publishes[0], \
        f'stream_publishes never advanced across the run -- the policy was not re-synced per window: {publishes}'


def _varied_content_reward(completions, **kwargs):
    """A deterministic ORM reward that VARIES with completion CONTENT (char-sum mod 11, scaled to [0,1)).

    GRPO's advantage is ``(r - group_mean) / group_std``, so a constant reward across a prompt's group gives
    std=0 -> advantage 0 -> zero gradient -> ``lora_B`` never moves and ``assert_rl_trained`` would fail for a
    reason unrelated to placement. Keying the reward on content gives most groups a non-zero std so the run
    really trains (mirrors ``test_algorithms_e2e``/``test_samplers_e2e``).
    """
    return [float(sum(ord(ch) for ch in c) % 11) / 11.0 for c in completions]

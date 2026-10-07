# Copyright (c) ModelScope Contributors. All rights reserved.
"""Adversarial cross-cutting dimension -- infeasible combinations must fail loudly (RL_PLAN §四/§五).

The contract under test is NOT "these configs train" but the opposite: a combination the system cannot
honour must be REJECTED at config-validation time with a message that points at the correct approach,
rather than silently degrading (basic principle G3 -- no temporary fallbacks, no "unsupported" silence --
and basic principle 1 -- backend equivalence, with in-place ``generate`` the one TransformersModel-only
capability). A silent degradation is the failure every case here kills: it would let a run "succeed" while
doing something other than what was asked.

These drive the REAL production validation entry -- ``process_and_validate_configs``, the first thing
``cli.rlhf.run_rlhf_configs`` calls -- and stop there. An infeasible combo is rejected during validation,
BEFORE ``run_rlhf`` builds any model/sampler, so this exercises the exact guard path the ``swift rlhf``
command hits without needing a GPU or a Ray session. Each case is reverse-verifiable: delete the guard and
the config validates cleanly (the ``pytest.raises`` goes red), restore it and it is refused again.

The five combinations (RL_PLAN §四 last row):

1. megatron x in-place generate   -- ``MegatronModel.generate`` refuses by design (its weights are sharded
   across TP/PP under Megatron names, a layout no inference engine reads). In RL this is doubly unreachable:
   the rollout sampler is restricted to the weight-syncable engines (case 2), so no RL path ever asks a
   policy to generate in place; the model-level refusal is the hard backstop that makes that safe.
2. transformers sampler x weight sync -- an in-process TransformersSampler has no CheckpointEngineMixin, so
   the per-step policy->sampler weight sync cannot happen; refused rather than run without syncing.
3. unsloth x megatron             -- unsloth is wired only into the transformers build; the megatron build
   ignores ``tuner_backend`` entirely, so this would silently train a stock peft LoRA and drop the unsloth
   kernels. Refused (the guard this suite added to ``_check_unsloth_strategy``).
4. colocate x async overlap       -- overlapping generation with training needs the sampler on its own GPUs;
   a colocated sampler time-shares one DeviceGroup and cannot overlap.
5. async GRPO without off-policy correction -- any staleness > 0 makes the trained batch off-policy, and
   GRPO trains on raw sampled tokens, so it requires ``rollout_importance_sampling_mode``.

Run with ``pytest swift/dev/tests/feature/rl/test_infeasible_combos.py`` (fast, no GPU).
"""
import pytest

from swift.dev.tests.feature.rl.conftest import _force_rl_placement, rl_configs


def _validate_only(configs):
    """Run the production RL config validation and stop before anything is built.

    ``process_and_validate_configs`` is exactly what ``run_rlhf_configs`` calls first; an infeasible combo
    raises inside it, so the heavy ``run_rlhf`` build that follows is never reached. ``_force_rl_placement``
    replicates the CLI argv parser's Ray/vLLM placement forcing (which a programmatic caller bypasses) so
    the config validated here is the one ``swift rlhf`` would validate.
    """
    from swift.dev.config import process_and_validate_configs
    _force_rl_placement(configs)
    process_and_validate_configs(configs)


def test_megatron_refuses_in_place_generate():
    """Case 1: ``MegatronModel`` refuses in-place generation instead of approximating it.

    The refusal is the design guarantee that keeps backend equivalence safe (basic principle 1: generate is
    the ONE TransformersModel-only capability). ``generate``/``generate_stream`` are plain methods (not
    ``@remote_function``) whose body raises before touching ``self`` -- a bare instance therefore exercises
    the exact refusing code with no GPU/Ray build, which is the faithful unit of this contract.
    """
    try:
        from twinkle.model.megatron import MegatronModel
    except Exception as exc:  # megatron-core / transformer-engine absent
        pytest.skip(f'megatron backend not importable in this env: {exc}')
    instance = MegatronModel.__new__(MegatronModel)  # __init__ not run: the refusal never reads self
    for method in ('generate', 'generate_stream'):
        with pytest.raises(NotImplementedError, match='cannot generate in place'):
            getattr(instance, method)()


def test_transformers_sampler_cannot_back_weight_synced_rollout(tiny_qwen2_5, rl_data, tmp_path):
    """Case 2: an in-process transformers sampler cannot back an online-RL rollout (no weight sync).

    The online loop syncs the trained policy into the sampler every step through a CheckpointEngineManager,
    which needs a CheckpointEngineMixin engine; TransformersSampler deliberately has none (it generates on
    the trainer's own weights). Setting it anyway must fail loudly, not silently run a rollout that never
    receives the updated weights (which would train against a stale behaviour policy with no correction).
    """
    configs = rl_configs(
        rlhf_type='grpo',
        model=tiny_qwen2_5,
        model_type='qwen2',
        template='qwen2_5',
        dataset=rl_data.prompt_only(n=2),
        out_dir=str(tmp_path / 'out_sampler'),
        max_steps=1,
        rollout_over={'rollout_sampler': 'transformers'},
    )
    with pytest.raises(ValueError, match='cannot back an online-RL rollout'):
        _validate_only(configs)


def test_unsloth_rejected_on_megatron_backend(tiny_qwen2_5, rl_data, tmp_path):
    """Case 3: ``tuner_backend='unsloth'`` on the megatron backend is refused, not silently downgraded.

    unsloth is wired ONLY into the transformers build (``builders/model.py`` swaps in ``UnslothModel``);
    ``_build_megatron_model`` never reads ``tuner_backend`` and ``adapter._build_adapter_config`` keys off
    ``tuner`` alone, so without the guard a megatron run would build a plain MegatronModel + a stock peft
    LoRA and silently drop the unsloth kernels the user asked for. ``optim`` is reset to its default
    (``adamw_torch_fused``) because ``rl_configs`` otherwise sets the transformers-only ``optim='adamw'``,
    which ``_check_backend_specific`` would reject first and mask the unsloth guard this case targets.

    The negative control proves the refusal is megatron-SPECIFIC, not a blanket unsloth rejection: the same
    tuner on the transformers backend is not refused by this guard (unsloth is a transformers construct).
    """
    common = dict(
        rlhf_type='dpo',
        model=tiny_qwen2_5,
        model_type='qwen2',
        template='qwen2_5',
        dataset=rl_data.preference(n=2),
        max_steps=1,
        train_over={'optim': 'adamw_torch_fused'},
        tuner_over={'tuner_backend': 'unsloth'},
    )
    megatron_cfg = rl_configs(out_dir=str(tmp_path / 'out_unsloth_megatron'), backend='megatron', **common)
    with pytest.raises(NotImplementedError, match='cannot run on the megatron backend'):
        _validate_only(megatron_cfg)

    # Negative control: backend=None is the transformers backend, where unsloth is legitimate. Validation
    # must NOT refuse it with the megatron guard (it may pass outright, or raise something unrelated -- the
    # assertion is only that the megatron-specific refusal does not fire on transformers).
    hf_cfg = rl_configs(out_dir=str(tmp_path / 'out_unsloth_hf'), backend=None, **common)
    try:
        _validate_only(hf_cfg)
    except NotImplementedError as exc:
        assert 'megatron backend' not in str(exc), \
            f'unsloth wrongly refused on the transformers backend (guard is not megatron-specific): {exc}'


def test_colocate_async_overlap_rejected(tiny_qwen2_5, rl_data, tmp_path):
    """Case 4: an overlapping async regime on a colocated sampler is refused (cannot overlap one device).

    ``async_mode='one_step_off'`` overlaps generation with training, which needs the sampler generating on
    its OWN GPUs while the trainer advances; ``vllm_mode='colocate'`` time-shares ONE DeviceGroup between
    trainer and sampler, so the two cannot run at once. ``rollout_importance_sampling_mode`` is set so the
    GRPO off-policy guard (case 5) passes first and this case isolates the colocate rejection.
    """
    configs = rl_configs(
        rlhf_type='grpo',
        model=tiny_qwen2_5,
        model_type='qwen2',
        template='qwen2_5',
        dataset=rl_data.prompt_only(n=2),
        out_dir=str(tmp_path / 'out_colocate_async'),
        max_steps=1,
        rollout_over={'async_mode': 'one_step_off', 'vllm_mode': 'colocate'},
        rlhf_over={'rollout_importance_sampling_mode': 'token_truncate'},
    )
    with pytest.raises(ValueError, match='needs the sampler on its own GPUs'):
        _validate_only(configs)


def test_async_grpo_requires_importance_sampling(tiny_qwen2_5, rl_data, tmp_path):
    """Case 5: async GRPO without ``rollout_importance_sampling_mode`` is refused (off-policy uncorrected).

    Any staleness > 0 trains on a rollout batch produced by an older policy, i.e. off-policy. GRPO trains on
    the raw sampled tokens, so it needs the token-level importance-sampling correction; without it the update
    is silently biased. ``vllm_mode='disaggregated'`` is set so the colocate guard (case 4) does not fire
    first, isolating the missing-IS-mode rejection. (PPO's clipped surrogate already bounds the update, so
    this requirement is GRPO-specific.)
    """
    configs = rl_configs(
        rlhf_type='grpo',
        model=tiny_qwen2_5,
        model_type='qwen2',
        template='qwen2_5',
        dataset=rl_data.prompt_only(n=2),
        out_dir=str(tmp_path / 'out_async_noIS'),
        max_steps=1,
        rollout_over={'async_mode': 'one_step_off', 'vllm_mode': 'disaggregated'},
    )
    with pytest.raises(ValueError, match='requires off-policy correction'):
        _validate_only(configs)

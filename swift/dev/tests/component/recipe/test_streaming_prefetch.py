"""Streaming recipes must leave generation work in flight when a training batch is handed off."""

from importlib import import_module
from types import SimpleNamespace

import pytest

from swift.dev.config import RewardConfig, RLConfig, RolloutConfig
from swift.dev.recipe.batching import streaming_batch_geometry


@pytest.mark.parametrize(
    "need_rows,generations,sync_steps,lag,expected",
    [
        (64, 8, 1, 0, (8, 64)),
        (64, 8, 1, 1, (8, 128)),
        (64, 8, 1, 2, (8, 192)),
        (64, 8, 2, 1, (16, 256)),
        (4, 3, 1, 1, (4, 24)),
        (4, 3, 2, 1, (4, 24)),
        (4, 3, 4, 1, (8, 48)),
        (4, 8, 1, 1, (1, 16)),
        (16, 1, 1, 1, (16, 32)),
    ],
)
def test_capacity_covers_complete_updates_and_generation_groups(need_rows, generations, sync_steps, lag, expected):
    assert (
        streaming_batch_geometry(
            need_rows=need_rows,
            num_generations=generations,
            parameter_sync_step=sync_steps,
            max_staleness=lag,
        )
        == expected
    )


@pytest.fixture
def driver_factory(monkeypatch):
    import twinkle_agentic.async_rl as kernel

    real_driver = kernel.StreamingDriver
    active, submissions, created = set(), [], []
    availability = {"all": False}

    def submit(prompt, index, policy):
        handle = len(submissions)
        submissions.append(handle)
        active.add(handle)
        return handle

    def collect(handle):
        active.remove(handle)
        return SimpleNamespace(source_handle=handle)

    def construct(**kwargs):
        # Keep the recipe's geometry, pull rule and real iterator/commit bridge. Replace only
        # remote execution: the first batch finishes, while prefetched requests remain in flight.
        kwargs.update(
            rollout=None,
            score=None,
            resource_adapter=None,
            submit=submit,
            poll=lambda handles: list(handles) if availability["all"] else [h for h in handles if h < 64],
            collect=collect,
        )
        driver = real_driver(**kwargs)
        created.append(driver)
        return driver

    monkeypatch.setattr(kernel, "StreamingDriver", construct)

    def build(algorithm, mode="one_step_off", sync_steps=1, complete_all=False, steps=1):
        availability["all"] = complete_all
        recipe = "distill" if algorithm in ("gkd", "opsd", "mopd") else algorithm
        module = import_module(f"swift.dev.recipe.{recipe}.assembly")
        builder = getattr(
            module, {"grpo": "build_driver", "ppo": "build_ppo_driver", "distill": "build_distill_driver"}[recipe]
        )
        kwargs = {
            "cluster": SimpleNamespace(model=None, rollout=None, worker=None, adapter=None, orm_scorers=[]),
            "rl_config": RLConfig(rl_type=algorithm, num_generations=8, lmbda=1),
            "rollout_config": RolloutConfig(
                async_mode=mode,
                max_staleness=2 if mode == "fully_async" else 1,
                weight_sync_strategy="in_place",
                allow_partial_rollout=mode != "none",
                parameter_sync_step=sync_steps,
            ),
            "tuner_config": None,
            "train_batch_size": 4,
            "gradient_accumulation_steps": 16,
            "prompt_dataloader": list(range(64)),
            "num_train_epochs": 1,
            "max_steps": steps,
            "output_dir": "unused",
        }
        if recipe == "grpo":
            kwargs["reward_config"] = RewardConfig()
        return builder(**kwargs)

    yield build, active, submissions
    for driver in created:
        driver.abort()


@pytest.mark.parametrize("algorithm", ["grpo", "ppo", "gkd", "opsd", "mopd"])
def test_one_step_off_keeps_next_batch_generating_during_training(driver_factory, algorithm):
    build, active, submissions = driver_factory
    driver = build(algorithm)
    batch = next(iter(driver))
    assert len(batch) == 64
    assert len(submissions) == 128
    assert len(active) == 64, "Training must not begin with an empty generation pipeline."
    assert driver._max_staleness == 1
    driver.commit(1)


@pytest.mark.parametrize("algorithm", ["grpo", "ppo", "gkd", "opsd", "mopd"])
@pytest.mark.parametrize("mode,expected_depth", [("none", 64), ("fully_async", 192)])
def test_other_regimes_keep_their_generation_capacity(driver_factory, algorithm, mode, expected_depth):
    build, active, submissions = driver_factory
    driver = build(algorithm, mode=mode)
    assert len(next(iter(driver))) == 64
    assert len(submissions) == expected_depth
    assert len(active) == expected_depth - 64
    driver.commit(1)


@pytest.mark.parametrize("algorithm", ["grpo", "ppo", "gkd", "opsd", "mopd"])
def test_sparse_publication_capacity_uses_optimizer_steps_not_prompt_groups(driver_factory, algorithm):
    build, active, submissions = driver_factory
    driver = build(algorithm, sync_steps=2)
    assert len(next(iter(driver))) == 64
    assert len(submissions) == 256
    assert len(active) == 192
    driver.commit(1)


@pytest.mark.parametrize("algorithm", ["grpo", "ppo", "gkd", "opsd", "mopd"])
def test_completed_prefetch_is_not_merged_into_one_training_handoff(driver_factory, algorithm):
    build, active, submissions = driver_factory
    driver = build(algorithm, complete_all=True, steps=2)
    iterator = iter(driver)
    first = next(iterator)
    assert len(first) == 64
    assert [sample.source_handle for sample in first] == list(range(64))
    driver.request_publish(lambda window: None)
    driver.commit(1)
    second = next(iterator)
    assert len(second) == 64
    assert [sample.source_handle for sample in second] == list(range(64, 128))
    assert driver.stats.publishes == 1
    driver.commit(1)
    driver.abort()
    assert driver.stats.optimizer_steps == driver.stats.train_pulls == 2
    assert driver.stats.consumed == 128
    assert driver.stats.off_policy_consumed == 64

"""Streaming recipes must leave generation work in flight when a training batch is handed off."""

from importlib import import_module
from types import SimpleNamespace

import pytest

from swift.dev.config import RewardConfig, RLConfig, RolloutConfig
from swift.dev.recipe.batching import publish_rollout_weights, streaming_batch_geometry


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


def test_in_place_publish_keeps_partial_rollout_parked_until_weights_arrive():
    import asyncio
    import threading

    from twinkle.data_format import SampledSequence, SampleResponse, SamplingParams
    from twinkle.sampler.partial_rollout import PartialRolloutMixin

    sampler = PartialRolloutMixin()
    sampler.template = SimpleNamespace(decode=lambda tokens: str(tokens))
    loop = asyncio.new_event_loop()
    sampler._async_loop = loop
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    started, seen_versions = threading.Event(), []
    version = [0]

    async def rollout():
        aborted = asyncio.Event()

        async def abort(request_id):
            aborted.set()

        sampler._abort_request = abort

        async def attempt(feature, state, params, index):
            seen_versions.append(version[0])
            if index == 0:
                sampler._register_inflight("pending")
                started.set()
                try:
                    await aborted.wait()
                finally:
                    sampler._unregister_inflight("pending")
            response = SampleResponse(
                prompt_token_ids=[99],
                sequences=[
                    SampledSequence(
                        stop_reason="abort" if index == 0 else "stop",
                        tokens=[index + 1],
                        decoded="",
                        new_input_feature={"input_ids": [99, 1]},
                    )
                ],
            )
            return response, None

        return await sampler._run_partial_rollout(
            {"input_ids": [99]},
            SamplingParams(max_tokens=8),
            run_attempt=attempt,
            allow_partial_rollout=True,
        )

    future = asyncio.run_coroutine_threadsafe(rollout(), loop)

    def transfer():
        assert sampler._generation_paused
        assert not future.done()
        assert seen_versions == [0]
        version[0] = 1

    try:
        assert started.wait(timeout=5)
        publish_rollout_weights(
            SimpleNamespace(
                sampler=sampler,
                engine=SimpleNamespace(sync_weights=transfer),
                adapter=None,
            )
        )
        outcome = future.result(timeout=5)
        assert seen_versions == [0, 1]
        assert outcome.response.sequences[0].tokens == [1, 2]
        assert outcome.resumed_partial_output
    finally:
        if not future.done():
            future.cancel()
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()


def test_failed_publish_does_not_resume_generation():
    events = []

    def transfer():
        events.append("transfer")
        raise RuntimeError("transfer failed")

    cluster = SimpleNamespace(
        adapter=None,
        engine=SimpleNamespace(sync_weights=transfer),
        sampler=SimpleNamespace(
            pause_generation=lambda: events.append("pause"),
            resume_generation=lambda: events.append("resume"),
        ),
    )
    with pytest.raises(RuntimeError, match="transfer failed"):
        publish_rollout_weights(cluster)
    assert events == ["pause", "transfer"]


def test_adaptive_publication_uses_its_own_pause_transaction():
    events = []
    publish_rollout_weights(SimpleNamespace(adapter=SimpleNamespace(publish=lambda: events.append("publish"))))
    assert events == ["publish"]

"""GRPO submits optimizer batches; Twinkle owns their micro-batches."""

import importlib
from types import SimpleNamespace

import pytest

from swift.dev.config import RLConfig
from swift.dev.recipe.utils import (
    rollout_step_budget,
    training_progress,
)

grpo = importlib.import_module("swift.dev.recipe.grpo.run_grpo")


class RecordingModel:
    def __init__(self, dp):
        self.device_mesh = SimpleNamespace(data_world_size=dp)
        self.calls, self.steps = [], 0

    def forward_backward(self, **kwargs):
        assert kwargs["gradient_accumulation_steps"] == 1
        self.calls.append(kwargs)

    def clip_grad_and_step(self, **kwargs):
        assert kwargs["gradient_accumulation_steps"] == 1
        self.steps += 1


def make_state(ga=4, dp=2, max_steps=100):
    state = SimpleNamespace(
        model=RecordingModel(dp), ga=ga, global_step=0, micro_step=0, max_grad_norm=1.0, max_steps=max_steps, saved=[]
    )
    state.reached_max = lambda: state.global_step >= state.max_steps
    state.should_stop = lambda: False

    def record():
        state.global_step += 1

    state.record = record
    state.step_tail = lambda save=None: save(str(state.global_step)) if save else None
    return state


def train(state, count, *, config=None, chord_fn=None, ema=None, offset=0, batching_kwargs=None):
    rows = [
        SimpleNamespace(
            encoded={"input_ids": [i], "labels": [i], "routed_experts": [i]},
            rollout_logps=[i + 0.1],
            truncated=i % 2 == 0,
            sampling_mask=f"mask-{i}",
        )
        for i in range(offset, offset + count)
    ]
    return grpo._train_mini_batches(
        state,
        config or RLConfig(rl_type="grpo"),
        rows,
        train_batch_size=2 * state.model.device_mesh.data_world_size,
        batching_kwargs=batching_kwargs,
        advantages=[i * 10.0 for i in range(offset, offset + count)],
        old_logps=[[i + 0.2] for i in range(offset, offset + count)],
        ref_logps=[[i + 0.3] for i in range(offset, offset + count)],
        teacher_logps=[[i + 0.4] for i in range(offset, offset + count)],
        temperature=0.8,
        chord_fn=chord_fn,
        ema_manager=ema,
        save=state.saved.append,
    )


def ids(call):
    return [row["input_ids"][0] for row in call["inputs"]]


def test_full_update_keeps_input_order_and_all_auxiliary_fields_aligned():
    state = make_state()
    cfg = RLConfig(
        rl_type="grpo", enable_sampling_replay=True, router_replay_mode="R3", sdar_loss_coef=0.1, overlong_filter=True
    )
    assert train(state, 16, config=cfg) == 1
    (call,) = state.model.calls
    expected = list(range(16))
    assert ids(call) == expected
    assert call["micro_batch_size"] == 2
    assert call["advantages"] == [i * 10.0 for i in expected]
    for key, offset in [("rollout_logps", 0.1), ("old_logps", 0.2), ("ref_logps", 0.3), ("teacher_logps", 0.4)]:
        assert call[key] == [[i + offset] for i in expected]
    assert call["sampling_masks"] == [f"mask-{i}" for i in expected]
    assert call["truncated"] == [i % 2 == 0 for i in expected]
    assert [row["routed_experts"] for row in call["inputs"]] == [[i] for i in expected]
    assert state.model.steps == state.global_step == 1
    assert state.micro_step == 4
    assert state.saved == ["1"]


def test_replays_share_update_windows_and_drop_only_the_final_tail():
    state = make_state()
    synced = []
    cfg = RLConfig(
        rl_type="grpo",
        num_iterations=2,
        ref_model_sync_steps=2,
        ref_model_mixup_alpha=0.5,
        enable_sampling_replay=True,
        router_replay_mode="R3",
        sdar_loss_coef=0.1,
    )
    ema = SimpleNamespace(sync_ema=synced.append)
    assert train(state, 28, config=cfg, ema=ema) == 3
    expected = (list(range(28)) * 2)[:48]
    assert [ids(call) for call in state.model.calls] == [expected[start : start + 16] for start in range(0, 48, 16)]
    for call in state.model.calls:
        indices = ids(call)
        assert call["advantages"] == [index * 10.0 for index in indices]
        assert call["old_logps"] == [[index + 0.2] for index in indices]
        assert call["ref_logps"] == [[index + 0.3] for index in indices]
        assert call["teacher_logps"] == [[index + 0.4] for index in indices]
        assert call["rollout_logps"] == [[index + 0.1] for index in indices]
        assert call["sampling_masks"] == [f"mask-{index}" for index in indices]
        assert [row["routed_experts"] for row in call["inputs"]] == [[index] for index in indices]
    assert state.model.steps == state.global_step == 3
    assert state.micro_step == 12
    assert state.saved == ["1", "2", "3"]
    assert synced == [0.5]


@pytest.mark.parametrize("count, iterations", [(0, 3), (4, 4), (12, 2), (28, 2), (36, 1)])
def test_replay_batching_drops_only_less_than_one_global_batch(count, iterations):
    state = make_state()
    config = RLConfig(rl_type="grpo", num_iterations=iterations, num_generations=4)
    expected_steps = count * iterations // 16
    assert train(state, count, config=config) == expected_steps
    expected = (list(range(count)) * iterations)[: expected_steps * 16]
    assert [index for call in state.model.calls for index in ids(call)] == expected
    assert state.global_step == expected_steps and state.micro_step == expected_steps * 4


def test_training_splits_generation_groups_after_advantages_are_calculated():
    state = make_state()
    assert train(state, 36, config=RLConfig(rl_type="grpo", num_generations=6)) == 2
    assert [ids(call) for call in state.model.calls] == [list(range(16)), list(range(16, 32))]


@pytest.mark.parametrize("count", [0, 3])
def test_rollout_smaller_than_one_global_micro_batch_is_dropped(count):
    state = make_state()
    assert train(state, count) == 0
    assert state.model.calls == []
    assert state.model.steps == state.global_step == state.micro_step == 0


@pytest.mark.parametrize(
    "count, sizes, micros", [(4, [], 0), (15, [], 0), (19, [16], 4), (20, [16], 4), (28, [16], 4), (32, [16, 16], 8)]
)
def test_updates_drop_every_incomplete_global_batch(count, sizes, micros):
    state = make_state()
    assert train(state, count) == len(sizes)
    assert [len(call["inputs"]) for call in state.model.calls] == sizes
    assert state.micro_step == micros
    assert state.global_step == state.model.steps == len(sizes)


def test_dropped_tails_do_not_carry_training_data_into_the_next_call():
    state = make_state()
    assert train(state, 12) == 0
    assert train(state, 12, offset=12) == 0
    assert state.model.calls == []
    assert train(state, 16, offset=24) == 1
    call = state.model.calls[0]
    expected = list(range(24, 40))
    assert ids(call) == expected
    assert call["advantages"] == [index * 10.0 for index in expected]
    assert call["old_logps"] == [[index + 0.2] for index in expected]
    assert state.micro_step == 4 and state.global_step == 1


def test_dynamic_batching_keeps_nominal_ga_without_reading_a_forward_result():
    state = make_state()
    options = {"dynamic_batching": True, "max_tokens_per_micro_batch": 1024, "packing_algorithm": "kk"}
    assert train(state, 20, batching_kwargs=options) == 1
    assert [len(call["inputs"]) for call in state.model.calls] == [16]
    assert all(call[key] == value for call in state.model.calls for key, value in options.items())
    assert state.global_step == 1 and state.micro_step == 4


def test_dbs_chord_keeps_the_sft_window_and_nominal_count():
    state, cursor = make_state(), [0]

    def chord():
        cursor[0] += 6
        return [{"input_ids": [i]} for i in range(cursor[0] - 6, cursor[0])]

    options = {"dynamic_batching": True, "max_tokens_per_micro_batch": 1024, "packing_algorithm": "ffd"}
    config = RLConfig(rl_type="grpo", num_iterations=2)
    assert train(state, 28, config=config, batching_kwargs=options, chord_fn=chord) == 3
    assert [len(call["inputs"]) for call in state.model.calls] == [16, 24] * 3
    assert cursor[0] == 72 and state.micro_step == 12
    assert state.global_step == state.model.steps == 3
    for call in state.model.calls:
        assert all(call[key] == value for key, value in options.items())
    assert ids(state.model.calls[2]) == list(range(16, 28)) + list(range(4))


def test_max_steps_stops_between_optimizer_updates():
    state = make_state(max_steps=1)
    assert train(state, 32, config=RLConfig(rl_type="grpo", num_iterations=2)) == 1
    assert len(state.model.calls) == 1
    assert state.global_step == 1
    assert state.micro_step == 4


def test_chord_submits_two_calls_per_update_using_its_independent_micro_size():
    state, offset = make_state(), [0]

    def chord():
        rows = [{"input_ids": [i], "labels": [i]} for i in range(offset[0], offset[0] + 6)]
        offset[0] += 6
        return rows

    cfg = RLConfig(rl_type="grpo", chord_mu_peak=0.3, chord_mu_valley=0.3)
    assert train(state, 32, config=cfg, chord_fn=chord) == 2
    rl, sft, next_rl, next_sft = state.model.calls
    assert [len(call["inputs"]) for call in state.model.calls] == [16, 24, 16, 24]
    assert ids(sft) == list(range(24))
    assert ids(next_sft) == list(range(24, 48))
    for call in (rl, next_rl):
        assert call["loss_key"] == "grpo" and call["loss_weight"] == 0.7
        assert call["finalize_gradients"] is False and call["micro_batch_size"] == 2
    for call in (sft, next_sft):
        assert call["loss_key"] == "sft" and call["loss_weight"] == 0.3
        assert call["finalize_gradients"] is True and call["micro_batch_size"] == 3
    assert offset[0] == 48
    assert state.model.steps == state.global_step == 2
    assert state.micro_step == 8


def test_checkpoint_restores_completed_update_counters_without_pending_samples(tmp_path):
    import torch

    state, checkpoints = make_state(), []

    def save(_):
        snapshot = {"cur_step": state.global_step, "component_states": {"training": {"micro_step": state.micro_step}}}
        path = tmp_path / f"{state.global_step}.pt"
        torch.save(snapshot, path)
        checkpoints.append(path)

    state.saved = SimpleNamespace(append=save)
    assert train(state, 28) == 1
    snapshot = torch.load(checkpoints[-1], weights_only=False)
    resumed = make_state()
    resumed.global_step, resumed.micro_step = training_progress(snapshot)
    assert (resumed.global_step, resumed.micro_step) == (1, 4)
    assert train(resumed, 16, offset=28) == 1
    assert (resumed.global_step, resumed.micro_step) == (2, 8)
    assert ids(resumed.model.calls[0]) == list(range(28, 44))


@pytest.mark.parametrize("iterations", [1, 2])
def test_colocate_replays_each_rollout_without_repeating_the_prompt_dataset(iterations, monkeypatch):
    from swift.dev.config import RewardConfig, RolloutConfig
    from swift.dev.recipe import offload

    state, generated, visits, scored_sizes = make_state(), [0], [], []

    class Loader:
        def __iter__(self):
            visits.append(True)
            for size in (4, 3):
                yield [{"messages": []} for _ in range(size)]

    def generate(prompts, *, num_samples, **kwargs):
        start = generated[0]
        generated[0] += len(prompts) * num_samples
        return [
            SimpleNamespace(encoded={"input_ids": [index], "labels": [index]}, rollout_logps=[0.0])
            for index in range(start, generated[0])
        ]

    def advantages(cluster, config, reward, rows, *args, **kwargs):
        scored_sizes.append(len(rows))
        assert len(rows) % config.num_generations == 0
        return [float(row.encoded["input_ids"][0]) for row in rows], [[0.0]] * len(rows), None, None

    monkeypatch.setattr(offload, "offload_for_rollout", lambda *args: None)
    monkeypatch.setattr(offload, "sleep_for_rollout", lambda *args: None)
    monkeypatch.setattr(grpo, "_advantage_pipeline", advantages)
    state.model.reload_to_gpu = lambda: None
    cluster = SimpleNamespace(
        model=state.model,
        sampler=SimpleNamespace(wake_up=lambda **kwargs: None),
        engine=SimpleNamespace(sync_weights=lambda **kwargs: None),
        rollout=SimpleNamespace(generate=generate),
        prm_scorer=None,
        orm_scorers=[],
        sampling_params={},
    )
    grpo._colocate_loop(
        cluster,
        RLConfig(rl_type="grpo", num_generations=4, num_iterations=iterations),
        RolloutConfig(),
        RewardConfig(),
        Loader(),
        state,
        train_batch_size=4,
        batching_kwargs={"micro_batch_size": 2},
    )
    expected = list(range(16)) * iterations
    tail = list(range(16, 28)) * iterations
    expected += tail[: (len(tail) // 16) * 16]
    assert [index for call in state.model.calls for index in ids(call)] == expected
    assert visits == [True] and generated[0] == 28 and scored_sizes == [16, 12]
    assert state.global_step == len(expected) // 16 and state.micro_step == len(expected) // 4
    assert (
        rollout_step_budget(
            num_prompts=7,
            num_generations=4,
            train_batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            generation_batch_size=4,
            num_iterations=iterations,
        )
        == state.global_step
    )


def test_disaggregated_loop_commits_actual_updates_before_publishing(monkeypatch, tmp_path):
    from swift.dev.config import RewardConfig, RolloutConfig
    from swift.dev.recipe.grpo import assembly

    state = make_state()
    state.tracker = SimpleNamespace(log_prompts=lambda *args: None)
    pulls = [
        [
            SimpleNamespace(encoded={"input_ids": [i], "labels": [i]}, rollout_logps=[0.0], truncated=False, prompt=[])
            for i in range(start, stop)
        ]
        for start, stop in ((0, 16), (16, 48))
    ]
    committed, published, aborted = [], [], []

    class Driver:
        def __iter__(self):
            return iter(pulls)

        commit = staticmethod(committed.append)
        request_publish = staticmethod(published.append)
        abort = staticmethod(lambda: aborted.append(True))

    driver = Driver()
    monkeypatch.setattr(assembly, "build_driver", lambda **kwargs: driver)
    monkeypatch.setattr(grpo, "_prepare_routing_replay", lambda *args: None)
    monkeypatch.setattr(
        grpo,
        "_advantage_pipeline",
        lambda cluster, config, reward, rows, *args, **kwargs: (
            [float(row.encoded["input_ids"][0]) for row in rows],
            [[0.0]] * len(rows),
            None,
            None,
        ),
    )
    cluster = SimpleNamespace(model=state.model, orm_scorers=[], sampling_params={"temperature": 1.0})
    grpo._disaggregated_loop(
        cluster,
        RLConfig(rl_type="grpo"),
        RolloutConfig(weight_sync_strategy="in_place", parameter_sync_step=1),
        RewardConfig(),
        None,
        None,
        state,
        train_batch_size=4,
        batching_kwargs={"micro_batch_size": 2},
        num_train_epochs=1,
        output_dir=str(tmp_path),
    )
    assert committed == [1, 2] and len(published) == 2 and aborted == [True]
    assert [len(call["inputs"]) for call in state.model.calls] == [16, 16, 16]
    assert state.global_step == 3 and state.micro_step == 12


def test_new_training_starts_with_zero_progress():
    assert training_progress({}) == (0, 0)
    assert training_progress({"cur_step": 0}) == (0, 0)


@pytest.mark.parametrize(
    "streaming, group_pull, expected",
    [
        (False, False, 2),
        (True, True, 2),
        (True, False, 2),
    ],
)
def test_grpo_budget_counts_only_deliverable_updates(streaming, group_pull, expected):
    assert (
        rollout_step_budget(
            num_prompts=5,
            num_generations=4,
            train_batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            generation_batch_size=4,
            num_iterations=2,
            streaming=streaming,
            group_pull=group_pull,
        )
        == expected
    )


@pytest.mark.parametrize("count", [16, 20])
def test_real_dp_dispatch_slices_full_updates_in_contiguous_order(count):
    from twinkle.infra.dispatch import _dispatch_args

    state = make_state()
    train(state, count)
    mesh = SimpleNamespace(
        world_size=2,
        get_data_rank_from_global_rank=lambda rank: rank,
        get_slice=lambda length, rank: slice(rank * length // 2, (rank + 1) * length // 2),
    )
    dispatched = _dispatch_args([None, None], "slice_dp", "all", mesh, (), state.model.calls[0])
    for rank in range(2):
        expected = list(range(rank * 8, (rank + 1) * 8))
        kwargs = dispatched[rank][2]
        assert ids(kwargs) == expected
        assert kwargs["advantages"] == [index * 10.0 for index in expected]
        assert kwargs["old_logps"] == [[index + 0.2] for index in expected]
        assert kwargs["ref_logps"] == [[index + 0.3] for index in expected]


@pytest.mark.parametrize("ga", [1, 4])
def test_resume_only_needs_global_step_regardless_of_backend_ga(ga):
    assert training_progress({"cur_step": 9, "gradient_accumulation_steps": ga}) == (9, 0)


def test_checkpoint_may_omit_the_micro_step_statistic():
    assert training_progress({"cur_step": 9, "component_states": {"training": {}}}) == (9, 0)


@pytest.mark.parametrize(
    "prompts, iterations, ga, streaming, group_pull, expected",
    [
        (4, 1, 4, False, False, 1),
        (8, 1, 4, False, False, 2),
        (8, 2, 4, False, False, 4),
        (4, 1, 1, False, False, 4),
        (8, 1, 4, True, True, 2),
        (5, 1, 4, False, False, 1),
        (5, 2, 4, False, False, 2),
        (5, 1, 4, True, True, 1),
        (5, 1, 4, True, False, 1),
    ],
)
def test_budget_counts_actual_optimizer_calls(prompts, iterations, ga, streaming, group_pull, expected):
    assert (
        rollout_step_budget(
            num_prompts=prompts,
            num_generations=4,
            train_batch_size=4,
            gradient_accumulation_steps=ga,
            num_train_epochs=1,
            generation_batch_size=4,
            num_iterations=iterations,
            streaming=streaming,
            group_pull=group_pull,
        )
        == expected
    )


@pytest.mark.parametrize(
    "prompts, generations, iterations, expected",
    [
        (3, 4, 2, 1),
        (12, 3, 2, 4),
        (7, 6, 2, 5),
    ],
)
def test_colocate_budget_matches_replay_updates_without_generation_group_alignment(
    prompts, generations, iterations, expected
):
    state = make_state()
    config = RLConfig(rl_type="grpo", num_generations=generations, num_iterations=iterations)
    assert train(state, prompts * generations, config=config) == expected
    assert (
        rollout_step_budget(
            num_prompts=prompts,
            num_generations=generations,
            train_batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            generation_batch_size=16,
            num_iterations=iterations,
        )
        == expected
    )


@pytest.mark.parametrize("iterations", [1, 2, 3])
@pytest.mark.parametrize("group_pull", [True, False])
@pytest.mark.parametrize("num_prompts", [2, 3, 4, 6, 10])
def test_streaming_budget_matches_the_drivers_actual_pull_rule(group_pull, num_prompts, iterations):
    from types import SimpleNamespace

    from twinkle_agentic.async_rl.streaming_driver import StreamingDriver

    num_generations, update_size = 3, 4
    records = [
        SimpleNamespace(group_key=prompt, score_ref=None)
        for prompt in range(num_prompts)
        for _ in range(num_generations)
    ]
    rule = (
        StreamingDriver._make_group_pull_rule(num_generations, update_size)
        if group_pull
        else StreamingDriver._make_no_group_pull_rule(update_size)
    )
    delivered = rule(records) or []
    expected = len(delivered) * iterations // update_size
    assert (
        rollout_step_budget(
            num_prompts=num_prompts,
            num_generations=num_generations,
            train_batch_size=2,
            gradient_accumulation_steps=2,
            num_train_epochs=1,
            generation_batch_size=1,
            num_iterations=iterations,
            streaming=True,
            group_pull=group_pull,
        )
        == expected
    )


@pytest.mark.parametrize(
    "streaming, group_pull, expected",
    [
        (False, False, 1),
        (True, False, 0),
        (True, True, 0),
    ],
)
def test_budget_repeats_colocate_tails_but_keeps_the_drivers_delivery_drop(streaming, group_pull, expected):
    assert (
        rollout_step_budget(
            num_prompts=3,
            num_generations=4,
            train_batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            generation_batch_size=4,
            num_iterations=2,
            streaming=streaming,
            group_pull=group_pull,
        )
        == expected
    )


@pytest.mark.parametrize(
    "streaming, group_pull, expected",
    [
        (False, False, 0),
        (True, False, 1),
        (True, True, 1),
    ],
)
def test_budget_drops_each_independent_round_tail(streaming, group_pull, expected):
    assert (
        rollout_step_budget(
            num_prompts=6,
            num_generations=1,
            train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=2,
            generation_batch_size=2,
            streaming=streaming,
            group_pull=group_pull,
        )
        == expected
    )

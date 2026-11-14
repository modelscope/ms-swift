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
        model=RecordingModel(dp),
        ga=ga,
        global_step=0,
        micro_step=0,
        max_grad_norm=1.0,
        max_steps=max_steps,
        saved=[],
        balance_metrics=[],
    )
    state.reached_max = lambda: state.global_step >= state.max_steps
    state.should_stop = lambda: False

    def record(*, extra_metrics=None):
        state.global_step += 1
        state.balance_metrics.append(extra_metrics)

    state.record = record
    state.step_tail = lambda save=None: save(str(state.global_step)) if save else None
    # The rollout statistics methods _train_mini_batches calls; the micro-batch harness
    # never sets up metrics, so they only need to exist and be harmless.
    from swift.dev.recipe.utils import RolloutStepState

    for name in ('setup_rollout_metrics', 'begin_rollout_iteration', 'observe_rollout_batch',
                 'record_rollout_training', 'rollout_memory_metrics', 'fold_iteration_metrics',
                 'finish_rollout_metrics'):
        setattr(state, name, getattr(RolloutStepState, name).__get__(state))
    state._metric_context = {}
    state._run_metric = None
    state._completed_batches = state._batch_wall_ns = 0
    state._rollout_groups = state._trained_samples = 0
    state._stream_previous = {}
    state._iteration_started_ns = None
    state._iteration_facts = None
    state._memory_rows = None
    return state


def train(
    state,
    count,
    *,
    config=None,
    chord_fn=None,
    ema=None,
    offset=0,
    batching_kwargs=None,
    lengths=None,
    padding_free=False,
):
    rows = [
        SimpleNamespace(
            encoded={"input_ids": [i], "labels": [i], "routed_experts": [i]},
            rollout_logps=[i + 0.1],
            truncated=i % 2 == 0,
            sampling_mask=f"mask-{i}",
            # Every per-sample training signal rides the trajectory itself; _forward_kwargs passes
            # only inputs/batching/temperature and the model lifts these off each row.
            advantages=i * 10.0,
            old_logps=[i + 0.2],
            ref_logps=[i + 0.3],
            teacher_logps=[i + 0.4],
        )
        for i in range(offset, offset + count)
    ]
    if lengths is not None:
        for row, length in zip(rows, lengths, strict=True):
            row.encoded["input_ids"] *= length
    row_lengths = list(lengths) if lengths is not None else [1] * count
    state.padding_free = padding_free
    return grpo._train_mini_batches(
        state,
        config or RLConfig(rl_type="grpo"),
        rows,
        row_lengths,
        train_batch_size=2 * state.model.device_mesh.data_world_size,
        batching_kwargs=batching_kwargs,
        temperature=0.8,
        chord_fn=chord_fn,
        ema_manager=ema,
        save=state.saved.append,
    )


def ids(call):
    """Row identity: a trajectory carries its encoding under .encoded, a prepared row is the dict."""
    return [(row["input_ids"] if isinstance(row, dict) else row.encoded["input_ids"])[0]
            for row in call["inputs"]]


@pytest.mark.parametrize("padding_free", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
def test_length_balance_reduces_rank_load_and_keeps_all_signals_aligned(padding_free, dynamic):
    state = make_state(ga=2)
    state.dp_balance = True
    config = RLConfig(
        rl_type="grpo",
        enable_sampling_replay=True,
        router_replay_mode="R3",
        sdar_loss_coef=0.1,
        overlong_filter=True,
    )
    batching = {"micro_batch_size": 2}
    if dynamic:
        batching.update(dynamic_batching=True, max_tokens_per_micro_batch=4000, packing_algorithm="kk")
    assert (
        train(
            state,
            8,
            config=config,
            lengths=[2000, 1900, 1800, 1700, 300, 250, 200, 150],
            padding_free=padding_free,
            batching_kwargs=batching,
        )
        == 1
    )
    (call,) = state.model.calls
    order = ids(call)
    assert sorted(order) == list(range(8))
    assert order != list(range(8))
    rows = call["inputs"]
    assert [row.advantages for row in rows] == [i * 10.0 for i in order]
    for key, delta in [("rollout_logps", 0.1), ("old_logps", 0.2), ("ref_logps", 0.3), ("teacher_logps", 0.4)]:
        assert [getattr(row, key) for row in rows] == [[i + delta] for i in order]
    assert [row.sampling_mask for row in rows] == [f"mask-{i}" for i in order]
    assert [row.truncated for row in rows] == [i % 2 == 0 for i in order]
    assert [row.encoded["routed_experts"] for row in call["inputs"]] == [[i] for i in order]
    metrics = state.balance_metrics[0]
    assert metrics["dp_balance/workload_imbalance_after"] < metrics["dp_balance/workload_imbalance_before"]
    assert ("dp_balance/padding_ratio_after" in metrics) == (not padding_free)
    assert state.global_step == state.model.steps == 1
    assert "padding_free" not in call  # This is the processor layout, not a model RPC option.


def test_balance_preserves_replay_update_membership_and_dropped_tail():
    state = make_state(ga=1)
    state.dp_balance = True
    state.padding_free = False
    config = RLConfig(rl_type="grpo", num_iterations=2)
    assert train(state, 7, config=config, lengths=[2000, 1900, 300, 200, 1800, 1700, 100]) == 3
    expected = (list(range(7)) * 2)[:12]
    for start, call in zip(range(0, 12, 4), state.model.calls, strict=True):
        assert sorted(ids(call)) == sorted(expected[start : start + 4])
        assert [row.advantages for row in call["inputs"]] == [i * 10.0 for i in ids(call)]
    assert state.micro_step == state.global_step == 3


def test_balance_disabled_keeps_skewed_batch_order():
    state = make_state(ga=2)
    train(state, 8, lengths=[2000, 1900, 1800, 1700, 300, 250, 200, 150])
    assert ids(state.model.calls[0]) == list(range(8))
    assert state.balance_metrics == [{}]


def test_balance_metrics_reach_tracker_and_jsonl_without_leaking_to_next_step(tmp_path):
    import json

    from swift.dev.recipe.utils import StepState

    state = StepState.__new__(StepState)
    state.model = SimpleNamespace(calculate_metric=lambda **kwargs: {"loss": 1.0})
    state.global_step, state.micro_step, state.epoch, state.best_metric = 0, 0, 0, None
    state.manual_gc, state.manual_gc_steps, state.empty_cache_steps = False, 0, None
    state._logging_config, state._extra_metrics_fn, state._log_formatter = None, None, None
    state._output_dir = str(tmp_path)
    state.history = []
    tracked = []

    def log(record, step):
        tracked.append(dict(record))
        return record

    state.tracker = SimpleNamespace(log=log)
    state.handler = SimpleNamespace(
        state=SimpleNamespace(), control=SimpleNamespace(should_log=True), on_log=lambda **kwargs: None
    )
    metric = "dp_balance/workload_imbalance_after"
    state.record(extra_metrics={metric: 1.01})
    state.record()
    saved = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert tracked[0][metric] == saved[0][metric] == state.history[0][metric] == 1.01
    assert all(metric not in record for record in (tracked[1], saved[1], state.history[1]))


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
    rows = call["inputs"]
    assert [row.advantages for row in rows] == [i * 10.0 for i in expected]
    for key, offset in [("rollout_logps", 0.1), ("old_logps", 0.2), ("ref_logps", 0.3), ("teacher_logps", 0.4)]:
        assert [getattr(row, key) for row in rows] == [[i + offset] for i in expected]
    assert [row.sampling_mask for row in rows] == [f"mask-{i}" for i in expected]
    assert [row.truncated for row in rows] == [i % 2 == 0 for i in expected]
    assert [row.encoded["routed_experts"] for row in call["inputs"]] == [[i] for i in expected]
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
        rows = call["inputs"]
        assert [row.advantages for row in rows] == [index * 10.0 for index in indices]
        assert [row.old_logps for row in rows] == [[index + 0.2] for index in indices]
        assert [row.ref_logps for row in rows] == [[index + 0.3] for index in indices]
        assert [row.teacher_logps for row in rows] == [[index + 0.4] for index in indices]
        assert [row.rollout_logps for row in rows] == [[index + 0.1] for index in indices]
        assert [row.sampling_mask for row in rows] == [f"mask-{index}" for index in indices]
        assert [row.encoded["routed_experts"] for row in call["inputs"]] == [[index] for index in indices]
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
    rows = call["inputs"]
    assert [row.advantages for row in rows] == [index * 10.0 for index in expected]
    assert [row.old_logps for row in rows] == [[index + 0.2] for index in expected]
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


def test_new_training_starts_with_zero_progress():
    assert training_progress({}) == (0, 0)
    assert training_progress({"cur_step": 0}) == (0, 0)


@pytest.mark.parametrize("streaming", [False, True])
def test_grpo_budget_counts_only_deliverable_updates(streaming):
    """The 4-row epoch tail is not an optimizer update on its own. The streaming driver folds it
    into the 20 generated rows and floors that to its 16-row batch, so both routes count two."""
    from swift.dev.recipe.utils import stream_batch_size

    rows = stream_batch_size(
        colocate=False, generation_batch_size=4, num_generations=4,
        train_batch_size=4, gradient_accumulation_steps=4)
    assert (
        rollout_step_budget(
            num_prompts=5,
            num_generations=4,
            train_batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            generation_batch_size=4,
            num_iterations=2,
            stream_batch_rows=rows if streaming else None,
        )
        == 2
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
        rows = kwargs["inputs"]
        assert [row.advantages for row in rows] == [index * 10.0 for index in expected]
        assert [row.old_logps for row in rows] == [[index + 0.2] for index in expected]
        assert [row.ref_logps for row in rows] == [[index + 0.3] for index in expected]


@pytest.mark.parametrize("ga", [1, 4])
def test_resume_only_needs_global_step_regardless_of_backend_ga(ga):
    assert training_progress({"cur_step": 9, "gradient_accumulation_steps": ga}) == (9, 0)


def test_checkpoint_may_omit_the_micro_step_statistic():
    assert training_progress({"cur_step": 9, "component_states": {"training": {}}}) == (9, 0)


@pytest.mark.parametrize(
    "prompts, iterations, ga, streaming, expected",
    [
        (4, 1, 4, False, 1),
        (8, 1, 4, False, 2),
        (8, 2, 4, False, 4),
        (4, 1, 1, False, 4),
        (8, 1, 4, True, 2),
        (5, 1, 4, False, 1),
        (5, 2, 4, False, 2),
        (5, 1, 4, True, 1),
    ],
)
def test_budget_counts_actual_optimizer_calls(prompts, iterations, ga, streaming, expected):
    from swift.dev.recipe.utils import stream_batch_size

    rows = None
    if streaming:
        rows = stream_batch_size(
            colocate=False, generation_batch_size=4, num_generations=4,
            train_batch_size=4, gradient_accumulation_steps=ga)
    assert (
        rollout_step_budget(
            num_prompts=prompts,
            num_generations=4,
            train_batch_size=4,
            gradient_accumulation_steps=ga,
            num_train_epochs=1,
            generation_batch_size=4,
            num_iterations=iterations,
            stream_batch_rows=rows,
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
@pytest.mark.parametrize("num_prompts", [2, 3, 4, 6, 10])
def test_streaming_budget_matches_the_drivers_actual_delivery(num_prompts, iterations):
    """The driver hands out whole ``batch_size`` rows at a time and that batch holds whole groups
    (both enforced when it is built), so the budget is the generated row count floored to the
    batch, replayed ``num_iterations`` times, then floored to the optimizer update."""
    from swift.dev.recipe.utils import stream_batch_size

    num_generations, update_size = 3, 4
    train_batch_size, gradient_accumulation_steps = 2, 2
    rows = stream_batch_size(
        colocate=False, generation_batch_size=1, num_generations=num_generations,
        train_batch_size=train_batch_size, gradient_accumulation_steps=gradient_accumulation_steps)
    # whole groups: the invariant the driver's constructor raises on if it does not hold
    assert rows % num_generations == 0
    delivered = (num_prompts * num_generations) // rows * rows
    expected = delivered * iterations // update_size
    assert (
        rollout_step_budget(
            num_prompts=num_prompts,
            num_generations=num_generations,
            train_batch_size=train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=1,
            generation_batch_size=1,
            num_iterations=iterations,
            stream_batch_rows=rows,
        )
        == expected
    )


@pytest.mark.parametrize("streaming, expected", [(False, 1), (True, 0)])
def test_budget_repeats_colocate_tails_but_keeps_the_drivers_delivery_drop(streaming, expected):
    """12 rows replayed twice make one full 16-row update. The streaming driver instead floors the
    delivery to its colocate batch first -- which amortizes the sleep/wake/sync hand-over over a
    whole generation batch, here 16 rows -- and 12 rows never reach one, so nothing trains."""
    from swift.dev.recipe.utils import stream_batch_size

    rows = stream_batch_size(
        colocate=True, generation_batch_size=4, num_generations=4,
        train_batch_size=4, gradient_accumulation_steps=4)
    assert (
        rollout_step_budget(
            num_prompts=3,
            num_generations=4,
            train_batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            generation_batch_size=4,
            num_iterations=2,
            stream_batch_rows=rows if streaming else None,
        )
        == expected
    )


@pytest.mark.parametrize("streaming, expected", [(False, 0), (True, 1)])
def test_budget_drops_each_independent_round_tail(streaming, expected):
    """Six independent 2-row rounds never fill an 8-row update on their own, so a round loop drops
    every tail. The streaming driver combines them into one delivery and floors it to its 8-row
    batch, which is exactly one update."""
    from swift.dev.recipe.utils import stream_batch_size

    rows = stream_batch_size(
        colocate=False, generation_batch_size=2, num_generations=1,
        train_batch_size=2, gradient_accumulation_steps=4)
    assert (
        rollout_step_budget(
            num_prompts=6,
            num_generations=1,
            train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=2,
            generation_batch_size=2,
            stream_batch_rows=rows if streaming else None,
        )
        == expected
    )

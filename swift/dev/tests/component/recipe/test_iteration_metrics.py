"""Recipe boundaries feed the existing metric/reporting components."""

from importlib import import_module
from types import SimpleNamespace as NS

import json

from swift.dev.recipe.utils import RunTracker
from swift.dev.utils.scoring import sample_token_lengths

grpo = import_module("swift.dev.recipe.grpo.run_grpo")


def sample(prompt=3, response=2, *, tool=0):
    ids = list(range(prompt + response + tool))
    labels = [-100] * prompt + ids[prompt : prompt + response] + [-100] * tool
    return NS(encoded={"input_ids": ids, "labels": labels[1:] + labels[:1]}, truncated=False)


def observed_state(path, now, groups=None):
    state = NS(global_step=0, model=NS(memory_stats=lambda **kw: []), tracker=RunTracker(None, str(path)))
    grpo._init_metrics(state, groups or [("model", [0])])
    grpo._begin_iteration(state, clock_ns=lambda: int(now[0] * 1e9))
    return state


def iteration_rows(path, state):
    state.tracker._metric_reporter.flush()
    rows = [json.loads(line) for line in (path / "metrics_raw.jsonl").read_text().splitlines()]
    return [
        dict(row["attributes"], data=row["values"], run_id=row["run_id"]) for row in rows if row["stage"] == "rollout"
    ]


def test_lengths_include_tool_observations_and_exclude_padding():
    assert sample_token_lengths(sample(tool=4)) == (3, 6)
    assert sample_token_lengths(sample(response=0)) == (3, 0)
    row = sample(tool=4)
    row.encoded["input_ids"] += [0, 0]
    labels = [-100] * 3 + [3, 4] + [-100] * 6
    row.encoded["labels"] = labels[1:] + labels[:1]
    row.encoded["attention_mask"] = [1] * 9 + [0, 0]
    assert sample_token_lengths(row) == (3, 6)


def test_replay_counts_visits_but_tokens_once_and_devices_without_duplicates(tmp_path):
    now = [0.0]
    state = observed_state(tmp_path, now, [("model", [0, 1]), ("shared", [0])])
    grpo._receive_iteration(state, [sample(), sample(prompt=4, response=3)])
    state._iteration.sample_visits = 8
    state._iteration.trained_indices = {0, 1}
    state.global_step = 4
    now[0] = 2
    grpo._finish_iteration(state, publications=1)
    row = iteration_rows(tmp_path, state)[0]
    assert row["data"]["perf/total_num_tokens"] == 12
    assert row["data"]["perf/throughput"] == 3
    assert row["data"]["training/sample_visits"] == 8
    assert row["data"]["training/unique_samples_trained"] == 2
    assert row["data"]["training/optimizer_steps_this_iteration"] == 4
    assert row["data"]["training/weight_publications_this_iteration"] == 1
    assert row["step_unit"] == "rl_iteration"
    state.tracker.close()


def test_nested_logging_excludes_overlap_once_and_keeps_raw_intervals(tmp_path):
    now = [0.0]
    state = observed_state(tmp_path, now)
    grpo._receive_iteration(state, [sample()])
    with grpo._stage(state, "update_actor"):
        now[0] = 1
        with grpo._stage(state, "_logging"):
            now[0] = 2
            with grpo._stage(state, "_logging"):
                now[0] = 3
            now[0] = 4
        now[0] = 6
    grpo._finish_iteration(state)
    row = iteration_rows(tmp_path, state)[0]
    assert row["data"]["timing_s/update_actor"] == row["data"]["timing_s/step"] == 3
    assert row["data"]["observability/iteration_wall_seconds"] == 6
    assert row["data"]["observability/excluded_logging_seconds"] == 3
    assert sum(event[2] == "_logging" for event in row["timing_spans"]) == 2
    assert any(event[2] == "update_actor" and event[7] == 6_000_000_000 for event in row["timing_spans"])
    state.tracker.close()


def test_publication_counter_deltas_and_terminal_summary_use_existing_tracker(tmp_path):
    now = [0.0]
    state = observed_state(tmp_path, now)
    for step, publish_time, publications in [(1, 0.5, 1), (2, 0.5, 1)]:
        if step > 1:
            grpo._begin_iteration(state, clock_ns=lambda: int(now[0] * 1e9))
        now[0] += 1
        grpo._receive_iteration(state, [sample()])
        now[0] += 1
        state.global_step = step
        grpo._finish_iteration(state, stream_stats={"publish_time": publish_time, "publishes": publications})
    rows = iteration_rows(tmp_path, state)
    assert [row["data"]["training/weight_publications_this_iteration"] for row in rows] == [1, 0]
    assert [row["data"]["timing_s/update_weights"] for row in rows] == [0.5, 0]
    assert all(row["data"]["timing_s/gen"] == 1 for row in rows)
    grpo._record_run_metrics(state, stream_stats={"dropped_incomplete": 3, "final_version": 2})
    state.tracker.close()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["result"]["completed_iterations"] == 2
    assert summary["result"]["stream/dropped_incomplete"] == 3
    assert summary["result"]["stream/final_version"] == 2
    assert "stream/dropped_incomplete" not in rows[0]["data"]


def test_default_worker_recorder_reset_does_not_erase_iteration_observations(tmp_path):
    from twinkle.perf import Perf, default_recorder

    now = [0.0]
    state = observed_state(tmp_path, now)
    grpo._receive_iteration(state, [sample()])
    with grpo._stage(state, "gen"):
        now[0] = 2
    enabled = default_recorder.enabled
    try:
        default_recorder.enable()
        with default_recorder.span("worker"):
            pass
        Perf.dump_process()
        grpo._finish_iteration(state)
    finally:
        if not enabled:
            default_recorder.disable()
    assert iteration_rows(tmp_path, state)[0]["data"]["timing_s/gen"] == 2
    state.tracker.close()

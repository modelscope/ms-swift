"""Recipe boundaries feed the existing metric/reporting components."""

from types import SimpleNamespace as NS

import json
import pytest

from swift.dev.recipe import utils as batch_metrics
from swift.dev.recipe.utils import RunTracker, rollout_metrics
from swift.dev.utils.scoring import sample_token_lengths


def sample(prompt=3, response=2, *, tool=0):
    ids = list(range(prompt + response + tool))
    labels = [-100] * prompt + ids[prompt : prompt + response] + [-100] * tool
    return NS(encoded={"input_ids": ids, "labels": labels[1:] + labels[:1]}, truncated=False)


def observed_state(path, now, groups=None):
    state = NS(global_step=0, model=NS(memory_stats=lambda **kw: []), tracker=RunTracker(None, str(path)))
    batch_metrics.init_batch_metrics(state, groups or [("model", [0])])
    batch_metrics.begin_batch(state, clock_ns=lambda: int(now[0] * 1e9))
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


def test_shared_batch_statistics_do_not_require_rewards_or_recipe_state():
    values = rollout_metrics(
        prompt_lengths=[3, 6, 4], response_lengths=[0, 2, 5],
        timings={'step': 5, 'gen': 3.5}, num_devices=2, wall_seconds=7,
        truncated=[False, True, False], turns=[1, 2, 3],
    )
    assert values['perf/total_num_tokens'] == 20
    assert values['perf/throughput'] == 2
    assert values['perf/gpu_seconds'] == 10
    assert values['observability/iteration_wall_seconds'] == 7
    assert values['observability/excluded_logging_seconds'] == 2
    assert values['response/truncated_ratio'] == pytest.approx(1 / 3)
    assert values['rollout/assistant_turns/mean'] == 2
    assert values['rollout/assistant_turns/min'] == 1
    assert values['rollout/assistant_turns/max'] == 3
    assert not any(key.startswith('train/') for key in values)


@pytest.mark.parametrize('turns', [(), (1, None), (1, '2')])
def test_unavailable_turn_counts_are_omitted(turns):
    values = rollout_metrics(
        prompt_lengths=[3, 3], response_lengths=[2, 2],
        timings={'step': 1}, num_devices=1, wall_seconds=1, turns=turns,
    )
    assert not any(key.startswith('rollout/assistant_turns/') for key in values)


def test_replay_counts_visits_but_tokens_once_and_devices_without_duplicates(tmp_path):
    now = [0.0]
    state = observed_state(tmp_path, now, [("model", [0, 1]), ("shared", [0])])
    batch_metrics.observe_batch(state, [sample(), sample(prompt=4, response=3)])
    for _ in range(4):
        batch_metrics.record_batch_training(state, [0, 1])
    state.global_step = 4
    now[0] = 2
    batch_metrics.finish_batch(state, publications=1)
    row = iteration_rows(tmp_path, state)[0]
    assert row["data"]["perf/total_num_tokens"] == 12
    assert row["data"]["perf/throughput"] == 3
    assert row["data"]["training/sample_visits"] == 8
    assert row["data"]["training/unique_samples_trained"] == 2
    assert row["data"]["training/optimizer_steps_this_iteration"] == 4
    assert row["data"]["training/weight_publications_this_iteration"] == 1
    assert row["step_unit"] == "rl_iteration"
    state.tracker.close()


def test_shared_batch_lifecycle_accepts_another_sample_format(tmp_path):
    now = [0.0]
    state = observed_state(tmp_path, now)
    rows = [NS(lengths=(3, 2), truncated=False), NS(lengths=(4, 3), truncated=True)]
    now[0] = 1
    batch_metrics.observe_batch(state, iter(rows), length_fn=lambda row: row.lengths)
    now[0] = 2
    state.global_step = 1
    batch_metrics.record_batch_training(state, [0])
    batch_metrics.finish_batch(state, publications=1)
    row = iteration_rows(tmp_path, state)[0]
    assert row['data']['perf/total_num_tokens'] == 12
    assert row['data']['response/truncated_ratio'] == 0.5
    assert row['data']['training/sample_visits'] == 1
    assert row['data']['training/samples_not_trained'] == 1
    assert row['prompt_lengths'] == [3, 4]
    assert 'adv_pipeline_scope' not in row
    state.tracker.close()


def test_batch_completion_can_run_on_a_different_thread(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    now = [0.0]
    state = observed_state(tmp_path, now)
    batch_metrics.observe_batch(state, [sample()])
    with batch_metrics.batch_stage(state, 'update_actor'):
        now[0] = 2
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(batch_metrics.finish_batch, state, memory={}).result()
    batch_metrics.finish_batch(state)  # repeated completion must not duplicate a record
    rows = iteration_rows(tmp_path, state)
    assert len(rows) == 1
    assert rows[0]['data']['timing_s/update_actor'] == 2
    assert state._batch_observation is None
    state.tracker.close()


def test_batch_helpers_are_noops_without_initialization():
    state = NS(global_step=0)
    batch_metrics.begin_batch(state)
    with batch_metrics.batch_stage(state, 'gen'):
        batch_metrics.observe_batch(state, [NS()])
    batch_metrics.record_batch_training(state, [0])
    assert batch_metrics.batch_memory(state) == {}
    batch_metrics.finish_batch(state)
    batch_metrics.finish_batch_metrics(state)
    assert not hasattr(state, '_batch_observation')


def test_nested_logging_excludes_overlap_once_and_keeps_raw_intervals(tmp_path):
    now = [0.0]
    state = observed_state(tmp_path, now)
    batch_metrics.observe_batch(state, [sample()])
    with batch_metrics.batch_stage(state, "update_actor"):
        now[0] = 1
        with batch_metrics.batch_stage(state, "_logging"):
            now[0] = 2
            with batch_metrics.batch_stage(state, "_logging"):
                now[0] = 3
            now[0] = 4
        now[0] = 6
    batch_metrics.finish_batch(state)
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
            batch_metrics.begin_batch(state, clock_ns=lambda: int(now[0] * 1e9))
        now[0] += 1
        batch_metrics.observe_batch(state, [sample()])
        now[0] += 1
        state.global_step = step
        batch_metrics.finish_batch(state, stream_stats={"publish_time": publish_time, "publishes": publications})
    rows = iteration_rows(tmp_path, state)
    assert [row["data"]["training/weight_publications_this_iteration"] for row in rows] == [1, 0]
    assert [row["data"]["timing_s/update_weights"] for row in rows] == [0.5, 0]
    assert all(row["data"]["timing_s/gen"] == 1 for row in rows)
    batch_metrics.finish_batch_metrics(state, stream_stats={"dropped_incomplete": 3, "final_version": 2})
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
    batch_metrics.observe_batch(state, [sample()])
    with batch_metrics.batch_stage(state, "gen"):
        now[0] = 2
    enabled = default_recorder.enabled
    try:
        default_recorder.enable()
        with default_recorder.span("worker"):
            pass
        Perf.dump_process()
        batch_metrics.finish_batch(state)
    finally:
        if not enabled:
            default_recorder.disable()
    assert iteration_rows(tmp_path, state)[0]["data"]["timing_s/gen"] == 2
    state.tracker.close()

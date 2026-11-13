"""Recipe boundaries feed the existing metric/reporting components.

The lifecycle helpers are razor-thin: timing is the twinkle dispatch layer's driver-side
auto-spans on the process default recorder (mirrored here by driving that recorder with a
fake clock), throughput formulas are twinkle.metric.ThroughputMetric, and only GRPO-specific
sample facts live in swift. These tests keep the statistics contract of the original
implementation (token counts, visit accounting, publication deltas, the logging exclusion).
"""

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


class _FakeClock:
    """Drives the process default recorder deterministically (seconds granularity)."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return int(self.now * 1e9)

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture()
def perf_clock(monkeypatch):
    """Enable the default recorder with a fake clock; restore on exit."""
    from twinkle.perf import default_recorder

    clock = _FakeClock()
    monkeypatch.setattr(default_recorder, '_clock_ns', clock)
    monkeypatch.setattr(default_recorder, '_enabled', True)
    default_recorder.reset()
    yield clock
    default_recorder.reset()
    monkeypatch.setattr(default_recorder, '_enabled', False)


def observed_state(path, groups=None):
    # A real RolloutStepState skips StepState.__init__ (callbacks/tracker wiring is not
    # under test); only the statistics fields + the pieces the methods read are mounted.
    from swift.dev.recipe.utils import RolloutStepState

    state = RolloutStepState.__new__(RolloutStepState)
    state.global_step = 0
    state.model = NS(memory_stats=lambda **kw: [])
    state.tracker = RunTracker(None, str(path))
    state.setup_rollout_metrics(groups or [("model", [0])])
    state.begin_rollout_iteration()
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


def test_replay_counts_visits_but_tokens_once_and_devices_without_duplicates(tmp_path, perf_clock):
    from twinkle.perf import default_recorder

    state = observed_state(tmp_path, [("model", [0, 1]), ("shared", [0])])
    state.observe_rollout_samples([sample(), sample(prompt=4, response=3)])
    for _ in range(4):
        state.record_rollout_training([0, 1])
    state.global_step = 4
    with default_recorder.span("forward_backward"):
        perf_clock.advance(2)
    state.fold_iteration_metrics(publications=1)
    row = iteration_rows(tmp_path, state)[0]
    assert row["data"]["perf/total_num_tokens"] == 12
    assert row["data"]["perf/throughput"] == 3
    assert row["data"]["training/sample_visits"] == 8
    assert row["data"]["training/unique_samples_trained"] == 2
    assert row["data"]["training/weight_publications_this_iteration"] == 1
    assert row["step_unit"] == "rl_iteration"
    state.tracker.close()


def test_shared_batch_lifecycle_accepts_another_sample_format(tmp_path):
    state = observed_state(tmp_path)
    rows = [NS(lengths=(3, 2), truncated=False), NS(lengths=(4, 3), truncated=True)]
    state.observe_rollout_samples(iter(rows), length_fn=lambda row: row.lengths)
    state.global_step = 1
    state.record_rollout_training([0])
    state.fold_iteration_metrics(publications=1)
    row = iteration_rows(tmp_path, state)[0]
    assert row['data']['perf/total_num_tokens'] == 12
    assert row['data']['response/truncated_ratio'] == 0.5
    assert row['data']['training/sample_visits'] == 1
    assert row['data']['training/samples_not_trained'] == 1
    assert 'adv_pipeline_scope' not in row
    state.tracker.close()


def test_balancing_and_batch_observations_record_the_same_update(tmp_path):
    from swift.dev.tests.component.recipe.test_grpo_micro_batch import make_state, train

    from swift.dev.recipe.utils import RolloutStepState

    lengths = [2000, 1900, 1800, 1700, 300, 250, 200, 150]
    state = make_state(ga=2)
    # Adopt the statistics methods without StepState.__init__ (make_state's namespace is
    # the micro-batch harness under test here): plain unbound-method attachment.
    for name in ('setup_rollout_metrics', 'begin_rollout_iteration', 'observe_rollout_samples',
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
    state.dp_balance = True
    state.model.memory_stats = lambda **kwargs: []
    state.tracker = RunTracker(None, str(tmp_path))
    state.setup_rollout_metrics([('model', [0, 1])], num_generations=8)
    state.begin_rollout_iteration()
    samples = [NS(encoded={'input_ids': [i] * n, 'labels': [i] * n}) for i, n in enumerate(lengths)]
    state.observe_rollout_samples(samples)
    try:
        assert train(state, 8, lengths=lengths) == 1
        state.fold_iteration_metrics()
        row = iteration_rows(tmp_path, state)[0]
        assert state.global_step == state.model.steps == 1
        balance = state.balance_metrics[0]
        assert balance['dp_balance/workload_imbalance_after'] < balance['dp_balance/workload_imbalance_before']
        assert row['data']['training/sample_visits'] == row['data']['training/unique_samples_trained'] == 8
        assert row['data']['perf/total_num_tokens'] == sum(lengths)
    finally:
        state.tracker.close()


def test_batch_completion_can_run_on_a_different_thread(tmp_path, perf_clock):
    from concurrent.futures import ThreadPoolExecutor

    from twinkle.perf import default_recorder

    state = observed_state(tmp_path)
    state.observe_rollout_samples([sample()])
    with default_recorder.span('forward_backward'):
        perf_clock.advance(2)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(state.fold_iteration_metrics).result()
    state.fold_iteration_metrics()  # repeated completion must not duplicate a record
    rows = iteration_rows(tmp_path, state)
    assert len(rows) == 1
    assert rows[0]['data']['timing_s/forward_backward'] == 2
    assert state._iteration_facts is None
    state.tracker.close()


def test_batch_helpers_are_noops_without_initialization():
    from swift.dev.recipe.utils import RolloutStepState

    # Fresh instance WITHOUT setup_rollout_metrics: every helper must be a harmless no-op.
    state = RolloutStepState.__new__(RolloutStepState)
    state._metric_context = {}
    state._run_metric = None
    state._completed_batches = state._batch_wall_ns = 0
    state._rollout_groups = state._trained_samples = 0
    state._stream_previous = {}
    state._iteration_started_ns = None
    state._iteration_facts = None
    state._memory_rows = None
    state.begin_rollout_iteration()
    state.observe_rollout_samples([NS()])
    state.record_rollout_training([0])
    assert state.rollout_memory_metrics() == {}
    assert state.fold_iteration_metrics() == {}
    state.finish_rollout_metrics()
    assert state._iteration_facts is None  # cleared, not deleted: the field is permanent now


def test_nested_logging_excludes_overlap_once(tmp_path, perf_clock):
    # The exclusion band lives in twinkle.perf (TimeRecorder.phase_totals); the raw span
    # stream stays in the recorder (dumpable via perf), no longer copied into records.
    from twinkle.perf import default_recorder

    state = observed_state(tmp_path)
    state.observe_rollout_samples([sample()])
    with default_recorder.span("update_actor"):
        perf_clock.advance(1)
        with default_recorder.span("_logging"):
            perf_clock.advance(1)
            with default_recorder.span("_logging"):
                perf_clock.advance(1)
            perf_clock.advance(1)
        perf_clock.advance(2)
    state.fold_iteration_metrics()
    row = iteration_rows(tmp_path, state)[0]
    assert row["data"]["timing_s/update_actor"] == row["data"]["timing_s/step"] == 3
    assert row["data"]["observability/iteration_wall_seconds"] == 6
    state.tracker.close()


def test_publication_counter_deltas_and_terminal_summary_use_existing_tracker(tmp_path, perf_clock):
    state = observed_state(tmp_path)
    for step, publish_time, publications in [(1, 0.5, 1), (2, 0.5, 1)]:
        if step > 1:
            state.begin_rollout_iteration()
        # one second until the pulled samples arrive: the 'gen' fallback window end
        perf_clock.advance(1)
        state.observe_rollout_samples([sample()])
        perf_clock.advance(1)
        state.global_step = step
        state.fold_iteration_metrics(
            stream_stats={"publish_time": publish_time, "publishes": publications})
    rows = iteration_rows(tmp_path, state)
    assert [row["data"]["training/weight_publications_this_iteration"] for row in rows] == [1, 0]
    assert [row["data"]["timing_s/update_weights"] for row in rows] == [0.5, 0]
    assert all(row["data"]["timing_s/gen"] == 1 for row in rows)
    state.finish_rollout_metrics(stream_stats={"dropped_incomplete": 3, "final_version": 2})
    state.tracker.close()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["result"]["completed_iterations"] == 2
    assert summary["result"]["stream/dropped_incomplete"] == 3
    assert summary["result"]["stream/final_version"] == 2
    assert "stream/dropped_incomplete" not in rows[0]["data"]


def test_default_worker_recorder_reset_does_not_erase_iteration_observations(tmp_path, perf_clock):
    from twinkle.perf import Perf, default_recorder

    state = observed_state(tmp_path)
    state.observe_rollout_samples([sample()])
    with default_recorder.span("gen"):
        perf_clock.advance(2)
    # The logging-cadence dump resets the window aggregates but must not erase the
    # event stream the iteration's phase reduction still reads.
    Perf.dump_process()
    state.fold_iteration_metrics()
    assert iteration_rows(tmp_path, state)[0]["data"]["timing_s/gen"] == 2
    state.tracker.close()

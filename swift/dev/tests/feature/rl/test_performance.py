"""RL iteration accounting: actual batches, physical resources, durable records."""

import importlib
from types import SimpleNamespace as NS

import json
import pytest

from swift.dev.recipe import utils as batch_metrics
from swift.dev.recipe.utils import RunTracker
from swift.dev.tests.component.recipe.test_iteration_metrics import iteration_rows, perf_clock



def attach_rollout_metrics(state):
    """Bind RolloutStepState's statistics methods onto a SimpleNamespace test state."""
    from swift.dev.recipe.utils import RolloutStepState

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

def sample(prompt=3, response=2, *, tool=0):
    ids = list(range(prompt + response + tool))
    labels = [-100] * prompt + ids[prompt : prompt + response] + [-100] * tool
    return NS(encoded={"input_ids": ids, "labels": labels[1:] + labels[:1]}, truncated=False)


def test_raw_training_metrics_survive_without_reporters_and_resume(tmp_path):
    for _ in range(2):
        tracker = RunTracker(None, str(tmp_path))
        tracker.log({"loss": float("nan"), "train/reward": 2.0}, 1)
        assert (tmp_path / "metrics_raw.jsonl").exists()
        tracker.close()
    rows = [json.loads(line) for line in (tmp_path / "metrics_raw.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and rows[0]["run_id"] != rows[1]["run_id"]
    assert rows[0]["attributes"]["non_numeric_values"]["loss"] is None
    assert rows[0]["values"]["train/reward"] == 2.0


def test_raw_optimizer_stream_does_not_replace_webui_records(tmp_path, monkeypatch):
    from swift.dev.recipe import utils

    monkeypatch.setattr(utils, "is_main_process", lambda: True)
    tracker = RunTracker(None, str(tmp_path))
    try:
        tracker.log({"loss": 0.5}, 1)
        utils.StepState._write_metrics_jsonl(NS(_output_dir=str(tmp_path)), {"step": 1, "loss": 0.5})
    finally:
        tracker.close()
    assert json.loads((tmp_path / "metrics.jsonl").read_text()) == {"step": 1, "loss": 0.5}
    assert json.loads((tmp_path / "metrics_raw.jsonl").read_text())["values"] == {"loss": 0.5}


def test_raw_training_record_uses_native_normalization_and_preserves_observations(tmp_path):
    tracker = RunTracker(None, str(tmp_path))
    try:
        tracker.log({
            'loss': float('nan'), 'count': 19, 'scalar': 'tensor(2.5)', 'accuracy': '85.0%',
            'note': 'pending%', 'details': [1, 2],
        }, 3)
    finally:
        tracker.close()
    row = json.loads((tmp_path / 'metrics_raw.jsonl').read_text())
    assert row['optimizer_step'] == 3
    assert row['values'] == {'count': 19, 'scalar': 2.5, 'accuracy': 85.0}
    assert row['attributes']['non_numeric_values'] == {'loss': None, 'note': 'pending%', 'details': [1, 2]}


def test_colocate_sync_boundary_and_no_double_wakeup(tmp_path, monkeypatch, perf_clock):
    mod = importlib.import_module("swift.dev.recipe.grpo.run_grpo")
    events = []
    awake = set()

    def wake(tags):
        for tag in tags:
            assert tag not in awake, f"double wake {tag}"
            awake.add(tag)
        events.append(("wake", tags))

    def sleep(level):
        awake.clear()
        events.append(("sleep", level))

    def sync(**kwargs):
        assert "weights" in awake
        events.append(("sync", state.global_step))

    model = NS(
        offload_to_cpu=lambda **kw: None,
        reload_to_gpu=lambda: None,
        memory_stats=lambda **kwargs: [{"max_memory_allocated_bytes": 2 * 1024**3}],
    )
    def _generate(*a, **kw):
        # generate's contract is (samples, encodings); the colocate loop re-attaches them.
        one = sample()
        return [one], [one.encoded]

    cluster = NS(
        model=model,
        sampler=NS(wake_up=wake, sleep=sleep),
        engine=NS(sync_weights=sync),
        sampling_params={},
        orm_scorers=[],
        prm_scorer=None,
        rollout=NS(generate=_generate),
        worker=NS(),  # _prepare_routing_replay takes it (router_replay disabled here)
    )
    state = NS(model=model, global_step=0, reached_max=lambda: state.global_step >= 2, should_stop=lambda: False)
    state.tracker = RunTracker(None, str(tmp_path))
    attach_rollout_metrics(state)
    state.setup_rollout_metrics([("trainer", [0])])
    monkeypatch.setattr(mod, "_advantage_pipeline", lambda *a, **kw: (None,) * 4)
    monkeypatch.setattr(mod, "_prepare_routing_replay", lambda *a: None)

    def train(*a, **kw):
        events.append(("train", state.global_step))
        perf_clock.advance(1)  # one second of optimizer work per train call
        state.global_step += 1
        return 1

    monkeypatch.setattr(mod, "_train_mini_batches", train)
    mod._colocate_loop(
        cluster,
        NS(num_generations=1, dynamic_sample=False, advantage_estimator="grpo", log_completions=False),
        NS(sleep_level=2, offload_model=True, offload_optimizer=True),
        NS(orm_weights=None),
        [[{"messages": []}]] * 2,
        state,
        train_batch_size=1,
        output_dir=str(tmp_path),
    )
    assert [e for e in events if e[0] in ("sync", "train")] == [
        ("sync", 0),
        ("train", 0),
        ("sync", 1),
        ("train", 1),
        ("sync", 2),
    ]
    assert not awake
    rows = iteration_rows(tmp_path, state)
    assert len(rows) == 2
    assert all("timing_s/update_weights" in row["data"] for row in rows)
    state.tracker.close()


def test_actor_memory_sample_precedes_periodic_checkpoint(tmp_path, monkeypatch):
    mod = importlib.import_module("swift.dev.recipe.grpo.run_grpo")
    events = []
    peak = [2]

    def memory(**kwargs):
        assert kwargs == {"synchronize": True}
        events.append("memory")
        return [{"max_memory_allocated_bytes": peak[0] * 1024**3}]

    model = NS(
        forward_backward=lambda **kw: None,
        clip_grad_and_step=lambda **kw: None,
        memory_stats=memory,
        device_mesh=NS(data_world_size=1),
    )
    state = NS(
        model=model,
        ga=1,
        micro_step=0,
        global_step=0,
        max_grad_norm=1,
        is_grad_sync_boundary=lambda: True,
        reached_max=lambda: False,
        should_stop=lambda: False,
    )

    def record():
        state.global_step += 1

    def tail(**kw):
        peak[0] = 9
        events.append("checkpoint")

    state.record = record
    state.step_tail = tail
    state.tracker = RunTracker(None, str(tmp_path))
    attach_rollout_metrics(state)
    state.setup_rollout_metrics([("trainer", [0])])
    state.begin_rollout_iteration()
    state.observe_rollout_samples([sample()])
    monkeypatch.setattr(mod, "_forward_kwargs", lambda *a, **kw: {})
    mod._train_mini_batches(
        state,
        NS(num_iterations=1),
        [sample()],
        train_batch_size=1,
        advantages=[0],
        old_logps=[0],
        ref_logps=None,
        teacher_logps=None,
        temperature=1,
    )
    result = state.rollout_memory_metrics()
    assert result["perf/max_memory_allocated_gb"] == 2
    assert events == ["memory", "checkpoint"]  # no post-checkpoint resampling
    state.tracker.close()


def test_dropped_optimizer_tail_does_not_count_as_trained(tmp_path, monkeypatch):
    mod = importlib.import_module("swift.dev.recipe.grpo.run_grpo")
    peak = [0]
    snapshots = []

    def forward(**kw):
        peak[0] += 1

    def memory(**kwargs):
        assert kwargs == {"synchronize": True}
        snapshots.append(peak[0])
        return [{"max_memory_allocated_bytes": peak[0] * 1024**3}]

    model = NS(
        forward_backward=forward,
        clip_grad_and_step=lambda **kw: None,
        memory_stats=memory,
        device_mesh=NS(data_world_size=1),
    )
    state = NS(
        model=model,
        ga=2,
        micro_step=0,
        global_step=0,
        max_grad_norm=1,
        is_grad_sync_boundary=lambda: state.micro_step % 2 == 0,
        reached_max=lambda: False,
        should_stop=lambda: False,
        step_tail=lambda **kw: None,
    )

    def record():
        state.global_step += 1

    state.record = record
    state.tracker = RunTracker(None, str(tmp_path))
    attach_rollout_metrics(state)
    state.setup_rollout_metrics([("trainer", [0])])
    state.begin_rollout_iteration()
    state.observe_rollout_samples([sample()] * 3)
    monkeypatch.setattr(mod, "_forward_kwargs", lambda *a, **kw: {})
    mod._train_mini_batches(
        state,
        NS(num_iterations=1),
        [sample()] * 3,
        train_batch_size=1,
        advantages=[0] * 3,
        old_logps=[0] * 3,
        ref_logps=None,
        teacher_logps=None,
        temperature=1,
    )
    assert state.rollout_memory_metrics()["perf/max_memory_allocated_gb"] == 1
    assert snapshots == [1]
    assert state._iteration_facts["sample_visits"] == 2
    assert state._iteration_facts["trained_indices"] == {0, 1}
    state.tracker.close()


def test_fresh_grpo_horizon_counts_first_complete_accumulation_window():
    from swift.dev.recipe.utils import rollout_step_budget

    assert (
        rollout_step_budget(
            num_prompts=40,
            num_generations=8,
            train_batch_size=8,
            gradient_accumulation_steps=8,
            num_train_epochs=1,
            generation_batch_size=8,
        )
        == 5
    )


def test_failed_optimizer_never_records_a_successful_update(tmp_path, monkeypatch):
    mod = importlib.import_module("swift.dev.recipe.grpo.run_grpo")

    def failed_step(**kwargs):
        # clip_grad_and_step is declared sync=True on both model backends, so a worker
        # failure surfaces AT the call site -- the recipe no longer wraps it in collect().
        raise RuntimeError("optimizer failed")

    model = NS(
        device_mesh=NS(data_world_size=1),
        forward_backward=lambda **kwargs: None,
        clip_grad_and_step=failed_step,
    )
    state = NS(
        model=model,
        ga=1,
        global_step=0,
        micro_step=0,
        max_grad_norm=1,
        reached_max=lambda: False,
        should_stop=lambda: False,
        record=lambda: pytest.fail("failed optimizer must not advance the step"),
    )
    monkeypatch.setattr(mod, "_forward_kwargs", lambda *args, **kwargs: {})
    with pytest.raises(RuntimeError, match="optimizer failed"):
        mod._train_mini_batches(
            state,
            NS(num_iterations=1),
            [sample()],
            train_batch_size=1,
            advantages=[0],
            old_logps=[0],
            ref_logps=None,
            teacher_logps=None,
            temperature=1,
        )
    assert state.global_step == state.micro_step == 0


def test_auxiliary_forwards_are_submitted_before_consuming_signals():
    mod = importlib.import_module("swift.dev.recipe.grpo.run_grpo")
    launches = []

    class Pending:
        _is_lazy_collect = True

        def __call__(self):
            pytest.fail("large forward payloads must stay off the controller")

    def lane(name):
        def forward(**kwargs):
            launches.append(name)
            return Pending()

        return forward

    def advantages(samples, scored, *args):
        assert launches == ["ref", "teacher", "old"]
        assert all(isinstance(value, Pending) for value in scored.values())
        return (1, 2, 3, 4)

    cluster = NS(
        model=NS(device_mesh=NS(data_world_size=1), forward_only=lane("old")),
        reference=NS(forward_only=lane("ref")),
        teacher=NS(forward_only=lane("teacher")),
        worker=NS(privileged_features=lambda rows: [], grpo_advantages=advantages),
        sampling_params={},
    )
    config = NS(rollout_importance_sampling_mode="token_truncate", log_rollout_offpolicy_metrics=False)
    assert mod._advantage_pipeline(cluster, config, NS(), [sample()], 1, 0) == (1, 2, 3, 4)

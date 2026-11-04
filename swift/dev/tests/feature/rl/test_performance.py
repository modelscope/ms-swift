"""RL iteration accounting: actual batches, physical resources, durable records."""

import importlib
from types import SimpleNamespace as NS

import json
import pytest

from twinkle.perf import RLPerformanceRecorder, reduce_memory, sample_lengths
from swift.dev.recipe.utils import RunTracker


def sample(prompt=3, response=2, *, tool=0):
    ids = list(range(prompt + response + tool))
    labels = [-100] * prompt + ids[prompt : prompt + response] + [-100] * tool
    return NS(encoded={"input_ids": ids, "labels": labels[1:] + labels[:1]}, truncated=False)


def test_raw_training_metrics_survive_without_reporters_and_resume(tmp_path):
    for _ in range(2):
        tracker = RunTracker(None, str(tmp_path))
        tracker.log({"loss": float("nan"), "train/reward": 2.0}, 1)
        assert (tmp_path / "optimizer_metrics.jsonl").exists()
        tracker.close()
    rows = [json.loads(line) for line in (tmp_path / "optimizer_metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and rows[0]["run_id"] != rows[1]["run_id"]
    assert rows[0]["data"]["loss"] is None
    assert rows[0]["data"]["train/reward"] == 2.0


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
    assert json.loads((tmp_path / "optimizer_metrics.jsonl").read_text())["data"] == {"loss": 0.5}


def test_colocate_sync_boundary_and_no_double_wakeup(tmp_path, monkeypatch):
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
        performance_memory=lambda: [{"perf/max_memory_allocated_gb": 2}],
    )
    cluster = NS(
        model=model,
        sampler=NS(wake_up=wake, sleep=sleep),
        engine=NS(sync_weights=sync),
        sampling_params={},
        orm_scorers=[],
        prm_scorer=None,
        rollout=NS(generate=lambda *a, **kw: [sample()]),
    )
    state = NS(model=model, global_step=0, reached_max=lambda: state.global_step >= 2, should_stop=lambda: False)
    state.performance = RLPerformanceRecorder(tmp_path, [("trainer", [0])], mode="colocate")
    monkeypatch.setattr(mod, "_advantage_pipeline", lambda *a, **kw: (None,) * 4)
    monkeypatch.setattr(mod, "_prepare_routing_replay", lambda *a: None)

    def train(*a, **kw):
        events.append(("train", state.global_step))
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
    rows = [json.loads(line) for line in (tmp_path / "performance.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert all("timing_s/update_weights" in row["data"] for row in rows)
    state.performance.close()


def test_actor_memory_sample_precedes_periodic_checkpoint(tmp_path, monkeypatch):
    mod = importlib.import_module("swift.dev.recipe.grpo.run_grpo")
    events = []
    peak = [2]

    def memory():
        events.append("memory")
        return [{"perf/max_memory_allocated_gb": peak[0]}]

    model = NS(
        forward_backward=lambda **kw: None,
        clip_grad_and_step=lambda **kw: None,
        performance_memory=memory,
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
    state.performance = RLPerformanceRecorder(tmp_path, [("trainer", [0])], mode="colocate")
    state.performance.begin(0)
    state.performance.received_lengths([3], [2])
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
    result = mod._iteration_memory(state)
    assert result["perf/max_memory_allocated_gb"] == 2
    assert events == ["memory", "checkpoint"]  # no post-checkpoint resampling
    state.performance.close()


def test_dropped_optimizer_tail_does_not_count_as_trained(tmp_path, monkeypatch):
    mod = importlib.import_module("swift.dev.recipe.grpo.run_grpo")
    peak = [0]
    snapshots = []

    def forward(**kw):
        peak[0] += 1

    def memory():
        snapshots.append(peak[0])
        return [{"perf/max_memory_allocated_gb": peak[0]}]

    model = NS(
        forward_backward=forward,
        clip_grad_and_step=lambda **kw: None,
        performance_memory=memory,
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
    state.performance = RLPerformanceRecorder(tmp_path, [("trainer", [0])], mode="colocate")
    state.performance.begin(0)
    state.performance.received_lengths([3] * 3, [2] * 3)
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
    assert mod._iteration_memory(state)["perf/max_memory_allocated_gb"] == 1
    assert snapshots == [1]
    assert state.performance.sample_visits == 2
    assert state.performance.trained_indices == {0, 1}
    state.performance.close()


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

"""defect40①: evaluation observations must persist under stage='evaluation', never 'train'.

PLAN (case | input | expected | kills):
 1 eval-stage | evaluate() val-loss, RunTracker disk readback | stage=evaluation step aligned | call site drops stage=
 2 train-control | tracker.log default stage | stage=train | over-correction: all records become evaluation
 3 eval-none | evaluate_loss returns None | no persisted record, returns None | phantom eval record
 4 stage-honored | tracker.log(stage='evaluation') direct | stage=evaluation | RunTracker.log hardcodes 'train'
 5 eval-step | global_step=12 | optimizer_step=12 | wrong step binding on eval record

Oracle = persisted metrics_raw.jsonl read back from disk (independent artifact, top-level
'stage' field per twinkle MetricRecord schema), not the in-memory return value.
Reverse-verify: drop stage='evaluation' at supervised.py evaluate call site -> case 1 RED;
hardcode stage='train' in RunTracker.log -> cases 1/4 RED.
"""

from types import SimpleNamespace as NS

import json

from swift.dev.recipe import supervised
from swift.dev.recipe.supervised import _Evaluation
from swift.dev.recipe.utils import RunTracker


def _read_records(tracker, tmp_path):
    # The jsonl reporter offloads I/O to a background writer; flush before reading back.
    if tracker._metric_reporter is not None:
        tracker._metric_reporter.flush(timeout_s=10)
    path = tmp_path / 'metrics_raw.jsonl'
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _mount_evaluation(tmp_path, monkeypatch, result, global_step):
    tracker = RunTracker(None, str(tmp_path))
    ev = _Evaluation.__new__(_Evaluation)
    ev.predict_with_generate = False
    ev.model = object()
    ev.cluster = NS(eval_dataloader=object())
    ev.eval_iters = 1
    ev.manual_gc = False
    ev.manual_gc_eval = False
    ev.micro_step_fn = None
    ev.micro_batch_size = 1
    ev.eval_group_size = 1
    ev.history = []
    ev.best_step, ev.best_ckpt, ev.warned_eval_dp = 0, None, False
    ev.metric_for_best_model = None
    ev.greater_is_better = False
    ev.output_dir = str(tmp_path)
    ev.state = NS(
        global_step=global_step,
        tracker=tracker,
        best_metric=None,
        publish_state=lambda metrics: None,
        handler=NS(on_evaluate=lambda metrics: None),
    )
    monkeypatch.setattr(supervised, 'evaluate_loss', lambda *a, **k: result)
    return ev


def test_eval_record_persists_under_evaluation_stage(tmp_path, monkeypatch):
    ev = _mount_evaluation(tmp_path, monkeypatch, {'eval_loss': 0.5, 'step': 7}, global_step=7)
    result = ev.evaluate()
    assert result == {'eval_loss': 0.5, 'step': 7}
    records = _read_records(ev.state.tracker, tmp_path)
    assert len(records) == 1
    assert records[0]['stage'] == 'evaluation'
    assert records[0]['optimizer_step'] == 7
    assert records[0]['values']['eval_loss'] == 0.5


def test_train_record_default_stage_control(tmp_path):
    tracker = RunTracker(None, str(tmp_path))
    tracker.log({'loss': 1.0}, 3)
    records = _read_records(tracker, tmp_path)
    assert len(records) == 1
    assert records[0]['stage'] == 'train'
    assert records[0]['optimizer_step'] == 3


def test_eval_none_persists_nothing(tmp_path, monkeypatch):
    ev = _mount_evaluation(tmp_path, monkeypatch, None, global_step=7)
    assert ev.evaluate() is None
    assert _read_records(ev.state.tracker, tmp_path) == []


def test_log_honors_explicit_stage(tmp_path):
    tracker = RunTracker(None, str(tmp_path))
    tracker.log({'eval_loss': 0.25}, 4, stage='evaluation')
    records = _read_records(tracker, tmp_path)
    assert len(records) == 1
    assert records[0]['stage'] == 'evaluation'


def test_eval_record_step_alignment(tmp_path, monkeypatch):
    ev = _mount_evaluation(tmp_path, monkeypatch, {'eval_loss': 0.9, 'step': 12}, global_step=12)
    ev.evaluate()
    records = _read_records(ev.state.tracker, tmp_path)
    assert len(records) == 1
    assert records[0]['optimizer_step'] == 12
    assert records[0]['stage'] == 'evaluation'

"""Fast tests for the dev training tracker lifecycle."""
from __future__ import annotations
import json
import math
import sys
from types import ModuleType, SimpleNamespace

import pytest

from swift.dev.config import LoggingConfig
from swift.dev.recipe.utils import RunTracker, _public_config, fit_supervised


class _TensorBoardWriter:

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.records = []
        self.closed = False

    def add_scalar(self, key, value, step):
        self.records.append((key, value, step))

    def close(self):
        self.closed = True


def test_step_schedule_and_non_finite_loss_filter(tmp_path):
    tracker = RunTracker(
        LoggingConfig(
            report_to=['none'], logging_steps=3, logging_first_step=True, logging_nan_inf_filter=True),
        str(tmp_path))

    assert tracker.should_log(1)
    assert not tracker.should_log(2)
    assert tracker.should_log(3)
    assert tracker.log({'loss': 2.0, 'step': 1}, 1)['loss'] == 2.0
    assert tracker.log({'loss': math.nan, 'step': 2}, 2)['loss'] == 2.0
    tracker.close()


def test_epoch_and_disabled_schedules(tmp_path):
    epoch = RunTracker(LoggingConfig(report_to=['none'], logging_strategy='epoch'), str(tmp_path))
    assert not epoch.should_log(1)
    assert epoch.should_log(1, epoch_end=True)

    disabled = RunTracker(LoggingConfig(report_to=['none'], logging_strategy='no'), str(tmp_path))
    assert not disabled.should_log(1, epoch_end=True)


def test_all_reporters_receive_metrics_and_close(monkeypatch, tmp_path):
    tensorboard = ModuleType('torch.utils.tensorboard')
    tensorboard.SummaryWriter = _TensorBoardWriter
    monkeypatch.setitem(sys.modules, 'torch.utils.tensorboard', tensorboard)

    wandb = SimpleNamespace(init_calls=[], records=[], finished=False)
    wandb.init = lambda **kwargs: wandb.init_calls.append(kwargs)
    wandb.log = lambda metrics, step: wandb.records.append((metrics, step))
    wandb.finish = lambda: setattr(wandb, 'finished', True)
    monkeypatch.setitem(sys.modules, 'wandb', wandb)

    swanlab = SimpleNamespace(init_calls=[], records=[], finished=False)
    swanlab.init = lambda **kwargs: swanlab.init_calls.append(kwargs)
    swanlab.log = lambda metrics, step: swanlab.records.append((metrics, step))
    swanlab.finish = lambda: setattr(swanlab, 'finished', True)
    swanlab.login = lambda _token: None
    monkeypatch.setitem(sys.modules, 'swanlab', swanlab)

    config = LoggingConfig(
        report_to=['tensorboard', 'wandb', 'swanlab'],
        logging_steps=1,
        logging_dir=str(tmp_path / 'logs'),
        run_name='run',
        swanlab_token='secret-token',
        swanlab_secret='secret-value',
        swanlab_webhook_url='secret-url')
    tracker = RunTracker(config, str(tmp_path))
    tracker.log({'step': 1, 'loss': 1.5, 'label': 'ignored'}, 1)

    tb_writer = tracker._writers['tensorboard']
    assert tb_writer.records == [('loss', 1.5, 1)]
    assert wandb.records == [({'loss': 1.5}, 1)]
    assert swanlab.records == [({'loss': 1.5}, 1)]
    assert 'swanlab_token' not in wandb.init_calls[0]['config']
    assert 'swanlab_secret' not in swanlab.init_calls[0]['config']

    tracker.close()
    assert tb_writer.closed
    assert wandb.finished
    assert swanlab.finished
    assert tracker._writers == {}


def test_wandb_unique_prompts_are_logged_once(tmp_path):
    writer = SimpleNamespace(records=[])
    writer.Table = lambda **kwargs: kwargs
    writer.log = lambda metrics, step: writer.records.append((metrics, step))
    tracker = RunTracker(
        LoggingConfig(report_to=['none'], wandb_log_unique_prompts=True), str(tmp_path))
    tracker._writers['wandb'] = writer
    # The tracker receives the rows CompletionLog already built -- a (prompt, decoded completion)
    # pair per sample -- so the decoding is the caller's job, not the tracker's.
    rows = [
        ([{'role': 'user', 'content': 'one'}], 'a'),
        ([{'role': 'user', 'content': 'two'}], 'b'),
    ]

    tracker.log_prompts(rows, 1)
    tracker.log_prompts(rows, 2)

    assert len(writer.records) == 1
    assert writer.records[0][0]['completions']['data'][0][1] == 'a'


def test_non_main_process_does_not_initialize_reporters(monkeypatch, tmp_path):
    monkeypatch.setenv('RANK', '1')
    missing = ModuleType('wandb')
    missing.init = lambda **_kwargs: (_ for _ in ()).throw(AssertionError('must not initialize'))
    monkeypatch.setitem(sys.modules, 'wandb', missing)

    tracker = RunTracker(LoggingConfig(report_to=['wandb']), str(tmp_path))
    assert tracker._writers == {}


def test_public_config_excludes_credentials():
    public = _public_config(
        LoggingConfig(swanlab_token='token', swanlab_secret='secret', swanlab_webhook_url='webhook'))
    assert 'swanlab_token' not in public
    assert 'swanlab_secret' not in public
    assert 'swanlab_webhook_url' not in public


# -- epoch-end re-emission must not re-persist a duplicate (queue 729 / fix 30f80d013) -------------
#
# PLAN (failure surface = fit_supervised's epoch-boundary call ``tracker.log(history[-1], step,
# epoch_end=True, persist=...)`` at utils.py:2357; the bug was persist=True writing a byte-identical
# second copy of each epoch's last optimizer step into metrics_raw.jsonl):
#
# | # | case | input | expected (contract) | failure it kills |
# |---|------|-------|---------------------|------------------|
# | 1 | multi-epoch boundary persistence | fit_supervised over 3 full epochs x 4 updates, REAL RunTracker -> metrics_raw.jsonl | train-stage optimizer_step read back from disk == [1..12], each exactly once | epoch-end re-emission persisting a duplicate of each epoch's last step (X07b step-16/step-30 重号) |
# | 2 | final partial epoch at reached_max | max_steps=10, 4 updates/epoch (epoch0=4, epoch1=8, epoch2 cut at 10) | optimizer_step == [1..10] each once; the reached_max break still fires epoch-end but step 10 is not duplicated | the final-step duplicate face when the last epoch is cut short by max_steps |
# | 3 | epoch-end fan-out survives (no over-correction) | logging_strategy='epoch', real tensorboard writer | the epoch-boundary record IS fanned out to the writer at epoch end even though persist=False | over-correction that drops the epoch-boundary emission entirely (epoch logging goes silent) |
# | 4 | reverse-verify sensitivity | case 1 with RunTracker.log forced to persist=True (the old bug) | a byte-identical duplicate DOES appear (optimizer_step 4 twice) | proves case 1's assertion is sensitive to the fix, not a tautology |
#
# The persistence oracle reads the persisted artifact back from disk (not a self-consistent round
# trip); the step schedule [1..N] is the independent oracle. These are unit guards locating the
# defect -- the end-to-end 旧红→新绿 is anchored by a real multi-epoch SFT run (METRICS_RAW_epoch_verify).


def _epoch_loop_state(tracker, *, max_steps):
    """A minimal StepState driving the REAL fit_supervised + REAL RunTracker (no fake tracker)."""
    state = SimpleNamespace(
        ga=1, global_step=0, micro_step=0, epoch=0, max_steps=max_steps, history=[], tracker=tracker)
    state.reached_max = lambda: state.max_steps > 0 and state.global_step >= state.max_steps
    state.should_stop = lambda: False
    state.publish_state = lambda: None
    state.handler = SimpleNamespace(on_epoch_begin=lambda: None, on_epoch_end=lambda: None)

    def record():
        state.global_step += 1
        rec = {'step': state.global_step, 'loss': float(state.global_step)}
        tracker.log(rec, state.global_step)  # persist=True: the per-step observation
        state.history.append(rec)

    state.record = record
    return state


def _raw_train_steps(output_dir):
    rows = [json.loads(line) for line in open(output_dir / 'metrics_raw.jsonl')]
    return [r['optimizer_step'] for r in rows if r.get('stage') == 'train']


def _run_epochs(tmp_path, *, num_epochs, max_steps, logging_strategy='steps'):
    tracker = RunTracker(
        LoggingConfig(report_to=['none'], logging_steps=1, logging_strategy=logging_strategy), str(tmp_path))
    state = _epoch_loop_state(tracker, max_steps=max_steps)
    dataloader = [[{'input_ids': [i]}] for i in range(4)]  # ga=1 -> 4 updates per epoch
    fit_supervised(
        state, dataloader, SimpleNamespace(num_train_epochs=num_epochs),
        start_epoch=0, update_step=lambda group: state.record())
    tracker.close()  # flush the async jsonl backend so the artifact is complete on disk
    return _raw_train_steps(tmp_path)


def test_epoch_end_does_not_repersist_last_step(tmp_path):
    steps = _run_epochs(tmp_path, num_epochs=3, max_steps=0)
    assert steps == list(range(1, 13)), f'epoch boundary re-persisted a duplicate: {steps}'


def test_epoch_end_no_duplicate_at_reached_max_final_step(tmp_path):
    steps = _run_epochs(tmp_path, num_epochs=100, max_steps=10)
    assert steps == list(range(1, 11)), f'final partial-epoch step re-persisted: {steps}'


def test_epoch_end_fanout_survives_persist_false(tmp_path, monkeypatch):
    tensorboard = ModuleType('torch.utils.tensorboard')
    tensorboard.SummaryWriter = _TensorBoardWriter
    monkeypatch.setitem(sys.modules, 'torch.utils.tensorboard', tensorboard)
    tracker = RunTracker(
        LoggingConfig(report_to=['tensorboard'], logging_strategy='epoch'), str(tmp_path))
    state = _epoch_loop_state(tracker, max_steps=0)
    dataloader = [[{'input_ids': [i]}] for i in range(4)]
    fit_supervised(
        state, dataloader, SimpleNamespace(num_train_epochs=2),
        start_epoch=0, update_step=lambda group: state.record())
    writer = tracker._writers['tensorboard']
    # epoch strategy: no per-step fan-out, only the two epoch-boundary emissions (steps 4 and 8).
    fanned_steps = [step for (_key, _value, step) in writer.records]
    assert fanned_steps == [4, 8], f'epoch-end fan-out was suppressed by persist=False: {fanned_steps}'
    tracker.close()
    # ...and yet nothing was re-persisted: each epoch's last step appears exactly once on disk.
    assert _raw_train_steps(tmp_path) == list(range(1, 9))


def test_epoch_end_duplicate_guard_is_sensitive(tmp_path, monkeypatch):
    original = RunTracker.log

    def force_persist(self, metrics, step, *, epoch_end=False, persist=True):
        return original(self, metrics, step, epoch_end=epoch_end, persist=True)  # the old bug

    monkeypatch.setattr(RunTracker, 'log', force_persist)
    steps = _run_epochs(tmp_path, num_epochs=3, max_steps=0)
    assert steps.count(4) == 2, f'guard did not catch the re-persisted epoch-boundary duplicate: {steps}'

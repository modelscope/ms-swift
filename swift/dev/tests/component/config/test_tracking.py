"""Fast tests for the dev training tracker lifecycle."""
from __future__ import annotations
import math
import sys
from types import ModuleType, SimpleNamespace

from swift.dev.config import LoggingConfig
from swift.dev.recipe.tracking import RunTracker, _public_config


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
    prompts = [[{'role': 'user', 'content': 'one'}], [{'role': 'user', 'content': 'two'}]]
    samples = [SimpleNamespace(decoded='a'), SimpleNamespace(decoded='b')]

    tracker.log_prompts(prompts, samples, step=1)
    tracker.log_prompts(prompts, samples, step=2)

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

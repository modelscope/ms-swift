"""Epoch saves cannot capture an incomplete gradient accumulation window."""
import importlib.util
import pytest
from types import SimpleNamespace
from unittest.mock import Mock

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec('fsdp_turbo') is None, reason='requires the optional FSDPTurbo package')


def test_epoch_checkpoint_defers_until_completed_optimizer_step():
    from swift.fsdpturbo.trainer import FSDPTurboTrainer

    trainer = object.__new__(FSDPTurboTrainer)
    trainer.train_args = SimpleNamespace(eval_strategy='no', save_strategy='epoch')
    trainer._global_step = 3
    trainer._accumulated_losses = {'loss': [object()]}
    trainer._ckpt_manager = None
    trainer._save_checkpoint = Mock()
    trainer._on_epoch_end()
    trainer._save_checkpoint.assert_not_called()
    trainer._accumulated_losses['loss'].clear()
    trainer._global_step = 4
    trainer._on_checkpoint_step()
    trainer._save_checkpoint.assert_called_once()


def test_epoch_checkpoint_saves_completed_final_window():
    from swift.fsdpturbo.trainer import FSDPTurboTrainer

    trainer = object.__new__(FSDPTurboTrainer)
    trainer.train_args = SimpleNamespace(eval_strategy='no', save_strategy='epoch')
    trainer._global_step = 4
    trainer._accumulated_losses = {'loss': []}
    trainer._save_checkpoint = Mock()
    trainer._on_epoch_end()
    trainer._save_checkpoint.assert_called_once()

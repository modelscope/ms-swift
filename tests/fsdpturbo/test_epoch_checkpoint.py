"""Epoch saves cannot capture an incomplete gradient accumulation window."""
from types import SimpleNamespace
from unittest.mock import Mock

from swift.fsdpturbo.trainer import FSDPTurboTrainer


def test_epoch_checkpoint_defers_until_completed_optimizer_step():
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
    trainer = object.__new__(FSDPTurboTrainer)
    trainer.train_args = SimpleNamespace(eval_strategy='no', save_strategy='epoch')
    trainer._global_step = 4
    trainer._accumulated_losses = {'loss': []}
    trainer._save_checkpoint = Mock()
    trainer._on_epoch_end()
    trainer._save_checkpoint.assert_called_once()

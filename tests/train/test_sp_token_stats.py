# Copyright (c) ModelScope Contributors. All rights reserved.
from types import SimpleNamespace

from swift.trainers.mixin import SwiftMixin


class _BaseTrainer:
    """Stand-in for ``transformers.Trainer`` with the raw (SP-replicated) token count."""

    def __init__(self, delta: int):
        self.delta = delta

    def _track_num_input_tokens(self, inputs):
        self.state.num_input_tokens_seen += self.delta


class _Trainer(SwiftMixin, _BaseTrainer):
    pass


def _make_trainer(sp_size: int, delta: int) -> _Trainer:
    trainer = _Trainer.__new__(_Trainer)
    trainer.template = SimpleNamespace(sequence_parallel_size=sp_size)
    trainer.state = SimpleNamespace(num_input_tokens_seen=0)
    _BaseTrainer.__init__(trainer, delta)
    return trainer


def test_sp_token_stats_are_deduplicated():
    trainer = _make_trainer(sp_size=4, delta=400)
    trainer._track_num_input_tokens({})
    # The base trainer gathers over all ranks, so every token is counted sp_size times.
    assert trainer.state.num_input_tokens_seen == 100


def test_non_sp_token_stats_are_unchanged():
    trainer = _make_trainer(sp_size=1, delta=400)
    trainer._track_num_input_tokens({})
    assert trainer.state.num_input_tokens_seen == 400


def test_sp_token_stats_accumulate_across_steps():
    trainer = _make_trainer(sp_size=2, delta=530)
    trainer._track_num_input_tokens({})
    trainer._track_num_input_tokens({})
    assert trainer.state.num_input_tokens_seen == 530

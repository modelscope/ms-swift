# Copyright (c) ModelScope Contributors. All rights reserved.
"""``BaseLoss`` and ``loss_map``: the contract behind every ``--loss_type`` entry.

``BaseLoss`` is what ``SwiftMixin.create_loss_and_eval_metric`` instantiates -- it receives the
training args and the trainer, and it flags ``is_megatron`` by looking for a Megatron trainer anywhere
in the trainer's MRO, which is why a dummy class *named* ``BaseMegatronTrainer`` is enough to pin it.
``loss_map`` is the ``--loss_type`` registry, so its keys are CLI surface and every entry must be
constructible the way the mixin constructs it.
"""
import pytest

from swift.loss import BaseLoss, loss_map


class _Trainer:
    pass


class BaseMegatronTrainer:
    """A stand-in named after the real trainer: the flag is set by *name*, from the MRO."""


class _MegatronTrainer(BaseMegatronTrainer):
    pass


def test_base_loss_is_abstract():
    with pytest.raises(TypeError, match='abstract'):
        BaseLoss(args=None, trainer=_Trainer())


def test_base_loss_keeps_the_args_and_the_trainer():
    args = object()
    trainer = _Trainer()
    loss = loss_map['cross_entropy'](args=args, trainer=trainer)
    assert loss.args is args
    assert loss.trainer is trainer


def test_is_megatron_is_detected_from_the_trainer_mro():
    assert loss_map['cross_entropy'](args=None, trainer=_Trainer()).is_megatron is False
    assert loss_map['cross_entropy'](args=None, trainer=_MegatronTrainer()).is_megatron is True


def test_loss_map_still_registers_the_long_standing_names():
    """The registry's keys are CLI surface (``TrainConfig.loss`` is a free string, so nothing else
    declares the accepted names): a long-standing one must not disappear or be renamed silently.

    Membership, not exact equality -- ``loss_map`` grows as losses are added (the decision family
    ``scoring`` / ``distill_rps`` / ``smooth_brier`` / ``ce_rps``), and an equality snapshot fires on
    every addition, which is maintenance noise rather than a defect. Removal is what breaks a user's
    ``--loss``, and that is what this pins.
    """
    assert set(loss_map) >= {
        'cross_entropy',
        'cosine_similarity',
        'contrastive',
        'online_contrastive',
        'infonce',
        'pointwise_reranker',
        'listwise_reranker',
    }


def test_every_registered_loss_is_instantiable_the_way_the_mixin_builds_it():
    for name, cls in loss_map.items():
        loss = cls(args=None, trainer=_Trainer())
        assert isinstance(loss, BaseLoss), name

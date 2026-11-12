"""Recipes pass references and lightweight metadata; TrainableModel owns balancing."""

from importlib import import_module
from types import SimpleNamespace

import pytest

from swift.dev.config import RLConfig
from swift.dev.tests.component.recipe.test_grpo_micro_batch import make_state
from swift.dev.tests.component.recipe.test_recipe_micro_batch import Model
from swift.dev.utils.scoring import batch_input_lengths

LENGTHS = [2000, 1900, 1800, 1700, 300, 250, 200, 150]


def samples():
    import ray

    return [
        SimpleNamespace(
            encoded=ray.ObjectRef.from_random(),
            token_lengths=(length - 1, 1),
            rollout_logps=[index],
            messages=[],
            extra={},
        )
        for index, length in enumerate(LENGTHS)
    ]


def test_streaming_grpo_delegates_reference_balance_without_fetching_or_permuting(monkeypatch):
    import ray

    recipe = import_module("swift.dev.recipe.grpo.run_grpo")
    monkeypatch.setattr("swift.dev.utils.scoring.should_copy_inputs", lambda: False)
    monkeypatch.setattr(ray, "get", lambda *args, **kwargs: pytest.fail("driver fetched a feature"))
    rows = samples()
    progress = make_state(ga=2)
    assert (
        recipe._train_mini_batches(
            progress,
            RLConfig(rl_type="grpo"),
            rows,
            train_batch_size=4,
            advantages=list(range(8)),
            old_logps=list(range(8)),
            ref_logps=None,
            teacher_logps=None,
            temperature=1.0,
        )
        == 1
    )
    call = progress.model.calls[0]
    assert call["inputs"] == [row.encoded for row in rows]
    assert call["advantages"] == list(range(8))
    assert call["input_lengths"] == LENGTHS
    assert progress.balance_metrics == [None]


def test_ppo_policy_and_value_delegate_independently_with_aligned_signals(monkeypatch):
    import ray

    recipe = import_module("swift.dev.recipe.ppo.run_ppo")
    monkeypatch.setattr(ray, "get", lambda *args, **kwargs: pytest.fail("driver fetched a feature"))
    rows = samples()
    progress = make_state(ga=2)
    value = Model(2)
    cluster = SimpleNamespace(model=progress.model, value_model=value)
    signals = [[float(i)] for i in range(8)]
    assert (
        recipe._train_ppo_batch(
            cluster,
            progress,
            RLConfig(rl_type="ppo", num_ppo_epochs=1),
            rows,
            signals,
            signals,
            signals,
            signals,
            train_batch_size=4,
            temperature=1.0,
            ema_manager=None,
            save=lambda *args: None,
        )
        == 1
    )
    for call in (progress.model.calls[0], value.calls[0]):
        assert call["inputs"] == [row.encoded for row in rows]
        assert call["input_lengths"] == LENGTHS
    assert progress.model.calls[0]["advantages"] == signals
    assert value.calls[0]["returns"] == signals


def test_metadata_missing_does_not_fetch_or_invent_lengths():
    assert batch_input_lengths([SimpleNamespace(encoded=object())]) is None
    assert batch_input_lengths(samples()) == LENGTHS


def test_distillation_retains_remote_row_metadata():
    recipe = import_module("swift.dev.recipe.distill.run_distill")
    rows = recipe._rows_from_samples(samples())
    assert batch_input_lengths(rows) == LENGTHS


def test_single_cli_switch_is_owned_by_train_config():
    from dataclasses import fields

    from swift.dev.cli.rl import parse_rl_configs
    from swift.dev.config import TrainConfig

    assert "balance_batch" not in {item.name for item in fields(RLConfig)}
    configs = parse_rl_configs(["--rl_type", "grpo", "--dp_balance", "false"])
    assert configs["train_config"].dp_balance is False
    assert TrainConfig().dp_balance is True

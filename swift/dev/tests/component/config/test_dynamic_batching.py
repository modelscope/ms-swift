"""GRPO dynamic batching keeps its settings in TrainConfig and refuses unconnected paths."""

import pytest

from swift.dev.config import RLConfig, TrainConfig
from swift.dev.config.validate import _check_dynamic_batching


def test_grpo_dynamic_batching_cli_parameters():
    from swift.dev.cli.rl import parse_rl_configs

    configs = parse_rl_configs(
        [
            "--model",
            "unused",
            "--load_args",
            "false",
            "--dataset",
            "unused",
            "--rl_type",
            "grpo",
            "--dynamic_batching",
            "true",
            "--max_tokens_per_micro_batch",
            "4096",
            "--packing_algorithm",
            "kk",
        ]
    )
    config = configs["train_config"]
    assert config.dynamic_batching is True
    assert config.max_tokens_per_micro_batch == 4096 and config.packing_algorithm == "kk"
    _check_dynamic_batching(config, configs["rl_config"], False)


@pytest.mark.parametrize("budget", [None, 0, -1])
def test_dynamic_batching_requires_a_token_budget(budget):
    with pytest.raises(ValueError, match="max_tokens_per_micro_batch must be positive"):
        _check_dynamic_batching(
            TrainConfig(dynamic_batching=True, max_tokens_per_micro_batch=budget), RLConfig(rl_type="grpo"), False
        )


@pytest.mark.parametrize("rl_type", [None, "ppo", "dpo", "gkd"])
def test_other_recipes_do_not_silently_accept_dynamic_batching(rl_type):
    with pytest.raises(ValueError, match="only to the GRPO recipe"):
        _check_dynamic_batching(
            TrainConfig(dynamic_batching=True, max_tokens_per_micro_batch=4096),
            RLConfig(rl_type=rl_type) if rl_type else None,
            False,
        )


def test_megatron_dynamic_batching_is_rejected():
    with pytest.raises(ValueError, match="Megatron uses fixed micro-batches"):
        _check_dynamic_batching(
            TrainConfig(dynamic_batching=True, max_tokens_per_micro_batch=4096), RLConfig(rl_type="grpo"), True
        )

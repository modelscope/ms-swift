"""Exercise post-rollout balancing and keep dataset-completion/SFT paths unchanged."""

from dataclasses import fields
from importlib import import_module
from types import SimpleNamespace

import pytest

from swift.dev.config import DistributedConfig, RLConfig, TrainConfig
from swift.dev.config.validate import _check_batch_balancing
from swift.dev.recipe.batching import plan_rollout_batch
from swift.dev.tests.component.recipe.test_recipe_micro_batch import Model, state

LENGTHS = [2000, 1900, 1800, 1700, 300, 250, 200, 150]
LENGTHS16 = [2000 - i * 90 for i in range(8)] + [800 - i * 90 for i in range(8)]
ONLINE_TYPES = ["grpo", "ppo", "gkd", "opsd", "mopd", "rft"]
OFFLINE_TYPES = ["dpo", "kto", "cpo", "orpo", "simpo", "rm"]


def features(lengths=LENGTHS):
    return [{"input_ids": [index] * length, "labels": [index] * length} for index, length in enumerate(lengths)]


def progress(model):
    result = state(model)
    result.balance_batch, result.padding_free = True, False
    result.records = []

    def record(*, extra_metrics=None):
        result.global_step += 1
        result.records.append(extra_metrics or {})

    result.record = record
    # The rollout statistics methods _train_mini_batches calls; no metrics are set up here,
    # so they only need to exist and be harmless.
    from swift.dev.recipe.utils import RolloutStepState

    for name in ('setup_rollout_metrics', 'begin_rollout_iteration', 'observe_rollout_samples',
                 'record_rollout_training', 'rollout_memory_metrics', 'fold_iteration_metrics',
                 'finish_rollout_metrics'):
        setattr(result, name, getattr(RolloutStepState, name).__get__(result))
    result._metric_context = {}
    result._run_metric = None
    result._completed_batches = result._batch_wall_ns = 0
    result._rollout_groups = result._trained_samples = 0
    result._stream_previous = {}
    result._iteration_started_ns = None
    result._iteration_facts = None
    result._memory_rows = None
    return result


def test_disabled_does_not_inspect_unencoded_rows_or_model():
    assert plan_rollout_batch(SimpleNamespace(), [{}]) is None


@pytest.mark.parametrize("algorithm", ONLINE_TYPES)
def test_online_recipes_accept_balancing(algorithm):
    _check_batch_balancing(RLConfig(rl_type=algorithm, balance_batch=True), DistributedConfig(mode="ray"))


@pytest.mark.parametrize("algorithm", OFFLINE_TYPES)
def test_offline_recipes_reject_rollout_balancing(algorithm):
    with pytest.raises(ValueError, match="post-rollout"):
        _check_batch_balancing(RLConfig(rl_type=algorithm, balance_batch=True), DistributedConfig(mode="ray"))


@pytest.mark.parametrize("mode", ["local", "remote"])
def test_rank_local_and_server_modes_are_rejected(mode):
    with pytest.raises(ValueError, match="controller-side"):
        _check_batch_balancing(RLConfig(rl_type="grpo", balance_batch=True), DistributedConfig(mode=mode))


@pytest.mark.parametrize("mode", ["local", "ray", "remote"])
def test_disabled_flag_does_not_restrict_other_recipes_or_modes(mode):
    _check_batch_balancing(None, DistributedConfig(mode=mode))
    for algorithm in ONLINE_TYPES + OFFLINE_TYPES:
        _check_batch_balancing(RLConfig(rl_type=algorithm), DistributedConfig(mode=mode))


@pytest.mark.parametrize("algorithm", ["gkd", "opsd", "mopd"])
@pytest.mark.parametrize("ratio", [0.0, 0.5, 1.0])
def test_distillation_requires_a_generated_branch(algorithm, ratio):
    config = RLConfig(rl_type=algorithm, balance_batch=True, lmbda=ratio)
    if ratio == 0:
        with pytest.raises(ValueError, match="dataset-only distillation"):
            _check_batch_balancing(config, DistributedConfig(mode="ray"))
    else:
        _check_batch_balancing(config, DistributedConfig(mode="ray"))


def test_flag_is_owned_by_rl_config_and_unavailable_for_sft():
    from swift.dev.cli.rl import parse_rl_configs
    from swift.dev.cli.sft import parse_sft_configs

    configs = parse_rl_configs(["--rl_type", "grpo", "--balance_batch", "true", "--mode", "ray"])
    assert configs["rl_config"].balance_batch is True
    assert "balance_batch" not in {item.name for item in fields(TrainConfig)}
    with pytest.raises(ValueError, match="Unrecognized arguments.*balance_batch"):
        parse_sft_configs(["--balance_batch", "true"])


def test_ppo_balances_policy_and_value_for_their_own_dp_meshes():
    recipe = import_module("swift.dev.recipe.ppo.run_ppo")
    policy, critic = Model(2), Model(4)
    p = progress(policy)
    rows = features(LENGTHS16)
    samples = [SimpleNamespace(encoded=row) for row in rows]
    advantages, returns, values, old = [[[i + offset] for i in range(16)] for offset in (100, 200, 300, 400)]
    assert (
        recipe._train_ppo_batch(
            SimpleNamespace(model=policy, value_model=critic),
            p,
            RLConfig(rl_type="ppo", num_ppo_epochs=1),
            samples,
            advantages,
            returns,
            values,
            old,
            train_batch_size=4,
            temperature=1.0,
            ema_manager=None,
            save=None,
        )
        == 1
    )
    for call, keys in [
        (policy.calls[0], [("advantages", advantages), ("old_logps", old)]),
        (critic.calls[0], [("returns", returns), ("old_values", values)]),
    ]:
        indices = [row["input_ids"][0] for row in call["inputs"]]
        for key, source in keys:
            assert call[key] == [source[index] for index in indices]
    metrics = p.records[0]
    assert metrics["dp_balance/rank_tokens_mean"] == sum(LENGTHS16) / 2
    assert metrics["dp_balance/value/rank_tokens_mean"] == sum(LENGTHS16) / 4


@pytest.mark.parametrize(
    "algorithm, signal", [("gkd", "teacher_logits"), ("opsd", "teacher_logps"), ("mopd", "teacher_logps")]
)
def test_mixed_distillation_only_balances_generated_rows_and_resets_metrics(algorithm, signal):
    from swift.dev.recipe.distill.worker import DistillRow

    recipe = import_module("swift.dev.recipe.distill.run_distill")
    _check_batch_balancing(RLConfig(rl_type=algorithm, balance_batch=True), DistributedConfig(mode="ray"))
    model, heavy = Model(2), object()
    p = progress(model)
    rows = [
        DistillRow(row, [{"id": index}], {"teacher_prompt": index}) for index, row in enumerate(features(LENGTHS16))
    ]
    received = []

    def forward(batch, use_student):
        received.append((list(batch), use_student))
        return {"inputs": [row.feature for row in batch], signal: heavy, "teacher_weights": (0.3, 0.7)}

    cluster = SimpleNamespace(model=model, worker=SimpleNamespace(forward_kwargs=forward))
    for use_student in (True, False):
        assert recipe._train_rows(cluster, p, rows, use_student, train_batch_size=4, save=None) == 1
        batch, actual_source = received[-1]
        order = [row.feature["input_ids"][0] for row in batch]
        assert actual_source is use_student
        assert sorted(order) == list(range(16))
        assert (order != list(range(16))) is use_student
        assert all(row.extra["teacher_prompt"] == row.messages[0]["id"] == row.feature["input_ids"][0] for row in batch)
        assert model.calls[-1][signal] is heavy
        assert model.calls[-1]["teacher_weights"] == (0.3, 0.7)
    assert "dp_balance/workload_imbalance_after" in p.records[0]
    assert p.records[1] == {}
    assert rows == received[1][0]
    assert model.steps == p.global_step == 2


def test_chord_balances_generated_samples_and_keeps_auxiliary_sft_order():
    recipe = import_module("swift.dev.recipe.grpo.run_grpo")
    model = Model(2)
    p = progress(model)
    rows = features(LENGTHS16)
    sft_rows = features(LENGTHS16)
    batches = iter([sft_rows[start : start + 4] for start in range(0, 16, 4)])
    assert (
        recipe._train_mini_batches(
            p,
            RLConfig(rl_type="grpo", chord_mu_peak=0.3, chord_mu_valley=0.3),
            [SimpleNamespace(encoded=row, rollout_logps=[0.0]) for row in rows],
            train_batch_size=4,
            advantages=list(range(16)),
            old_logps=None,
            ref_logps=None,
            teacher_logps=None,
            temperature=1.0,
            chord_fn=lambda: next(batches),
        )
        == 1
    )
    rl, sft = model.calls
    order = [row["input_ids"][0] for row in rl["inputs"]]
    assert sorted(order) == list(range(16)) and order != list(range(16))
    assert rl["advantages"] == order
    assert sft["inputs"] == sft_rows
    assert rl["loss_weight"] == 0.7 and rl["finalize_gradients"] is False
    assert sft["loss_weight"] == 0.3 and sft["finalize_gradients"] is True
    assert "dp_balance/workload_imbalance_after" in p.records[0]
    assert not any(key.startswith("dp_balance/chord_sft/") for key in p.records[0])
    assert model.steps == p.global_step == 1 and p.micro_step == 4


def test_rft_balances_only_the_selected_update():
    recipe = import_module("swift.dev.recipe.rft.run_rft")
    model = Model(2)
    p = progress(model)
    selected = [SimpleNamespace(encoded=row) for row in features(LENGTHS16)]
    recipe._train_selected_samples(SimpleNamespace(model=model), p, selected, 4, None)
    order = [row["input_ids"][0] for row in model.calls[0]["inputs"]]
    assert sorted(order) == list(range(16)) and order != list(range(16))
    assert p.records[0]["dp_balance/workload_imbalance_after"] < p.records[0]["dp_balance/workload_imbalance_before"]

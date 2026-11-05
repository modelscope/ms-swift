"""Exercise shared balancing through real recipe loops and loss-specific record layouts."""

from importlib import import_module
from types import SimpleNamespace

import pytest

from swift.dev.config import (
    CheckpointConfig,
    DatasetConfig,
    DistributedConfig,
    ModelConfig,
    RLConfig,
    TemplateConfig,
    TrainConfig,
)
from swift.dev.recipe.batching import plan_training_batch
from swift.dev.recipe.rlhf.encoder import PreferenceBatchEncoder
from swift.dev.recipe.sft.run_sft import _supervised_micro_kwargs
from swift.dev.recipe.supervised import run_supervised
from swift.dev.tests.component.recipe.test_recipe_micro_batch import Model, state

LENGTHS = [2000, 1900, 1800, 1700, 300, 250, 200, 150]
LENGTHS16 = [2000 - i * 90 for i in range(8)] + [800 - i * 90 for i in range(8)]


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
    return result


def test_disabled_does_not_inspect_unencoded_rows_or_model():
    assert plan_training_batch(SimpleNamespace(), [{}]) is None


@pytest.mark.parametrize("mode", ["local", "remote"])
def test_rank_local_and_server_modes_are_rejected(mode):
    from swift.dev.config.validate import _check_batch_balancing

    with pytest.raises(ValueError, match="controller-side"):
        _check_batch_balancing(TrainConfig(balance_batch=True), DistributedConfig(mode=mode))
    _check_batch_balancing(TrainConfig(balance_batch=False), DistributedConfig(mode=mode))


@pytest.mark.parametrize("command", ["sft", "rl"])
def test_shared_cli_flag_is_owned_by_train_config(command):
    module = import_module(f"swift.dev.cli.{command}")
    config = getattr(module, f"parse_{command}_configs")(["--balance_batch", "true", "--mode", "ray"])
    assert config["train_config"].balance_batch is True
    assert not hasattr(config.get("rl_config"), "balance_batch")


def supervised(tmp_path, forward, rows, *, task=None, loss=None):
    model = forward.model if isinstance(forward, PreferenceBatchEncoder) else Model(2)
    cluster = SimpleNamespace(
        model=model,
        dataloader=[rows],
        total_opt_steps=1,
        eval_dataloader=None,
        template=None,
        processor=None,
        task_type=task,
    )
    history = run_supervised(
        cluster,
        ModelConfig(model="unused", task_type=task),
        TemplateConfig(),
        DatasetConfig(),
        TrainConfig(balance_batch=True, loss=loss, per_device_train_batch_size=2, max_steps=1),
        DistributedConfig(mode="ray"),
        CheckpointConfig(),
        None,
        None,
        None,
        None,
        micro_step_fn=forward,
        output_dir=str(tmp_path),
        _save_final=False,
    )
    assert len(history) == model.steps == 1
    assert "dp_balance/workload_imbalance_after" in history[0]
    return model.calls[0], history[0]


@pytest.mark.parametrize("task", ["causal_lm", "seq_cls"])
def test_supervised_entry_balances_encoded_records(tmp_path, task):
    rows = features()
    call, record = supervised(tmp_path, _supervised_micro_kwargs(task), rows, task=task)
    order = [row["input_ids"][0] for row in call["inputs"]]
    assert order != list(range(8))
    assert sorted(order) == list(range(8))
    assert all(row["labels"][0] == row["input_ids"][0] for row in call["inputs"])
    assert record["dp_balance/workload_imbalance_after"] < record["dp_balance/workload_imbalance_before"]


def test_supervised_packing_units_are_never_split(tmp_path):
    rows = [
        [
            {"input_ids": [index] * length, "labels": [index] * length},
            {"input_ids": [index + 10] * 10, "labels": [index + 10] * 10},
        ]
        for index, length in enumerate(LENGTHS)
    ]
    call, record = supervised(tmp_path, _supervised_micro_kwargs("causal_lm"), rows)
    order = [pack[0]["input_ids"][0] for pack in call["inputs"]]
    assert sorted(order) == list(range(8)) and order != list(range(8))
    assert all(pack[1]["input_ids"][0] == pack[0]["input_ids"][0] + 10 for pack in call["inputs"])
    assert record["dp_balance/grouped_input_cost_proxy"] == 1


@pytest.mark.parametrize("algorithm", ["dpo", "cpo", "orpo", "simpo", "rm", "kto"])
def test_preference_entry_preserves_pairs_reference_logps_and_labels(tmp_path, algorithm):
    import torch

    model = Model(2)
    reference = SimpleNamespace(
        forward_only=lambda *, inputs, **kwargs: {"logps": torch.tensor([[row["input_ids"][0]] for row in inputs])}
    )

    def encode(row):
        index = row["index"]
        return {
            "chosen_input_ids": [index * 2] * LENGTHS[index],
            "chosen_labels": [index * 2] * LENGTHS[index],
            "rejected_input_ids": [index * 2 + 1] * LENGTHS[index],
            "rejected_labels": [index * 2 + 1] * LENGTHS[index],
            "label": index % 2 == 0,
        }

    encoder = PreferenceBatchEncoder(
        model,
        SimpleNamespace(encode=encode),
        rl_type=algorithm,
        reference=None if algorithm == "rm" else reference,
        calculate_kl=False,
        data_parallel_size=2,
    )
    rows = [{"index": index, "messages": [{"role": "assistant", "content": str(index)}]} for index in range(8)]
    call, _ = supervised(tmp_path, encoder, rows)
    order = [row["input_ids"][0] for row in call["inputs"]]
    assert order != list(range(0, 16, 2)) if algorithm == "kto" else order != list(range(16))
    if algorithm == "kto":
        assert call["label"] == [value // 2 % 2 == 0 for value in order]
    else:
        assert all(order[i + 1] == order[i] + 1 and order[i] % 2 == 0 for i in range(0, 16, 2))
    if algorithm != "rm":
        assert call["ref_logps"].flatten().tolist() == order


@pytest.mark.parametrize("loss", ["infonce", "online_contrastive", "cosine_similarity", "contrastive"])
def test_grouped_embedding_keeps_loss_pools_and_whole_records(tmp_path, loss):
    rows = [
        {
            "anchor_input_ids": [index] * length,
            "positive_input_ids": [index + 10] * 20,
            "negative_input_ids": [[index + 20] * 30],
            "labels": [1.0, 0.0],
        }
        for index, length in enumerate(LENGTHS)
    ]
    call, record = supervised(tmp_path, _supervised_micro_kwargs("embedding"), rows, task="embedding", loss=loss)
    order = [row["anchor_input_ids"][0] for row in call["inputs"]]
    assert sorted(order) == list(range(8))
    for row in call["inputs"]:
        assert row["positive_input_ids"][0] == row["anchor_input_ids"][0] + 10
        assert row["negative_input_ids"][0][0] == row["anchor_input_ids"][0] + 20
    assert record["dp_balance/grouped_input_cost_proxy"] == 1
    assert not any("padding_ratio" in key for key in record)
    if loss == "infonce":
        for start in (0, 2):
            assert set(order[start : start + 2] + order[4 + start : 6 + start]) == set(
                list(range(start, start + 2)) + list(range(4 + start, 6 + start))
            )
    elif loss == "online_contrastive":
        assert {tuple(order[i : i + 2]) for i in range(0, 8, 2)} == {(0, 1), (2, 3), (4, 5), (6, 7)}


@pytest.mark.parametrize("task", ["reranker", "generative_reranker"])
def test_reranker_keeps_local_candidate_loss_batches(tmp_path, task):
    rows = [
        {"positive_input_ids": [index] * length, "negative_input_ids": [[index + 10] * 30], "labels": [1.0, 0.0]}
        for index, length in enumerate(LENGTHS)
    ]
    call, record = supervised(tmp_path, _supervised_micro_kwargs(task), rows, task=task)
    order = [row["positive_input_ids"][0] for row in call["inputs"]]
    assert {tuple(order[i : i + 2]) for i in range(0, 8, 2)} == {(0, 1), (2, 3), (4, 5), (6, 7)}
    assert record["dp_balance/grouped_input_cost_proxy"] == 1


@pytest.mark.parametrize("count", [2, 6, 10])
def test_info_nce_preserves_short_tail_negative_pools(count):
    p = progress(Model(2))
    data = features(list(range(count, 0, -1)))
    plan = plan_training_batch(p, data, micro_batch_kwargs={"micro_batch_size": 2}, keep_microbatch_pool=True)
    size = count // 2
    for start in range(0, size, 2):
        width = min(2, size - start)
        before = set(list(range(start, start + width)) + list(range(size + start, size + start + width)))
        after = set(plan.order[start : start + width] + plan.order[size + start : size + start + width])
        assert after == before


def test_chord_balances_the_independent_sft_lane_without_changing_loss_controls():
    recipe = import_module("swift.dev.recipe.grpo.run_grpo")
    model = Model(2)
    p = progress(model)
    rows = features()
    chunks = iter([rows[:4], rows[4:]])
    config = RLConfig(rl_type="grpo", chord_mu_peak=0.3, chord_mu_valley=0.3)
    plan = recipe._chord_forward_backward(
        model,
        config,
        {"inputs": rows, "gradient_accumulation_steps": 1, "micro_batch_size": 2},
        lambda: list(next(chunks)),
        step=0,
        num_micro_batches=2,
        state=p,
    )
    first, second = model.calls
    assert first["finalize_gradients"] is False and second["finalize_gradients"] is True
    assert first["loss_weight"] == pytest.approx(0.7) and second["loss_weight"] == pytest.approx(0.3)
    assert second["loss_key"] == "sft" and second["micro_batch_size"] == 2
    assert sorted(row["input_ids"][0] for row in second["inputs"]) == list(range(8))
    assert [row["input_ids"][0] for row in second["inputs"]] != list(range(8))
    assert "dp_balance/chord_sft/workload_imbalance_after" in plan.metrics


@pytest.mark.parametrize("kind", ["infonce", "online_contrastive"])
def test_contrastive_objective_and_embedding_gradients_survive_balancing(kind):
    import torch
    from twinkle.loss import InfonceLoss, OnlineContrastiveLoss

    p = progress(Model(2))
    p.padding_free = True  # This also makes a nontrivial permutation useful within one global micro-batch.
    lengths = LENGTHS if kind == "infonce" else LENGTHS16
    data = features(lengths)
    plan = plan_training_batch(
        p,
        data,
        micro_batch_kwargs={"micro_batch_size": 2},
        group_size=1 if kind == "infonce" else 2,
        keep_microbatch_pool=kind == "infonce",
    )
    assert plan.order != list(range(len(data)))
    count, per_record = len(data), 3 if kind == "infonce" else 2
    generator = torch.Generator().manual_seed(31)
    embeddings = torch.randn(count, per_record, 8, generator=generator, requires_grad=True)
    rank_size = count // 2

    def objective(order):
        total = embeddings.sum() * 0
        if kind == "infonce":
            loss = InfonceLoss(temperature=0.7)
            groups = [
                order[start : start + 2] + order[rank_size + start : rank_size + start + 2]
                for start in range(0, rank_size, 2)
            ]
        else:
            loss = OnlineContrastiveLoss()
            groups = [order[start : start + 2] for start in range(0, count, 2)]
        for indices in groups:
            labels = (
                torch.tensor([1, 0, 0] * len(indices))
                if kind == "infonce"
                else torch.tensor([index % 2 for index in indices])
            )
            total = total + loss({"labels": labels}, {"embeddings": embeddings[indices].reshape(-1, 8)})["loss"]
        return total

    before, after = objective(list(range(count))), objective(plan.order)
    before_grad = torch.autograd.grad(before, embeddings, retain_graph=True)[0]
    after_grad = torch.autograd.grad(after, embeddings)[0]
    torch.testing.assert_close(before, after)
    torch.testing.assert_close(before_grad, after_grad, atol=1e-6, rtol=1e-5)


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


@pytest.mark.parametrize("signal", ["teacher_logits", "teacher_logps"])
def test_distillation_reorders_rows_before_building_heavy_signals(signal):
    recipe = import_module("swift.dev.recipe.distill.run_distill")
    from swift.dev.recipe.distill.worker import DistillRow

    model, heavy = Model(2), object()
    p = progress(model)
    rows = [
        DistillRow(row, [{"id": index}], {"teacher_prompt": index}) for index, row in enumerate(features(LENGTHS16))
    ]
    received = []

    def forward(rows, use_student):
        received.extend(rows)
        return {"inputs": [row.feature for row in rows], signal: heavy, "teacher_weights": (0.3, 0.7)}

    assert (
        recipe._train_rows(
            SimpleNamespace(model=model, worker=SimpleNamespace(forward_kwargs=forward)),
            p,
            rows,
            True,
            train_batch_size=4,
            save=None,
        )
        == 1
    )
    assert [row.feature["input_ids"][0] for row in received] != list(range(16))
    assert all(row.extra["teacher_prompt"] == row.messages[0]["id"] == row.feature["input_ids"][0] for row in received)
    assert model.calls[0][signal] is heavy
    assert model.calls[0]["teacher_weights"] == (0.3, 0.7)


def test_rft_balances_only_the_selected_update():
    recipe = import_module("swift.dev.recipe.rft.run_rft")
    model = Model(2)
    p = progress(model)
    selected = [SimpleNamespace(encoded=row) for row in features(LENGTHS16)]
    recipe._train_selected_samples(SimpleNamespace(model=model), p, selected, 4, None)
    order = [row["input_ids"][0] for row in model.calls[0]["inputs"]]
    assert sorted(order) == list(range(16)) and order != list(range(16))
    assert p.records[0]["dp_balance/workload_imbalance_after"] < p.records[0]["dp_balance/workload_imbalance_before"]

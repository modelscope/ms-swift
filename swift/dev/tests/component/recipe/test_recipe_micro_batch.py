"""Exercise update submission through the real recipe entry points and shared epoch loop."""

from importlib import import_module
from types import SimpleNamespace

import pytest

from swift.dev.config import (
    CheckpointConfig,
    DatasetConfig,
    DistributedConfig,
    ModelConfig,
    RLConfig,
    RolloutConfig,
    TemplateConfig,
    TrainConfig,
)
from swift.dev.recipe.rlhf.encoder import PreferenceBatchEncoder
from swift.dev.recipe.sft.run_sft import _supervised_micro_kwargs
from swift.dev.recipe.supervised import run_supervised
from swift.dev.recipe.utils import (
    fit_supervised,
    plan_supervised_steps,
    resume_training,
    rollout_step_budget,
    training_progress,
)


class Model:
    def __init__(self, dp=1):
        self.device_mesh = SimpleNamespace(data_world_size=dp)
        self.calls, self.steps = [], 0
        self.saved = []

    def forward_backward(self, **kwargs):
        assert kwargs["gradient_accumulation_steps"] == 1
        self.calls.append(kwargs)

    def clip_grad_and_step(self, **kwargs):
        assert kwargs["gradient_accumulation_steps"] == 1
        self.steps += 1

    def calculate_metric(self, **kwargs):
        return {"loss": 0.0}

    def save(self, name, **kwargs):
        self.saved.append({"cur_step": self.steps, **kwargs})


def state(model=None, max_steps=100):
    result = SimpleNamespace(
        model=model or Model(2), ga=4, global_step=0, micro_step=0, max_steps=max_steps, max_grad_norm=1.0, history=[]
    )
    result.reached_max = lambda: result.max_steps > 0 and result.global_step >= result.max_steps
    result.should_stop = lambda: False
    result.publish_state = lambda: None
    result.handler = SimpleNamespace(
        on_epoch_begin=lambda: None, on_epoch_end=lambda: None, control=SimpleNamespace(should_save=False)
    )
    result.tracker = SimpleNamespace(log=lambda *args, **kwargs: None)

    def record():
        result.global_step += 1

    result.record = record
    result.step_tail = lambda **kwargs: None
    return result


@pytest.mark.parametrize("preference", [False, True])
@pytest.mark.parametrize("mode, dp", [("local", 1), ("ray", 2)])
def test_offline_entry_submits_full_and_short_updates(tmp_path, preference, mode, dp):
    model = Model(dp)
    train_batch_size = 2 * dp
    update_batch_size = train_batch_size * 4
    rows = [{"input_ids": [index], "labels": [index]} for index in range(update_batch_size + train_batch_size)]
    batches = [rows[index : index + train_batch_size] for index in range(0, len(rows), train_batch_size)]
    cluster = SimpleNamespace(
        model=model, dataloader=batches, total_opt_steps=2, eval_dataloader=None, template=None, processor=None
    )
    if preference:
        template = SimpleNamespace(
            encode=lambda row: {
                "chosen_input_ids": row["input_ids"],
                "chosen_labels": row["labels"],
                "rejected_input_ids": [-row["input_ids"][0]],
                "rejected_labels": row["labels"],
            }
        )
        forward = PreferenceBatchEncoder(model, template, rl_type="dpo")
    else:
        forward = _supervised_micro_kwargs("causal_lm")
    history = run_supervised(
        cluster,
        ModelConfig(model="unused"),
        TemplateConfig(),
        DatasetConfig(),
        TrainConfig(per_device_train_batch_size=2, gradient_accumulation_steps=4, max_steps=2),
        DistributedConfig(mode=mode),
        CheckpointConfig(save_steps=1),
        None,
        None,
        None,
        None,
        micro_step_fn=forward,
        output_dir=str(tmp_path),
        _save_final=False,
    )
    factor = 2 if preference else 1
    assert [len(call["inputs"]) for call in model.calls] == [update_batch_size * factor, train_batch_size * factor]
    assert [call["micro_batch_size"] for call in model.calls] == [2 * factor] * 2
    assert model.steps == len(history) == 2
    assert [training_progress(checkpoint) for checkpoint in model.saved] == [(1, 4), (2, 5)]
    if preference:
        assert [row["input_ids"][0] for row in model.calls[0]["inputs"]] == [
            value for index in range(update_batch_size) for value in (index, -index)
        ]


def test_epoch_tail_counts_actual_micro_steps_across_epochs():
    progress = state(max_steps=0)
    updates = []

    def update(group):
        updates.append((list(group), progress.micro_step))
        progress.record()

    batches = [[i * 2, i * 2 + 1] for i in range(5)]
    fit_supervised(
        progress, batches, TrainConfig(num_train_epochs=2), start_epoch=0, train_batch_size=2, update_step=update
    )
    assert updates == [(list(range(8)), 4), ([8, 9], 5), (list(range(8)), 9), ([8, 9], 10)]
    assert progress.micro_step == 10 and progress.global_step == 4


def test_partial_dataloader_batch_counts_the_actual_micro_batches():
    progress = state(max_steps=1)
    updates = []

    def update(group):
        updates.append(group)
        progress.record()

    fit_supervised(
        progress,
        [[1, 2], [3, 4], [5, 6], [7]],
        TrainConfig(),
        start_epoch=0,
        train_batch_size=2,
        update_step=update,
    )
    assert updates == [list(range(1, 8))]
    assert progress.micro_step == 4 and progress.global_step == 1


def test_resume_at_the_epoch_boundary_continues_with_full_updates():
    class Loader:
        def set_epoch(self, epoch):
            self.epoch = epoch

        def __iter__(self):
            if self.epoch == 0:
                return iter([])
            return iter([[0, 1], [2, 3], [4, 5], [6, 7]])

    progress = state(max_steps=2)
    progress.global_step, progress.micro_step = 1, 4
    submitted = []

    def update(group):
        submitted.append(group)
        progress.record()

    fit_supervised(progress, Loader(), TrainConfig(), start_epoch=0, train_batch_size=2, update_step=update)
    assert submitted == [list(range(8))]
    assert (progress.global_step, progress.micro_step, progress.epoch) == (2, 8, 1)


def test_supervised_budget_includes_short_updates_and_honors_explicit_max_steps():
    config = TrainConfig(gradient_accumulation_steps=4, num_train_epochs=2)
    assert plan_supervised_steps("sft", config, [None] * 8) == 4
    assert plan_supervised_steps("sft", config, [None] * 5) == 4
    assert plan_supervised_steps("sft", config, [None] * 1) == 2
    config.max_steps = 10
    assert plan_supervised_steps("sft", config, [None] * 5) == 10
    assert training_progress({"cur_step": 2, "component_states": {"training": {"micro_step": 5}}}) == (2, 5)


@pytest.mark.parametrize("ignore_data_skip", [False, True])
def test_public_resume_restores_update_micro_and_data_positions(tmp_path, ignore_data_skip):
    (tmp_path / "trainer_state.json").write_text("{}")
    data_position = {"epoch": 2, "offset": 8, "consumed_train_samples": 28}
    checkpoint = {"cur_step": 7, "component_states": {"training": {"micro_step": 28}, "dataloader": data_position}}
    loaded = []
    loader = SimpleNamespace(load_state=loaded.append, get_state=lambda: loaded[-1] if loaded else {})
    model = SimpleNamespace(resume_from_checkpoint=lambda *args, **kwargs: checkpoint)
    result = resume_training(
        model,
        checkpoint_config=CheckpointConfig(resume_from_checkpoint=str(tmp_path)),
        distributed_config=DistributedConfig(),
        megatron_config=None,
        tuner_config=None,
        dataloader=loader,
        ignore_data_skip=ignore_data_skip,
    )
    assert result == (7, 28, 0 if ignore_data_skip else 2)
    assert loaded == ([] if ignore_data_skip else [data_position])


def test_preference_dispatch_rejects_a_tail_that_would_split_pairs():
    encoder = PreferenceBatchEncoder(Model(2), None, rl_type="dpo", data_parallel_size=2)
    with pytest.raises(ValueError, match="whole chosen/rejected pairs"):
        encoder([{}, {}, {}], micro_batch_size=2)


@pytest.mark.parametrize("rl_type", ["dpo", "kto"])
@pytest.mark.parametrize("shared_reference", [False, True])
def test_preference_auxiliary_forwards_use_per_call_micro_batch_size(rl_type, shared_reference):
    import torch

    from swift.dev.processor import InputProcessor

    processor = InputProcessor(framework="megatron", padding_free=False)
    calls = []

    class Scorer:
        def __init__(self, reference=False):
            self.reference = reference

        def forward_only(self, *, inputs, micro_batch_size, **kwargs):
            # Exercise Megatron's actual collator, including its minimum input-size assertion.
            processor.collate_fn(
                processor.prepare_inputs(inputs), micro_batch_size=micro_batch_size, variable_seq_lengths=True
            )
            disabled = kwargs.get("disable_lora", False)
            calls.append((len(inputs), micro_batch_size, disabled))
            value = 0.0 if self.reference or disabled else 1.0
            return {"logps": torch.full((len(inputs), 2), value)}

    model = Scorer()
    reference = model if shared_reference else Scorer(reference=True)
    template = SimpleNamespace(
        encode=lambda row: {
            "chosen_input_ids": [int(row["messages"][-1]["content"]), 99],
            "chosen_labels": [int(row["messages"][-1]["content"]), -100],
            "rejected_input_ids": [int(row.get("rejected_response", "0")), 99],
            "rejected_labels": [int(row.get("rejected_response", "0")), -100],
            "label": True,
        }
    )
    encoder = PreferenceBatchEncoder(model, template, rl_type=rl_type, reference=reference)
    for batch_size in (4, 1, 4):  # Training -> evaluation -> training on the same encoder.
        calls.clear()
        rows = [{"messages": [{"role": "assistant", "content": str(i + 1)}]} for i in range(batch_size)]
        kwargs = encoder(rows, micro_batch_size=batch_size)
        sequence_batch_size = batch_size * (1 if rl_type == "kto" else 2)
        assert kwargs["micro_batch_size"] == sequence_batch_size
        expected = [(sequence_batch_size, sequence_batch_size, shared_reference)]
        if rl_type == "kto":
            expected += [(batch_size, batch_size, False), (batch_size, batch_size, shared_reference)]
            assert kwargs["z_kl"] == pytest.approx(1.0)
        assert calls == expected


def test_ppo_updates_policy_and_critic_with_aligned_auxiliary_data():
    recipe = import_module("swift.dev.recipe.ppo.run_ppo")
    policy, critic = Model(2), Model(2)
    progress = state(policy)
    samples = [SimpleNamespace(encoded={"input_ids": [i]}) for i in range(44)]
    signals = [[float(i)] for i in range(44)]
    assert (
        recipe._train_ppo_batch(
            SimpleNamespace(model=policy, value_model=critic),
            progress,
            RLConfig(rl_type="ppo", num_ppo_epochs=1),
            samples,
            signals,
            signals,
            signals,
            signals,
            train_batch_size=4,
            temperature=0.7,
            ema_manager=None,
            save=None,
        )
        == 2
    )
    assert [len(call["inputs"]) for call in policy.calls] == [16, 16]
    assert [call["micro_batch_size"] for call in policy.calls + critic.calls] == [2] * 4
    assert critic.calls[1]["returns"] == signals[16:32]
    assert policy.calls[1]["old_logps"] == signals[16:32]
    assert policy.steps == critic.steps == progress.global_step == 2
    assert progress.micro_step == 8


@pytest.mark.parametrize("count,replays", [(0, 2), (12, 2), (28, 2), (44, 3)])
def test_ppo_continuous_replay_keeps_all_signals_aligned(count, replays):
    recipe = import_module("swift.dev.recipe.ppo.run_ppo")
    policy, critic = Model(2), Model(2)
    progress = state(policy)
    samples = [SimpleNamespace(encoded={"input_ids": [i]}) for i in range(count)]
    advantages, returns, values, old_logps = [
        [[float(offset + i)] for i in range(count)] for offset in (100, 200, 300, 400)
    ]
    steps = recipe._train_ppo_batch(
        SimpleNamespace(model=policy, value_model=critic),
        progress,
        RLConfig(rl_type="ppo", num_ppo_epochs=replays),
        samples,
        advantages,
        returns,
        values,
        old_logps,
        train_batch_size=4,
        temperature=0.7,
        ema_manager=None,
        save=None,
    )
    expected = (list(range(count)) * replays)[: count * replays // 16 * 16]
    assert steps == policy.steps == critic.steps == len(expected) // 16
    assert progress.micro_step == steps * 4
    assert [row["input_ids"][0] for call in policy.calls for row in call["inputs"]] == expected
    for policy_call, critic_call in zip(policy.calls, critic.calls):
        indices = [row["input_ids"][0] for row in policy_call["inputs"]]
        assert critic_call["inputs"] == policy_call["inputs"]
        assert policy_call["advantages"] == [advantages[i] for i in indices]
        assert policy_call["old_logps"] == [old_logps[i] for i in indices]
        assert critic_call["returns"] == [returns[i] for i in indices]
        assert critic_call["old_values"] == [values[i] for i in indices]
    if count:
        assert (
            rollout_step_budget(
                num_prompts=count,
                num_generations=1,
                train_batch_size=4,
                gradient_accumulation_steps=4,
                num_train_epochs=1,
                generation_batch_size=count,
                num_iterations=replays,
            )
            == steps
        )


def test_distillation_submits_one_teacher_signal_batch_per_update():
    recipe = import_module("swift.dev.recipe.distill.run_distill")
    model = Model(2)
    progress = state(model)
    worker = SimpleNamespace(
        forward_kwargs=lambda rows, use_student: {
            "inputs": rows,
            "teacher_logits": tuple(rows),
        }
    )
    assert (
        recipe._train_rows(
            SimpleNamespace(model=model, worker=worker), progress, list(range(44)), True, train_batch_size=4, save=None
        )
        == 2
    )
    assert [call["inputs"] for call in model.calls] == [list(range(16)), list(range(16, 32))]
    assert [call["micro_batch_size"] for call in model.calls] == [2, 2]
    assert progress.micro_step == 8


@pytest.mark.parametrize("use_student", [False, True])
@pytest.mark.parametrize("colocate", [False, True])
@pytest.mark.parametrize("batch_size,generations", [(1, 4), (4, 4), (9, 4), (2, 3), (3, 6)])
def test_distill_prepares_one_global_batch_per_update(use_student, colocate, batch_size, generations, monkeypatch):
    recipe = import_module("swift.dev.recipe.distill.run_distill")
    offload = import_module("swift.dev.recipe.offload")
    from swift.dev.recipe.distill.worker import DistillRow

    progress, epochs, borrowed, generated = state(), [], [False], [0]
    sources, generation_sizes, generated_prompts = [], [], []
    prompt_count = 9

    class Loader:
        set_epoch = staticmethod(epochs.append)

        def __iter__(self):
            for start in range(0, prompt_count, batch_size):
                yield [
                    {"messages": [{"role": "user", "content": str(i)}], "row_id": i}
                    for i in range(start, min(start + batch_size, prompt_count))
                ]

    def borrow(*args):
        borrowed[0] = True

    def reload():
        borrowed[0] = False

    def forward_kwargs(rows, on_policy):
        assert not borrowed[0], "Teacher/student forwards must run after the rollout device is returned."
        sources.append(on_policy)
        return {"inputs": [row.feature for row in rows]}

    def generate(prompts, *, num_samples, **kwargs):
        prompt_ids = [int(messages[0]["content"]) for messages in prompts]
        assert prompt_ids == [extra["row_id"] for extra in kwargs["prompt_extras"]]
        generated_prompts.extend(i for i in prompt_ids for _ in range(num_samples))
        generation_sizes.append(len(prompts) * num_samples)
        start = generated[0]
        generated[0] += len(prompts) * num_samples
        return [
            SimpleNamespace(encoded={"input_ids": [i]}, messages=[], prompt=[], extra={})
            for i in range(start, generated[0])
        ]

    monkeypatch.setattr(offload, "offload_for_rollout", borrow)
    monkeypatch.setattr(offload, "sleep_for_rollout", lambda *args: None)
    progress.model.reload_to_gpu = reload
    cluster = SimpleNamespace(
        model=progress.model,
        worker=SimpleNamespace(forward_kwargs=forward_kwargs),
        colocate=colocate,
        sampler=SimpleNamespace(wake_up=lambda **kwargs: None),
        engine=SimpleNamespace(sync_weights=lambda **kwargs: None),
        rollout=SimpleNamespace(generate=generate),
        sampling_params={},
    )
    dataset = [DistillRow({"input_ids": [i]}, [], {}) for i in range(5)]
    recipe._distill_round_loop(
        cluster,
        RLConfig(rl_type="gkd", lmbda=float(use_student), num_generations=generations),
        SimpleNamespace(sleep_level=0),
        dataset,
        Loader(),
        progress,
        train_batch_size=4,
        num_train_epochs=2,
        seed=0,
        save=None,
    )
    updates_per_epoch = prompt_count * generations // 16
    expected_steps = updates_per_epoch * 2
    assert epochs == [0, 1] and sources == [use_student] * expected_steps
    assert [len(call["inputs"]) for call in progress.model.calls] == [16] * expected_steps
    assert [call["micro_batch_size"] for call in progress.model.calls] == [2] * expected_steps
    assert progress.model.steps == progress.global_step == expected_steps
    assert progress.micro_step == expected_steps * 4
    actual_ids = [row["input_ids"][0] for call in progress.model.calls for row in call["inputs"]]
    kept = list(range(expected_steps * 16))
    assert actual_ids == (kept if use_student else [i % 5 for i in kept])
    expected_prompts = [i for i in range(prompt_count) for _ in range(generations)][: updates_per_epoch * 16] * 2
    assert generated_prompts == (expected_prompts if use_student else [])
    assert generation_sizes == ([16] * expected_steps if use_student else [])
    assert (
        rollout_step_budget(
            num_prompts=prompt_count,
            num_generations=generations,
            train_batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=2,
        )
        == progress.global_step
    )


@pytest.mark.parametrize("max_steps", [1, 4])
def test_distill_selects_each_update_and_advances_only_the_off_policy_cursor(monkeypatch, max_steps):
    recipe = import_module("swift.dev.recipe.distill.run_distill")
    from swift.dev.recipe.distill.worker import DistillRow

    progress = state(Model(), max_steps=max_steps)
    progress.ga = 2
    sources, choices, synced_at, generated_prompts = [], [], [], []
    draws = iter([0.75, 0.25, 0.75, 0.25])

    def choose(seed):
        # Source selection must happen once after each completed optimizer update.
        choices.append((seed, progress.model.steps))
        return SimpleNamespace(random=lambda: next(draws))

    def generate(prompts, *, num_samples, **kwargs):
        generated_prompts.append(prompts)
        return [
            SimpleNamespace(encoded={"input_ids": [-int(messages[0]["content"]) - 1]}, messages=messages, extra=extra)
            for messages, extra in zip(prompts, kwargs["prompt_extras"])
            for _ in range(num_samples)
        ]

    def forward_kwargs(rows, use_student):
        sources.append(use_student)
        return {"inputs": [row.feature for row in rows]}

    class Loader:
        def set_epoch(self, epoch):
            assert epoch == 0

        def __iter__(self):
            for start in (0, 3, 6):
                yield [{"messages": [{"role": "user", "content": str(i)}]} for i in range(start, min(start + 3, 8))]

    monkeypatch.setattr(recipe.random, "Random", choose)
    cluster = SimpleNamespace(
        model=progress.model,
        worker=SimpleNamespace(forward_kwargs=forward_kwargs),
        colocate=False,
        engine=SimpleNamespace(sync_weights=lambda: synced_at.append(progress.model.steps)),
        rollout=SimpleNamespace(generate=generate),
        sampling_params={},
    )
    dataset = [DistillRow({"input_ids": [i]}, [], {}) for i in range(7)]
    recipe._distill_round_loop(
        cluster,
        RLConfig(rl_type="gkd", lmbda=0.5, num_generations=2),
        SimpleNamespace(),
        dataset,
        Loader(),
        progress,
        train_batch_size=2,
        num_train_epochs=2,
        seed=10,
        save=None,
    )
    assert choices == [(10 + i, i) for i in range(max_steps)]
    assert sources == [False, True, False, True][:max_steps]
    expected_rows = [[0, 1, 2, 3], [-3, -3, -4, -4], [4, 5, 6, 0], [-7, -7, -8, -8]][:max_steps]
    assert [[row["input_ids"][0] for row in call["inputs"]] for call in progress.model.calls] == expected_rows
    assert [len(prompts) for prompts in generated_prompts] == [4] * (max_steps // 2)
    assert synced_at == [1, 3][: max_steps // 2]
    assert progress.global_step == progress.model.steps == max_steps
    assert progress.micro_step == max_steps * 2


def test_distill_generates_only_four_rows_for_a_global_batch_of_four():
    recipe = import_module("swift.dev.recipe.distill.run_distill")
    progress = state(Model())
    progress.ga = 2
    generated = []

    def generate(prompts, *, num_samples, **kwargs):
        ids = [int(messages[0]["content"]) for messages in prompts for _ in range(num_samples)]
        generated.append(ids)
        return [SimpleNamespace(encoded={"input_ids": [i]}, messages=[], extra={}) for i in ids]

    # One input batch has three prompts, but it must result in three separate generation/update calls.
    batch = [{"messages": [{"role": "user", "content": str(i)}]} for i in range(3)]
    cluster = SimpleNamespace(
        model=progress.model,
        colocate=False,
        rollout=SimpleNamespace(generate=generate),
        sampling_params={},
        engine=SimpleNamespace(sync_weights=lambda: None),
        worker=SimpleNamespace(forward_kwargs=lambda rows, use_student: {"inputs": [row.feature for row in rows]}),
    )

    class Loader:
        def set_epoch(self, epoch):
            pass

        def __iter__(self):
            return iter([batch])

    recipe._distill_round_loop(
        cluster,
        RLConfig(rl_type="gkd", lmbda=1, num_generations=4),
        SimpleNamespace(),
        [],
        Loader(),
        progress,
        train_batch_size=2,
        num_train_epochs=1,
        seed=0,
        save=None,
    )
    assert generated == [[0] * 4, [1] * 4, [2] * 4]
    assert [len(call["inputs"]) for call in progress.model.calls] == [4, 4, 4]
    assert [call["micro_batch_size"] for call in progress.model.calls] == [2, 2, 2]
    assert progress.global_step == progress.model.steps == 3


@pytest.mark.parametrize(
    "sampler_mode,lmbda,expected", [("colocate", 1.0, 6), ("disaggregated", 0.5, 6), ("disaggregated", 1.0, 7)]
)
def test_distill_entry_budgets_updates_across_prompt_loader_batches(monkeypatch, sampler_mode, lmbda, expected):
    recipe = import_module("swift.dev.recipe.distill.run_distill")
    prompts = [[{"role": "user", "content": str(i)}] for i in range(5)]
    monkeypatch.setattr(recipe, "prompt_rows_from_dataset", lambda config: ([], prompts, [{}] * 5))
    monkeypatch.setattr(recipe, "initialize_rl_devices", lambda **kwargs: (None, None, None))
    monkeypatch.setattr(recipe, "distill_dataset_rows", lambda *args: [object()])
    monkeypatch.setattr("swift.dev.builders.build_dp_mesh", lambda *args: SimpleNamespace(data_world_size=1))

    class BudgetCaptured(Exception):
        pass

    def initialize(**kwargs):
        assert kwargs["max_steps"] == expected
        raise BudgetCaptured

    monkeypatch.setattr(recipe, "initialize_distill_cluster", initialize)
    with pytest.raises(BudgetCaptured):
        recipe.run_gkd(
            ModelConfig(model="unused"),
            TemplateConfig(),
            DatasetConfig(),
            TrainConfig(per_device_train_batch_size=2, gradient_accumulation_steps=2, num_train_epochs=2, max_steps=-1),
            DistributedConfig(),
            CheckpointConfig(),
            RolloutConfig(sampler_mode=sampler_mode, generation_batch_size=2),
            RLConfig(rl_type="gkd", num_generations=3, lmbda=lmbda),
        )


def test_ppo_colocate_preserves_a_single_prompt_pass_for_both_models(monkeypatch):
    recipe = import_module("swift.dev.recipe.ppo.run_ppo")
    offload = import_module("swift.dev.recipe.offload")
    progress, visits, generated = state(), [], [0]
    critic = Model(2)

    class Loader:
        def __iter__(self):
            visits.append(True)
            for size in (4, 1):
                yield [{"messages": []} for _ in range(size)]

    def generate(prompts, *, num_samples, **kwargs):
        start = generated[0]
        generated[0] += len(prompts) * num_samples
        return [SimpleNamespace(encoded={"input_ids": [i]}) for i in range(start, generated[0])]

    def gae(cluster, reward, config, samples, *args):
        signals = [[float(sample.encoded["input_ids"][0])] for sample in samples]
        return samples, signals, signals, signals, signals

    monkeypatch.setattr(offload, "offload_for_rollout", lambda *args: None)
    monkeypatch.setattr(offload, "sleep_for_rollout", lambda *args: None)
    monkeypatch.setattr(recipe, "_gae_pipeline", gae)
    progress.model.reload_to_gpu = lambda: None
    critic.reload_to_gpu = lambda: None
    cluster = SimpleNamespace(
        model=progress.model,
        value_model=critic,
        worker=SimpleNamespace(orm_scorers=[]),
        sampler=SimpleNamespace(wake_up=lambda **kwargs: None),
        engine=SimpleNamespace(sync_weights=lambda **kwargs: None),
        rollout=SimpleNamespace(generate=generate),
        sampling_params={},
    )
    recipe._colocate_loop(
        cluster,
        RLConfig(rl_type="ppo", num_generations=4, num_ppo_epochs=1),
        SimpleNamespace(),
        SimpleNamespace(sleep_level=0),
        Loader(),
        progress,
        None,
        train_batch_size=4,
        ema_manager=None,
        save=None,
    )
    assert visits == [True]
    for model in (progress.model, critic):
        assert [len(call["inputs"]) for call in model.calls] == [16]
        assert [row["input_ids"][0] for call in model.calls for row in call["inputs"]] == (list(range(16)))
        assert model.steps == 1
    assert progress.global_step == 1 and progress.micro_step == 4


def test_rft_keeps_round_saves_deferred_when_dropping_the_tail():
    recipe = import_module("swift.dev.recipe.rft.run_rft")
    progress = state()
    samples = [SimpleNamespace(encoded={"input_ids": [i]}) for i in range(47)]
    due = recipe._train_selected_samples(SimpleNamespace(model=progress.model), progress, samples, 4, 2)
    assert due is True
    assert [len(call["inputs"]) for call in progress.model.calls] == [16, 16]
    assert [call["micro_batch_size"] for call in progress.model.calls] == [2, 2]
    assert progress.global_step == 2 and progress.micro_step == 8


@pytest.mark.parametrize("recipe_name", ["ppo", "distill", "rft"])
@pytest.mark.parametrize("count", [0, 3, 4, 15, 16, 20, 32, 35])
def test_online_updates_drop_all_samples_outside_full_ga_windows(recipe_name, count):
    progress = state()
    rows = [SimpleNamespace(encoded={"input_ids": [i]}) for i in range(count)]
    worker = SimpleNamespace(forward_kwargs=lambda rows, use_student: {"inputs": rows})
    cluster = SimpleNamespace(model=progress.model, value_model=Model(2), worker=worker)
    if recipe_name == "ppo":
        recipe = import_module("swift.dev.recipe.ppo.run_ppo")
        signals = [[float(i)] for i in range(count)]
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
            save=None,
        )
    elif recipe_name == "distill":
        recipe = import_module("swift.dev.recipe.distill.run_distill")
        recipe._train_rows(cluster, progress, rows, True, train_batch_size=4, save=None)
    else:
        recipe = import_module("swift.dev.recipe.rft.run_rft")
        recipe._train_selected_samples(cluster, progress, rows, 4, 0)
    expected_sizes = [16] * (count // 16)
    assert [len(call["inputs"]) for call in progress.model.calls] == expected_sizes
    assert progress.model.steps == progress.global_step == len(expected_sizes)
    assert progress.micro_step == len(expected_sizes) * 4
    if recipe_name == "ppo":
        assert [len(call["inputs"]) for call in cluster.value_model.calls] == expected_sizes


def test_unsized_dataloader_submits_its_short_ga_tail():
    from swift.dev.recipe.utils import grouped_training_batches

    batches = iter([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
    assert list(grouped_training_batches(batches, ga=4)) == [list(range(8)), [8, 9]]

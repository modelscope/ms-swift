"""GRPO submits optimizer batches; Twinkle owns their micro-batches."""

import importlib
from types import SimpleNamespace

import pytest

from swift.dev.config import RLConfig

grpo = importlib.import_module("swift.dev.recipe.grpo.run_grpo")


class RecordingModel:
    def __init__(self, dp):
        self.device_mesh = SimpleNamespace(data_world_size=dp)
        self.calls, self.steps = [], 0

    def forward_backward(self, **kwargs):
        assert kwargs["gradient_accumulation_steps"] == 1
        self.calls.append(kwargs)

    def clip_grad_and_step(self, **kwargs):
        assert kwargs["gradient_accumulation_steps"] == 1
        self.steps += 1


def make_state(ga=4, dp=2, max_steps=100):
    state = SimpleNamespace(
        model=RecordingModel(dp), ga=ga, global_step=0, micro_step=0, max_grad_norm=1.0, max_steps=max_steps, saved=[]
    )
    state.reached_max = lambda: state.global_step >= state.max_steps
    state.should_stop = lambda: False

    def record():
        state.global_step += 1

    state.record = record
    state.step_tail = lambda save=None: save(str(state.global_step)) if save else None
    return state


def train(state, count, *, config=None, chord_fn=None, ema=None):
    rows = [
        SimpleNamespace(
            encoded={"input_ids": [i], "labels": [i], "routed_experts": [i]},
            rollout_logps=[i + 0.1],
            truncated=i % 2 == 0,
            sampling_mask=f"mask-{i}",
        )
        for i in range(count)
    ]
    return grpo._train_mini_batches(
        state,
        config or RLConfig(rl_type="grpo"),
        rows,
        train_batch_size=2 * state.model.device_mesh.data_world_size,
        advantages=[i * 10.0 for i in range(count)],
        old_logps=[[i + 0.2] for i in range(count)],
        ref_logps=[[i + 0.3] for i in range(count)],
        teacher_logps=[[i + 0.4] for i in range(count)],
        temperature=0.8,
        chord_fn=chord_fn,
        ema_manager=ema,
        save=state.saved.append,
    )


def ids(call):
    return [row["input_ids"][0] for row in call["inputs"]]


def test_full_update_keeps_input_order_and_all_auxiliary_fields_aligned():
    state = make_state()
    cfg = RLConfig(
        rl_type="grpo", enable_sampling_replay=True, router_replay_mode="R3", sdar_loss_coef=0.1, overlong_filter=True
    )
    assert train(state, 16, config=cfg) == 1
    (call,) = state.model.calls
    assert ids(call) == list(range(16))
    assert call["micro_batch_size"] == 2
    assert call["advantages"] == [i * 10.0 for i in range(16)]
    for key, offset in [("rollout_logps", 0.1), ("old_logps", 0.2), ("ref_logps", 0.3), ("teacher_logps", 0.4)]:
        assert call[key] == [[i + offset] for i in range(16)]
    assert call["sampling_masks"] == [f"mask-{i}" for i in range(16)]
    assert call["truncated"] == [i % 2 == 0 for i in range(16)]
    assert [row["routed_experts"] for row in call["inputs"]] == [[i] for i in range(16)]
    assert state.model.steps == state.global_step == 1
    assert state.micro_step == 4
    assert state.saved == ["1"]


def test_short_update_is_submitted_immediately_and_replay_updates_are_counted():
    state = make_state()
    synced = []
    cfg = RLConfig(rl_type="grpo", num_iterations=2, ref_model_sync_steps=2, ref_model_mixup_alpha=0.5)
    ema = SimpleNamespace(sync_ema=synced.append)
    assert train(state, 28, config=cfg, ema=ema) == 4
    assert [ids(call) for call in state.model.calls] == [
        list(range(16)),
        list(range(16, 28)),
        list(range(16)),
        list(range(16, 28)),
    ]
    assert state.model.steps == state.global_step == 4
    assert state.micro_step == 14  # Two replays, each with four full-update and three short-update micros.
    assert state.saved == ["1", "2", "3", "4"]
    assert synced == [0.5, 0.5]


def test_original_global_micro_tail_is_dropped_and_too_small_a_pull_still_errors():
    state = make_state()
    assert train(state, 7) == 1
    assert ids(state.model.calls[0]) == [0, 1, 2, 3]
    assert state.micro_step == 1
    with pytest.raises(RuntimeError, match="no full mini-batch"):
        train(state, 3)


def test_max_steps_stops_between_optimizer_updates():
    state = make_state(max_steps=1)
    assert train(state, 32, config=RLConfig(rl_type="grpo", num_iterations=2)) == 1
    assert len(state.model.calls) == 1
    assert state.global_step == 1
    assert state.micro_step == 4


def test_chord_submits_two_calls_per_update_using_its_independent_micro_size():
    state, offset = make_state(), [0]

    def chord():
        rows = [{"input_ids": [i], "labels": [i]} for i in range(offset[0], offset[0] + 6)]
        offset[0] += 6
        return rows

    cfg = RLConfig(rl_type="grpo", chord_mu_peak=0.3, chord_mu_valley=0.3)
    assert train(state, 20, config=cfg, chord_fn=chord) == 2
    rl, sft, short_rl, short_sft = state.model.calls
    assert [len(call["inputs"]) for call in state.model.calls] == [16, 24, 4, 6]
    assert ids(sft) == list(range(24))
    assert ids(short_sft) == list(range(24, 30))
    for call in (rl, short_rl):
        assert call["loss_key"] == "grpo" and call["loss_weight"] == 0.7
        assert call["finalize_gradients"] is False and call["micro_batch_size"] == 2
    for call in (sft, short_sft):
        assert call["loss_key"] == "sft" and call["loss_weight"] == 0.3
        assert call["finalize_gradients"] is True and call["micro_batch_size"] == 3
    assert offset[0] == 30
    assert state.model.steps == state.global_step == 2
    assert state.micro_step == 5  # Four RL/SFT pairs, then one pair in the short update.


def test_actual_micro_steps_are_saved_at_each_optimizer_boundary_and_resume_independently():
    state, checkpoints = make_state(), []
    state.saved = SimpleNamespace(
        append=lambda _: checkpoints.append(
            {
                "cur_step": state.global_step,
                "gradient_accumulation_steps": 1,
                "component_states": {"grpo": {"micro_step": state.micro_step}},
            }
        )
    )
    assert train(state, 28) == 2
    assert [grpo._grpo_training_progress(saved) for saved in checkpoints] == [(1, 4), (2, 7)]
    resumed = make_state()
    resumed.global_step, resumed.micro_step = grpo._grpo_training_progress(checkpoints[-1])
    assert train(resumed, 12) == 1
    assert (resumed.global_step, resumed.micro_step) == (3, 10)


def test_new_training_starts_with_zero_progress():
    assert grpo._grpo_training_progress({}) == (0, 0)
    assert grpo._grpo_training_progress({"cur_step": 0}) == (0, 0)


@pytest.mark.parametrize("ga", [1, 4])
def test_resume_requires_grpo_micro_steps_regardless_of_backend_ga(ga):
    with pytest.raises(ValueError, match=r"component_states\.grpo\.micro_step"):
        grpo._grpo_training_progress({"cur_step": 9, "gradient_accumulation_steps": ga})


@pytest.mark.parametrize(
    "prompts, iterations, ga, group_pull, expected",
    [
        (4, 1, 4, False, 1),
        (5, 1, 4, False, 2),  # The final prompt batch makes a short update.
        (5, 2, 4, False, 4),
        (4, 1, 1, False, 4),
        (5, 1, 4, True, 1),  # Streaming pulls require a complete group quantum.
    ],
)
def test_budget_counts_actual_optimizer_calls(prompts, iterations, ga, group_pull, expected):
    assert (
        grpo._grpo_step_budget(
            num_prompts=prompts,
            num_generations=4,
            train_batch_size=4,
            gradient_accumulation_steps=ga,
            num_train_epochs=1,
            generation_batch_size=4,
            num_iterations=iterations,
            group_pull=group_pull,
        )
        == expected
    )

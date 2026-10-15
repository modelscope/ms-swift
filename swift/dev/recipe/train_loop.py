"""Step-budget helpers for the flat online-RL recipes.

The loop classes are gone: every online recipe (GRPO / PPO / RFT / GKD / OPSD / MOPD) runs a flat body that
composes ``StepRecorder`` / ``CheckpointSaver`` (in ``swift.dev.recipe``), the per-sample streaming driver
(``_streaming``) and its algorithm component. The prompt source is no longer a hand-rolled scheduler either:
both the streaming driver and the blocking round loop ride a twinkle ``DataLoader`` over the prompt rows
(``swift.dev.builders.build_prompt_dataloader``), which owns the seeded shuffle, the epoch passes and the
resume position natively -- so the retired ``PromptBatchScheduler`` / ``PromptRowStream`` / ``PromptPosition``
classes are deleted. What remains here is the rollout-side geometry those bodies share: the step-budget
helpers (``prompt_batch_count`` / ``rollout_step_budget`` / ``resolve_rollout_max_steps``) that size the LR
horizon from the prompt set. The offline recipes and the offline preference family (run_dpo's dpo / kto /
cpo / orpo / simpo / rm) never subclassed here either -- they compose the thin StepRecorder / Evaluator /
CheckpointSaver and run their epoch loop in ``offline_loop.py``.
"""
from __future__ import annotations
import math
from typing import Optional

from swift.dev.recipe._loop_utils import num_optimizer_steps


def prompt_batch_count(num_prompts: int,
                       num_train_epochs: float,
                       generation_batch_size: Optional[int] = None) -> int:
    """Generation batches one ``num_train_epochs`` sweep of the prompt set spans.

    One batch rolls out ``generation_batch_size`` prompts (None -> the whole set, so one batch per epoch);
    an epoch spans ``ceil(num_prompts / batch_size)`` batches, and a fractional epoch still runs one
    (truncated) pass. :func:`rollout_step_budget` multiplies this count by the per-batch optimizer steps to
    size the LR horizon. This mirrors the blocking round loop's ``ceil(num_train_epochs)`` DataLoader passes.
    """
    batch_size = min(generation_batch_size or num_prompts, num_prompts)
    batches_per_epoch = math.ceil(num_prompts / batch_size)
    return max(1, math.ceil(num_train_epochs * batches_per_epoch))


def rollout_step_budget(*,
                        num_prompts: int,
                        num_generations: int,
                        train_batch_size: int,
                        gradient_accumulation_steps: int,
                        num_train_epochs: float,
                        generation_batch_size: Optional[int] = None,
                        num_iterations: int = 1) -> int:
    """Optimizer steps for ``num_train_epochs`` passes over the prompt set (the LR-scheduler horizon).

    The rollout-side mirror of ``TrainAssembly``'s dataloader budget: the blocking round loop draws
    ``prompt_batch_count`` generation batches; each rolls out ``batch_size * num_generations`` completions,
    splits into full ``train_batch_size`` mini-batches (an undersized tail is dropped, exactly as the flat
    bodies' mini-batch split does), and replays them ``num_iterations`` times. Those micro-steps become
    optimizer steps under twinkle GA (a continuous stream, since ``micro_step`` never resets across
    batches). Returns 0 when a generation batch cannot fill even one mini-batch, so the caller fails loudly
    rather than sizing a 0-step schedule.
    """
    batch_size = min(generation_batch_size or num_prompts, num_prompts)
    total_batches = prompt_batch_count(num_prompts, num_train_epochs, generation_batch_size)
    mini_batches = (batch_size * num_generations) // max(1, train_batch_size)
    micro_per_batch = mini_batches * max(1, num_iterations)
    return num_optimizer_steps(micro_per_batch * total_batches, gradient_accumulation_steps)


def resolve_rollout_max_steps(explicit_max_steps: int, budget: int, *, recipe: str) -> int:
    """The step budget an on-policy recipe trains for: explicit ``--max_steps`` wins, else the epoch budget.

    Mirrors ``TrainAssembly``'s dataloader branch (``max_steps`` if set, else derived from
    ``num_train_epochs``). Every online recipe counts optimizer steps, so ``budget`` is always
    ``rollout_step_budget`` (GRPO/RFT/GKD pass ``num_iterations=1``-or-``num_iterations``; PPO passes its
    ``num_ppo_epochs``). A non-positive budget means a generation batch cannot fill one ``train_batch_size``
    mini-batch, so no optimizer step is possible -- fail loudly (the same condition the flat bodies'
    mini-batch split guards at runtime) instead of sizing a 0-step schedule.
    """
    if explicit_max_steps and explicit_max_steps > 0:
        return explicit_max_steps
    if budget <= 0:
        raise ValueError(
            f'{recipe}: derived {budget} training steps from num_train_epochs -- a generation batch cannot '
            'fill one train_batch_size (per_device_train_batch_size * dp_size) mini-batch, so no optimizer '
            'step is possible. Enlarge the prompt dataset or num_generations, lower '
            'per_device_train_batch_size / the DP world size, or set --max_steps explicitly.')
    return budget

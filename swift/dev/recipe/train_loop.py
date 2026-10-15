"""Prompt-batch scheduling and step-budget helpers for the flat online-RL recipes.

The loop classes are gone: every online recipe (GRPO / PPO / RFT / GKD / OPSD / MOPD) runs a flat body
that composes ``StepRunner`` / ``CheckpointSaver`` (in ``swift.dev.recipe``), the per-sample streaming
driver (``_streaming``) and its algorithm component. What remains here is the rollout-side geometry those
bodies share: :class:`PromptBatchScheduler` (shuffled generation batches for the synchronous round loops),
:class:`PromptStream` (one prompt index at a time for the streaming driver) and the step-budget helpers
(``prompt_batch_count`` / ``rollout_step_budget`` / ``resolve_rollout_max_steps``) that size the LR
horizon from the prompt set. The offline recipes and the offline preference family (run_dpo's dpo / kto /
cpo / orpo / simpo / rm) never subclassed here either -- they compose the thin StepRunner / Evaluator /
CheckpointSaver and run their epoch loop in ``offline_loop.py``.
"""
from __future__ import annotations
import math
import random
from typing import Iterator, List, Optional

from swift.dev.recipe._loop_utils import num_optimizer_steps


class PromptBatchScheduler:
    """Iterate the prompt dataset in shuffled generation batches for ``num_train_epochs`` full passes.

    A *generation batch* is the slice of prompts one rollout regenerates. The on-policy loops (GRPO/PPO/
    GKD, hence RFT/OPSD/MOPD) historically rolled out the WHOLE prompt set every step, so ``num_train_epochs``
    had no meaning and only ``max_steps`` bounded training. This scheduler makes an epoch a real pass over
    the dataset: ``generation_batch_size`` prompts are drawn per batch, so one epoch spans
    ``ceil(num_prompts / generation_batch_size)`` batches, and the stream is exhausted after
    ``num_train_epochs`` passes. ``generation_batch_size=None`` (the default) draws the whole set per batch,
    which reproduces the historical one-rollout-per-epoch behaviour exactly -- sharding is opt-in.

    Batches are drawn from a seeded reshuffled permutation, so a run is reproducible; the permutation is
    refilled cyclically, so every batch is exactly ``generation_batch_size`` long (never a short tail that
    could starve a slice_dp rank). Yielded values are the GLOBAL prompt indices, so downstream ``prompt_id``
    and dataset-column lookups stay correct whether or not the set is sharded.
    """

    def __init__(self,
                 num_prompts: int,
                 *,
                 generation_batch_size: Optional[int] = None,
                 num_train_epochs: float = 1.0,
                 seed: int = 42):
        if num_prompts <= 0:
            raise ValueError(f'PromptBatchScheduler needs at least one prompt, got {num_prompts}.')
        if generation_batch_size is not None and generation_batch_size <= 0:
            raise ValueError(f'generation_batch_size must be > 0 or None, got {generation_batch_size}.')
        if num_train_epochs <= 0:
            raise ValueError(f'num_train_epochs must be > 0, got {num_train_epochs}.')
        self.num_prompts = num_prompts
        self.batch_size = min(generation_batch_size or num_prompts, num_prompts)
        self.batches_per_epoch = math.ceil(num_prompts / self.batch_size)
        # A fractional epoch still runs one (truncated) pass, so ceil -- matching assembly's dataloader budget.
        self.total_batches = max(1, math.ceil(num_train_epochs * self.batches_per_epoch))
        self._rng = random.Random(seed)
        self._pending: List[int] = []
        self._emitted = 0

    def __len__(self) -> int:
        return self.total_batches

    def __iter__(self) -> Iterator[List[int]]:
        return self

    def __next__(self) -> List[int]:
        if self._emitted >= self.total_batches:
            raise StopIteration
        while len(self._pending) < self.batch_size:
            permutation = list(range(self.num_prompts))
            self._rng.shuffle(permutation)
            self._pending.extend(permutation)
        batch = self._pending[:self.batch_size]
        del self._pending[:self.batch_size]
        self._emitted += 1
        return batch


class PromptStream:
    """Iterate the prompt dataset ONE PROMPT INDEX at a time for the streaming driver.

    The per-sample counterpart of :class:`PromptBatchScheduler`: the streaming driver admits each
    prompt's ``num_generations`` trajectories individually and backfills as completions free buffer
    slots, so it needs a stream of single prompt indices rather than fixed batches -- a batch iterator
    would rebuild exactly the head-of-line blocking the streaming driver exists to remove.

    Reproducibility mirrors the scheduler: each pass is a seeded reshuffled permutation of every global
    prompt index, and ``num_train_epochs`` passes are yielded before the stream is exhausted (a
    fractional epoch still runs one truncated-to-whole-prompts pass, ceil like the scheduler). Yielded
    values are GLOBAL prompt indices, so ``prompt_id`` stamping and dataset-column lookups stay correct.
    """

    def __init__(self, num_prompts: int, *, num_train_epochs: float = 1.0, seed: int = 42):
        if num_prompts <= 0:
            raise ValueError(f'PromptStream needs at least one prompt, got {num_prompts}.')
        if num_train_epochs <= 0:
            raise ValueError(f'num_train_epochs must be > 0, got {num_train_epochs}.')
        self.num_prompts = num_prompts
        #: Total prompts to yield across all passes (one epoch = one full permutation).
        self.total_prompts = max(1, math.ceil(num_train_epochs)) * num_prompts
        self._rng = random.Random(seed)
        self._pending: List[int] = []
        self._emitted = 0

    def __len__(self) -> int:
        return self.total_prompts

    def __iter__(self) -> Iterator[int]:
        return self

    def __next__(self) -> int:
        if self._emitted >= self.total_prompts:
            raise StopIteration
        if not self._pending:
            self._pending = list(range(self.num_prompts))
            self._rng.shuffle(self._pending)
        self._emitted += 1
        return self._pending.pop()


def prompt_batch_count(num_prompts: int,
                       num_train_epochs: float,
                       generation_batch_size: Optional[int] = None) -> int:
    """Generation batches :class:`PromptBatchScheduler` yields for ``num_train_epochs`` dataset passes.

    One batch rolls out ``generation_batch_size`` prompts (None -> the whole set, so one batch per epoch);
    an epoch spans ``ceil(num_prompts / batch_size)`` batches, and a fractional epoch still runs one
    (truncated) pass. :func:`rollout_step_budget` multiplies this count by the per-batch optimizer steps to
    size the LR horizon.
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

    The rollout-side mirror of ``TrainAssembly``'s dataloader budget: :class:`PromptBatchScheduler` draws
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

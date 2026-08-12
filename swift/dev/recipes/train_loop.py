"""SFTLoop: minimal transparent SFT training loop.

The thinnest possible SFT recipe: just iterate the dataloader, forward_backward,
clip_grad_and_step, periodically log/save. No RL policies, no mode hooks.
Backend-agnostic: works with any TrainableModel (twinkle-derived TransformersModel /
MegatronModel).

Cookbook users may copy this loop or write their own; CLI uses it as the default
SFT orchestration.
"""
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from swift.dev.model import TrainableModel

logger = logging.getLogger(__name__)


def _is_megatron_model(model) -> bool:
    """True for the dev MegatronModel (its forward_backward derives GA from the microbatch list)."""
    try:
        from swift.dev.model.megatron.model import MegatronModel
    except Exception:
        return False
    return isinstance(model, MegatronModel)


# GA step arithmetic (twinkle's grad-sync gate lags one micro-step). Lives here because the
# only current consumer is the SFT loop; not inherently SFT-specific, so if another training
# path needs it later, lift it to a shared home.
def num_optimizer_steps(num_micro_batches: int, gradient_accumulation_steps: int) -> int:
    """Optimizer-update count for a given number of micro-batches under twinkle GA.

    twinkle's do_grad_sync fires at cur_step = ga+1, 2*ga+1, ... (one micro-step late),
    so over N micro-batches the number of optimizer steps is floor((N-1)/ga) for ga>1,
    and N for ga==1. Used to size the LR scheduler's num_training_steps so it matches the
    actual number of steps taken.
    """
    ga = max(1, gradient_accumulation_steps)
    if ga == 1:
        return num_micro_batches
    return max(0, (num_micro_batches - 1) // ga)


class SFTLoop:
    """Minimal SFT training loop over a dataloader yielding list[InputFeature]."""

    def __init__(
        self,
        model: TrainableModel,
        dataloader: Any,
        *,
        max_steps: int = -1,
        num_train_epochs: float = 1.0,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        logging_steps: int = 1,
        save_steps: Optional[int] = None,
        output_dir: str = 'output',
        eval_dataloader: Any = None,
        eval_steps: Optional[int] = None,
        task: str = 'causal_lm',
    ):
        self.model = model
        self.dataloader = dataloader
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.max_grad_norm = max_grad_norm
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.output_dir = output_dir
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.eval_dataloader = eval_dataloader
        self.eval_steps = eval_steps
        # Forwarded verbatim to twinkle's forward_backward/forward_only. task='embedding' swaps the
        # lm_head for the pooling patch, so the loss reads outputs['embeddings'] instead of logits;
        # 'causal_lm' (the default) keeps the SFT path byte-identical.
        self.task = task
        # Megatron derives GA from the microbatch-list length inside forward_backward and stores
        # forward kwargs for metric accumulation; passing gradient_accumulation_steps in would
        # collide with its own accumulate() call. Detect once so fit() omits the kwarg for Megatron.
        self._is_megatron = _is_megatron_model(model)

        # optimizer step counter (increments once per GA window)
        self.global_step = 0
        self.micro_step = 0
        self.history: list = []
        self.eval_history: list = []
        # First epoch to run; advanced by resume() so cross-epoch resume doesn't replay
        # already-consumed epochs.
        self._start_epoch = 0

    def _reached_max(self) -> bool:
        return self.max_steps > 0 and self.global_step >= self.max_steps

    def _is_grad_sync_boundary(self) -> bool:
        """Whether the current micro_step is an optimizer-update boundary.

        Replicates twinkle optimizer_group.do_grad_sync loop-side (using self.micro_step as
        twinkle's cur_step, which resume() keeps aligned): boundary when ga==1, else one micro-step
        late at cur_step = ga+1, 2ga+1, .... Computed here rather than read off model.optimizer_group
        so it works for BOTH the in-process transformers model and the Ray-remote Megatron model
        (whose optimizer_group lives on the workers, not the driver handle).
        """
        ga = self.gradient_accumulation_steps
        return ga == 1 or ((self.micro_step - 1) % ga == 0 and self.micro_step > 1)

    def evaluate(self) -> Optional[dict]:
        """Run one eval pass over eval_dataloader; return normalized per-token eval metrics.

        Mirrors the train metric path on the eval side (calculate_metrics(False)):
        for each val batch call forward_only (no grad, sets eval_status + accumulates the
        prior batch's metric) then calculate_loss (fills THIS batch's eval loss/num_tokens).
        A final calculate_metrics(False) aggregates over all batches (num_tokens-normalized)
        and resets. Returns None if no eval_dataloader.
        """
        if self.eval_dataloader is None:
            return None
        for batch in self.eval_dataloader:
            self.model.forward_only(inputs=batch, task=self.task)
            # Fill eval_status loss/num_tokens for this batch. On the transformers backend the CE
            # loss is computed here (forward_only only stores inputs/outputs). On Megatron the
            # pipeline scheduler already produced the loss inside forward_only and calculate_loss
            # raises NotImplementedError -- treat that as "already populated" (recipes rely only
            # on forward_backward/forward_only, never the 3-way split).
            try:
                self.model.calculate_loss()
            except NotImplementedError:
                pass
        # model.calculate_metric is the driver-callable metric path (works for the in-process
        # transformers model and the Ray-remote Megatron model alike; it resolves the active
        # optimizer group internally on the worker).
        metrics = self.model.calculate_metric(is_training=False)
        result = {'step': self.global_step}
        if metrics.get('loss') is not None:
            result['eval_loss'] = float(metrics['loss'])
        self.eval_history.append(result)
        logger.info(f"step {self.global_step}  eval_loss={result.get('eval_loss', float('nan')):.4f}")
        return result

    def fit(self) -> list:
        """Run the loop; returns the per-optimizer-step loss history.

        The two backends express gradient accumulation differently, so fit dispatches:
          - transformers: ONE micro-batch per forward_backward; twinkle's grad-sync gate no-ops on
            non-boundary micro-steps and steps the optimizer every ga-th call. The loop mirrors that
            boundary (_is_grad_sync_boundary) only to log / count.
          - Megatron: GA is internal to a single forward_backward -- it splits an inputs LIST into
            len(inputs) microbatches and accumulates their grads before one optimizer step. So the
            loop groups ga dataloader batches into one list and calls forward_backward ONCE per
            optimizer step (cross-microbatch loss normalization is handled inside twinkle/Megatron).
        """
        if self._is_megatron:
            return self._fit_megatron()
        return self._fit_transformers()

    def _epochs(self) -> int:
        return math.ceil(self.num_train_epochs) if self.max_steps <= 0 else 10**9

    def _record_step(self) -> None:
        """Count one completed optimizer step + log / periodic save / periodic eval (shared).

        Reads the NORMALIZED per-token loss (+ grad_norm) via the driver-callable
        model.calculate_metric (works for transformers + Ray-remote Megatron; resolves the active
        optimizer group on the worker). Raw forward_backward loss under reduction='sum' is a
        token-sum, not comparable across steps -- calculate_metric divides by num_tokens and resets.
        """
        self.global_step += 1
        metrics = self.model.calculate_metric(is_training=True)
        loss = float(metrics['loss']) if metrics.get('loss') is not None else float('nan')
        record = {'step': self.global_step, 'loss': loss}
        if metrics.get('grad_norm') is not None:
            record['grad_norm'] = float(metrics['grad_norm'])
        self.history.append(record)
        if self.logging_steps and self.global_step % self.logging_steps == 0:
            gn = record.get('grad_norm')
            gn_str = f'  grad_norm={gn:.4f}' if gn is not None else ''
            logger.info(f'step {self.global_step}  loss={loss:.4f}{gn_str}')
        if self.save_steps and self.global_step % self.save_steps == 0:
            self.save(f'checkpoint-{self.global_step}')
        if (self.eval_dataloader is not None and self.eval_steps and self.global_step % self.eval_steps == 0):
            self.evaluate()

    def _final_eval(self) -> None:
        """Final eval pass so training always ends with an eval point (unless just evaluated)."""
        if self.eval_dataloader is not None and (not self.eval_history
                                                 or self.eval_history[-1].get('step') != self.global_step):
            self.evaluate()

    def _fit_transformers(self) -> list:
        """transformers GA: one micro-batch per forward_backward, twinkle grad-sync gate steps."""
        ga = self.gradient_accumulation_steps
        for epoch in range(self._start_epoch, self._epochs()):
            if self._reached_max():
                break
            # Reshuffle each epoch (BatchSamplerShard.curr_seed only advances via set_epoch);
            # the resumable wrapper relies on this to reproduce a given epoch's order.
            if hasattr(self.dataloader, 'set_epoch'):
                self.dataloader.set_epoch(epoch)
            for batch in self.dataloader:
                self.micro_step += 1
                self.model.forward_backward(inputs=batch, gradient_accumulation_steps=ga, task=self.task)
                # Boundary computed loop-side (backend-agnostic); clip_grad_and_step self-guards via
                # the same predicate on the worker, so calling it every micro-step is safe.
                is_boundary = self._is_grad_sync_boundary()
                self.model.clip_grad_and_step(max_grad_norm=self.max_grad_norm, gradient_accumulation_steps=ga)
                if is_boundary:
                    self._record_step()
                    if self._reached_max():
                        break
        self._final_eval()
        return self.history

    def _fit_megatron(self) -> list:
        """Megatron GA: group ga dataloader batches into one microbatch-list, one step per group.

        Each dataloader batch is a list[InputFeature] of size per_device_train_batch_size; ga of them
        concatenated form one optimizer step's microbatch list. forward_backward derives
        num_microbatches from that list length and accumulates internally, so exactly ONE optimizer
        step is taken per group (ga>1 is gradient-equivalent to ga=1 with a proportionally larger
        batch). gradient_accumulation_steps is NOT passed: Megatron reads GA from len(inputs) and the
        kwarg would collide with its internal metric accumulate(). micro_batch_size is NOT passed
        either: each worker picks micro_batch_size=min(2, per_rank_len) itself, and a driver-scope
        value would exceed the per-rank length (assert len(inputs) >= micro_batch_size).

        DP sharding differs by mode, and the group this loop feeds forward_backward is already the
        per-DP-rank slice in BOTH cases:
          - Ray: forward_backward's dispatch='slice_dp' splits the group across DP ranks on the
            driver before each worker runs; the loop here runs once on the driver.
          - local (torchrun): slice_dp is a no-op (no driver), so the dataloader shards by DP rank
            up front via _MegatronDPBatchSampler; this loop runs per rank over its own slice.

        On Megatron the dataloader is built with drop_last=True (see builders/dataset.py::_drop_last),
        so a trailing partial group cannot occur there: global_batch_size is an exact invariant and
        legacy drops the remainder too. The loop still handles a short group for the transformers
        path, which keeps drop_last=False.
        """
        ga = self.gradient_accumulation_steps
        for epoch in range(self._start_epoch, self._epochs()):
            if self._reached_max():
                break
            if hasattr(self.dataloader, 'set_epoch'):
                self.dataloader.set_epoch(epoch)
            group: list = []
            batches_in_group = 0
            for batch in self.dataloader:
                self.micro_step += 1
                group.extend(batch)
                batches_in_group += 1
                if batches_in_group < ga:
                    continue
                self._megatron_step(group)
                group, batches_in_group = [], 0
                if self._reached_max():
                    break
            if group and not self._reached_max():  # trailing partial group
                self._megatron_step(group)
        self._final_eval()
        return self.history

    def _megatron_step(self, microbatch_list: list) -> None:
        """One Megatron optimizer step over a microbatch list (GA internal to forward_backward)."""
        self.model.forward_backward(inputs=microbatch_list, task=self.task)
        self.model.clip_grad_and_step(max_grad_norm=self.max_grad_norm)
        self._record_step()

    def save(self, name: str = 'checkpoint-final') -> str:
        """Persist the model + full training state via twinkle's native save.

        Passes save_optimizer=True so twinkle writes optimizer.pt / scheduler.pt / scaler.pt /
        rng_state.pt / trainer_state.json (twinkle schema: cur_step / gradient_accumulation_steps
        / consumed_train_samples). dev does NOT hand-roll trainer_state; it feeds the wrapper's
        consumed_samples so the resume position is recoverable.
        """
        consumed = getattr(self.dataloader, 'consumed_samples', 0)
        return self.model.save(name, output_dir=self.output_dir, save_optimizer=True, consumed_train_samples=consumed)

    def resume(self, state: dict) -> None:
        """Seed loop counters + dataloader position from a restored twinkle trainer_state.

        state = {'cur_step', 'consumed_train_samples', 'gradient_accumulation_steps'} (twinkle
        schema; single counting source). Weights/optimizer/scheduler/RNG are already restored by
        model.resume_from_checkpoint BEFORE this call. Here we only align the loop's own counters
        and the dataloader's skip position so training + GA phase continue exactly.
        """
        cur_step = int(state['cur_step'])
        ga = self.gradient_accumulation_steps
        # micro_step must equal twinkle's cur_step so do_grad_sync phase stays aligned
        # (twinkle's optimizer_config.cur_step was already restored to cur_step).
        self.micro_step = cur_step
        # global_step (optimizer steps taken) derived from cur_step + ga (single source).
        self.global_step = num_optimizer_steps(cur_step, ga)
        # dataloader skip: reproduce the exact epoch/offset from consumed_train_samples.
        consumed = int(state['consumed_train_samples'])
        if hasattr(self.dataloader, 'skip_consumed_samples'):
            self.dataloader.skip_consumed_samples(consumed)
        # Start the epoch loop at the resume epoch so cross-epoch resume does not replay
        # already-consumed epochs (the wrapper only skips the offset within its resume epoch).
        self._start_epoch = getattr(self.dataloader, '_resume_epoch', 0)

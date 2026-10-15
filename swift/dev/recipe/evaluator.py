"""The evaluation half of a flat offline recipe: val-loss / generative eval + best-model tracking.

``Evaluator`` is the thin re-home of the eval path the old ``OfflineLoop`` carried: periodic val-loss
scoring over ``eval_dataloader``, the optional generative EvalScope path through a resident sampler,
``best_model`` tracking + reload, and the ``eval_history`` a test reads back. It owns no loop -- the
recipe (via :func:`swift.dev.recipe.offline_loop.run_offline`) decides WHEN to evaluate (``due_for_eval`` /
``final_eval`` / ``eval_on_start``) and supplies the one leaf-specific piece the val-loss forward needs:
the ``micro_step_kwargs`` callable that turns a raw eval batch into ``forward_only`` arguments.

It reads the counters / tracker / callback handler off the linked :class:`StepRunner`, so eval metrics
are logged and published onto the same shared state the training steps use.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from swift.dev.utils.logger import get_logger

if TYPE_CHECKING:
    from swift.dev.model import TrainableModel
    from swift.dev.recipe.step_runner import StepRunner

logger = get_logger()

__all__ = ['Evaluator']


class Evaluator:
    """Run val-loss / generative eval passes and track the best checkpoint for an offline recipe."""

    def __init__(
        self,
        model: 'TrainableModel',
        recorder: 'StepRunner',
        *,
        eval_dataloader: Any = None,
        eval_steps: Optional[int] = None,
        eval_iters: int = -1,
        output_dir: str = 'output',
        manual_gc: bool = False,
        manual_gc_eval: bool = True,
        predict_with_generate: bool = False,
        eval_sampler: Any = None,
        eval_template: Any = None,
        eval_datasets: Optional[List[str]] = None,
        eval_model_id: Optional[str] = None,
        eval_task_config: Optional[Dict[str, Any]] = None,
        eval_enter: Optional[Any] = None,
        eval_exit: Optional[Any] = None,
        eval_delay: int = 0,
        eval_on_start: bool = False,
        metric_for_best_model: Optional[str] = None,
        greater_is_better: Optional[bool] = None,
        load_best_model_at_end: bool = False,
        best_model_adapter_name: Optional[str] = None,
    ):
        self.model = model
        self.recorder = recorder
        self.output_dir = output_dir
        self.eval_dataloader = eval_dataloader
        self.eval_steps = eval_steps
        self.eval_iters = eval_iters
        self.manual_gc = manual_gc
        self.manual_gc_eval = manual_gc_eval
        # Warn once about the 1/N generative-eval throughput under torchrun (see _evaluate_generate).
        self._warned_eval_dp = False
        self.eval_history: list = []

        # --- generative eval: predict_with_generate swaps the val-loss path for an EvalScope benchmark
        # driven through the resident sampler the assembly wired. eval_sampler is None on the val-loss
        # path, so eval_capable() gates the generative dispatch on it. ---
        self.predict_with_generate = predict_with_generate
        self.eval_sampler = eval_sampler
        self.eval_template = eval_template
        self.eval_datasets = eval_datasets
        self.eval_model_id = eval_model_id
        self.eval_task_config = eval_task_config
        # Optional bracketing of the generative eval run with the assembly's weight-sync + memory schedule
        # for a co-resident vllm/sglang engine; both None on the transformers-live path.
        self.eval_enter = eval_enter
        self.eval_exit = eval_exit
        self.eval_delay = eval_delay
        self.eval_on_start = eval_on_start

        # --- best-model tracking: the watched metric, its best value/step, and the checkpoint that
        # produced it, reloaded at train end when load_best_model_at_end is set. ---
        self.metric_for_best_model = metric_for_best_model
        # An explicit flag wins; otherwise a metric with 'loss' in its name is smaller-is-better.
        self.greater_is_better = (greater_is_better if greater_is_better is not None else
                                  not bool(metric_for_best_model and 'loss' in metric_for_best_model.lower()))
        self.load_best_model_at_end = load_best_model_at_end
        # LoRA checkpoints reload through their adapter group; the assembly passes 'default' (the name
        # apply_tuner created) so load() targets it rather than the full-parameter weights.
        self.best_model_adapter_name = best_model_adapter_name
        self.best_metric: Optional[float] = None
        self.best_step = 0
        self.best_ckpt: Optional[str] = None

    # --- cadence gates the recipe reads -----------------------------------------------------------
    def eval_capable(self) -> bool:
        """Whether an eval path is actually wired: val-loss needs a dataloader, generative a sampler."""
        return self.eval_sampler is not None if self.predict_with_generate else self.eval_dataloader is not None

    def due_for_eval(self) -> bool:
        """Whether the current optimizer step should trigger a periodic evaluation pass."""
        if not self.eval_steps or not self.eval_capable():
            return False
        if self.recorder.global_step % self.eval_steps != 0:
            return False
        # eval_delay skips the first N optimizer steps (0 -> no skip).
        return self.recorder.global_step > self.eval_delay

    def evaluate(self, micro_step_kwargs: Callable[..., Dict[str, Any]]) -> Optional[dict]:
        """Run one evaluation pass and return its metrics dict (or None when no path is wired).

        Dispatches on ``predict_with_generate``: the default val-loss path scores ``eval_dataloader`` with
        the training loss; the generative path runs an EvalScope benchmark through the resident sampler.
        Both append to ``eval_history``, write to the tracker, update best-model tracking, and fire
        ``on_evaluate``. ``micro_step_kwargs(batch, owns_ga=True)`` is the recipe's leaf hook that encodes
        one eval batch into ``forward_only`` arguments (val-loss path only).
        """
        result = self._evaluate_generate() if self.predict_with_generate else self._evaluate_loss(micro_step_kwargs)
        if result is None:
            return None
        self.eval_history.append(result)
        self.recorder.tracker.log(result, self.recorder.global_step)
        self._track_best(result)
        self.recorder.publish_state(metrics=result)
        self.recorder.handler.on_evaluate(metrics=result)
        scalars = {
            key: value
            for key, value in result.items() if key != 'step' and isinstance(value, (int, float))
        }
        logger.info(f'step {self.recorder.global_step}  eval: {scalars}')
        return result

    def final_eval(self, micro_step_kwargs: Callable[..., Dict[str, Any]]) -> None:
        """Final eval pass so training always ends with an eval point (unless just evaluated)."""
        if self.eval_capable() and (not self.eval_history
                                     or self.eval_history[-1].get('step') != self.recorder.global_step):
            self.evaluate(micro_step_kwargs)

    def load_best_model(self) -> None:
        """Reload the best checkpoint at train end when ``load_best_model_at_end`` is set."""
        if not self.load_best_model_at_end:
            return
        if not self.best_ckpt:
            logger.warning('load_best_model_at_end is set but no eval produced metric_for_best_model; '
                           'keeping the final weights.')
            return
        if not os.path.isdir(self.best_ckpt):
            raise FileNotFoundError(
                f'load_best_model_at_end wants {self.best_ckpt} (best {self.metric_for_best_model}='
                f'{self.best_metric} at step {self.best_step}), but it does not exist: save_steps must align '
                'with eval_steps so the best step is checkpointed, and save_total_limit must not have pruned it.')
        logger.info(f'load_best_model_at_end: reloading best checkpoint {self.best_ckpt} '
                    f'({self.metric_for_best_model}={self.best_metric} at step {self.best_step}).')
        kwargs = {'adapter_name': self.best_model_adapter_name} if self.best_model_adapter_name else {}
        self.model.load(self.best_ckpt, **kwargs)

    def shutdown_eval_sampler(self) -> None:
        """Release the resident eval sampler: built once, it lives across every eval."""
        shutdown = getattr(self.eval_sampler, 'shutdown', None)
        if callable(shutdown):
            shutdown()

    # --- the two eval paths -----------------------------------------------------------------------
    def _evaluate_loss(self, micro_step_kwargs: Callable[..., Dict[str, Any]]) -> Optional[dict]:
        """Val-loss eval: normalized per-token loss over eval_dataloader (None if there is none).

        Mirrors the train metric path on the eval side (calculate_metrics(False)): for each val batch call
        forward_only (no grad, sets eval_status + accumulates the prior batch's metric) then calculate_loss
        (fills THIS batch's eval loss/num_tokens). A final calculate_metric(False) aggregates over all
        batches (num_tokens-normalized) and resets.
        """
        if self.eval_dataloader is None:
            return None
        if self.manual_gc and self.manual_gc_eval:
            import gc
            gc.collect()
        for index, batch in enumerate(self.eval_dataloader):
            if self.eval_iters > 0 and index >= self.eval_iters:
                break
            # forward_only takes the SAME arguments forward_backward does, minus gradient accumulation, so
            # the eval batch is encoded through the recipe's one hook (owns_ga=True): SFT re-scores its
            # pre-encoded batch. Without this the eval path would score a differently-shaped batch.
            self.model.forward_only(**micro_step_kwargs(batch, owns_ga=True))
            # Fill eval_status loss/num_tokens for this batch. On the transformers backend the CE loss is
            # computed here (forward_only only stores inputs/outputs). On Megatron the pipeline scheduler
            # already produced the loss inside forward_only and calculate_loss raises NotImplementedError --
            # treat that as "already populated" (recipes rely only on forward_backward/forward_only).
            try:
                self.model.calculate_loss()
            except NotImplementedError:
                pass
        # model.calculate_metric is the driver-callable metric path (works for the in-process transformers
        # model and the Ray-remote Megatron model alike; it resolves the active optimizer group internally).
        metrics = self.model.calculate_metric(is_training=False)
        result = {'step': self.recorder.global_step}
        if metrics.get('loss') is not None:
            result['eval_loss'] = float(metrics['loss'])
        for key, value in metrics.items():
            if key.startswith('loss_'):
                result[f'eval_{key}'] = float(value)
        if self.manual_gc and self.manual_gc_eval:
            import gc
            gc.collect(0)
        return result

    def _evaluate_generate(self) -> Optional[dict]:
        """Generative eval: run an EvalScope benchmark over the resident sampler (None if unwired).

        Only rank 0 drives EvalScope (a single process owns the report files); the other ranks wait at a
        barrier so the next training step stays collective-safe. The sampler -- a transformers-live wrapper
        or a weight-synced vllm/sglang engine -- is built and owned by the assembly, so this path only fires
        ``on_predict`` and runs it.
        """
        if self.eval_sampler is None:
            return None
        import torch.distributed as dist

        from swift.dev.eval import run_evaluator
        from swift.dev.recipe._loop_utils import flatten_evalscope_report
        self.recorder.handler.on_predict()
        distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        is_main = (not distributed) or dist.get_rank() == 0
        if distributed and is_main and not self._warned_eval_dp:
            # A multi-rank local/torchrun launch can only be on the transformers-live sampler (vllm/sglang
            # need ray, and a param-sharding strategy is refused by _check_eval_generation), so this path
            # runs EvalScope on rank 0 alone while the peers idle at the barrier below: correct, but at 1/N
            # of the launched GPUs. Surface it once rather than let the throughput drop pass unexplained.
            world_size = dist.get_world_size()
            logger.warning(
                f'predict_with_generate under torchrun drives EvalScope from rank 0 only (world_size='
                f'{world_size}): generation uses 1/{world_size} of the launched GPUs while the rest wait at '
                'the eval barrier. For full-throughput generative eval, run under --mode ray with '
                '--eval_sampler_backend vllm|sglang.')
            self._warned_eval_dp = True
        result: Dict[str, Any] = {'step': self.recorder.global_step}
        # eval_enter/eval_exit bracket the run with the assembly's weight-sync + memory schedule for a
        # co-resident vllm/sglang engine (None on the transformers-live path, which needs no sync). They run
        # on every rank: under ray the driver drives the remote workers, under torchrun they are collective.
        # Only rank 0 drives EvalScope itself; the others wait at the barrier.
        if self.eval_enter is not None:
            self.eval_enter()
        try:
            if is_main:
                summary = run_evaluator(
                    self.eval_sampler,
                    self.eval_template,
                    datasets=self.eval_datasets,
                    model_id=self.eval_model_id,
                    task_config=self.eval_task_config)
                result.update(flatten_evalscope_report(summary))
                result['eval_report'] = summary
            if distributed:
                dist.barrier()
        finally:
            if self.eval_exit is not None:
                self.eval_exit()
        return result

    def _track_best(self, metrics: dict) -> None:
        """Update best_metric / best_step / best_ckpt when the watched metric improves."""
        if not self.metric_for_best_model:
            return
        value = metrics.get(self.metric_for_best_model)
        if value is None:
            logger.warning(f'metric_for_best_model={self.metric_for_best_model!r} was not produced by this eval '
                           f'(got {sorted(key for key in metrics if key != "step")}); best-model tracking skipped '
                           'for this pass.')
            return
        value = float(value)
        improved = (self.best_metric is None
                    or (value > self.best_metric if self.greater_is_better else value < self.best_metric))
        if improved:
            self.best_metric = value
            self.best_step = self.recorder.global_step
            self.best_ckpt = os.path.join(self.output_dir, f'checkpoint-{self.recorder.global_step}')

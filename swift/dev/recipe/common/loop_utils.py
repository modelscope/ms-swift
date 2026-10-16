"""Backend-agnostic training primitives shared by the offline recipes and the flat online recipes.

These are the small, model/backend-agnostic building blocks both halves of ``swift.dev.recipe`` need --
the gradient-accumulation boundary + step arithmetic (twinkle's grad-sync gate lags one micro-step), the
twinkle-native checkpoint write, the manual-GC / empty-cache cadence, the EvalScope report flattening and
the "who owns gradient accumulation" capability probe. They live here rather than in ``rl/prompts.py``
(which holds the prompt-batch scheduling / step-budget helpers) so the dependency runs one way: the loop
and recipe modules both import from here, and this module imports from neither.

Perf orchestration is deliberately NOT here: turning profiling on, ticking a step and folding a window's
stage/module summary are twinkle's own capability and live in :mod:`twinkle.perf` (``Perf.initialize`` /
``Perf.advance_step`` / ``Perf.collect_step_summary``), which both loops call directly. The one perf-adjacent
helper that IS dev's own -- selecting which of the objects an assembly / loop built are genuine
``@remote_class`` actors to hand ``Perf.initialize`` -- lives here (:func:`filter_perf_actors`), because both
halves need it and it duplicates no twinkle logic.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from swift.dev.model import TrainableModel

__all__ = [
    'is_grad_sync_boundary',
    'num_optimizer_steps',
    'save_training_checkpoint',
    'flatten_evalscope_report',
    'backend_owns_gradient_accumulation',
    'filter_perf_actors',
    'start_manual_gc',
    'collect_manual_gc',
    'collect_empty_cache',
    'finish_manual_gc',
]


def filter_perf_actors(candidates: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only the genuine ``@remote_class`` actors from a ``label -> object`` candidate map.

    A dev assembly / loop owns a mix of remote actors and plain in-process objects (a local dataloader, the
    template). ``twinkle.infra`` injects the ``perf_on`` control handle onto every ``@remote_class`` and only
    those, so ``hasattr(handle, 'perf_on')`` is exactly the "is a profileable actor" test -- and
    ``Perf.initialize`` must be handed only actors. Nothing is lost by dropping a plain object: under local
    mode its ``@timed`` spans still land in the one shared driver recorder (labelled ``driver``), and under Ray
    only the actors that own worker ranks have per-rank timings to dump. ``None`` candidates are dropped too, so
    a caller can list every object it might have built without guarding each one.
    """
    return {label: handle for label, handle in candidates.items() if handle is not None and hasattr(handle, 'perf_on')}


def backend_owns_gradient_accumulation(model) -> bool:
    """True when the backend's ``forward_backward`` internalizes gradient accumulation.

    fit() forks on exactly one axis: who schedules the GA micro-steps. Most backends (transformers) leave
    it to the loop -- one micro-batch per ``forward_backward``, the optimizer stepped every ga-th call. A
    backend that owns its own micro-batch schedule needs the loop to instead group the ga batches and call
    ``forward_backward`` once per optimizer step; in dev that is Megatron, whose 1F1B pipeline derives GA
    from the length of the micro-batch LIST it is handed. A future such backend (e.g. torchtitan/veomni
    running pipeline parallelism) is recognized HERE, so the dispatch stays a two-way fork on this
    capability rather than growing a branch per backend name.
    """
    try:
        from twinkle.model.megatron import MegatronModel
    except Exception:
        return False
    return isinstance(model, MegatronModel)


def is_grad_sync_boundary(micro_step: int, gradient_accumulation_steps: int) -> bool:
    """Whether ``micro_step`` is an optimizer-update boundary, computed driver-side.

    Replicates twinkle's ``OptimizerGroup.do_grad_sync`` (a pure function of ``cur_step`` and the
    accumulation size: boundary when ga==1, else one micro-step late at cur_step = ga+1, 2ga+1, ...) using
    the loop's own ``micro_step`` as ``cur_step``, which the loops advance once per ``forward_backward`` and
    ``resume()`` keeps aligned. It is a module-level helper rather than a read of ``model.optimizer_group``
    because that group lives on the WORKERS: under Ray the driver-side model is a proxy with no such
    attribute, so both the in-process transformers loop and the Ray-remote loops (SFT/GRPO/RFT) must derive
    the boundary here instead of reaching into the proxy.
    """
    ga = gradient_accumulation_steps
    return ga == 1 or ((micro_step - 1) % ga == 0 and micro_step > 1)


def flatten_evalscope_report(summary: Any) -> Dict[str, float]:
    """Flatten EvalScope report rows into a ``{metric_name: score}`` dict.

    ``summarize`` returns one report dict per benchmark (an ``evalscope.report.Report``): its overall
    ``score`` is keyed by the benchmark ``name``, and each entry in ``metrics`` contributes its own
    ``name``/``score``. ``metric_for_best_model`` on the generative path names one of these keys, so
    best-model tracking and ``on_evaluate`` read a flat mapping rather than the nested report.
    """
    flat: Dict[str, float] = {}
    for row in summary or []:
        if not isinstance(row, dict):
            continue
        if row.get('name') is not None and row.get('score') is not None:
            flat[str(row['name'])] = float(row['score'])
        for metric in row.get('metrics') or []:
            if isinstance(metric, dict) and metric.get('name') is not None and metric.get('score') is not None:
                flat[str(metric['name'])] = float(metric['score'])
    return flat


# GA step arithmetic (twinkle's grad-sync gate lags one micro-step). Shared by the SFT loop (which sizes
# its LR-scheduler horizon from a dataloader) and the on-policy RL budget (which sizes it from a
# prompt-set pass count), so it lives here rather than in either consumer.
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


def start_manual_gc(enabled: bool) -> Optional[bool]:
    """Disable automatic GC for a training loop and perform one full collection."""
    if not enabled:
        return None
    import gc
    was_enabled = gc.isenabled()
    gc.disable()
    gc.collect()
    return was_enabled


def collect_manual_gc(enabled: bool, interval: int, step: int) -> None:
    """Run the configured periodic full collection after an optimizer step."""
    if enabled and interval and step % interval == 0:
        import gc
        gc.collect()


def collect_empty_cache(model: 'TrainableModel', interval: Optional[int], step: int) -> None:
    """Release the device caching allocator every ``interval`` optimizer steps (torch_empty_cache_steps).

    A remote model call, not a driver-local ``torch.cuda.empty_cache()``: under Ray the loop runs on the
    driver while the tensors live on the workers, so the release has to be dispatched to them (the model
    method is ``dispatch='all'``). None/0 disables it.
    """
    if interval and step % interval == 0:
        model.empty_cache()


def finish_manual_gc(was_enabled: Optional[bool]) -> None:
    """Restore the interpreter's automatic-GC state after a loop exits."""
    if was_enabled:
        import gc
        gc.enable()


def save_training_checkpoint(model: 'TrainableModel', name: str, *, output_dir: str,
                             consumed_train_samples: int = 0, component_states: Optional[Dict[str, dict]] = None,
                             no_save_optim: bool = False, no_save_rng: bool = False,
                             safe_serialization: bool = True, max_shard_size: str = '5GB',
                             save_total_limit: Optional[int] = None,
                             async_save: bool = False) -> str:
    """Save one trainable model, then retain only the newest numbered/final checkpoints."""
    kwargs = {
        'consumed_train_samples': consumed_train_samples,
        'component_states': component_states or {},
        'safe_serialization': safe_serialization,
        'max_shard_size': max_shard_size,
        'save_total_limit': save_total_limit,
        # TransformersModel accepts extra save kwargs; MegatronModel consumes this one.
        'no_save_rng': no_save_rng,
        'async_save': async_save,
    }
    return model.save(name, output_dir=output_dir, save_optimizer=not no_save_optim, **kwargs)

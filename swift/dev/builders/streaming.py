# Copyright (c) ModelScope Contributors. All rights reserved.
"""The streaming rollout over already-built resources: rollout, checkpoint manager, driver.

Every online recipe (GRPO / PPO / distillation / RFT) and the infer component build their stream
here, so the config -> knob mapping (``async_mode`` -> staleness, ``parameter_sync_step`` -> sync
interval, the session and thread bounds of the rollout and env replicas, the placement groups) is
written once. Scoring, the group verdicts (``on_group``) and the advantage stay with the callers.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

#: CPU groups of a remote run: the rollout replicas (``rollout_num_workers`` ranks), the sandbox env
#: replicas (``sandbox_num_workers``) and one rank for the driver, the checkpoint manager and the
#: scoring / reward workers (they only call the handles they hold).
ROLLOUT_GROUP = 'rollout'
ENV_GROUP = 'env'
CONTROL_GROUP = 'control'


def max_staleness_of(rollout_config: Any) -> int:
    """How many policy versions a trained group may lag: 0 / 1 / ``max_staleness`` by ``async_mode``."""
    if rollout_config.async_mode == 'none':
        return 0
    if rollout_config.async_mode == 'one_step_off':
        return 1
    return rollout_config.max_staleness


def inflight_rows(rollout_config: Any, batch_size: int, num_generations: int) -> int:
    """The episodes the driver may run at once: its admission bound plus one baseline per group.

    The driver admits ``(max_staleness + 1) * parameter_sync_step * batch_size`` rows ahead of the
    trainer; a remax baseline adds one episode per group of ``num_generations``.
    """
    rows = (max_staleness_of(rollout_config) + 1) * rollout_config.parameter_sync_step * batch_size
    return rows + rows // max(1, num_generations)


def rollout_sampling_params(max_tokens: Optional[int], generation_config: Any = None) -> Dict[str, Any]:
    """The rollout's sampling knobs as a dict: ``max_tokens`` plus the set temperature / top_p / top_k."""
    params: Dict[str, Any] = {}
    if max_tokens is not None:
        params['max_tokens'] = max_tokens
    for key in ('temperature', 'top_p', 'top_k'):
        value = getattr(generation_config, key, None)
        if value is not None:
            params[key] = value
    return params


def offload_kwargs(rollout_config: Any) -> Dict[str, Any]:
    """What a colocated trainer moves off the device while the sampler generates (``offload_to_cpu``).

    Both switches false keeps parameters and optimizer state resident and still releases the
    trainer's caching allocator so the colocated engine can map its KV cache.
    """
    kwargs: Dict[str, Any] = {
        'offload_model': bool(rollout_config.offload_model),
        'offload_optimizer': bool(rollout_config.offload_optimizer),
    }
    reservation = getattr(rollout_config, 'vllm_gpu_memory_utilization', None)
    if reservation is not None and not rollout_config.offload_model and not rollout_config.offload_optimizer:
        kwargs['gc_min_free_fraction'] = min(1.0, float(reservation) + 0.05)
    return kwargs


def build_rollout(sampler: Any,
                  template: Any,
                  rollout_config: Any,
                  *,
                  sampling_params: Dict[str, Any],
                  remote: bool = False,
                  inflight: Optional[int] = None,
                  **kwargs: Any) -> Any:
    """One ``MultiTurnRollout`` with the run's tools; ``kwargs`` go to its constructor.

    ``remote`` places it on :data:`ROLLOUT_GROUP` (and the sandbox env on :data:`ENV_GROUP`);
    ``inflight`` (see :func:`inflight_rows`) sizes the threads of each replica so every running
    episode can hold a call. A single-turn rollout is ``max_turns`` unset.
    """
    from twinkle.data_format import SamplingParams
    from twinkle_agentic.rollout import MultiTurnRollout

    from .sandbox import build_tools

    env, tool_manager = build_tools(rollout_config, env_group=(ENV_GROUP if remote else None), inflight=inflight)
    placement: Dict[str, Any] = {}
    if remote:
        placement['remote_group'] = ROLLOUT_GROUP
        if inflight is not None:
            placement['max_concurrency'] = inflight
    return MultiTurnRollout(
        sampler,
        template,
        sampling_params=SamplingParams(**sampling_params),
        max_turns=(getattr(rollout_config, 'max_turns', None) or 1),
        max_trajectory_tokens=getattr(rollout_config, 'max_trajectory_tokens', None),
        env=env,
        tool_manager=tool_manager,
        **kwargs,
        **placement)


def build_checkpoint_manager(model: Any, sampler: Any, rollout_config: Any, *, remote: bool = False) -> Any:
    """The policy -> sampler weight sync: colocate wakes / offloads around it, standalone pauses generation."""
    from twinkle.checkpoint_engine import CheckpointEngineManager

    colocate = rollout_config.sampler_mode == 'colocate'
    return CheckpointEngineManager(
        model=model,
        sampler=sampler,
        mode=('colocate' if colocate else 'standalone'),
        sync_backend=rollout_config.weight_sync_backend,
        sleep_level=(rollout_config.sleep_level if colocate else 0),
        offload_kwargs=(offload_kwargs(rollout_config) if colocate else None),
        **({'remote_group': CONTROL_GROUP} if remote else {}))


def build_streaming_driver(*,
                           dataloader: Any,
                           rollout: Any,
                           rollout_config: Any,
                           batch_size: int,
                           num_generations: int = 1,
                           num_train_epochs: float = 1.0,
                           max_steps: Optional[int] = None,
                           checkpoint_manager: Any = None,
                           orms: Sequence[Any] = (),
                           orm_names: Optional[Sequence[str]] = None,
                           prms: Sequence[Any] = (),
                           prm_names: Optional[Sequence[str]] = None,
                           on_sample: Optional[Callable[[Any], Optional[bool]]] = None,
                           on_group: Optional[Callable[[Any, List[Any], int], List[Any]]] = None,
                           max_group_regenerations: int = 0,
                           cluster_adapter: Any = None,
                           max_staleness: Optional[int] = None,
                           remote: bool = False) -> Any:
    """One ``StreamingDriver`` handing out trajectory batches of ``batch_size`` rows.

    Each batch arrives with its named score columns already written (``rewards.orm`` / ``env`` /
    ``baseline``, ``prm_segments``) and its identity facts; the light batch facts ride
    ``driver.batch_summary``. ``orm_names`` / ``prm_names`` name the scorers in those columns
    (``None`` falls back to the driver's positional names).

    Staleness follows ``async_mode`` (:func:`max_staleness_of`) unless ``max_staleness`` is given
    (without a checkpoint manager it only sets how many batches run ahead), the sync interval is
    ``parameter_sync_step``; ``remote`` places the driver on :data:`CONTROL_GROUP`.
    """
    from twinkle_agentic.async_rl import StreamingDriver

    return StreamingDriver(
        dataloader,
        rollout,
        batch_size=batch_size,
        num_generations=num_generations,
        num_epochs=num_train_epochs,
        max_steps=(max_steps or None),
        max_staleness=(max_staleness_of(rollout_config) if max_staleness is None else max_staleness),
        sync_interval=rollout_config.parameter_sync_step,
        checkpoint_manager=checkpoint_manager,
        orms=orms,
        orm_names=orm_names,
        prms=prms,
        prm_names=prm_names,
        on_sample=on_sample,
        on_group=on_group,
        max_group_regenerations=max_group_regenerations,
        cluster_adapter=cluster_adapter,
        **({'remote_group': CONTROL_GROUP} if remote else {}))


def build_cluster_adapter(resource_plan: Any, rollout_config: Any, *, model: Any, sampler: Any,
                          checkpoint_manager: Any, lend: bool = True) -> Any:
    """``resource_mode='adaptive'``: grow the sampler onto the flex ranks, and onto the trainer's while it waits.

    A new sampler replica takes the current weights through ``checkpoint_manager.warm`` before it
    serves; the hysteresis window and the hold are counted in weight syncs. ``lend=False`` keeps the
    trainer's ranks out (another trained model shares them and would not be offloaded).
    """
    from twinkle_agentic.async_rl import ClusterAdapter

    return ClusterAdapter(
        {'sampler': sampler},
        model=model,
        lend_ranks=(list(resource_plan.trainer.ranks) if lend else None),
        spare_ranks=list(resource_plan.flex_ranks),
        warmups={'sampler': checkpoint_manager.warm},
        widths={'sampler': resource_plan.sampler.gpus_per_worker},
        window=rollout_config.resource_window_steps,
        cooldown=rollout_config.resource_min_hold_steps)

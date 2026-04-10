# Copyright (c) ModelScope Contributors. All rights reserved.
"""WorkerGroup — manages a set of MegatronWorker Ray actors.

Dispatch/collect patterns (inspired by verl's ``@register`` and
twinkle's ``@remote_function``):

Dispatch modes:
  - broadcast: same data to every worker
  - dp: data keyed by dp_rank → each worker gets its slice
  - dp_split: single global batch → chunk(dp_size) → dispatch by dp

Collect modes:
  - all: return list of all results
  - dp: return {dp_rank: result} from collector ranks only
  - first: return first collector's result (scalar metrics, etc.)

Use ``@dispatch_collect`` on ``MegatronWorker`` methods to declare
default dispatch/collect semantics.  ``WorkerGroup.build_dispatch_info``
scans the worker class and auto-binds decorated methods as direct
callables on the group: ``wg.train_step(batch)`` instead of
``wg.execute('train_step', batch, dispatch=..., collect=...)``.
"""
import ray
import torch
from typing import Any, Dict, List, Optional

from swift.utils import get_logger

logger = get_logger()

DispatchMode = str  # 'broadcast' | 'dp' | 'dp_split'
CollectMode = str  # 'all' | 'dp' | 'first'

_DC_ATTR = '_dispatch_collect_meta'

_DP_DISPATCHED = object()


class DPDispatchedDict(dict):
    """Marker dict for dp-dispatched data, keyed by dp_rank."""
    _dp_dispatched = _DP_DISPATCHED


def dispatch_collect(
    dispatch: DispatchMode = 'broadcast',
    collect: CollectMode = 'all',
):
    """Decorator declaring default dispatch/collect for a worker method.

    ``WorkerGroup.build_dispatch_info`` reads this metadata and
    creates a bound method ``wg.<method_name>(*args, **kwargs)``
    that automatically dispatches, invokes remote, and collects.
    """

    def decorator(fn):
        setattr(fn, _DC_ATTR, {'dispatch': dispatch, 'collect': collect})
        return fn

    return decorator


def _is_dp_dispatched(value) -> bool:
    """Check if a value is a dp-dispatched dict."""
    return isinstance(value, DPDispatchedDict) or getattr(value, '_dp_dispatched', None) is _DP_DISPATCHED


def _slice_dp(value: Any, dp_size: int) -> Any:
    """Recursively split *value* into DPDispatchedDict {dp_rank: chunk}.

    Handles tensors, lists, and dicts (recursed into).
    Scalars and unsplittable values are broadcast as-is.
    """
    if isinstance(value, torch.Tensor):
        if value.shape[0] >= dp_size:
            parts = value.chunk(dp_size)
            return DPDispatchedDict({i: p for i, p in enumerate(parts)})
        return value
    if isinstance(value, list):
        if len(value) >= dp_size:
            k, m = divmod(len(value), dp_size)
            result = DPDispatchedDict()
            offset = 0
            for i in range(dp_size):
                size = k + (1 if i < m else 0)
                result[i] = value[offset:offset + size]
                offset += size
            return result
        return value
    if isinstance(value, dict):
        if _is_dp_dispatched(value):
            return value
        splits = {k: _slice_dp(v, dp_size) for k, v in value.items()}
        any_split = any(_is_dp_dispatched(v) for v in splits.values())
        if not any_split:
            return value
        result = DPDispatchedDict()
        for k, v in splits.items():
            if _is_dp_dispatched(v):
                for dp_r, chunk in v.items():
                    result.setdefault(dp_r, {})[k] = chunk
            else:
                for dp_r in range(dp_size):
                    result.setdefault(dp_r, {})[k] = v
        return result
    return value


class WorkerGroup:
    """A group of ``MegatronWorker`` Ray actors.

    After ``init_model`` on all workers, call ``build_dispatch_info()``
    to populate the DP-rank map and collector mask.
    """

    def __init__(self, name: str, worker_handles: List[Any]):
        self.name = name
        self._workers = list(worker_handles)
        self._dp_rank_map: Optional[List[int]] = None
        self._collect_mask: Optional[List[bool]] = None
        self._dp_size: Optional[int] = None

    @property
    def world_size(self) -> int:
        return len(self._workers)

    @property
    def dp_size(self) -> int:
        if self._dp_size is None:
            raise RuntimeError('Call build_dispatch_info() first.')
        return self._dp_size

    def __len__(self) -> int:
        return len(self._workers)

    def build_dispatch_info(self, worker_cls=None):
        """Query workers for DP rank / collector info, then bind decorated methods."""
        infos = ray.get([w.get_parallel_info.remote() for w in self._workers])
        self._dp_rank_map = [i['dp_rank'] for i in infos]
        self._collect_mask = [i['is_collector'] for i in infos]
        self._dp_size = infos[0]['dp_size']

        if worker_cls is None:
            from .megatron_worker import MegatronWorker
            worker_cls = MegatronWorker
        self._bind_decorated_methods(worker_cls)

    def _bind_decorated_methods(self, worker_cls):
        """Scan *worker_cls* and bind methods with ``@dispatch_collect``."""
        for name in dir(worker_cls):
            if name.startswith('_'):
                continue
            method = getattr(worker_cls, name, None)
            if method is None or not callable(method):
                continue
            meta = getattr(method, _DC_ATTR, None)
            if meta is None:
                continue
            bound = self._make_bound_method(name, meta['dispatch'], meta['collect'])
            setattr(self, name, bound)

    def _make_bound_method(self, method_name: str, default_dispatch: str, default_collect: str):
        """Create a callable that dispatches/collects when invoked."""
        wg = self

        class _BoundMethod:
            """Callable wrapping a remote method with dispatch/collect."""

            def __call__(self_, *args, dispatch=None, collect=None, blocking=True, **kwargs):
                d = dispatch if dispatch is not None else default_dispatch
                c = collect if collect is not None else default_collect
                return wg.execute(method_name, *args, dispatch=d, collect=c, blocking=blocking, **kwargs)

            def __repr__(self_):
                return f'<BoundMethod {wg.name}.{method_name} dispatch={default_dispatch} collect={default_collect}>'

        return _BoundMethod()

    # ------------------------------------------------------------------
    # Core: execute with dispatch + collect
    # ------------------------------------------------------------------

    def execute(
        self,
        method_name: str,
        *args,
        dispatch: DispatchMode = 'broadcast',
        collect: CollectMode = 'all',
        blocking: bool = True,
        **kwargs,
    ) -> Any:
        """Call a remote method with configurable dispatch/collect.

        Args:
            method_name: Remote method on each worker.
            dispatch: How to distribute arguments to workers.
            collect: How to aggregate results.
            blocking: If False, return raw Ray futures without waiting.
            *args, **kwargs: Arguments (interpretation depends on dispatch mode).

        Returns:
            Aggregated results (type depends on collect mode), or
            futures list if blocking=False.
        """
        per_worker_args = self._dispatch(dispatch, args, kwargs)
        futures = [getattr(w, method_name).remote(*a, **kw) for w, (a, kw) in zip(self._workers, per_worker_args)]
        if not blocking:
            return futures
        results = ray.get(futures)
        return self._collect(collect, results)

    # ------------------------------------------------------------------
    # Dispatch logic
    # ------------------------------------------------------------------

    def _dispatch(
        self,
        mode: DispatchMode,
        args: tuple,
        kwargs: dict,
    ) -> List[tuple]:
        """Return [(args_i, kwargs_i)] for each worker."""
        if mode == 'broadcast':
            return [(args, kwargs)] * self.world_size

        if mode == 'dp':
            return self._dispatch_dp(args, kwargs)

        if mode == 'dp_split':
            return self._dispatch_dp_split(args, kwargs)

        raise ValueError(f'Unknown dispatch mode: {mode!r}')

    def _dispatch_dp(self, args: tuple, kwargs: dict) -> List[tuple]:
        """Dispatch by DP rank.

        Each positional arg and kwarg value should be a DPDispatchedDict
        ``{dp_rank: value}``. Each worker receives the entry
        matching its DP rank.
        """
        result = []
        for dp_r in self._dp_rank_map:
            worker_args = tuple(a[dp_r] if _is_dp_dispatched(a) else a for a in args)
            worker_kwargs = {k: v[dp_r] if _is_dp_dispatched(v) else v for k, v in kwargs.items()}
            result.append((worker_args, worker_kwargs))
        return result

    def _dispatch_dp_split(self, args: tuple, kwargs: dict) -> List[tuple]:
        """Split a global batch by dp_size, then dispatch by DP rank."""
        dp_size = self._dp_size
        split_args = tuple(_slice_dp(a, dp_size) for a in args)
        split_kwargs = {k: _slice_dp(v, dp_size) for k, v in kwargs.items()}
        return self._dispatch_dp(split_args, split_kwargs)

    # ------------------------------------------------------------------
    # Collect logic
    # ------------------------------------------------------------------

    def _collect(self, mode: CollectMode, results: List[Any]) -> Any:
        if mode == 'all':
            return results

        if mode == 'dp':
            return self._collect_dp(results)

        if mode == 'first':
            return self._collect_first(results)

        raise ValueError(f'Unknown collect mode: {mode!r}')

    def _collect_dp(self, results: List[Any]) -> Dict[int, Any]:
        """Return {dp_rank: result} from collector ranks only."""
        collected = {}
        for r, dp_r, is_coll in zip(
                results,
                self._dp_rank_map,
                self._collect_mask,
        ):
            if is_coll and r is not None:
                collected[dp_r] = r
        return collected

    def _collect_first(self, results: List[Any]) -> Any:
        """Return the result from the first collector rank."""
        for r, is_coll in zip(results, self._collect_mask):
            if is_coll and r is not None:
                return r
        return None

    # ------------------------------------------------------------------
    # Convenience shortcuts
    # ------------------------------------------------------------------

    def broadcast(self, method_name: str, *args, **kwargs) -> List[Any]:
        """Call same method with same args on all workers, block."""
        return self.execute(method_name, *args, dispatch='broadcast', collect='all', **kwargs)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_pool(
        cls,
        name: str,
        resource_pool: 'ResourcePool',
        worker_cls: Any = None,
        num_gpus: float = 1.0,
        master_port: Optional[int] = None,
    ) -> 'WorkerGroup':
        """Spawn actors on a ``ResourcePool``."""
        from ray.runtime_env import RuntimeEnv
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        if worker_cls is None:
            from .megatron_worker import MegatronWorker
            worker_cls = ray.remote(num_gpus=num_gpus)(MegatronWorker)

        placements = resource_pool.get_placements(master_port=master_port)
        workers = []

        node_local_ranks: Dict[str, int] = {}
        for p in placements:
            node_key = '%s:%s' % (p['master_addr'], p['master_port'])
            local_rank = node_local_ranks.get(node_key, 0)
            node_local_ranks[node_key] = local_rank + 1

            env_vars = {
                'RANK': str(p['rank']),
                'LOCAL_RANK': str(local_rank),
                'WORLD_SIZE': str(p['world_size']),
                'MASTER_ADDR': str(p['master_addr']),
                'MASTER_PORT': str(p['master_port']),
                'CUDA_DEVICE_MAX_CONNECTIONS': '1',
                'RAY_SWIFT_GROUP': 'default,%s' % name,
            }
            w = worker_cls.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=p['pg'], placement_group_bundle_index=p['bundle_idx']),
                runtime_env=RuntimeEnv(env_vars=env_vars),
            ).remote()
            workers.append(w)
        return cls(name, workers)

    def ping(self) -> List[str]:
        return self.broadcast('ping')

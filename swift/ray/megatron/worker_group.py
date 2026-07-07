# Copyright (c) ModelScope Contributors. All rights reserved.
import ray
import socket
import torch
from enum import Enum
from ray.runtime_env import RuntimeEnv
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers.utils import is_torch_npu_available
from typing import TYPE_CHECKING, Any, Dict, List, Union

from swift.utils.logger import get_logger

if TYPE_CHECKING:
    from .resource_pool import ResourcePool

logger = get_logger()


class DispatchMode(str, Enum):
    BROADCAST = 'broadcast'
    DP = 'dp'
    DP_SPLIT = 'dp_split'


class CollectMode(str, Enum):
    ALL = 'all'
    DP = 'dp'
    DP_FLAT = 'dp_flat'
    FIRST = 'first'


_DC_ATTR = '_dispatch_collect_meta'


class DPDispatchedDict(dict):
    """Marker dict for dp-dispatched data, keyed by dp_rank."""


def _is_dp_dispatched(value) -> bool:
    return isinstance(value, DPDispatchedDict)


def dispatch_collect(
    dispatch: Union[DispatchMode, str] = DispatchMode.BROADCAST,
    collect: Union[CollectMode, str] = CollectMode.ALL,
):
    """Decorator declaring default dispatch/collect for a worker method."""
    dispatch = DispatchMode(dispatch) if isinstance(dispatch, str) else dispatch
    collect = CollectMode(collect) if isinstance(collect, str) else collect

    def decorator(fn):
        setattr(fn, _DC_ATTR, {'dispatch': dispatch, 'collect': collect})
        return fn

    return decorator


def _slice_dp(value: Any, dp_size: int) -> Any:
    """Split *value* into ``DPDispatchedDict``{dp_rank: chunk}.

    Handles tensors, lists, and dicts (recursed into); scalars and
    un-sliceable values are broadcast as-is.  Raises on empty or
    sub-dp_size inputs rather than silently collapsing to broadcast —
    callers must pad upstream.
    """
    if isinstance(value, torch.Tensor):
        n = value.shape[0]
        if n == 0:
            raise ValueError(f'_slice_dp got empty tensor (shape={tuple(value.shape)})')
        if n < dp_size:
            raise ValueError(f'_slice_dp: tensor first dim {n} < dp_size {dp_size}.  '
                             f'Pad the batch upstream or use dispatch="broadcast".')
        if n % dp_size != 0:
            raise ValueError(f'_slice_dp: tensor first dim {n} not divisible by dp_size {dp_size}. '
                             f'Pad the batch upstream or use dispatch="broadcast".')
        parts = value.chunk(dp_size)
        return DPDispatchedDict({i: p for i, p in enumerate(parts)})
    if isinstance(value, list):
        n = len(value)
        if n == 0:
            raise ValueError('_slice_dp got empty list')
        if n < dp_size:
            raise ValueError(f'_slice_dp: list length {n} < dp_size {dp_size}. '
                             f'Pad the batch upstream or use dispatch="broadcast".')
        if n % dp_size != 0:
            raise ValueError(f'_slice_dp: list length {n} not divisible by dp_size {dp_size}. '
                             f'Pad the batch upstream or use dispatch="broadcast".')
        chunk_size = n // dp_size
        result = DPDispatchedDict()
        for i in range(dp_size):
            result[i] = value[i * chunk_size:(i + 1) * chunk_size]
        return result
    if isinstance(value, tuple):
        n = len(value)
        if n == 0:
            raise ValueError('_slice_dp got empty tuple')
        if n < dp_size:
            raise ValueError(f'_slice_dp: tuple length {n} < dp_size {dp_size}. '
                             f'Pad the batch upstream or use dispatch="broadcast".')
        if n % dp_size != 0:
            raise ValueError(f'_slice_dp: tuple length {n} not divisible by dp_size {dp_size}. '
                             f'Pad the batch upstream or use dispatch="broadcast".')
        chunk_size = n // dp_size
        result = DPDispatchedDict()
        for i in range(dp_size):
            result[i] = value[i * chunk_size:(i + 1) * chunk_size]
        return result
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


_DISPATCH_FNS: Dict[str, Any] = {}
_COLLECT_FNS: Dict[str, Any] = {}


def _register_builtin_modes():
    _DISPATCH_FNS[DispatchMode.BROADCAST] = WorkerGroup._dispatch_broadcast
    _DISPATCH_FNS[DispatchMode.DP] = WorkerGroup._dispatch_dp
    _DISPATCH_FNS[DispatchMode.DP_SPLIT] = WorkerGroup._dispatch_dp_split
    _COLLECT_FNS[CollectMode.ALL] = WorkerGroup._collect_all
    _COLLECT_FNS[CollectMode.DP] = WorkerGroup._collect_dp
    _COLLECT_FNS[CollectMode.DP_FLAT] = WorkerGroup._collect_dp_flat
    _COLLECT_FNS[CollectMode.FIRST] = WorkerGroup._collect_first


class WorkerGroup:
    """A group of Ray actors with dispatch / collect helpers.

    After workers are up, call :meth:`build_dispatch_info` to query
    ``get_parallel_info`` on every actor, cache the dp layout, and
    bind decorated methods on the group instance.  Subsequent calls
    like ``wg.train_step(batch)`` dispatch + collect automatically
    using the metadata attached by :func:`dispatch_collect`.
    """

    def __init__(self, name: str, worker_handles: List[Any]):
        self.name = name
        self._workers = list(worker_handles)
        self._dp_rank_map: List[int] = []
        self._collect_mask: List[bool] = []
        self._dp_size: int = 0

    @property
    def world_size(self) -> int:
        return len(self._workers)

    @property
    def workers(self) -> List[Any]:
        return list(self._workers)

    @property
    def dp_size(self) -> int:
        if self._dp_size == 0:
            raise RuntimeError(f'WorkerGroup[{self.name}]: build_dispatch_info() has '
                               'not been called yet (dp_size is unknown).')
        return self._dp_size

    def __len__(self) -> int:
        return len(self._workers)

    def build_dispatch_info(self, worker_cls, rpc: str = 'get_parallel_info'):
        """Query workers' parallel info and bind decorated methods."""
        infos = ray.get([getattr(w, rpc).remote() for w in self._workers])
        if len(infos) != self.world_size:
            raise RuntimeError(f'WorkerGroup[{self.name}]: expected {self.world_size} info entries, got {len(infos)}')
        self._dp_rank_map = [int(i['dp_rank']) for i in infos]
        self._collect_mask = [bool(i['is_collector']) for i in infos]
        sizes = {int(i['dp_size']) for i in infos}
        if len(sizes) != 1:
            raise ValueError(f'WorkerGroup[{self.name}]: inconsistent dp_size across workers: {sizes}')
        self._dp_size = sizes.pop()
        self._bind_decorated_methods(worker_cls)

        logger.info('WorkerGroup[%s]: dp_size=%d, world_size=%d, collectors=%d', self.name, self._dp_size,
                    self.world_size, sum(self._collect_mask))

    def _bind_decorated_methods(self, worker_cls) -> List[str]:
        """Bind methods carrying ``_dispatch_collect_meta`` onto the group.

        We only bind methods that explicitly opt in via
        :func:`dispatch_collect`; other helpers on the worker class
        stay private to the actor so callers can't accidentally talk
        to them through the group.  Any name already defined on the
        group (methods like ``execute`` / ``broadcast``, properties
        like ``dp_size``) is skipped with a warning — the caller can
        still reach it via ``wg.execute(name, ...)``.
        """
        bound = []
        for attr_name in dir(worker_cls):
            if attr_name.startswith('_'):
                continue
            method = getattr(worker_cls, attr_name, None)
            if method is None or not callable(method):
                continue
            meta = getattr(method, _DC_ATTR, None)
            if meta is None:
                continue
            if hasattr(type(self), attr_name) or attr_name in self.__dict__:
                logger.warning(
                    'WorkerGroup[%s]: worker method %r collides with an existing '
                    'attribute on WorkerGroup; call wg.execute(%r, ...) instead.', self.name, attr_name, attr_name)
                continue
            setattr(self, attr_name, self._make_bound(attr_name, meta['dispatch'], meta['collect']))
            bound.append(attr_name)
        return bound

    def _make_bound(self, method_name: str, default_dispatch: str, default_collect: str):
        wg = self

        def _bound(*args, dispatch=None, collect=None, **kwargs):
            return wg.execute(
                method_name,
                *args,
                dispatch=dispatch if dispatch is not None else default_dispatch,
                collect=collect if collect is not None else default_collect,
                **kwargs,
            )

        _bound.__name__ = method_name
        _bound.__qualname__ = f'{wg.name}.{method_name}'
        _bound.__doc__ = (f'Bound remote method {method_name} '
                          f'(dispatch={default_dispatch}, collect={default_collect})')
        return _bound

    def execute(
        self,
        method_name: str,
        *args,
        dispatch: Union[DispatchMode, str] = DispatchMode.BROADCAST,
        collect: Union[CollectMode, str] = CollectMode.ALL,
        **kwargs,
    ) -> Any:
        """Call a remote method with configurable dispatch/collect."""
        dispatch = DispatchMode(dispatch) if isinstance(dispatch, str) else dispatch
        collect = CollectMode(collect) if isinstance(collect, str) else collect
        per_worker = self._dispatch(dispatch, args, kwargs)
        futures = [getattr(w, method_name).remote(*a, **kw) for w, (a, kw) in zip(self._workers, per_worker)]
        return self._collect(collect, ray.get(futures))

    def broadcast(self, method_name: str, *args, **kwargs) -> List[Any]:
        """Same args to every worker, block, return list."""
        return self.execute(method_name, *args, dispatch=DispatchMode.BROADCAST, collect=CollectMode.ALL, **kwargs)

    def _dispatch(self, mode: Union[DispatchMode, str], args: tuple, kwargs: dict) -> List[tuple]:
        mode = DispatchMode(mode) if isinstance(mode, str) else mode
        fn = _DISPATCH_FNS.get(mode)
        if fn is None:
            raise ValueError(f'Unknown dispatch mode: {mode!r}; registered: {list(_DISPATCH_FNS)}')
        return fn(self, args, kwargs)

    def _dispatch_broadcast(self, args: tuple, kwargs: dict) -> List[tuple]:
        return [(args, kwargs)] * self.world_size

    def _dispatch_dp(self, args: tuple, kwargs: dict) -> List[tuple]:
        result = []
        for dp_r in self._dp_rank_map:
            worker_args = tuple(a[dp_r] if _is_dp_dispatched(a) else a for a in args)
            worker_kwargs = {k: v[dp_r] if _is_dp_dispatched(v) else v for k, v in kwargs.items()}
            result.append((worker_args, worker_kwargs))
        return result

    def _dispatch_dp_split(self, args: tuple, kwargs: dict) -> List[tuple]:
        dp = self.dp_size
        split_args = tuple(_slice_dp(a, dp) for a in args)
        split_kwargs = {k: _slice_dp(v, dp) for k, v in kwargs.items()}
        return self._dispatch_dp(split_args, split_kwargs)

    def _collect(self, mode: Union[CollectMode, str], results: List[Any]) -> Any:
        mode = CollectMode(mode) if isinstance(mode, str) else mode
        fn = _COLLECT_FNS.get(mode)
        if fn is None:
            raise ValueError(f'Unknown collect mode: {mode!r}; registered: {list(_COLLECT_FNS)}')
        return fn(self, results)

    def _collect_all(self, results: List[Any]) -> List[Any]:
        return results

    def _collect_dp(self, results: List[Any]) -> Dict[int, Any]:
        collected: Dict[int, Any] = {}
        for r, dp_r, is_coll in zip(results, self._dp_rank_map, self._collect_mask):
            if is_coll and r is not None:
                collected[dp_r] = r
        return collected

    def _collect_dp_flat(self, results: List[Any]) -> List[Any]:
        """Collect from DP collectors and flatten into a single ordered list.

        Equivalent to ``_collect_dp`` followed by sorting by rank and
        concatenating lists, which is the most common access pattern on
        the driver side.
        """
        per_rank = self._collect_dp(results)
        flat: List[Any] = []
        for rk in sorted(per_rank.keys()):
            part = per_rank[rk]
            if isinstance(part, list):
                flat.extend(part)
            elif part is not None:
                flat.append(part)
        return flat

    def _collect_first(self, results: List[Any]) -> Any:
        for r, is_coll in zip(results, self._collect_mask):
            if is_coll and r is not None:
                return r
        return None

    def shutdown(self, timeout: float = 30.0) -> None:
        """Best-effort shutdown of every worker and kill the Ray actors."""
        if not self._workers:
            return
        pending = []
        for w in self._workers:
            fn = getattr(w, 'shutdown', None)
            if fn is None:
                continue
            try:
                pending.append(fn.remote())
            except Exception as e:  # noqa: BLE001
                logger.warning('WorkerGroup[%s] shutdown dispatch failed: %s', self.name, e)
        if pending:
            try:
                ray.get(pending, timeout=timeout)
            except Exception as e:  # noqa: BLE001
                logger.warning('WorkerGroup[%s] shutdown timed out / raised: %s', self.name, e)
        for w in self._workers:
            try:
                ray.kill(w, no_restart=True)
            except Exception as e:  # noqa: BLE001
                logger.warning('WorkerGroup[%s] ray.kill failed: %s', self.name, e)
        self._workers = []

    @staticmethod
    def _get_device_env_config() -> Dict[str, str]:
        """Return platform-specific environment variable names.

        Supports CUDA (GPU) and Ascend (NPU).
        """
        if is_torch_npu_available():
            return {
                'visible_devices_key': 'ASCEND_RT_VISIBLE_DEVICES',
                'device_max_connections_key': 'HCCL_DEVICE_MAX_CONNECTIONS',
            }
        return {
            'visible_devices_key': 'CUDA_VISIBLE_DEVICES',
            'device_max_connections_key': 'CUDA_DEVICE_MAX_CONNECTIONS',
        }

    @classmethod
    def from_pool(
        cls,
        name: str,
        resource_pool: 'ResourcePool',
        worker_cls: str,
    ) -> 'WorkerGroup':
        """Spawn actors on a :class:`ResourcePool`.

        Iterates PGs, discovers master on PG[0], creates workers with
        env vars for distributed init.

        Swift uses ``num_gpus=0`` + ``NOSET_CVD`` + explicit CVD
        (torchrun style for Megatron TP/PP visibility).
        """
        master_addr, master_port = cls._discover_master(resource_pool)
        dev_cfg = cls._get_device_env_config()
        vis = resource_pool.visible_devices
        world_size = resource_pool.world_size
        workers = []

        rank = 0
        for pg_idx, pg in enumerate(resource_pool.pgs):
            local_ws = resource_pool.process_on_nodes[pg_idx]
            node_gpu_ids = [str(vis[rank + j]) for j in range(local_ws)]
            node_cvd = ','.join(node_gpu_ids)

            for local_rank in range(local_ws):
                env_vars = {
                    'RANK': str(rank),
                    'LOCAL_RANK': str(local_rank),
                    'WORLD_SIZE': str(world_size),
                    'LOCAL_WORLD_SIZE': str(local_ws),
                    'MASTER_ADDR': master_addr,
                    'MASTER_PORT': str(master_port),
                    dev_cfg['device_max_connections_key']: '1',
                    dev_cfg['visible_devices_key']: node_cvd,
                    'RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES': '1',
                    'RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES': '1',
                    'RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES': '1',
                    'NCCL_CUMEM_ENABLE': '0',
                    'RAY_SWIFT_GROUP': f'default,{name}',
                }
                w = worker_cls.options(
                    num_gpus=0,
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg, placement_group_bundle_index=local_rank),
                    runtime_env=RuntimeEnv(env_vars=env_vars),
                ).remote()
                workers.append(w)
                rank += 1
        return cls(name, workers)

    @staticmethod
    def _discover_master(resource_pool: 'ResourcePool'):
        """Find master IP + free port on PG[0] bundle[0].

        Discovers a free port on the first bundle of the first PG.
        """

        @ray.remote(num_gpus=0, num_cpus=0.01)
        def _probe():
            addr = ray.util.get_node_ip_address()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('', 0))
            port = sock.getsockname()[1]
            sock.close()
            return addr, port

        pg = resource_pool.pgs[0]
        return ray.get(
            _probe.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_bundle_index=0), ).remote())


_register_builtin_modes()

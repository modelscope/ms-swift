# Copyright (c) ModelScope Contributors. All rights reserved.
import ray
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from swift.rlhf_trainers.args_mixin import VllmArguments
from swift.utils.logger import get_logger

if TYPE_CHECKING:
    from ..resource_pool import ResourcePool

logger = get_logger()


class RolloutMode(str, Enum):
    HYBRID = 'hybrid'
    STANDALONE = 'standalone'


@dataclass
class VllmEngineConfig(VllmArguments):
    model: str = ''
    sleep_level: int = 0
    vllm_enable_lora: bool = False
    trust_remote_code: bool = True
    dtype: str = 'auto'
    load_format: str = 'auto'

    # override
    vllm_enable_prefix_caching: bool = True

    def __post_init__(self):
        VllmArguments.__post_init__(self)

    @classmethod
    def from_rollout_cfg(cls, rollout_cfg: Dict[str, Any], *, sleep_level: int = 0) -> 'VllmEngineConfig':
        """Build from merged rollout config dict.

        ``rollout_cfg`` is produced by ``merge_group_dict(shared, rollout_group)``
        and already contains top-level shared keys like ``model``.
        Unset fields fall through to ``VllmEngineConfig`` dataclass defaults.
        """
        cfg = rollout_cfg or {}
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}

        kwargs: Dict[str, Any] = {}
        for key, val in cfg.items():
            if key in known_fields and val is not None:
                kwargs[key] = val

        if sleep_level > 0:
            kwargs['sleep_level'] = sleep_level

        if cfg.get('tuner_type', 'full') == 'lora':
            kwargs.setdefault('vllm_enable_lora', True)

        return cls(**kwargs)

    def to_launch_kwargs(self, rollout_mode: str) -> Dict[str, Any]:
        # _prepare_engine_kwargs
        kw: Dict[str, Any] = {
            'model_id': self.model,
            'dtype': self.dtype,
            'rollout_mode': rollout_mode,
            'tensor_parallel_size': self.vllm_tensor_parallel_size,
            'gpu_memory_utilization': self.vllm_gpu_memory_utilization,
            'max_num_seqs': self.vllm_max_num_seqs,
            'enforce_eager': self.vllm_enforce_eager,
            'trust_remote_code': self.trust_remote_code,
            'load_format': self.load_format,
            'enable_sleep_mode': (self.sleep_level > 0),
            'enable_lora': self.vllm_enable_lora,
            'max_lora_rank': self.vllm_max_lora_rank,
            'data_parallel_size': self.vllm_data_parallel_size,
            'enable_prefix_caching': self.vllm_enable_prefix_caching,
        }
        if self.vllm_max_model_len is not None:
            kw['max_model_len'] = self.vllm_max_model_len
        extra = self.vllm_engine_kwargs
        if isinstance(extra, dict):
            kw.update(extra)
        return kw


class RolloutReplica:
    """One vLLM rollout replica living on Ray.

    A replica may span multiple nodes for large TP configurations. In
    that case one ``VllmServer`` actor is launched per node: node_rank=0
    runs the full HTTP server, other nodes run headless workers. This
    mirrors a standard multi-node vLLM architecture.

    For single-node replicas, there is exactly one ``VllmServer`` actor.
    """

    def __init__(
        self,
        config: VllmEngineConfig,
        mode: RolloutMode = RolloutMode.HYBRID,
        replica_rank: int = 0,
    ) -> None:
        self.config = config
        self.mode = mode
        self.replica_rank = replica_rank
        self._servers: List[Any] = []

    @classmethod
    def create_replicas(
        cls,
        rollout_cfg: Dict[str, Any],
        rollout_gpus: int,
        pool: 'ResourcePool',
        is_hybrid: bool,
        sleep_level: int = 0,
    ) -> List['RolloutReplica']:
        """Factory: create all rollout replicas from pipeline config."""
        config = VllmEngineConfig.from_rollout_cfg(rollout_cfg, sleep_level=sleep_level)
        world_size_per_replica = config.vllm_tensor_parallel_size * config.vllm_data_parallel_size
        if world_size_per_replica > rollout_gpus:
            raise ValueError(f'tp*dp ({world_size_per_replica}) exceeds rollout GPUs ({rollout_gpus})')
        if rollout_gpus % world_size_per_replica != 0:
            raise ValueError(f'rollout GPUs ({rollout_gpus}) must be divisible by '
                             f'tp*dp ({world_size_per_replica})')
        n_replicas = rollout_gpus // world_size_per_replica
        mode = RolloutMode.HYBRID if is_hybrid else RolloutMode.STANDALONE

        replicas: List['RolloutReplica'] = []
        bundle_infos = pool.bundle_infos
        for i in range(n_replicas):
            offset = i * world_size_per_replica
            replica_infos = bundle_infos[offset:offset + world_size_per_replica]
            nodes = {info[0] for info in replica_infos}
            gpus_per_node = (world_size_per_replica if len(nodes) == 1 else world_size_per_replica // len(nodes))
            replica = cls(config, mode=mode, replica_rank=i)
            replica._launch_servers(replica_infos, gpus_per_node)
            replicas.append(replica)

        logger.info('Rollout: %d replica(s) in %s mode (tp=%d, dp=%d, total_gpus=%d)', n_replicas, mode.value.upper(),
                    config.vllm_tensor_parallel_size, config.vllm_data_parallel_size, rollout_gpus)
        return replicas

    def _launch_servers(
        self,
        worker_infos: List[Tuple[str, str]],
        gpus_per_node: int,
    ) -> None:
        """Launch one ``VllmServer`` per node (hybrid or standalone).

        ``num_gpus=0`` + ``NOSET_CVD`` + explicit visible-device env so
        the actor sees exactly the GPUs from pool bundles; NodeAffinity
        pins each actor to the correct node.
        """
        from ray.runtime_env import RuntimeEnv
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
        from transformers.utils import is_torch_npu_available

        from .vllm_server import VllmServer

        visible_key = 'ASCEND_RT_VISIBLE_DEVICES' if is_torch_npu_available() else 'CUDA_VISIBLE_DEVICES'

        node_groups = self._group_by_node(worker_infos)
        nnodes = len(node_groups)
        tp = self.config.vllm_tensor_parallel_size

        actor_cls = ray.remote(num_gpus=0, num_cpus=1)(VllmServer)

        for node_rank, (node_id, gpu_ids) in enumerate(node_groups):
            cvd = ','.join(gpu_ids)
            env_vars: Dict[str, str] = {
                'VLLM_USE_V1': '1',
                'RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES': '1',
                'RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES': '1',
                'NCCL_CUMEM_ENABLE': '0',
                visible_key: cvd,
            }
            handle = actor_cls.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
                runtime_env=RuntimeEnv(env_vars=env_vars),
                name=f'swift_rollout_server_{self.replica_rank}_{node_rank}',
                max_concurrency=10,
            ).remote(
                node_rank=node_rank,
                nnodes=nnodes,
                gpus_per_node=gpus_per_node,
                cuda_visible_devices=cvd,
            )
            self._servers.append(handle)

        self._launch_all_servers(nnodes, tp)

    def _launch_all_servers(self, nnodes: int, tp: int) -> None:
        """Get master info from server[0], then launch_server on all."""
        launch_kw = self.config.to_launch_kwargs(self.mode.value)

        if nnodes > 1:
            master_address, master_port, dp_rpc_port = ray.get(self._servers[0].get_master_address.remote())
            refs = [
                server.launch_server.remote(
                    master_address=master_address, master_port=master_port, dp_rpc_port=dp_rpc_port, **launch_kw)
                for server in self._servers
            ]
            ray.get(refs)
        else:
            ray.get(self._servers[0].launch_server.remote(**launch_kw))

        logger.info('RolloutReplica[replica=%d, mode=%s]: launched %d server(s) (tp=%d, model=%s)', self.replica_rank,
                    self.mode.value, len(self._servers), tp, self.config.model)

    @staticmethod
    def _group_by_node(worker_infos: List[Tuple[str, str]]) -> List[Tuple[str, List[str]]]:
        """Group worker infos by node, preserving order.

        Returns list of ``(node_id, [accelerator_id, ...])`` in node
        encounter order.
        """
        ordered_nodes: List[str] = []
        node_gpus: Dict[str, List[str]] = defaultdict(list)
        for node_id, acc_id in worker_infos:
            if node_id not in node_gpus:
                ordered_nodes.append(node_id)
            node_gpus[node_id].append(acc_id)
        return [(nid, node_gpus[nid]) for nid in ordered_nodes]

    @property
    def primary(self) -> Any:
        """The node_rank=0 ``VllmServer`` actor handle.

        Callers that need ``sleep`` / ``wake_up`` / ``reset_prefix_cache``
        / ``update_weights_ipc`` / ``update_weights_direct`` talk to
        this handle directly.
        """
        if not self._servers:
            raise RuntimeError('RolloutReplica: not launched yet')
        return self._servers[0]

    @property
    def servers(self) -> List[Any]:
        """All server actor handles (one per node)."""
        return list(self._servers)

    def generate(
        self,
        infer_requests: List[Any],
        request_config: Any = None,
    ) -> ray.ObjectRef:
        """Submit generation to the primary server, returns an ObjectRef."""
        return self.primary.generate.remote(infer_requests, request_config)

    def shutdown(self) -> None:
        for server in self._servers:
            try:
                ray.get(server.shutdown.remote(), timeout=30)
            except Exception as e:  # noqa: BLE001
                logger.warning('RolloutReplica shutdown error: %s', e)
            try:
                ray.kill(server, no_restart=True)
            except Exception:  # noqa: BLE001
                pass
        self._servers = []

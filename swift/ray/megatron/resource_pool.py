# Copyright (c) ModelScope Contributors. All rights reserved.
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from swift.utils import get_logger

logger = get_logger()


def sort_pgs_by_node_ip(pgs: List[Any]) -> List[Any]:
    """Sort placement groups by node IP for deterministic rank assignment.

    Sorts PGs by node IP for deterministic ordering.
    """
    import ray

    node_ip = {node['NodeID']: node['NodeManagerAddress'] for node in ray.nodes()}
    pg_ip = {}
    for pg in pgs:
        specs = ray._private.state.state.placement_group_table(pg.id)
        node_id = specs['bundles_to_node_id'][0]
        pg_ip[pg.id] = node_ip[node_id]
    return sorted(pgs, key=lambda pg: pg_ip[pg.id])


@dataclass
class ResourcePool:
    """A pool of GPU resources backed by multiple Ray placement groups.

    Args:
        process_on_nodes: GPUs per node. ``[8]`` = 8 GPUs on 1 node,
            ``[4, 4]`` = 8 GPUs across 2 nodes.
        max_colocate_count: How many WorkerGroups share these GPUs.
    """

    process_on_nodes: List[int]
    max_colocate_count: int = 1

    pgs: List[Any] = field(default_factory=list, repr=False, init=False)
    node_ips: List[str] = field(default_factory=list, repr=False, init=False)
    bundle_infos: List[Tuple[str, str]] = field(default_factory=list, repr=False, init=False)

    @property
    def world_size(self) -> int:
        return sum(self.process_on_nodes)

    @property
    def num_nodes(self) -> int:
        return len(self.process_on_nodes)

    @property
    def visible_devices(self) -> List[int]:
        """Physical GPU ordinals: flat list across all nodes."""
        return [int(info[1]) if info[1] else i for i, info in enumerate(self.bundle_infos)]

    def create(self, device_name: str = 'GPU'):
        """Create one PG per node with STRICT_PACK strategy."""
        import ray
        from ray.util.placement_group import placement_group

        if device_name == 'npu':
            device_name = 'NPU'
        elif device_name == 'cuda':
            device_name = 'GPU'

        bundle_template = {
            device_name: 1,
            'CPU': max(self.max_colocate_count, 1),
        }

        pgs = []
        for n_gpus in self.process_on_nodes:
            bundles = [bundle_template.copy() for _ in range(n_gpus)]
            pg = placement_group(bundles, strategy='STRICT_PACK')
            pgs.append(pg)

        ray.get([pg.ready() for pg in pgs])

        self.pgs = sort_pgs_by_node_ip(pgs)
        self._discover_bundle_infos()

    def _discover_bundle_infos(self):
        """Probe each bundle's accelerator_id via lightweight actors.

        Swift-specific: needed because Swift uses num_gpus=0 + explicit
        CUDA_VISIBLE_DEVICES (torchrun style).
        """
        import os
        import ray
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
        from transformers.utils import is_torch_npu_available

        @ray.remote(num_gpus=0.01, num_cpus=0.01)
        def _probe_bundle():
            ctx = ray.get_runtime_context()
            acc_ids = ctx.get_accelerator_ids()
            gpu_id = ''
            for key in ('GPU', 'NPU'):
                ids = acc_ids.get(key, [])
                if ids:
                    gpu_id = ids[0]
                    break
            return ctx.get_node_id(), gpu_id

        all_infos: List[Tuple[str, str]] = []
        node_id_to_ip = {node['NodeID']: node['NodeManagerAddress'] for node in ray.nodes()}

        for pg_idx, pg in enumerate(self.pgs):
            refs = [
                _probe_bundle.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg, placement_group_bundle_index=i), ).remote()
                for i in range(self.process_on_nodes[pg_idx])
            ]
            results = ray.get(refs)
            for r in results:
                all_infos.append(r)

        vis_key = 'ASCEND_RT_VISIBLE_DEVICES' if is_torch_npu_available() else 'CUDA_VISIBLE_DEVICES'
        parent_cvd = os.environ.get(vis_key, '')
        if parent_cvd:
            phys_ids = [x.strip() for x in parent_cvd.split(',')]
            all_infos = [(nid, phys_ids[int(gid)] if gid.isdigit() and int(gid) < len(phys_ids) else gid)
                         for nid, gid in all_infos]

        self.bundle_infos = all_infos
        seen: set = set()
        node_ips = []
        for nid, _ in all_infos:
            if nid not in seen:
                seen.add(nid)
                node_ips.append(node_id_to_ip.get(nid, ''))
        self.node_ips = node_ips
        logger.info('ResourcePool: %d PG(s), %d bundles, node_ips=%s', len(self.pgs), len(all_infos), self.node_ips)

    def destroy(self):
        if self.pgs:
            import ray
            for pg in self.pgs:
                try:
                    ray.util.remove_placement_group(pg)
                except Exception:  # noqa: BLE001
                    pass
            self.pgs = []


class ResourcePoolManager:
    """Manages multiple ResourcePools, deduplicating shared pools (colocate)."""

    def __init__(self, pool_mapping: Dict[str, 'ResourcePool']):
        self._pools = pool_mapping

    def get_pool(self, group_name: str) -> 'ResourcePool':
        return self._pools[group_name]

    def create_all(self):
        seen: set = set()
        for pool in self._pools.values():
            if id(pool) not in seen:
                seen.add(id(pool))
                pool.create()

    def destroy_all(self):
        seen: set = set()
        for pool in self._pools.values():
            if id(pool) not in seen:
                seen.add(id(pool))
                pool.destroy()

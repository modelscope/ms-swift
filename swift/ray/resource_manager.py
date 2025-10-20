import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Any

import ray
from ray.util.placement_group import PlacementGroup

from swift.utils import find_free_port
from swift.utils.utils import find_node_ip


@dataclass
class NodeGroup:
    device_count: int
    nodes: List[Any] = field(default_factory=list)


@ray.remote
def get_node_rank():
    return int(os.environ.get("NODE_RANK", "0"))


@ray.remote
def get_node_address():
    return find_node_ip(), find_free_port()


class ResourceManager:

    def __init__(self, groups: Dict[str, Any]):
        nproc_per_node = int(groups['nproc_per_node'])
        device_types = set([group['device'] for group in groups.values()]) - {'CPU'}
        assert len(device_types) == 1
        device_type = next(iter(device_types))
        all_ranks = []
        last_rank = -1
        for group in groups.values():
            ranks = group['ranks']
            device = group['device']
            if device == 'CPU':
                continue
            try:
                ranks = int(ranks)
                ranks = list(range(last_rank+1, last_rank+1+ranks))
            except ValueError:
                ranks = eval(ranks)
            finally:
                all_ranks.extend(ranks)
                group['ranks'] = ranks
                last_rank = ranks[-1]

        assert len(set(all_ranks)) == len(all_ranks)
        groups['nnodes'] = math.ceil(len(all_ranks) / nproc_per_node)

        self.nodes = []
        for node in ray.nodes():
            resource = node["Resources"]
            node_gpu_num = int(resource.get(device_type, 0))
            if node_gpu_num >= nproc_per_node:
                self.nodes.append(node)

        bundles = []
        for i in range(groups['nnodes']):
            node = self.nodes[i]
            node_cpu = int(node["Resources"]["CPU"])
            bundles.append({device_type: nproc_per_node, "CPU": node_cpu // 2 + 1})

        self.placement_groups = [ray.util.placement_group([bundle]) for bundle in bundles]
        ray.get([pg.ready() for pg in self.placement_groups])

        self.node_ranks = ray.get(
            [get_node_rank.options(placement_group=pg).remote() for pg in self.placement_groups])
        if self.node_ranks.count(0) > 1:
            self.node_ranks = list(range(len(self.placement_groups)))

        self.node2pg: Dict[int, PlacementGroup] = {}
        ip, port = None, None
        for node_rank, placement_group in zip(self.node_ranks, self.placement_groups):
            if node_rank == 0:
                ip, port = get_node_address.options(placement_group=placement_group).remote()
            self.node2pg[node_rank] = placement_group

        groups['master_addr'] = (ip, port)
        self.device_groups = {}
        ray_address = str(ray.get_runtime_context().gcs_address)
        for group_name, group in groups.items():
            if group_name == 'nproc_per_node':
                continue
            ranks = group['ranks']
            local_device_groups = []
            for rank in ranks:
                node_rank = rank // nproc_per_node
                gpu_rank = rank % nproc_per_node
                local_device_groups.append(
                    dict(node_rank=node_rank, gpu_rank=gpu_rank,
                         placement_group=self.node2pg[node_rank], ray_address=ray_address)
                )
            for worker in group['workers']:
                self.device_groups[worker] = local_device_groups

        self.groups = groups

    def resource(self, worker):
        return self.device_groups[worker]

    def destroy_placement_group(self):
        [ray.util.remove_placement_group(pg) for pg in self.placement_groups]

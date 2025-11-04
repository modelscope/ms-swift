# Copyright (c) Alibaba, Inc. and its affiliates.
# Some code borrowed from ROLL: https://github.com/alibaba/ROLL
import ast
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class NodeGroup:
    device_count: int
    nodes: List[Any] = field(default_factory=list)


def get_node_rank():
    return int(os.environ.get('NODE_RANK', '0'))


class ResourceManager:

    possible_keys = ['nproc_per_node', 'nnodes']

    def __init__(self, groups: Dict[str, Any]):
        import ray
        from ray.util.placement_group import PlacementGroup
        nproc_per_node = int(groups['nproc_per_node'])
        device_types = set([group['device'].upper()
                            for group in groups.values() if hasattr(group, '__getitem__')]) - {'CPU'}
        assert len(device_types) == 1
        device_type = next(iter(device_types))
        all_ranks = []
        last_rank = -1
        cpu_proc_count = 0
        for group_name, group in groups.items():
            if group_name in self.possible_keys:
                continue
            ranks = group['ranks']
            device = group['device'].upper()
            if device == 'CPU':
                assert isinstance(ranks, int), 'CPU group only supports integer ranks'
                cpu_proc_count += ranks
                continue
            try:
                ranks = int(ranks)  # int type
                ranks = list(range(last_rank + 1, last_rank + 1 + ranks))
            except Exception:  # noqa
                if isinstance(ranks, str):
                    ranks = eval(ranks, {'__builtins__': {'list': list, 'range': range}})
            finally:
                all_ranks.extend(ranks)
                group['ranks'] = ranks
                last_rank = ranks[-1]

        assert len(set(all_ranks)) == len(all_ranks)
        groups['nnodes'] = math.ceil(len(all_ranks) / nproc_per_node)

        self.nodes = []
        for node in ray.nodes():
            resource = node['Resources']
            node_gpu_num = int(resource.get(device_type, 0))
            if node_gpu_num >= nproc_per_node:
                self.nodes.append(node)

        bundles = []
        cpu_bundles = []
        for i in range(groups['nnodes']):
            node = self.nodes[i]
            node_cpu = int(node['Resources']['CPU'])
            bundles.append({device_type: nproc_per_node, 'CPU': node_cpu // 2 + 1})
            cpu_bundles.append({'CPU': node_cpu // 4 + 1})  # TODO dynamic scheduling

        nproc_cpu_per_node = cpu_proc_count // len(cpu_bundles) + 1
        self.cpu_node_map = {}
        for i in range(cpu_proc_count):
            node_idx = i // nproc_cpu_per_node
            cpu_cnt = cpu_bundles[node_idx]['CPU']
            self.cpu_node_map[i] = (node_idx, cpu_cnt // nproc_cpu_per_node)

        self.placement_groups = [ray.util.placement_group([bundle]) for bundle in bundles]
        self.cpu_placement_groups = [ray.util.placement_group([bundle]) for bundle in cpu_bundles]
        cpu_bundles.sort(key=lambda bundle: bundle['CPU'], reverse=True)
        ray.get([pg.ready() for pg in self.placement_groups])
        ray.get([pg.ready() for pg in self.cpu_placement_groups])

        self.node_ranks = ray.get(
            [ray.remote(get_node_rank).options(placement_group=pg).remote() for pg in self.placement_groups])
        if self.node_ranks.count(0) > 1:
            self.node_ranks = list(range(len(self.placement_groups)))

        self.node2pg: Dict[int, PlacementGroup] = {}
        for node_rank, placement_group in zip(self.node_ranks, self.placement_groups):
            self.node2pg[node_rank] = placement_group

        self.device_groups = {}
        ray_address = str(ray.get_runtime_context().gcs_address)
        for group_name, group in groups.items():
            if group_name in self.possible_keys:
                continue

            if group['device'] != 'CPU':
                ranks = group['ranks']
                local_device_groups = []
                for rank in ranks:
                    node_rank = rank // nproc_per_node
                    gpu_rank = rank % nproc_per_node
                    local_device_groups.append(
                        dict(
                            node_rank=node_rank,
                            gpu_rank=[gpu_rank],
                            placement_group=self.node2pg[node_rank],
                            ray_address=ray_address))
                for worker in group['workers']:
                    self.device_groups[worker] = local_device_groups
            else:
                ranks = group['ranks']
                local_device_groups = []
                global_cpu_proc_idx = 0
                for _ in range(ranks):
                    local_device_groups.append(
                        dict(
                            placement_group=self.cpu_placement_groups[self.cpu_node_map[global_cpu_proc_idx][0]],
                            ray_address=ray_address))
                    global_cpu_proc_idx += 1
                for worker in group['workers']:
                    self.device_groups[worker] = local_device_groups

        self.groups = groups

    def resource(self, worker):
        return self.device_groups[worker]

    def destroy_placement_group(self):
        import ray
        for pg in self.placement_groups:
            ray.util.remove_placement_group(pg)
        for pg in self.cpu_placement_groups:
            ray.util.remove_placement_group(pg)

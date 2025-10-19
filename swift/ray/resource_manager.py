import math
from dataclasses import dataclass, field
from typing import Dict, List, Any

import ray
from ray.util.placement_group import PlacementGroup


@dataclass
class NodeGroup:
    device_count: int
    nodes: List[Any] = field(default_factory=list)


class ResourceManager:

    def __init__(self, groups: Dict[str, Dict[str, Any]]):
        nproc_per_node = groups['nproc_per_node']
        device_types = set([group['device'] for group in groups.values()])
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

        all_resources: Dict[str, List[NodeGroup]] = {}
        for node in ray.nodes():
            resource = node["Resources"]
            for device in device_types:
                if device not in all_resources:
                    all_resources[device] = []

                device_count = int(resource.get(device, 0))
                node_group = [ng for ng in all_resources[device] if ng.device_count == device_count]
                if not node_group:
                    ng = NodeGroup(device_count=device_count)
                    all_resources[device].append(ng)
                    node_group = [ng]

                assert len(node_group) == 1
                node_group = node_group[0]
                node_group.nodes.append(node)

        bundles = []
        for i in range(self.num_nodes):
            node = nodes_maybe_used[i]
            node_cpu = int(node["Resources"]["CPU"])
            bundles.append({current_platform.ray_device_key: self.gpu_per_node, "CPU": max(node_cpu / 2, 1)})

        self.placement_groups = [ray.util.placement_group([bundle]) for bundle in bundles]
        ray.get([pg.ready() for pg in self.placement_groups])

        for resource in all_resources.values():
            resource.sort(key=lambda x: x.device_count)

        device_groups = []
        for group_name, group in groups.items():
            node_groups = all_resources[group['device']]
            nproc_per_node = group['nproc_per_node']
            nnodes = group['nnodes']
            assert len(node_groups) >= nnodes, 'More nodes required to commit this training task.'
            for i in range(nnodes):
                for ng in node_groups:
                    if ng.device_count >= nproc_per_node:
                        bundle = ray.util.placement_group([bundle])
                        {current_platform.ray_device_key: self.gpu_per_node, "CPU": max(node_cpu / 2, 1)}
                       device_groups.append(dict(node_rank=node_rank,
                                                 gpu_rank=gpu_rank,
                                                 placement_group=pg,
                                                 ray_address=ray_address))

        nodes_maybe_used = []
        ray_nodes = ray.nodes()
        for node in ray_nodes:

            if node_gpu_num >= num_gpus_per_node:
                nodes_maybe_used.append(node)
        nodes_maybe_used = sorted(nodes_maybe_used, key=lambda n: n["Resources"]["CPU"])

        ray_num_nodes = len(nodes_maybe_used)
        if num_nodes is None:
            num_nodes = ray_num_nodes

        assert num_nodes <= ray_num_nodes, (f"The Ray clusters(ray_num_nodes: {ray_num_nodes}) cannot meet the "
                                            f"required number of nodes (`num_nodes`{num_nodes}).")
        self.num_nodes = num_nodes
        self.gpu_per_node = num_gpus_per_node
        self.num_gpus = self.gpu_per_node * self.num_nodes

        if self.gpu_per_node > 0:
            assert self.num_gpus <= available_gpu, f"num_gpus {self.num_gpus} > available_gpu {available_gpu}"
            bundles = []
            for i in range(self.num_nodes):
                node = nodes_maybe_used[i]
                node_cpu = int(node["Resources"]["CPU"])
                bundles.append({current_platform.ray_device_key: self.gpu_per_node, "CPU": max(node_cpu / 2, 1)})

            self.placement_groups = [ray.util.placement_group([bundle]) for bundle in bundles]
            ray.get([pg.ready() for pg in self.placement_groups])
            gpu_ranks = ray.get([
                get_visible_gpus.options(
                    placement_group=pg,
                    **(
                        {"num_gpus": self.gpu_per_node}
                        if current_platform.ray_device_key == "GPU"
                        else {"resources": {current_platform.ray_device_key: self.gpu_per_node}}
                    )
                ).remote(current_platform.device_control_env_var)
                for pg in self.placement_groups
            ])
            print(f"gpu ranks: {gpu_ranks}")
            self.node_ranks = ray.get(
                [get_node_rank.options(placement_group=pg).remote() for pg in self.placement_groups])
            if self.node_ranks.count(0) > 1:
                # NODE_RANK environment variable is not set in the cluster, so a default value is used for NODE_RANK.
                self.node_ranks = list(range(len(self.placement_groups)))

            self.gpu_ranks = [int(gpu_rank[0]) for gpu_rank in gpu_ranks]
            self.node2pg: Dict[int, PlacementGroup] = {}
            for node_rank, placement_group in zip(self.node_ranks, self.placement_groups):
                self.node2pg[node_rank] = placement_group
            print(f"node2pg: {self.node2pg}")
        else:
            assert self.num_nodes == 1
            node = nodes_maybe_used[0]
            node_cpu = int(node["Resources"]["CPU"])
            bundles = [{"CPU": node_cpu}] * self.num_nodes
            self.placement_groups = [ray.util.placement_group([bundle]) for bundle in bundles]
            ray.get([pg.ready() for pg in self.placement_groups])
            self.node_ranks = [0]
            self.node2pg: Dict[int, PlacementGroup] = {}
            for node_rank, placement_group in zip(self.node_ranks, self.placement_groups):
                self.node2pg[node_rank] = placement_group

    def nodes_placement_group(self, node_rank) -> PlacementGroup:
        """
        mesh table是 m×n，获取第node_rank nodel上gpu_rank的PlacementGroup，用于把ray.Actor部署到指定的GPU上
        """
        return self.node2pg[node_rank]

    def destroy_placement_group(self):
        [ray.util.remove_placement_group(pg) for pg in self.placement_groups]

    def allocate_placement_group(self, world_size, device_mapping: List[int] = None) -> List[List[Dict]]:
        """
            Allocate resources according to device_mapping (numbered by GPU RANK)
            - GPUs: Specify required GPU indices via device_mapping
            - CPUs: Specify via world_size

            Return Type: List[List[Dict]]
              Dict Keys:
                - node_rank
                - gpu_rank
                - placement_group
              List[Dict]: Represents GPUs allocated to a worker and access to placement groups
              Example: If num_gpus_per_worker=8, then len(List[Dict])=8

            A Worker is defined as a group of resource owners (can span multiple machines) that can independently use allocated resources to execute computation operations.
        """
        allocated_pg = []
        ray_address = f"{ray.get_runtime_context().gcs_address}"
        if device_mapping:
            num_gpus_per_worker = len(device_mapping) // world_size
            grouped_ranks = [
                list(device_mapping[i : i + num_gpus_per_worker])
                for i in range(0, len(device_mapping), num_gpus_per_worker)
            ]
            for group in grouped_ranks:
                pg_list = []
                for rank in group:
                    node_rank = rank // self.gpu_per_node
                    gpu_rank = rank % self.gpu_per_node

                    assert node_rank < self.num_nodes, (f"device_mapping used gpus are more than "
                                                        f"num_nodes×num_gpus_per_node={self.num_nodes}×{self.gpu_per_node}")

                    pg = self.nodes_placement_group(node_rank)
                    pg_list.append(
                        dict(node_rank=node_rank, gpu_rank=gpu_rank, placement_group=pg, ray_address=ray_address)
                    )
                allocated_pg.append(pg_list)
        else:
            # Try to spread the CPU workers across various nodes to avoid the out-of-memory (OOM) situation caused
            # by the concentration of CPU workers in one place and the resulting peak memory usage.
            for rank in range(world_size):
                node_rank = rank % self.num_nodes
                allocated_pg.append(
                    [
                        dict(
                            node_rank=node_rank,
                            gpu_rank=None,
                            placement_group=self.nodes_placement_group(node_rank),
                            ray_address=ray_address,
                        )
                    ]
                )

        assert len(allocated_pg) == world_size

        return allocated_pg

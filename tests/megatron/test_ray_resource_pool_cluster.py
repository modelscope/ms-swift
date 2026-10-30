# Copyright (c) ModelScope Contributors. All rights reserved.
import importlib
import os
import pytest
import ray
import sys
import threading
import time

from swift.ray.megatron.resource_pool import ResourcePool


@pytest.mark.skipif(sys.platform != 'linux', reason='Local multi-node Ray cluster requires Linux')
def test_pool_uses_distinct_ray_nodes_under_temporary_contention(monkeypatch):
    from ray.cluster_utils import Cluster
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    if ray.is_initialized():
        pytest.skip('Requires an isolated Ray driver')
    # GPU resources are logical; the probe performs no CUDA operations.
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    cluster = Cluster()
    pool = ResourcePool([2, 2])
    lease = None
    created = []
    submitted = threading.Event()
    stopping = threading.Event()
    released = threading.Event()
    release_errors = []
    thread = None
    try:
        cluster.add_node(
            num_cpus=1,
            num_gpus=0,
            object_store_memory=80 << 20,
            temp_dir=f'/tmp/swift-pool-test-{os.getpid()}',
            include_dashboard=False)
        for _ in range(2):
            cluster.add_node(num_cpus=4, num_gpus=4, object_store_memory=80 << 20)
        cluster.wait_for_nodes()
        ray.init(address=cluster.address, log_to_driver=False)
        nodes = [node for node in ray.nodes() if node['Resources'].get('GPU', 0)]

        @ray.remote(num_cpus=4)
        class CpuLease:

            def ready(self):
                return ray.get_runtime_context().get_node_id()

        target = nodes[1]['NodeID']
        lease = CpuLease.options(scheduling_strategy=NodeAffinitySchedulingStrategy(target, soft=False)).remote()
        assert ray.get(lease.ready.remote(), timeout=90) == target
        pg_module = importlib.import_module('ray.util.placement_group')
        placement = pg_module.placement_group
        original_get = ray.get

        def track_placement(*args, **kwargs):
            pg = placement(*args, **kwargs)
            created.append(pg)
            if len(created) == 2:
                submitted.set()
            return pg

        def bounded_get(*args, **kwargs):
            kwargs.setdefault('timeout', 90)
            return original_get(*args, **kwargs)

        def release_cpu():
            try:
                if not submitted.wait(90):
                    return
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and not stopping.is_set():
                    tables = [ray._private.state.state.placement_group_table(pg.id) for pg in created]
                    if all(table['state'] == 'CREATED' for table in tables):
                        break
                    time.sleep(.05)
                if not stopping.is_set():
                    ray.kill(lease, no_restart=True)
                    released.set()
            except BaseException as error:
                release_errors.append(error)

        monkeypatch.setattr(pg_module, 'placement_group', track_placement)
        monkeypatch.setattr(ray, 'get', bounded_get)
        thread = threading.Thread(target=release_cpu, daemon=True)
        thread.start()
        pool.create()
        assert not release_errors
        assert len({node for node, _ in pool.bundle_infos}) == 2
        assert len(pool.pgs) == pool.num_nodes == 2
        assert len(pool.bundle_infos) == pool.world_size == 4
    finally:
        stopping.set()
        submitted.set()
        if thread is not None:
            thread.join(timeout=10)
        if lease is not None and not released.is_set():
            ray.kill(lease, no_restart=True)
        pool.destroy()
        for pg in created:
            ray.util.remove_placement_group(pg)
        ray.shutdown()
        cluster.shutdown()

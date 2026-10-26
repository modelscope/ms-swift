# Copyright (c) ModelScope Contributors. All rights reserved.
"""Exercise the checkpoint protocol with real TCP ZMQ sockets and CPU collectives."""
import ast
import asyncio
import faulthandler
import importlib.util
import logging
import multiprocessing
import pytest
import queue
import signal
import time
import torch
import torch.distributed as dist
import traceback
import zmq
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


@dataclass
class _MasterMetadata:
    zmq_ip: str
    zmq_port: int
    nccl_store_host: str = ''
    nccl_store_port: int = 0
    auth_key: str = ''


def _load_checkpoint_module():
    # Load the communication code without importing the Ray/training stack.
    # Class and function bodies are compiled unchanged; only device/runtime
    # dependencies are supplied by the CPU test harness.
    import os
    import socket
    root = Path(os.environ.get('SWIFT_CKPT_TEST_ROOT', Path(__file__).resolve().parents[2]))
    engine_dir = root / 'swift/ray/megatron/checkpoint_engine'
    spec = importlib.util.spec_from_file_location('checkpoint_zmq_auth', engine_dir / 'zmq_auth.py')
    auth = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auth)

    def free_port():
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            return sock.getsockname()[1]

    module = ModuleType('checkpoint_nccl')
    module.__dict__.update(
        asyncio=asyncio,
        time=time,
        torch=torch,
        dist=dist,
        ray=SimpleNamespace(util=SimpleNamespace(get_node_ip_address=lambda: '127.0.0.1')),
        CheckpointEngine=object,
        MasterMetadata=_MasterMetadata,
        get_current_device=lambda: 'cpu',
        synchronize=lambda: None,
        _find_free_port=free_port,
        _is_valid_ipv6_address=lambda address: False,
        generate_auth_key=auth.generate_auth_key,
        send_auth_pyobj=auth.send_auth_pyobj,
        recv_auth_pyobj=auth.recv_auth_pyobj,
        logger=logging.getLogger(__name__),
    )
    source_path = engine_dir / 'nccl.py'
    tree = ast.parse(source_path.read_text())
    tree.body = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef,
                             ast.ClassDef)) or isinstance(node, ast.ImportFrom) and node.module == '__future__'
    ]
    exec(compile(tree, str(source_path), 'exec'), module.__dict__)
    return module


def _checkpoint_worker(rank, world_size, metadata, start, events, trace_path):
    nccl = _load_checkpoint_module()

    torch.set_num_threads(1)
    with open(trace_path, 'w') as trace:
        faulthandler.register(signal.SIGUSR1, file=trace, all_threads=True)
        original_broadcast = nccl._pg_broadcast

        def broadcast(pg, tensor, src=0):
            if tensor.numel() != 1:
                events.put(('bucket_enter', rank))
            original_broadcast(pg, tensor, src)
            if tensor.numel() == 1:
                events.put(('barrier_done', rank))

        with patch.object(nccl, 'get_current_device', return_value='cpu'), \
                patch.object(nccl, 'synchronize'), \
                patch.object(nccl.dist, 'ProcessGroupNCCL', dist.ProcessGroupGloo), \
                patch.object(nccl.ray.util, 'get_node_ip_address', return_value='127.0.0.1'), \
                patch.object(nccl, '_pg_broadcast', side_effect=broadcast):
            engine = nccl.NCCLCheckpointEngine(bucket_size=32)
            engine.is_master = rank == 0
            try:
                prepared = engine.prepare()
                if rank == 0:
                    metadata = prepared
                    events.put(('metadata', metadata))
                    if not start.wait(30):
                        raise TimeoutError('Test receivers did not start')
                else:
                    metadata = _MasterMetadata(**metadata)

                async def synchronize_weights():
                    for step in range(2):
                        engine.init_process_group(rank, world_size, metadata)
                        events.put(('initialized', rank))
                        weights = [(name, torch.arange(size, dtype=torch.float32) + step)
                                   for name, size in [('a', 3), ('b', 4), ('c', 5)]]
                        if rank == 0:
                            await engine.send_weights(iter(weights))
                        else:
                            received = {name: tensor.clone() async for name, tensor in engine.receive_weights()}
                            assert set(received) == {name for name, _ in weights}
                            for name, expected in weights:
                                torch.testing.assert_close(received[name], expected, rtol=0, atol=0)
                        engine.finalize()
                    events.put(('done', rank))

                asyncio.run(synchronize_weights())
            except BaseException:
                events.put(('error', rank, traceback.format_exc()))
            finally:
                if engine.socket is not None:
                    context = engine.socket.context
                    engine.socket.close(linger=0)
                    context.term()
                engine._pg = None
                engine._store = None


@pytest.mark.skipif(not hasattr(signal, 'SIGUSR1'), reason='Stack diagnostics require POSIX signals')
@pytest.mark.parametrize('receiver_count,subscription_delay', [(1, 0), (1, 0.5), (2, 0.5)])
def test_delayed_subscription_does_not_hang_weight_sync(tmp_path, receiver_count, subscription_delay):
    """Delay only the last receiver's subscription; never discard data in the proxy."""
    ctx = multiprocessing.get_context('spawn')
    start, events = ctx.Event(), ctx.Queue()
    processes = []
    context = zmq.Context()
    sockets = []
    observed = []
    trace_paths = [tmp_path / f'rank-{rank}.log' for rank in range(receiver_count + 1)]
    subscriptions_released = False
    forwarded_messages = 0

    def launch(rank, metadata):
        process = ctx.Process(
            target=_checkpoint_worker,
            args=(rank, receiver_count + 1, metadata, start, events, str(trace_paths[rank])),
        )
        process.start()
        processes.append(process)

    try:
        launch(0, None)
        event = events.get(timeout=30)
        assert event[0] == 'metadata', event
        master = event[1]
        poller = zmq.Poller()
        pairs = []
        for rank in range(1, receiver_count + 1):
            frontend, backend = context.socket(zmq.XPUB), context.socket(zmq.XSUB)
            sockets.extend([frontend, backend])
            port = frontend.bind_to_random_port('tcp://127.0.0.1')
            backend.connect(f'tcp://{master.zmq_ip}:{master.zmq_port}')
            poller.register(frontend, zmq.POLLIN)
            poller.register(backend, zmq.POLLIN)
            pairs.append((rank, frontend, backend))
            launch(rank, vars(replace(master, zmq_port=port)))
        start.set()

        held = []
        barrier_ranks, done_ranks = set(), set()
        barrier_time = released_time = None
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and len(done_ranks) != receiver_count + 1:
            while True:
                try:
                    event = events.get_nowait()
                except queue.Empty:
                    break
                observed.append(event)
                if event[0] == 'barrier_done':
                    barrier_ranks.add(event[1])
                elif event[0] == 'done':
                    done_ranks.add(event[1])
                elif event[0] == 'error':
                    pytest.fail(str(event))
            if len(barrier_ranks) == receiver_count + 1 and barrier_time is None:
                barrier_time = time.monotonic()
            if barrier_time is not None and not subscriptions_released:
                if time.monotonic() - barrier_time >= subscription_delay:
                    for backend, subscription in held:
                        backend.send(subscription)
                    held.clear()
                    subscriptions_released = True
                    released_time = time.monotonic()
            ready = dict(poller.poll(10))
            for rank, frontend, backend in pairs:
                if frontend in ready:
                    subscription = frontend.recv()
                    if rank == receiver_count and not subscriptions_released and subscription_delay:
                        held.append((backend, subscription))
                    else:
                        backend.send(subscription)
                if backend in ready:
                    frontend.send_multipart(backend.recv_multipart())
                    forwarded_messages += 1
            if released_time is not None and time.monotonic() - released_time > 4:
                break

        alive = [p.is_alive() for p in processes]
        assert len(done_ranks) == receiver_count + 1, (
            f'Weight sync hung after subscription delivery resumed: alive={alive}, '
            f'forwarded_messages={forwarded_messages}, events={observed}')
        for process in processes:
            process.join(timeout=5)
            assert process.exitcode == 0
    finally:
        # Gather stacks before the test watchdog cleans up blocked workers.
        for process in processes:
            if process.is_alive():
                import os
                os.kill(process.pid, signal.SIGUSR1)
        if any(process.is_alive() for process in processes):
            time.sleep(0.1)
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        for socket in sockets:
            socket.close(linger=0)
        context.term()
        events.close()
        for path in trace_paths:
            if path.exists() and path.stat().st_size:
                print(f'Worker stack at watchdog deadline ({path}):\n{Path(path).read_text()}')


def test_missing_rank_subscription_times_out():
    nccl = _load_checkpoint_module()
    engine = nccl.NCCLCheckpointEngine(bucket_size=32)
    engine.is_master = True
    metadata = engine.prepare()
    engine.rank, engine.world_size = 0, 2
    engine._store = dist.HashStore()
    engine._store.set_timeout(timedelta(milliseconds=100))
    engine._pg = object()
    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.connect(f'tcp://{metadata.zmq_ip}:{metadata.zmq_port}')
    # A common-topic subscription must not stand in for a rank's readiness.
    subscriber.setsockopt_string(zmq.SUBSCRIBE, engine.topic)
    try:
        with patch.object(nccl, '_pg_broadcast'):
            with pytest.raises(TimeoutError, match='bucket_metadata/ready/1'):
                engine.init_process_group(0, 2, metadata)
        assert not engine._group_initialized
    finally:
        subscriber.close(linger=0)
        context.term()
        publisher_context = engine.socket.context
        engine.socket.close(linger=0)
        publisher_context.term()

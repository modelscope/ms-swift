# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import asyncio
import ray
import time
import torch
import torch.distributed as dist
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Tuple

from swift.utils import get_current_device, synchronize
from swift.utils.logger import get_logger
from .base import CheckpointEngine, MasterMetadata, TensorMeta, _find_free_port, _is_valid_ipv6_address

logger = get_logger()


def _pg_broadcast(pg, tensor, src: int = 0):
    """Broadcast tensor via raw ProcessGroupNCCL (unregistered PG)."""
    opts = dist.BroadcastOptions()
    opts.rootRank = src
    work = pg.broadcast([tensor], opts)
    work.wait()


class BroadcastOperation:
    """Async NCCL broadcast in a thread pool executor."""

    def __init__(self, rank, pg, bucket, metadata, zmq_socket, topic):
        self.rank = rank
        self.pg = pg
        self.bucket = bucket
        self.metadata = metadata
        self.socket = zmq_socket
        self.topic = topic

        loop = asyncio.get_running_loop()
        self._task = loop.run_in_executor(None, self._run)

    def _run(self):
        import zmq
        if self.rank == 0:
            self.socket.send_string(self.topic, flags=zmq.SNDMORE)
            self.socket.send_pyobj(self.metadata)
        else:
            self.socket.recv_string()
            self.metadata = self.socket.recv_pyobj()
        _pg_broadcast(self.pg, self.bucket, src=0)

    async def wait_for_complete(self) -> dict:
        await self._task
        return self.metadata


class NCCLCheckpointEngine(CheckpointEngine):

    def __init__(
        self,
        bucket_size: int = 3072 << 20,
        group_name: str = 'swift_ckpt',
        rebuild_group: bool = False,
        **kwargs,
    ) -> None:
        self.bucket_size = bucket_size
        self.group_name = group_name
        self.rebuild_group = rebuild_group

        self.is_master = False
        self.topic = 'bucket_metadata'

        self.rank = None
        self.world_size = None
        self.send_buf = None
        self.recv_buf = None
        self.socket = None

        self._pg = None
        self._store = None
        self._prepared = False
        self._group_initialized = False

    def _start_zmq_server(self):
        import zmq
        self.ip = ray.util.get_node_ip_address().strip('[]')
        self.listen_port = _find_free_port()

        context = zmq.Context()
        self.socket = context.socket(zmq.PUB)
        if _is_valid_ipv6_address(self.ip):
            address = f'tcp://[{self.ip}]:{self.listen_port}'
            self.socket.setsockopt(zmq.IPV6, 1)
        else:
            address = f'tcp://{self.ip}:{self.listen_port}'
        self.socket.bind(address)

    def _connect_zmq_client(self, metadata: MasterMetadata):
        import zmq
        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)
        if _is_valid_ipv6_address(metadata.zmq_ip):
            address = f'tcp://[{metadata.zmq_ip}]:{metadata.zmq_port}'
            self.socket.setsockopt(zmq.IPV6, 1)
        else:
            address = f'tcp://{metadata.zmq_ip}:{metadata.zmq_port}'
        self.socket.connect(address)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, self.topic)

    def prepare(self) -> Optional[MasterMetadata]:
        """Allocate buffers and start ZMQ server (master only). Idempotent."""
        if self._prepared:
            if self.is_master:
                return MasterMetadata(
                    zmq_ip=self.ip,
                    zmq_port=self.listen_port,
                    nccl_store_host=self._nccl_store_host,
                    nccl_store_port=self._nccl_store_port,
                )
            return None

        device = get_current_device()
        self.send_buf = torch.zeros(self.bucket_size, dtype=torch.uint8, device=device)
        self.recv_buf = torch.zeros(self.bucket_size, dtype=torch.uint8, device=device)

        if self.is_master:
            self._start_zmq_server()
            self._nccl_store_host = self.ip
            self._nccl_store_port = _find_free_port()
            self._prepared = True
            return MasterMetadata(
                zmq_ip=self.ip,
                zmq_port=self.listen_port,
                nccl_store_host=self._nccl_store_host,
                nccl_store_port=self._nccl_store_port,
            )
        else:
            self._prepared = True
            return None

    def finalize(self):
        """Clean up resources. Full teardown only when rebuild_group=True."""
        if self.rebuild_group:
            if self.socket is not None:
                try:
                    self.socket.close()
                except Exception as e:
                    logger.warning('Error closing ZMQ socket: %s', e)
                self.socket = None

            if self._pg is not None:
                self._pg = None
                self._store = None

            self.rank = None
            self.world_size = None
            self.send_buf = None
            self.recv_buf = None
            self._prepared = False
            self._group_initialized = False

    @classmethod
    def build_topology(
        cls,
        trainer_world_size: int,
        rollout_world_size: int,
        metadata: List[Dict],
    ) -> Tuple[Dict[str, List[Any]], Dict[str, List[Any]]]:
        """Build NCCL broadcast topology: trainer rank0 as source, rollout as receivers."""
        master_metadata = metadata[0]

        trainer_kwargs = {
            'rank': [0] + [-1] * (trainer_world_size - 1),
            'world_size': [rollout_world_size + 1] * trainer_world_size,
            'master_metadata': [master_metadata] * trainer_world_size,
        }
        rollout_kwargs = {
            'rank': list(range(1, rollout_world_size + 1)),
            'world_size': [rollout_world_size + 1] * rollout_world_size,
            'master_metadata': [master_metadata] * rollout_world_size,
        }
        return trainer_kwargs, rollout_kwargs

    def init_process_group(self, rank: int, world_size: int, master_metadata: MasterMetadata):
        """Initialize a dedicated NCCL process group for weight synchronization.

        Creates a ``ProcessGroupNCCL`` directly (without registering it in
        the default ``_World``), using a ``TCPStore`` hosted by the master
        for rendezvous.

        Idempotent when ``rebuild_group=False``.
        """
        import os

        if rank < 0:
            self.rank = rank
            self.world_size = world_size
            self._group_initialized = True
            return

        if self._group_initialized and not self.rebuild_group:
            return

        if self._pg is None:
            self.rank = rank
            self.world_size = world_size

            os.environ['NCCL_CUMEM_ENABLE'] = '0'

            is_store_master = (rank == 0)
            self._store = dist.TCPStore(
                host_name=master_metadata.nccl_store_host,
                port=master_metadata.nccl_store_port,
                world_size=world_size,
                is_master=is_store_master,
                wait_for_workers=True,
            )
            self._pg = dist.ProcessGroupNCCL(
                self._store,
                rank,
                world_size,
            )
        else:
            assert self.rank == rank, f'rank {rank} != self.rank {self.rank}'
            assert self.world_size == world_size

        if self.rank > 0 and self.socket is None:
            self._connect_zmq_client(master_metadata)

        barrier_tensor = torch.zeros(1, dtype=torch.int32, device=get_current_device())
        _pg_broadcast(self._pg, barrier_tensor, src=0)
        synchronize()

        # ZMQ PUB/SUB "slow joiner" mitigation: after NCCL barrier confirms
        # all participants are connected, give SUB sockets time to fully
        # establish the subscription before PUB sends metadata.
        if self.rank == 0 and self.socket is not None:
            time.sleep(0.1)

        self._group_initialized = True

    # ── Send / Receive ───────────────────────────────────────────────────

    @torch.no_grad()
    async def send_weights(
        self,
        weights: Generator[Tuple[str, 'torch.Tensor'], None, None],
    ):
        """Send model weights to rollout workers via NCCL broadcast.

        Uses double buffering: fill send_buf while the previous bucket
        is being broadcast, then swap buffers.
        """
        assert self.rank is not None and self.rank <= 0

        if self.rank < 0:
            for name, weight in weights:
                pass
            return

        send_buf, recv_buf = self.send_buf, self.recv_buf
        broadcast_op = None
        start_time = time.time()
        bucket_meta: Dict[str, TensorMeta] = {}
        offset = 0

        for name, weight in weights:
            if offset + weight.nbytes > self.bucket_size:
                synchronize()
                if broadcast_op is not None:
                    await broadcast_op.wait_for_complete()

                broadcast_op = BroadcastOperation(
                    rank=self.rank,
                    pg=self._pg,
                    bucket=send_buf,
                    metadata={
                        'bucket_meta': bucket_meta,
                        'is_last': False
                    },
                    zmq_socket=self.socket,
                    topic=self.topic,
                )
                send_buf, recv_buf = recv_buf, send_buf
                bucket_meta = {}
                offset = 0

            assert offset + weight.nbytes <= self.bucket_size, (
                f'Weight {name}({weight.shape}, {weight.dtype}) is too large '
                f'for bucket ({self.bucket_size / 1e6:.1f} MB). Increase bucket_size.')

            bucket_meta[name] = {
                'name': name,
                'shape': weight.shape,
                'dtype': weight.dtype,
                'offset': offset,
            }
            send_buf[offset:offset + weight.nbytes].copy_(weight.view(-1).view(torch.uint8), non_blocking=True)
            offset += weight.nbytes

        synchronize()
        if broadcast_op is not None:
            await broadcast_op.wait_for_complete()

        broadcast_op = BroadcastOperation(
            rank=self.rank,
            pg=self._pg,
            bucket=send_buf,
            metadata={
                'bucket_meta': bucket_meta,
                'is_last': True
            },
            zmq_socket=self.socket,
            topic=self.topic,
        )
        await broadcast_op.wait_for_complete()

        logger.debug('Rank %d send weights done, time cost: %.2fs', self.rank, time.time() - start_time)

    @torch.no_grad()
    async def receive_weights(self) -> AsyncGenerator[Tuple[str, 'torch.Tensor'], None]:
        """Receive model weights from trainer via NCCL broadcast.

        Uses double buffering: receive into recv_buf while processing
        send_buf, then swap.

        Yields:
            Tuples of (name, tensor).  The tensor is a *view* into the
            receive buffer — callers that need to keep it should clone it.
        """
        assert self.rank is not None and self.rank > 0

        send_buf, recv_buf = self.send_buf, self.recv_buf
        total_bytes, total_params = 0, 0

        start_time = time.time()
        broadcast_op = BroadcastOperation(
            rank=self.rank,
            pg=self._pg,
            bucket=recv_buf,
            metadata=None,
            zmq_socket=self.socket,
            topic=self.topic,
        )
        metadata = await broadcast_op.wait_for_complete()
        total_bytes += self.bucket_size
        total_params += len(metadata['bucket_meta'])
        send_buf, recv_buf = recv_buf, send_buf

        while not metadata['is_last']:
            broadcast_op = BroadcastOperation(
                rank=self.rank,
                pg=self._pg,
                bucket=recv_buf,
                metadata=None,
                zmq_socket=self.socket,
                topic=self.topic,
            )
            for name, meta in metadata['bucket_meta'].items():
                dtype, shape = meta['dtype'], meta['shape']
                size = dtype.itemsize * shape.numel()
                tensor = send_buf[meta['offset']:meta['offset'] + size].view(dtype=dtype).view(shape)
                yield name, tensor

            metadata = await broadcast_op.wait_for_complete()
            total_bytes += self.bucket_size
            total_params += len(metadata['bucket_meta'])
            synchronize()
            send_buf, recv_buf = recv_buf, send_buf

        for name, meta in metadata['bucket_meta'].items():
            dtype, shape = meta['dtype'], meta['shape']
            size = dtype.itemsize * shape.numel()
            tensor = send_buf[meta['offset']:meta['offset'] + size].view(dtype=dtype).view(shape)
            yield name, tensor

        elapsed = time.time() - start_time
        bandwidth = total_bytes / elapsed / (1024 * 1024 * 1024) if elapsed > 0 else 0
        logger.debug('receive_weights done: rank=%d, params=%d, time=%.2fs, bandwidth=%.2f GB/s', self.rank,
                     total_params, elapsed, bandwidth)

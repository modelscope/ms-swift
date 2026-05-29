# Copyright (c) ModelScope Contributors. All rights reserved.
# Adapted from twinkle/src/twinkle/checkpoint_engine/hccl_checkpoint_engine.py
"""HCCL-based checkpoint engine for Ascend NPU.

Uses HCCL for weight payload transfer and ZMQ REQ/REP for bucket
metadata handshakes (reliable, with timeout).
"""
from __future__ import annotations

import os
import time
import torch
import zmq
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Generator, List, Optional, Tuple

from swift.utils import get_current_device, synchronize
from swift.utils.logger import get_logger
from .base import CheckpointEngine, TensorMeta, _find_free_port, _is_valid_ipv6_address

logger = get_logger()


def _configure_zmq_socket(socket: zmq.Socket, timeout_ms: int, linger: int = 0) -> None:
    """Apply timeout/linger options to a ZMQ socket."""
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    socket.setsockopt(zmq.LINGER, linger)


@dataclass
class HCCLMasterMetadata:
    """Metadata from the master for HCCL process group initialization."""
    zmq_ip: str
    zmq_port: int
    dist_ip: str
    dist_port: int


def _stateless_init_hccl(
    master_address: str,
    master_port: int,
    rank: int,
    world_size: int,
    device: int,
):
    """Create a stateless HCCL communicator via vLLM's StatelessProcessGroup."""
    import socket as _socket
    from datetime import timedelta
    from torch.distributed import TCPStore
    from vllm.distributed.utils import StatelessProcessGroup
    from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator

    launch_server = (rank == 0)
    listen_socket = None
    listen_fd = None

    if launch_server:
        if _is_valid_ipv6_address(master_address):
            listen_socket = _socket.socket(_socket.AF_INET6, _socket.SOCK_STREAM)
        else:
            listen_socket = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        listen_socket.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
        listen_socket.bind((master_address, master_port))
        listen_socket.listen()
        listen_fd = listen_socket.fileno()

    store = TCPStore(
        host_name=master_address,
        port=master_port,
        world_size=world_size,
        is_master=launch_server,
        timeout=timedelta(seconds=300),
        use_libuv=False,
        master_listen_fd=listen_fd,
    )

    pg = StatelessProcessGroup(
        rank=rank,
        world_size=world_size,
        store=store,
        socket=listen_socket,
        data_expiration_seconds=3600,
    )

    return PyHcclCommunicator(pg, device=device)


class HCCLCheckpointEngine(CheckpointEngine):
    """HCCL checkpoint engine for Ascend NPU weight synchronization."""

    def __init__(
        self,
        bucket_size: int = 3072 << 20,
        group_name: str = 'swift_ckpt',
        rebuild_group: bool = True,
        **kwargs,
    ) -> None:
        self.bucket_size = bucket_size
        self.group_name = group_name
        self.rebuild_group = rebuild_group
        self.pyhccl = None

        self.meta_timeout_s = int(os.environ.get('SWIFT_CKPT_HCCL_META_TIMEOUT_S', '300'))
        self.meta_timeout_ms = self.meta_timeout_s * 1000

        self.device = get_current_device()

        self.is_master = False
        self.rank: Optional[int] = None
        self.world_size: Optional[int] = None
        self.send_buf: Optional[torch.Tensor] = None
        self.recv_buf: Optional[torch.Tensor] = None
        self.socket: Optional[zmq.Socket] = None
        self._zmq_ctx: Optional[zmq.Context] = None

        self._prepared = False
        self._group_initialized = False
        self.ip: Optional[str] = None
        self.zmq_port: Optional[int] = None
        self.dist_port: Optional[int] = None

    def _new_socket(self, socket_type: int) -> zmq.Socket:
        assert self._zmq_ctx is not None
        socket = self._zmq_ctx.socket(socket_type)
        _configure_zmq_socket(socket, timeout_ms=self.meta_timeout_ms)
        return socket

    def _recv_pyobj(self, where: str) -> Any:
        assert self.socket is not None
        try:
            return self.socket.recv_pyobj()
        except zmq.error.Again as e:
            raise RuntimeError(f'HCCL metadata timeout ({self.meta_timeout_s}s) waiting at {where}.') from e

    def _send_pyobj(self, payload: Any, where: str) -> None:
        assert self.socket is not None
        try:
            self.socket.send_pyobj(payload)
        except zmq.error.Again as e:
            raise RuntimeError(f'HCCL metadata timeout ({self.meta_timeout_s}s) sending at {where}.') from e

    def _start_zmq_server(self):
        import ray
        self.ip = ray.util.get_node_ip_address().strip('[]')
        self.zmq_port = _find_free_port()
        self.dist_port = _find_free_port()

        self._zmq_ctx = zmq.Context()
        self.socket = self._new_socket(zmq.REP)
        if _is_valid_ipv6_address(self.ip):
            address = f'tcp://[{self.ip}]:{self.zmq_port}'
            self.socket.setsockopt(zmq.IPV6, 1)
        else:
            address = f'tcp://{self.ip}:{self.zmq_port}'
        self.socket.bind(address)

    def _connect_zmq_client(self, metadata: HCCLMasterMetadata):
        self._zmq_ctx = zmq.Context()
        self.socket = self._new_socket(zmq.REQ)
        if _is_valid_ipv6_address(metadata.zmq_ip):
            address = f'tcp://[{metadata.zmq_ip}]:{metadata.zmq_port}'
            self.socket.setsockopt(zmq.IPV6, 1)
        else:
            address = f'tcp://{metadata.zmq_ip}:{metadata.zmq_port}'
        self.socket.connect(address)

    # ── Core lifecycle ───────────────────────────────────────────────────

    def prepare(self) -> Optional[HCCLMasterMetadata]:
        if self._prepared:
            if self.is_master:
                return HCCLMasterMetadata(
                    zmq_ip=self.ip, zmq_port=self.zmq_port, dist_ip=self.ip, dist_port=self.dist_port)
            return None

        self.send_buf = torch.zeros(self.bucket_size, dtype=torch.uint8, device='npu')
        self.recv_buf = torch.zeros(self.bucket_size, dtype=torch.uint8, device='npu')

        if self.is_master:
            self._start_zmq_server()
            self._prepared = True
            return HCCLMasterMetadata(zmq_ip=self.ip, zmq_port=self.zmq_port, dist_ip=self.ip, dist_port=self.dist_port)

        self._prepared = True
        return None

    def finalize(self):
        if self.rebuild_group:
            if self.socket is not None:
                try:
                    self.socket.close()
                except Exception as e:
                    logger.warning('Error closing ZMQ socket: %s', e)
                self.socket = None

            if self._zmq_ctx is not None:
                try:
                    self._zmq_ctx.term()
                except Exception as e:
                    logger.warning('Error terminating ZMQ context: %s', e)
                self._zmq_ctx = None

            if self.rank is not None and self.rank >= 0 and self.pyhccl is not None:
                try:
                    self.pyhccl.destroyComm(self.pyhccl.comm)
                except Exception:
                    pass
                self.pyhccl = None

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
        metadata: List[Any],
    ) -> Tuple[dict, dict]:
        master_metadata = None
        for m in metadata:
            if m is not None:
                master_metadata = m
                break

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

    def init_process_group(self, rank: int, world_size: int, master_metadata: HCCLMasterMetadata):
        if rank < 0:
            self.rank = rank
            self.world_size = world_size
            self._group_initialized = True
            return

        if self._group_initialized and not self.rebuild_group:
            return

        if self.rebuild_group or self.pyhccl is None:
            self.pyhccl = _stateless_init_hccl(
                master_address=master_metadata.dist_ip,
                master_port=master_metadata.dist_port,
                rank=rank,
                world_size=world_size,
                device=self.device,
            )
            self.rank = rank
            self.world_size = world_size
        else:
            assert self.rank == rank
            assert self.world_size == world_size

        if self.rank > 0 and self.socket is None:
            self._connect_zmq_client(master_metadata)

        signal = torch.tensor([1], dtype=torch.int8, device=get_current_device())
        self.pyhccl.all_reduce(signal)

        self._group_initialized = True
        logger.info('HCCL init_process_group: rank=%d, world_size=%d', self.rank, self.world_size)

    # ── Metadata exchange ────────────────────────────────────────────────

    def _serve_bucket_requests(self, bucket_id: int, metadata: dict) -> None:
        """Master serves bucket metadata to all receivers via REQ/REP."""
        assert self.rank == 0 and self.world_size is not None
        if self.world_size <= 1:
            return

        pending = set(range(1, self.world_size))
        while pending:
            req = self._recv_pyobj(f'NEXT request for bucket={bucket_id}')
            if not isinstance(req, dict) or req.get('type') != 'NEXT':
                self._send_pyobj({'ok': False, 'error': f'unexpected: {req}'}, 'NEXT reply')
                continue

            req_rank = int(req.get('rank', -1))
            req_bucket_id = int(req.get('bucket_id', -1))

            if req_rank not in pending or req_bucket_id != bucket_id:
                self._send_pyobj({'ok': False, 'error': 'rank/bucket mismatch'}, 'NEXT reply')
                continue

            self._send_pyobj({'ok': True, 'metadata': metadata}, 'NEXT reply')
            pending.remove(req_rank)

    def _request_bucket(self, bucket_id: int) -> dict:
        """Receiver requests bucket metadata from master via REQ/REP."""
        assert self.rank is not None and self.rank > 0
        self._send_pyobj({'type': 'NEXT', 'rank': self.rank, 'bucket_id': bucket_id}, f'NEXT send bucket={bucket_id}')
        resp = self._recv_pyobj(f'NEXT recv bucket={bucket_id}')
        if not isinstance(resp, dict) or not resp.get('ok', False):
            raise RuntimeError(f'Metadata request failed for bucket {bucket_id}: {resp}')
        return resp['metadata']

    # ── Send / Receive ───────────────────────────────────────────────────

    @torch.no_grad()
    async def send_weights(
        self,
        weights: Generator[Tuple[str, torch.Tensor], None, None],
    ):
        assert self.rank is not None and self.rank <= 0
        if self.rank < 0:
            for _ in weights:
                pass
            return

        send_buf = self.send_buf
        start_time = time.time()
        bucket_meta: List[dict] = []
        offset = 0
        bucket_id = 0
        total_params = 0
        total_bytes = 0

        def _flush(is_last: bool):
            nonlocal bucket_meta, offset, bucket_id, total_bytes
            if not bucket_meta and not is_last:
                return
            metadata = {
                'bucket_id': bucket_id,
                'is_last': is_last,
                'bucket_meta': bucket_meta,
                'payload_size': offset,
            }
            self._serve_bucket_requests(bucket_id, metadata)
            self.pyhccl.broadcast(send_buf, src=0)
            synchronize()
            total_bytes += offset
            bucket_id += 1
            bucket_meta = []
            offset = 0

        for name, weight in weights:
            total_params += 1
            if weight.device.type == 'cpu':
                weight = weight.to(get_current_device())
            if not weight.is_contiguous():
                weight = weight.contiguous()

            weight_u8 = weight.view(-1).view(torch.uint8)
            nbytes = weight_u8.numel()
            if nbytes == 0:
                if offset >= self.bucket_size:
                    _flush(is_last=False)
                bucket_meta.append({
                    'name': name,
                    'shape': weight.shape,
                    'dtype': weight.dtype,
                    'offset': offset,
                    'nbytes': 0,
                    'chunk_offset': 0,
                    'total_nbytes': 0,
                })
                continue

            chunk_offset = 0
            while chunk_offset < nbytes:
                if offset >= self.bucket_size:
                    _flush(is_last=False)
                chunk_nbytes = min(self.bucket_size - offset, nbytes - chunk_offset)
                send_buf[offset:offset + chunk_nbytes].copy_(weight_u8[chunk_offset:chunk_offset + chunk_nbytes])
                bucket_meta.append({
                    'name': name,
                    'shape': weight.shape,
                    'dtype': weight.dtype,
                    'offset': offset,
                    'nbytes': chunk_nbytes,
                    'chunk_offset': chunk_offset,
                    'total_nbytes': nbytes,
                })
                offset += chunk_nbytes
                chunk_offset += chunk_nbytes

        _flush(is_last=True)

        elapsed = time.time() - start_time
        bandwidth = total_bytes / elapsed / (1024**3) if elapsed > 0 else 0.0
        logger.debug('HCCL send_weights done: rank=%d, params=%d, time=%.2fs, bw=%.2f GB/s', self.rank, total_params,
                     elapsed, bandwidth)

    @torch.no_grad()
    async def receive_weights(self) -> AsyncGenerator[Tuple[str, torch.Tensor], None]:
        assert self.rank is not None and self.rank > 0

        recv_buf = self.recv_buf
        bucket_id = 0
        total_params = 0
        total_bytes = 0
        start_time = time.time()
        partial_tensors: dict = {}

        while True:
            metadata = self._request_bucket(bucket_id)
            self.pyhccl.broadcast(recv_buf, src=0)
            synchronize()

            bucket_meta = metadata['bucket_meta']
            entries = bucket_meta.values() if isinstance(bucket_meta, dict) else bucket_meta
            total_bytes += int(metadata.get('payload_size', self.bucket_size))

            for meta in entries:
                name = meta['name']
                dtype = meta['dtype']
                shape = meta['shape']
                if not isinstance(shape, torch.Size):
                    shape = torch.Size(shape)
                offset = int(meta['offset'])
                nbytes = int(meta.get('nbytes', dtype.itemsize * shape.numel()))
                chunk_offset = int(meta.get('chunk_offset', 0))
                total_nbytes = int(meta.get('total_nbytes', dtype.itemsize * shape.numel()))

                if nbytes == total_nbytes and chunk_offset == 0:
                    tensor = recv_buf[offset:offset + nbytes].view(dtype=dtype).view(shape)
                    yield name, tensor
                    total_params += 1
                    continue

                state = partial_tensors.get(name)
                if state is None:
                    state = {
                        'buffer': torch.empty(total_nbytes, dtype=torch.uint8, device=recv_buf.device),
                        'dtype': dtype,
                        'shape': shape,
                        'total': total_nbytes,
                        'received': 0,
                    }
                    partial_tensors[name] = state

                if nbytes > 0:
                    state['buffer'][chunk_offset:chunk_offset + nbytes].copy_(recv_buf[offset:offset + nbytes])
                state['received'] += nbytes

                if state['received'] == state['total']:
                    full_size = dtype.itemsize * shape.numel()
                    tensor = state['buffer'][:full_size].view(dtype=dtype).view(shape)
                    yield name, tensor
                    total_params += 1
                    del partial_tensors[name]

            if metadata['is_last']:
                if partial_tensors:
                    pending = ', '.join(sorted(partial_tensors.keys())[:8])
                    raise RuntimeError(f'Incomplete chunked weights: {len(partial_tensors)} pending: {pending}')
                break
            bucket_id += 1

        elapsed = time.time() - start_time
        bandwidth = total_bytes / elapsed / (1024**3) if elapsed > 0 else 0.0
        logger.debug('HCCL receive_weights done: rank=%d, params=%d, time=%.2fs, bw=%.2f GB/s', self.rank, total_params,
                     elapsed, bandwidth)

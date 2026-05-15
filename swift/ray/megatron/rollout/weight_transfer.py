# Copyright (c) ModelScope Contributors. All rights reserved.
"""Weight transfer utilities for training → rollout weight synchronization.

BucketedWeightSender is used by the **training** side (MegatronWorker
via RolloutAdapter) to ship weights to vLLM's WeightSyncWorkerExtension
through ZMQ IPC.  It was originally in vllm_server.py but belongs here
because it is a training-side concern, not a rollout engine concern.
"""
from __future__ import annotations

import asyncio
import os
import torch
import uuid
from typing import Any, Dict

from swift.utils.logger import get_logger

logger = get_logger()


class BucketedWeightSender:
    """Streams model weights into the vLLM worker extension via ZMQ IPC.

    Shape of a single ``update_weights_ipc`` round-trip::

        async with BucketedWeightSender(zmq_handle, device_uuid) as sender:
            # The caller typically launches the collective_rpc on the
            # vLLM side FIRST so the worker subprocess has time to
            # connect back before the handshake.
            worker_rpc_task = ...
            await sender.handshake()          # ship IPC handle
            await sender.send_weights(iter)   # stream buckets
            await worker_rpc_task

    The sender is intentionally device-aware: it allocates its CUDA IPC
    bucket on the GPU identified by ``device_uuid``.  Process A's
    "current device" must already be that physical GPU — Ray + the
    ``RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES`` env var in
    ``RolloutReplica`` arrange for MegatronWorker actor and VllmServer
    actor to see the same physical GPU with different ordinals, so we
    pass the UUID explicitly to guarantee the two ends address the
    same physical device.
    """

    def __init__(
        self,
        zmq_handle: str,
        bucket_size_mb: int = 512,
        use_shm: bool = False,
        timeout_s: int = 600,
    ):
        self.zmq_handle = zmq_handle
        self.bucket_size = int(bucket_size_mb) << 20
        self.use_shm = use_shm
        self.timeout_ms = int(timeout_s * 1000)
        self.socket = None
        self.buffer = None
        self.shm = None
        self._pending_handshake = None

    async def __aenter__(self):
        self._init_socket_and_buffer()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.cleanup()

    def _init_socket_and_buffer(self):
        """Bind the REQ socket and allocate the bucket buffer.

        The first REQ/REP exchange is deliberately deferred to
        :meth:`handshake`.  The caller drives this from the asyncio
        loop that also pumps ``AsyncLLM.collective_rpc``; a blocking
        socket op here would pin the loop before the vLLM worker has
        a chance to schedule its receiver, causing a handshake
        deadlock.  :meth:`handshake` runs the blocking call on an
        executor thread instead.
        """
        import zmq

        ctx = zmq.Context.instance()
        self.socket = ctx.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(self.zmq_handle)

        from torch.multiprocessing.reductions import reduce_tensor

        if not self.use_shm:
            self.buffer = torch.empty(self.bucket_size, dtype=torch.uint8, device='cuda')
            self._pending_handshake = reduce_tensor(self.buffer)
        else:
            from multiprocessing import shared_memory
            shm_name = f'swift_weights_{uuid.uuid4().hex}'
            self.shm = shared_memory.SharedMemory(name=shm_name, create=True, size=self.bucket_size)
            self.buffer = torch.frombuffer(self.shm.buf, dtype=torch.uint8)
            self._pending_handshake = {'name': shm_name, 'size': self.bucket_size}

    async def handshake(self):
        if self._pending_handshake is None:
            raise RuntimeError('BucketedWeightSender.handshake() called before enter or twice: '
                               'the handshake payload is consumed on first call, a second handshake '
                               'within the same ``async with`` block would ship stale metadata.')
        loop = asyncio.get_running_loop()
        payload = self._pending_handshake
        self._pending_handshake = None

        def _send_recv():
            self.socket.send_pyobj(payload)
            return self.socket.recv()

        await loop.run_in_executor(None, _send_recv)

    async def _stream_weights_inner(self, items_iter, is_async: bool = False):
        """Shared bucketing logic for sync and async weight iterators."""
        loop = asyncio.get_running_loop()
        offset = 0
        bucket_meta: Dict[str, Dict[str, Any]] = {}
        n_weights = 0

        def _zmq_send_recv(payload):
            self.socket.send_pyobj(payload)
            return self.socket.recv()

        async def _flush(is_last: bool):
            nonlocal offset, bucket_meta
            if not bucket_meta and not is_last:
                return
            if self.buffer.device.type != 'cpu':
                from swift.utils import synchronize
                synchronize()
            await loop.run_in_executor(None, _zmq_send_recv, {'bucket_meta': bucket_meta, 'is_last': is_last})
            offset = 0
            bucket_meta = {}

        async def _process(name, weight):
            nonlocal offset, n_weights
            if self.use_shm and weight.device.type != 'cpu':
                weight = weight.cpu()
            if not weight.is_contiguous():
                weight = weight.contiguous()
            nbytes = int(weight.nbytes)
            if nbytes > self.bucket_size:
                raise RuntimeError(f'Weight {name} ({tuple(weight.shape)}, {weight.dtype}) '
                                   f'is {nbytes} bytes, exceeding bucket size ({self.bucket_size}).')
            if offset + nbytes > self.bucket_size:
                await _flush(False)
            bucket_meta[name] = {
                'name': name,
                'shape': weight.shape,
                'dtype': weight.dtype,
                'offset': offset,
            }
            self.buffer[offset:offset + nbytes].copy_(weight.view(-1).view(torch.uint8), non_blocking=True)
            offset += nbytes
            n_weights += 1

        if is_async:
            async for name, weight in items_iter:
                await _process(name, weight)
        else:
            for name, weight in items_iter:
                await _process(name, weight)

        await _flush(True)
        logger.debug('BucketedWeightSender: sent %d weights', n_weights)

    async def send_weights(self, weights):
        """Stream weights into buckets.  Accepts ``dict`` or iterator."""
        items = weights.items() if isinstance(weights, dict) else weights
        await self._stream_weights_inner(items, is_async=False)

    async def send_weights_async(self, async_weights):
        """Stream weights from an async generator into buckets."""
        await self._stream_weights_inner(async_weights, is_async=True)

    def cleanup(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        if self.zmq_handle.startswith('ipc://'):
            ipc_path = self.zmq_handle[len('ipc://'):]
            try:
                if os.path.exists(ipc_path):
                    os.remove(ipc_path)
            except OSError:
                pass
        del self.buffer
        self.buffer = None
        if self.shm is not None:
            try:
                self.shm.close()
                self.shm.unlink()
            except (FileNotFoundError, BufferError):
                pass
            self.shm = None
        self._pending_handshake = None
        from swift.utils import gc_collect, ipc_collect
        gc_collect()
        ipc_collect()

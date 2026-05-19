# Copyright (c) ModelScope Contributors. All rights reserved.
"""VllmServer — Ray Actor hosting a vLLM engine via :class:`RayVllmEngine`.

Thin Ray-actor shell that delegates to :class:`RayVllmEngine` for all
engine operations.  ``RolloutReplica`` wraps this class with
``ray.remote`` when spawning real actors.

Multi-node topology::

    Node 0 (node_rank=0)
    ├── MegatronWorker actors        — training processes
    └── VllmServer actor (primary)   — runs full server + API

    Node 1 (node_rank=1)
    ├── MegatronWorker actors
    └── VllmServer actor (headless)  — participates in TP, no API

"""
import asyncio
import os
import socket
import torch
from transformers.utils import is_torch_npu_available
from typing import Any, Dict, List, Optional, Tuple

from swift.utils import gc_collect, get_logger
from ..checkpoint_engine import CheckpointEngineMixin

logger = get_logger()


def _parse_bool_env(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() in ('1', 'true')


def _get_free_port(address: str = '') -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((address, 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class VllmServer(CheckpointEngineMixin):

    def __init__(
        self,
        node_rank: int = 0,
        nnodes: int = 1,
        gpus_per_node: int = 8,
        cuda_visible_devices: str = '',
    ) -> None:
        self._engine = None
        self._node_rank = node_rank
        self._nnodes = nnodes
        self._gpus_per_node = gpus_per_node

        if cuda_visible_devices:
            key = 'ASCEND_RT_VISIBLE_DEVICES' if is_torch_npu_available() else 'CUDA_VISIBLE_DEVICES'
            os.environ[key] = cuda_visible_devices

        self._server_address = None
        self._master_address: Optional[str] = None
        self._master_port: Optional[int] = None
        self._dp_rpc_port: Optional[int] = None

        if node_rank == 0:
            import ray as _ray
            self._server_address = _ray.util.get_node_ip_address()
            self._master_address = self._server_address
            self._master_port = _get_free_port(self._server_address)
            self._dp_rpc_port = _get_free_port(self._server_address)

    def get_master_address(self) -> Tuple[str, int, int]:
        """Return ``(master_address, master_port, dp_rpc_port)`` from node_rank=0."""
        return self._master_address, self._master_port, self._dp_rpc_port

    def launch_server(
        self,
        *,
        model_id: str,
        rollout_mode: str = 'hybrid',
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        max_num_seqs: int = 256,
        enable_sleep_mode: bool = False,
        enable_lora: bool = False,
        max_lora_rank: int = 8,
        enable_prefix_caching: bool = False,
        enforce_eager: bool = False,
        trust_remote_code: bool = True,
        dtype: str = 'auto',
        load_format: str = 'auto',
        master_address: Optional[str] = None,
        master_port: Optional[int] = None,
        dp_rpc_port: Optional[int] = None,
        data_parallel_size: int = 1,
        template_kwargs: Optional[Dict[str, Any]] = None,
        **engine_kwargs,
    ) -> Dict[str, Any]:
        if self._node_rank != 0:
            self._master_address = master_address
            self._master_port = master_port
            self._dp_rpc_port = dp_rpc_port
            import ray as _ray
            self._server_address = _ray.util.get_node_ip_address()

        extra_engine_kwargs = dict(engine_kwargs)

        if self._nnodes > 1:
            extra_engine_kwargs['nnodes'] = self._nnodes
            extra_engine_kwargs['node_rank'] = self._node_rank
            extra_engine_kwargs['master_addr'] = self._master_address
            extra_engine_kwargs['master_port'] = self._master_port

        if data_parallel_size > 1:
            assert self._gpus_per_node % tensor_parallel_size == 0, (
                'gpus_per_node should be divisible by vllm_tensor_parallel_size')
            dp_size_local = self._gpus_per_node // tensor_parallel_size
            extra_engine_kwargs['data_parallel_size'] = data_parallel_size
            extra_engine_kwargs['data_parallel_size_local'] = dp_size_local
            extra_engine_kwargs['data_parallel_start_rank'] = self._node_rank * dp_size_local
            extra_engine_kwargs['data_parallel_address'] = self._master_address
            extra_engine_kwargs['data_parallel_rpc_port'] = self._dp_rpc_port

        from .ray_vllm_engine import RayVllmEngine

        self._engine = RayVllmEngine(
            model_id,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            enable_sleep_mode=enable_sleep_mode,
            enable_lora=enable_lora,
            max_lora_rank=max_lora_rank,
            enable_prefix_caching=enable_prefix_caching,
            enforce_eager=enforce_eager,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            load_format=load_format,
            template_kwargs=template_kwargs,
            **extra_engine_kwargs,
        )
        logger.info(
            'VllmServer[mode=%s, node_rank=%d/%d]: engine initialized (model=%s, tp=%d)',
            rollout_mode,
            self._node_rank,
            self._nnodes,
            model_id,
            tensor_parallel_size,
        )
        return {
            'model_id': model_id,
            'tp_size': tensor_parallel_size,
            'node_rank': self._node_rank,
        }

    def generate(
        self,
        infer_requests: List[Any],
        request_config: Any = None,
    ) -> List[Any]:
        return self._engine.generate_batch(infer_requests, request_config)

    def sleep(self, level: int = 2) -> None:
        self._engine.sleep(level=level)

    def wake_up(self, tags: Optional[List[str]] = None) -> None:
        self._engine.wake_up(tags=tags)

    def get_model_param_names(self) -> List[str]:
        return self._engine.get_model_param_names()

    def reset_prefix_cache(self) -> None:
        self._engine.reset_prefix_cache()

    def update_weights_ipc(
        self,
        zmq_handle: str,
        use_shm: bool = False,
        timeout_s: int = 600,
        peft_config: Optional[dict] = None,
        base_sync_done: bool = False,
    ) -> None:
        self._engine.update_weights_ipc(
            zmq_handle, use_shm=use_shm, timeout_s=timeout_s, peft_config=peft_config, base_sync_done=base_sync_done)

    def receive_checkpoint_weights(
        self,
        base_sync_done: bool = False,
        peft_config: Optional[dict] = None,
    ) -> None:
        """Receive weights via NCCL broadcast and stream into vLLM."""
        engine = self._get_or_create_checkpoint_engine()
        # CUDA defaults to IPC (SHM disabled) to avoid frequent SharedMemory
        # warnings and long-run instability. NPU keeps SHM as default.
        use_shm = _parse_bool_env('SWIFT_RAY_NCCL_RECV_USE_SHM', default=is_torch_npu_available())

        async def _receive_and_load():
            from .weight_transfer import BucketedWeightSender
            zmq_handle = f'ipc:///tmp/swift-nccl-recv-{os.getpid()}.sock'
            bucket_mb = int(os.environ.get('SWIFT_RAY_WEIGHT_BUCKET_MB', '2048'))
            sender = BucketedWeightSender(
                zmq_handle=zmq_handle,
                bucket_size_mb=bucket_mb,
                use_shm=use_shm,
            )

            async def _weight_stream():
                async for name, tensor in engine.receive_weights():
                    if use_shm and tensor.device.type != 'cpu':
                        tensor = tensor.cpu()
                    yield name, tensor

            try:
                async with sender:
                    rpc_kwargs = {
                        'use_shm': use_shm,
                        'zmq_handle': zmq_handle,
                        'timeout_s': 600,
                    }
                    if peft_config is not None and base_sync_done:
                        rpc_kwargs['peft_config'] = peft_config
                        rpc_kwargs['base_sync_done'] = base_sync_done
                    rpc_task = asyncio.ensure_future(
                        self._engine.engine.collective_rpc(
                            'update_weights_from_ipc',
                            kwargs=rpc_kwargs,
                        ))
                    # Allow collective_rpc to schedule and bind its ZMQ socket
                    await asyncio.sleep(0)
                    await sender.handshake()
                    await sender.send_weights_async(_weight_stream())
                    await rpc_task
            finally:
                sender.cleanup()

        self._engine._run_in_loop(_receive_and_load())

    def shutdown(self) -> None:
        self._checkpoint_engine = None
        if self._engine is not None:
            self._engine.shutdown()
            self._engine = None
        gc_collect()

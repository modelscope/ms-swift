# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import asyncio
import logging
import ray
import time
import torch
from typing import Any, Generator, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _is_ipc_supported() -> bool:
    """Check if CUDA IPC is supported (GPU=True, NPU=fallback to SHM)."""
    from transformers.utils import is_torch_npu_available
    return not is_torch_npu_available()


class RolloutAdapter:

    def __init__(
        self,
        *,
        replica_rank: int = 0,
        rollout_rank: int = 0,
        bucket_size_mb: int = 2048,
    ):
        self.replica_rank = replica_rank
        self.rollout_rank = rollout_rank
        self.bucket_size_mb = bucket_size_mb
        self.is_primary = (rollout_rank == 0)
        self.use_shm = not _is_ipc_supported()
        self.zmq_handle = (f'ipc:///tmp/swift-rollout-zmq-replica-{replica_rank}-rank-{rollout_rank}.sock')
        self._server_handle = None
        # Persistent CUDA IPC buffer reused across all update_weights() syncs so the IPC
        # handle stays stable (the vLLM worker's mapping cache hits) and no IPC mapping
        # leaks per step.
        self._ipc_buffer: Optional[torch.Tensor] = None

    @property
    def server_handle(self) -> Any:
        if self._server_handle is None:
            self._server_handle = ray.get_actor(f'swift_rollout_server_{self.replica_rank}_0')
        return self._server_handle

    def update_weights(
        self,
        weight_iter: Generator[Tuple[str, torch.Tensor], None, None],
        *,
        vllm_lora_param_names: Optional[set] = None,
        peft_config: Optional[dict] = None,
        base_sync_done: bool = False,
    ) -> None:
        """Push training weights to vLLM via ZMQ IPC.

        Only the primary rank (rollout_rank == 0) sends weights;
        other ranks are no-op.

        Args:
            weight_iter: Iterator of (name, tensor) from bridge.export_weights.
            vllm_lora_param_names: When set, remap dense param names to
                LoRA-wrapped names (*.base_layer.weight) for full-weight sync
                with vllm_enable_lora.
            peft_config: When provided with base_sync_done=True, vLLM loads
                weights as a LoRA adapter via TensorLoRARequest.
            base_sync_done: Indicates this is an adapter-only sync after
                the initial full weight sync has completed.
        """
        if vllm_lora_param_names:
            from swift.rlhf_trainers.utils import add_base_layer_suffix_by_param_names
            weight_iter = add_base_layer_suffix_by_param_names(weight_iter, vllm_lora_param_names)

        if not self.is_primary:
            for _ in weight_iter:
                pass
            return

        from ..rollout.weight_transfer import BucketedWeightSender

        start_time = time.time()

        # Lazily (re)allocate the persistent IPC buffer; reused across syncs so the
        # handle signature stays stable and the worker-side IPC cache hits.
        external_buffer = None
        if not self.use_shm:
            from swift.utils import get_current_device
            bucket_size = self.bucket_size_mb << 20
            if self._ipc_buffer is None or self._ipc_buffer.numel() < bucket_size:
                self._ipc_buffer = torch.empty(bucket_size, dtype=torch.uint8, device=get_current_device())
            external_buffer = self._ipc_buffer

        async def _do_ipc_sync():
            sender = BucketedWeightSender(
                zmq_handle=self.zmq_handle,
                bucket_size_mb=self.bucket_size_mb,
                use_shm=self.use_shm,
                external_buffer=external_buffer,
            )
            try:
                async with sender:
                    rpc_ref = self.server_handle.update_weights_ipc.remote(self.zmq_handle, self.use_shm, 600,
                                                                           peft_config, base_sync_done)
                    await sender.handshake()
                    await sender.send_weights(weight_iter)
                    await asyncio.get_running_loop().run_in_executor(None, ray.get, rpc_ref)
            finally:
                sender.cleanup()

        asyncio.run(_do_ipc_sync())
        logger.debug('RolloutAdapter: update_weights done (replica=%d, adapter_only=%s, %.2fs)', self.replica_rank,
                     base_sync_done,
                     time.time() - start_time)

    def reset_prefix_cache(self) -> None:
        if not self.is_primary:
            return
        ray.get(self.server_handle.reset_prefix_cache.remote())

    def get_model_param_names(self) -> List[str]:
        if not self.is_primary:
            return []
        return ray.get(self.server_handle.get_model_param_names.remote()) or []

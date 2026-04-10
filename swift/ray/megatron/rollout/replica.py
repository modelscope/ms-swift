# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import torch
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from swift.utils import get_logger

logger = get_logger()


class RolloutMode(Enum):
    HYBRID = 'hybrid'
    SEPARATED = 'separated'
    # TODO: colocated for separate process


class RolloutReplica:

    def __init__(self, mode: RolloutMode = RolloutMode.SEPARATED):
        self.mode = mode
        self.server = None
        self._engine_kwargs = {}
        self._initialized = False

    def init_engine(self, **kwargs) -> Dict[str, Any]:
        """Initialize the underlying VllmServer.

        Accepts the same kwargs as ``VllmServer.init_engine``.

        For HYBRID mode, ``load_format`` defaults to 'dummy' and
        ``enable_sleep_mode`` defaults to True.
        """
        from .vllm_server import VllmServer

        if self.mode == RolloutMode.HYBRID:
            kwargs.setdefault('load_format', 'dummy')
            kwargs.setdefault('enable_sleep_mode', True)

        self.server = VllmServer()
        result = self.server.init_engine(**kwargs)
        self._engine_kwargs = kwargs
        self._initialized = True

        logger.info('RolloutReplica[%s]: engine initialized', self.mode.value)
        return result

    def generate(
        self,
        batch: List[Dict],
        sampling_params: Optional[Dict] = None,
    ) -> List[Dict]:
        """Generate completions. Delegates to VllmServer.generate."""
        self._check_initialized()
        return self.server.generate(batch, sampling_params)

    def update_weights_from_megatron(
        self,
        bridge,
        models,
        target_device: Optional[str] = None,
        use_ipc: bool = True,
    ) -> Dict[str, str]:
        """Sync weights from Megatron bridge to vLLM engine.

        Args:
            bridge: mcore_bridge instance with export_weights().
            models: Megatron model list.
            target_device: Device for exported weights ('cpu' or None for GPU).
            use_ipc: If True, use ZMQ+IPC path (efficient for large models).
                     If False, use direct load_weights via collective_rpc.
        """
        self._check_initialized()

        weight_iter = bridge.export_weights(models, target_device=target_device)

        if use_ipc:
            return self.server.update_weights_ipc(weight_iter)
        else:
            weight_list = list(weight_iter)
            return self.server.update_weights_direct(weight_list)

    def update_weights_from_iter(
        self,
        weight_iter,
        use_ipc: bool = True,
    ) -> Dict[str, str]:
        """Sync weights from an arbitrary iterator of (name, tensor) pairs."""
        self._check_initialized()

        if use_ipc:
            return self.server.update_weights_ipc(weight_iter)
        else:
            weight_list = list(weight_iter) if not isinstance(weight_iter, list) else weight_iter
            return self.server.update_weights_direct(weight_list)

    def sleep(self, level: int = 2) -> None:
        self._check_initialized()
        self.server.sleep(level)

    def wake_up(self, tags: Optional[List[str]] = None) -> None:
        self._check_initialized()
        self.server.wake_up(tags)

    def reset_prefix_cache(self) -> None:
        self._check_initialized()
        self.server.reset_prefix_cache()

    def shutdown(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server = None
            self._initialized = False

    def _check_initialized(self):
        if not self._initialized:
            raise RuntimeError('RolloutReplica not initialized. Call init_engine() first.')

    def ping(self) -> str:
        if self.server is not None:
            return f'replica[{self.mode.value}]:{self.server.ping()}'
        return f'replica[{self.mode.value}]:uninitialized'

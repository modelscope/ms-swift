# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import socket
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, Generator, List, Optional, Tuple, TypedDict

if TYPE_CHECKING:
    import torch


class TensorMeta(TypedDict):
    """Metadata for a tensor in the weight bucket."""
    name: str
    shape: 'torch.Size'
    dtype: 'torch.dtype'
    offset: int


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('', 0))
        return s.getsockname()[1]


def _is_valid_ipv6_address(addr: str) -> bool:
    try:
        socket.inet_pton(socket.AF_INET6, addr)
        return True
    except (OSError, socket.error):
        return False


@dataclass
class MasterMetadata:
    """Metadata from the master (trainer rank 0) for topology building."""
    zmq_ip: str
    zmq_port: int
    nccl_store_host: str = ''
    nccl_store_port: int = 0


class CheckpointEngine(ABC):
    rank: Optional[int] = None

    @abstractmethod
    def prepare(self) -> Dict[str, Any]:
        """Prepare the checkpoint engine before weight synchronization.

        Allocate weight transfer buffers, setup communication channels,
        and return metadata needed for topology building.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def build_topology(
        cls,
        trainer_world_size: int,
        rollout_world_size: int,
        metadata: List[Dict],
    ) -> Tuple[Dict[str, List[Any]], Dict[str, List[Any]]]:
        """Build communication topology between trainer and rollout workers.

        Returns (trainer_kwargs, rollout_kwargs) for init_process_group().
        """
        raise NotImplementedError

    @abstractmethod
    def init_process_group(self, **kwargs):
        """Initialize the process group for weight synchronization."""
        raise NotImplementedError

    @abstractmethod
    def finalize(self):
        """Finalize: free buffers, optionally destroy the process group."""
        raise NotImplementedError

    @abstractmethod
    async def send_weights(self, weights: Generator[Tuple[str, 'torch.Tensor'], None, None]):
        """Send model weights to rollout workers."""
        raise NotImplementedError

    @abstractmethod
    async def receive_weights(self) -> AsyncGenerator[Tuple[str, 'torch.Tensor'], None]:
        """Receive model weights from trainer."""
        raise NotImplementedError

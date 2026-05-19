# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from typing import Optional

from .base import CheckpointEngine, MasterMetadata


class CheckpointEngineMixin:
    """Mixin providing checkpoint engine lifecycle methods for Ray actors.

    Add this to Ray actors (MegatronWorker, VllmServer) to give them
    checkpoint engine capabilities.  The Manager calls these methods
    via ``ray.remote()`` to coordinate NCCL weight sync.

    Applied to ``MegatronWorker`` (send side) and ``VllmServer``
    (receive side).
    """

    _checkpoint_engine: Optional[CheckpointEngine] = None
    _bucket_size: int = 3072 << 20

    def _get_or_create_checkpoint_engine(self) -> CheckpointEngine:
        """Get or create the checkpoint engine instance (lazy singleton)."""
        if self._checkpoint_engine is None:
            from transformers.utils import is_torch_npu_available
            if is_torch_npu_available():
                from .hccl import HCCLCheckpointEngine
                self._checkpoint_engine = HCCLCheckpointEngine(self._bucket_size, rebuild_group=False)
            else:
                from .nccl import NCCLCheckpointEngine
                self._checkpoint_engine = NCCLCheckpointEngine(self._bucket_size)
        return self._checkpoint_engine

    def prepare_checkpoint_engine(self, is_master: bool) -> Optional[MasterMetadata]:
        """Prepare checkpoint engine for weight sync."""
        engine = self._get_or_create_checkpoint_engine()
        engine.is_master = is_master
        return engine.prepare()

    def init_checkpoint_process_group(
        self,
        rank: int,
        world_size: int,
        master_metadata: MasterMetadata,
    ) -> None:
        engine = self._get_or_create_checkpoint_engine()
        engine.init_process_group(
            rank=rank,
            world_size=world_size,
            master_metadata=master_metadata,
        )

    def finalize_checkpoint_engine(self) -> None:
        """Finalize checkpoint engine: release buffers, optionally destroy group."""
        if self._checkpoint_engine is not None:
            self._checkpoint_engine.finalize()

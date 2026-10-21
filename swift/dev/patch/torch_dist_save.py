# Copyright (c) ModelScope Contributors. All rights reserved.
"""Force a shard-writer ``thread_count`` onto mcore's ``TorchDistSaveShardedStrategy``.

mcore's torch_dist save strategy (``megatron.core.dist_checkpointing.strategies.torch``) defaults to
``thread_count=1``, i.e. it writes checkpoint shards single-threaded. When the strategy is constructed
*internally* -- e.g. by ``get_default_save_sharded_strategy()`` deep inside a checkpoint writer -- the
caller cannot pass ``thread_count`` in, so the only seam to parallelize the write is to patch the
constructor. This replaces legacy swift's ``patch_torch_dist_shard`` (swift/megatron/utils/patcher.py)
on the dev convert path, whose ``save_mcore_checkpoint`` builds the strategy internally.

Usage (only when the strategy is out of reach for direct construction)::

    from twinkle.patch import apply_patch
    from swift.dev.patch.torch_dist_save import TorchDistSaveThreadCountPatch
    apply_patch(None, TorchDistSaveThreadCountPatch(thread_count=8))
"""
from twinkle.patch import Patch
from twinkle.utils import get_logger

logger = get_logger()

_MARKER = '_swift_origin_dist_save_init'


class TorchDistSaveThreadCountPatch(Patch):
    """Inject ``thread_count`` into every ``TorchDistSaveShardedStrategy(...)``. Idempotent, reversible."""

    def __init__(self, thread_count: int = 2):
        self.thread_count = thread_count

    def __call__(self, module=None, *args, **kwargs):
        from megatron.core.dist_checkpointing.strategies.torch import TorchDistSaveShardedStrategy
        if hasattr(TorchDistSaveShardedStrategy, _MARKER):
            return module

        origin_init = TorchDistSaveShardedStrategy.__init__
        thread_count = self.thread_count

        def __init__(self, *a, **k):
            k['thread_count'] = thread_count
            return origin_init(self, *a, **k)

        setattr(TorchDistSaveShardedStrategy, _MARKER, origin_init)
        TorchDistSaveShardedStrategy.__init__ = __init__
        logger.info(f'Patched TorchDistSaveShardedStrategy to write shards with thread_count={thread_count}.')
        return module

    def unpatch(self, module=None, *args, **kwargs):
        from megatron.core.dist_checkpointing.strategies.torch import TorchDistSaveShardedStrategy
        origin = getattr(TorchDistSaveShardedStrategy, _MARKER, None)
        if origin is not None:
            TorchDistSaveShardedStrategy.__init__ = origin
            delattr(TorchDistSaveShardedStrategy, _MARKER)
        return module

# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..artifacts import ModelArtifact
from ..base import RecipeComponent

if TYPE_CHECKING:
    from swift.dev.config import CheckpointConfig, ModelConfig


class BaseWeightTransform(RecipeComponent):
    """Template for a weight transform; subclasses override ``run`` (and optionally the hooks).

    ``run`` returns the directory it wrote, which is not necessarily ``output_dir`` -- a transform
    that delegates to a worker propagates that worker's answer rather than assuming.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        *,
        output_dir: Optional[str] = None,
        checkpoint_config: Optional[CheckpointConfig] = None,
    ):
        self.model_config = model_config
        #: Explicit destination; wins over ``checkpoint_config.output_dir``.
        self.output_dir = output_dir
        self.checkpoint_config = checkpoint_config

    def __call__(self) -> ModelArtifact:
        self.validate()
        output_dir = self.resolve_output_dir()
        if self.skip(output_dir):
            return ModelArtifact(output_dir)
        return ModelArtifact(self.run(output_dir))

    def validate(self) -> None:
        """Reject a configuration that cannot produce a result, before any model is loaded."""

    def resolve_output_dir(self) -> str:
        """``output_dir`` -> ``checkpoint_config.output_dir`` -> ``default_output_dir()``."""
        return (self.output_dir or (self.checkpoint_config.output_dir if self.checkpoint_config else None)
                or self.default_output_dir())

    def default_output_dir(self) -> str:
        """Destination used when neither the caller nor the checkpoint config named one."""
        return 'output'

    def skip(self, output_dir: str) -> bool:
        """Whether to leave ``output_dir`` as it is and report it as the result.

        False by default: rewriting weights is the expected outcome. Overridden by transforms whose
        output is reproducible from their inputs, where an accidental multi-GB rewrite costs more
        than a stale directory.
        """
        return False

    def run(self, output_dir: str) -> str:
        """Load, transform, save; return the directory holding the result."""
        raise NotImplementedError


def _stream_partition(transform, partition_keys, rank, world_size, output_dir):
    """Ray task body: run one keyspace partition of a streaming transform, returning its manifest.

    A module-level function (not a bound method or a closure) so Ray can pickle and ship it by
    reference; all state travels in the ``transform`` clone and the scalar arguments.
    """
    return transform.run_partition(partition_keys, rank, world_size, output_dir)


class StreamingWeightTransform(BaseWeightTransform):
    """A weight transform that reads and writes tensors one at a time instead of holding the model.

    ``run`` fixes the streaming pipeline once -- open the source directory lazily, iterate
    ``stream_tensors`` (which yields each output tensor after reading only what it needs from the lazy
    source), and hand every tensor to a streaming saver that flushes a shard whenever it fills up.
    Peak resident weight memory is therefore one shard plus the single tensor in flight, independent of
    the model's total size; a 100B model is streamed on one GPU exactly as a 0.5B one is.

    Subclasses implement three hooks: ``source_model_dir`` (where the lazy tensors come from),
    ``stream_tensors`` (the transform itself, as a generator of ``(key, tensor)``), and
    ``save_metadata`` (the non-weight files: config, tokenizer, extra artifacts). ``prepare`` is an
    optional one-time setup step run before the source is opened.

    ``parallel_size > 1`` splits the keyspace across that many Ray workers, each streaming its own
    shards; the driver assembles their manifests into one global index (see ``_run_parallel``). It is a
    pure data-parallel speed-up of the same per-tensor transform -- ``parallel_size == 1`` streams the
    whole keyspace in-process and produces byte-identical output.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        *,
        output_dir: Optional[str] = None,
        checkpoint_config: Optional[CheckpointConfig] = None,
        parallel_size: int = 1,
    ):
        super().__init__(model_config, output_dir=output_dir, checkpoint_config=checkpoint_config)
        #: Number of data-parallel workers the keyspace is split across; 1 streams in-process (no Ray).
        self.parallel_size = parallel_size

    def run(self, output_dir: str) -> str:
        if self.parallel_size > 1:
            return self._run_parallel(output_dir)
        return self._run_single(output_dir)

    def _run_single(self, output_dir: str) -> str:
        """Stream the whole keyspace in this process (the ``parallel_size == 1`` path)."""
        from twinkle.utils.safetensors import SafetensorLazyLoader, StreamingSafetensorSaver

        self.prepare()
        with SafetensorLazyLoader(self.source_model_dir()) as loader:
            state = loader.get_state_dict()
            # save_rank='master': the writer rank is decided by the device-mesh env, which is trivially
            # the only rank in a single-process run.
            saver = StreamingSafetensorSaver(output_dir, max_shard_size=self.max_shard_size(), save_rank='master')
            for key, tensor in self.stream_tensors(state):
                saver.add_tensor(key, tensor)
            saver.finalize()
        self.save_metadata(output_dir)
        return output_dir

    def _run_parallel(self, output_dir: str) -> str:
        """Split the keyspace across ``parallel_size`` Ray workers, each streaming its own shards.

        Keyspace streaming is embarrassingly parallel -- the partitions are disjoint and no worker reads
        a tensor another holds -- so it needs no collective and runs through ``remote_map`` CPU tasks,
        unlike the model-parallel torchrun path. The driver resolves the source keyspace, ships each worker an
        unprepared clone (config only, cheap to pickle) plus its key list, then assembles the per-rank
        manifests into one global index and writes the metadata once.
        """
        from twinkle.infra import remote_map
        from twinkle.utils.safetensors import SafetensorLazyLoader, StreamingSafetensorSaver

        # prepare() on the driver only resolves the source directory: a streaming transform's prepare is
        # memory-safe by construction (merge builds a meta base), so the per-tensor work stays in workers.
        self.prepare()
        with SafetensorLazyLoader(self.source_model_dir()) as loader:
            partitions = loader.partition_keyspace(self.parallel_size)
        world_size = len(partitions)

        manifests = [
            manifest for manifest in remote_map(
                _stream_partition,
                [(self._clone_for_worker(), keys, rank, world_size, output_dir)
                 for rank, keys in enumerate(partitions)]) if manifest is not None
        ]
        StreamingSafetensorSaver.assemble_sharded_index(output_dir, manifests)
        self.save_metadata(output_dir)
        return output_dir

    def run_partition(self, partition_keys, rank: int, world_size: int, output_dir: str):
        """Worker entry: stream only ``partition_keys`` into rank-tagged shards; return the manifest.

        Runs inside a Ray worker process. ``prepare`` rebuilds the transform from the shipped config,
        then a state dict filtered to this rank's keys is streamed exactly as the single-process path
        does: the partitions are disjoint and group each module's tensors together, so a per-module
        transform (merge folds a module's weight and bias) never needs a key held by another rank. The
        saver writes rank-disjoint shard names and ``finalize`` returns a manifest instead of an index,
        because no single rank sees the whole model.
        """
        from twinkle.utils.safetensors import SafetensorLazyLoader, StreamingSafetensorSaver

        self.prepare()
        with SafetensorLazyLoader(self.source_model_dir()) as loader:
            full_state = loader.get_state_dict()
            state = {key: full_state[key] for key in partition_keys}
            saver = StreamingSafetensorSaver(
                output_dir, max_shard_size=self.max_shard_size(), rank=rank, world_size=world_size)
            for key, tensor in self.stream_tensors(state):
                saver.add_tensor(key, tensor)
            return saver.finalize()

    def _clone_for_worker(self):
        """A fresh, unprepared copy carrying only serializable config, for a Ray worker to ``prepare``.

        The prepared state (meta base, peft model, materialized adapters) is neither picklable nor worth
        shipping; the worker rebuilds it from config. Subclasses with extra constructor arguments
        override this to pass them through.
        """
        return type(self)(self.model_config, checkpoint_config=self.checkpoint_config)

    def prepare(self) -> None:
        """One-time setup before the source is opened (resolve dirs, build helper objects)."""

    def source_model_dir(self) -> str:
        """Local directory holding the source checkpoint's safetensors to read lazily."""
        raise NotImplementedError

    def max_shard_size(self) -> str:
        """Shard size handed to the streaming saver; mirrors ``save_checkpoint``'s default."""
        return '5GB'

    def stream_tensors(self, state: dict):
        """Yield ``(key, tensor)`` for every tensor of the output, reading inputs lazily from ``state``.

        ``state`` maps each source key to an unmaterialized ``LazyTensor``; call ``.load()`` on only what
        a given output tensor needs so the source is never fully resident. Tensors may be yielded in any
        order and grouping -- the saver assembles shards from the stream.
        """
        raise NotImplementedError

    def save_metadata(self, output_dir: str) -> None:
        """Write the non-weight files (config, tokenizer, extra artifacts) next to the streamed shards."""
        raise NotImplementedError

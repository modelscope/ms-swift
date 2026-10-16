# Copyright (c) ModelScope Contributors. All rights reserved.
"""The write side of the storage layer: the writer family a run streams rows into a store through.

The registry (:mod:`.base`) answers what a store *is* -- read / write / detect / suffix; this module
answers how a run *feeds* one. Which writers a format may use is a property of the format, not of the
caller: only an ``incremental`` (line-appendable) format can be written batch by batch while the run
progresses, and only it can checkpoint or resume; every other writable format is serialised once at
finish. The family:

- :class:`_IncrementalWriter` -- append each finished batch as it completes, gathering across DP ranks
  first; with no ``output_path`` it just holds everything until :meth:`finish`;
- :class:`_CheckpointWriter` -- the resumable variant a 'dpo' or resumed run uses, over the four-file
  scheme of :class:`_CheckpointPaths`;
- :class:`_BatchWriter` -- hold everything and serialise once at :meth:`finish`, for a non-incremental
  format.

:func:`open_store_writer` is the one seam a caller needs: it folds the format capability and the
run-level policy (dpo / resume / output path) into the writer choice, and :func:`store_output_complete`
is the "output already finished" probe a new run checks first. ``write_dataset_store`` stays a lazy
import inside :meth:`_BatchWriter.finish` -- this module is loaded while ``swift.dev.dataset`` is itself
initialising, so importing that package back at module level would be circular.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from swift.dev.utils import get_logger

from .base import STORAGE_MAPPING

logger = get_logger()

__all__ = ['open_store_writer', 'store_output_complete']


def _gather_rows(rows: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """All ranks' rows on the writing rank, None elsewhere. A no-op without torch.distributed."""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return rows

    from twinkle.utils import framework_util, is_master

    gathered = framework_util.gather_object(rows, device_mesh=None)
    return gathered if is_master() else None


def _write_jsonl(path: str, rows: List[Dict[str, Any]], append: bool = False) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, 'a' if append else 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


class _CheckpointPaths:
    """The four-file resume scheme, kept in one place so the ordering cannot be got wrong.

    ``final`` only ever appears via the closing move, which is what makes its existence mean "this run
    finished" -- :func:`store_output_complete`, the probe the infer run's early return uses,
    depends on that.
    """

    def __init__(self, output_dir: str, output_file: str):
        self.final = os.path.join(output_dir, output_file)
        self.tmp = self.final + '.tmp'
        self.resume = self.final + '.resume'
        self.state = os.path.join(output_dir, 'sampling_state.json')

    def prepare(self, resume: bool) -> Tuple[int, str]:
        """Returns ``(last_finished_batch_index, open_mode)``; -1 means start from the first batch.

        Resuming copies the snapshot back over ``tmp`` first: ``tmp`` may hold a half-written batch
        from the crash, and appending to that would emit a truncated row.
        """
        if not resume:
            for path in (self.tmp, self.resume, self.state):
                if os.path.exists(path):
                    os.remove(path)
            return -1, 'w'

        if os.path.exists(self.resume):
            shutil.copyfile(self.resume, self.tmp)
        last = -1
        if os.path.exists(self.state):
            with open(self.state, 'r', encoding='utf-8') as f:
                last = json.load(f).get('batch_index', -1)
            logger.info(f'run_infer: resuming after batch index {last}')
        return last, 'a'

    def checkpoint(self, batch_index: int) -> None:
        """Snapshot, then record. In this order: a snapshot without a state file re-does one batch,
        whereas a state file without its snapshot would skip a batch whose rows were never saved."""
        shutil.copyfile(self.tmp, self.resume)
        with open(self.state, 'w', encoding='utf-8') as f:
            json.dump({'batch_index': batch_index}, f)

    def finalize(self) -> None:
        """Publish the snapshot as the output file, keeping any previous one under a timestamp."""
        if os.path.exists(self.final):
            shutil.move(self.final, f'{self.final}.{int(time.time())}')
        source = self.resume if os.path.exists(self.resume) else self.tmp
        shutil.move(source, self.final)
        for path in (self.tmp, self.state):
            if os.path.exists(path):
                os.remove(path)


class _IncrementalWriter:
    """Append result rows to jsonl as batches complete, gathering across DP ranks first.

    Why gather: with DP > 1 each rank holds only its own slice, so a naive per-rank write either
    interleaves partial files or has every rank overwrite the same path. Legacy solved this with
    ``JsonlWriter(gather_obj=True)``; twinkle's ``gather_object`` is the same idea.

    This is the plain-'all' writer. When ``batch_size`` is falsy it does nothing until :meth:`finish`,
    which writes everything at once. :meth:`skip` / :meth:`checkpoint` are no-ops, so the generative
    loop drives this and :class:`_CheckpointWriter` through one interface.
    """

    def __init__(self, output_path: Optional[str], batch_size: Optional[int]):
        self.output_path = output_path
        self.incremental = bool(output_path and batch_size)
        self._started = bool(output_path and os.path.exists(output_path))

    def skip(self, index: int) -> bool:
        return False

    def write(self, batch: List[Dict[str, Any]]) -> None:
        if not self.incremental:
            return
        rows = _gather_rows(batch)
        if rows is None:
            return  # not the writing rank
        _write_jsonl(self.output_path, rows, append=self._started)
        self._started = True
        logger.info(f'run_infer: flushed {len(rows)} rows to {self.output_path}')

    def checkpoint(self, index: int) -> None:
        pass

    def finish(self, results: List[Dict[str, Any]]) -> None:
        if self.incremental or not self.output_path:
            return
        rows = _gather_rows(results)
        if rows is None:
            return
        _write_jsonl(self.output_path, rows, append=os.path.exists(self.output_path))
        logger.info(f'run_infer: wrote {len(rows)} rows to {self.output_path}')


class _CheckpointWriter:
    """The checkpointed jsonl writer for a 'dpo' or resumed run, over :class:`_CheckpointPaths`.

    ``tmp`` is the live write, ``resume`` the snapshot taken after each completed batch, ``state`` the
    last finished batch index, and the closing move to ``final`` is what marks a run complete -- so a
    multi-hour run that crashes resumes instead of restarting. Rows arrive as dicts and are serialized
    here, so it and :class:`_IncrementalWriter` share one emit surface. Under DP this assumes a single
    driving process (the ray sampler-group layout); the incremental writer is the one that gathers
    across torchrun DP ranks.
    """

    def __init__(self, output_path: str, resume: bool):
        output_dir = os.path.dirname(output_path) or 'output'
        os.makedirs(output_dir, exist_ok=True)
        self.paths = _CheckpointPaths(output_dir, os.path.basename(output_path))
        self._resume_from, write_mode = self.paths.prepare(resume)
        self._f = open(self.paths.tmp, write_mode, encoding='utf-8')

    def skip(self, index: int) -> bool:
        return index <= self._resume_from

    def write(self, rows: List[Dict[str, Any]]) -> None:
        for row in rows:
            self._f.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
        self._f.flush()

    def checkpoint(self, index: int) -> None:
        self.paths.checkpoint(index)

    def finish(self, results: List[Dict[str, Any]]) -> None:
        self._f.close()
        self.paths.finalize()
        logger.info(f'run_infer: wrote {self.paths.final}')


class _BatchWriter:
    """Write the run's rows as one store at the end, via the shared registry, for any non-jsonl format.

    The batch counterpart of :class:`_IncrementalWriter`: same emit surface, but the rows are serialised
    once at :meth:`finish` through :func:`~swift.dev.dataset.write_dataset_store` into whichever container
    ``store_format`` names -- ``arrow`` by default, or any writable format a ``@register_storage`` plugin
    added -- the same store ``swift export --to_cached_dataset`` produces
    (:mod:`swift.dev.dataset.storage`). There is no incremental flush, so a crash loses the run -- which is
    why any non-jsonl ``store_format`` is rejected together with ``resume``
    (:func:`swift.dev.config.validate.validate_infer_config`). jsonl keeps its own resumable line writers
    (:class:`_IncrementalWriter` / :class:`_CheckpointWriter`) instead of coming through here.

    The token payload stays in the NPZ sidecar the rows reference via ``rollout_tokens``; inlining those
    arrays as native columns belongs to the offline-RL reader (a later phase), so this unifies the
    container, not the token layout.
    """

    def __init__(self, output_path: str, store_format: str, store_fields: Optional[List[str]] = None):
        # The store sits beside the jsonl name the user gave: strip a trailing ``.jsonl`` and append the
        # format's own suffix, so the sidecar-relative ``rollout_tokens`` paths embedded in the rows stay
        # valid. A directory format has no ``store_suffix`` (arrow is detected by shape, not name), so it
        # takes the format name as its marker -- ``arrow`` -> ``X.arrow`` -- preserving the existing path.
        base = output_path[:-len('.jsonl')] if output_path.endswith('.jsonl') else output_path
        self.store_path = base + (STORAGE_MAPPING[store_format]().store_suffix or f'.{store_format}')
        self.store_format = store_format
        self.store_fields = store_fields

    def skip(self, index: int) -> bool:
        return False

    def write(self, batch: List[Dict[str, Any]]) -> None:
        pass  # rows are serialised once, at finish

    def checkpoint(self, index: int) -> None:
        pass

    def finish(self, results: List[Dict[str, Any]]) -> None:
        rows = _gather_rows(results)
        if rows is None:
            return  # not the writing rank under DP
        from swift.dev.dataset import write_dataset_store
        write_dataset_store(rows, self.store_path, backend=self.store_format, fields=self.store_fields)


def open_store_writer(output_path: Optional[str], store_format: str, *, store_fields: Optional[List[str]] = None,
                      checkpoint: bool = False, resume: bool = False) -> Union[_BatchWriter, _CheckpointWriter,
                                                                               _IncrementalWriter]:
    """The writer for this run, chosen from the format capability and the run-level policy.

    A non-incremental ``store_format`` with an ``output_path`` gets the batch writer (one serialisation
    at :meth:`finish`). On an incremental format, ``checkpoint`` -- set by a 'dpo' or resumed run -- gets
    the resumable writer, and anything else gets the plain incremental one, which with no ``output_path``
    just holds everything until its :meth:`finish`. ``checkpoint`` on a non-incremental format raises:
    the caller's policy and ``validate_infer_config`` already make it unreachable, so the invariant is
    kept local and loud.
    """
    incremental = STORAGE_MAPPING[store_format].incremental
    if checkpoint and not incremental:
        raise ValueError(
            f'checkpoint=True needs an incremental (line-appendable) store_format, got {store_format!r}, '
            'which writes the whole run in one go at the end and cannot checkpoint. Use --store_format '
            'jsonl for a resumable run.')
    if output_path and not incremental:
        return _BatchWriter(output_path, store_format, store_fields)
    if checkpoint:
        return _CheckpointWriter(output_path, resume)
    return _IncrementalWriter(output_path, 1 if output_path else None)


def store_output_complete(output_path: str) -> bool:
    """Whether a previous run already published ``output_path``: the ``.final`` marker probe.

    Rebuilds the checkpoint layout paths (the same directory normalisation the writers use) and checks
    the closing marker alone: ``final`` only ever appears through the atomic finalize, so its existence
    is exactly "a checkpointed run over this output finished".
    """
    paths = _CheckpointPaths(os.path.dirname(output_path) or 'output', os.path.basename(output_path))
    return os.path.exists(paths.final)

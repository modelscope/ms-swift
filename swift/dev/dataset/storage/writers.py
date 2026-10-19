# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
import json
import os
import shutil
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from swift.dev.utils import get_logger

from .base import STORAGE_MAPPING

logger = get_logger()

__all__ = ['StoreWriter', 'open_store_writer', 'store_output_complete', 'store_writer_kind']


def store_writer_kind(output_path: Optional[str], store_format: str, *, checkpoint: bool = False) -> str:
    """Which writer mode this run gets -- the single place the flush/checkpoint policy is decided.

    Folds the format capability (``incremental``: only a line-appendable format can flush mid-run,
    checkpoint, or resume) with the run's stated intent (``checkpoint`` = 'dpo' or resume wants
    crash protection). A format that cannot honour it degrades to 'batch'; open_store_writer
    refuses the ``resume`` combination. No ``output_path`` = 'memory'.
    """
    if not output_path:
        return 'memory'
    if not STORAGE_MAPPING[store_format].incremental:
        return 'batch'
    return 'checkpoint' if checkpoint else 'incremental'


def _gather_rows(rows: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """All ranks' rows on the writing rank, None elsewhere. A no-op without torch.distributed."""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return rows

    from twinkle.utils import framework_util, is_master

    gathered = framework_util.gather_object(rows, device_mesh=None)
    return gathered if is_master() else None


class _CheckpointPaths:
    """The four-file resume scheme, kept in one place so the ordering cannot be got wrong.

    ``final`` only ever appears via the closing move, which is what makes its existence mean "this
    run finished" -- the probe store_output_complete depends on that.
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


class StoreWriter:
    """The write side a run streams its result rows through -- one class, four modes.

    ``memory`` (no output path; all no-ops), ``incremental`` (:meth:`write` appends each finished
    batch via the format's ``append``, gathering across DP ranks), ``checkpoint`` (the resumable
    four-file scheme a 'dpo' or resumed run uses; single driving process, gathers nothing), and
    ``batch`` (a non-incremental format such as arrow: one ``write`` at :meth:`finish`).
    """

    def __init__(self, output_path: Optional[str], store_format: str, *,
                 store_fields: Optional[Sequence[str]] = None, checkpoint: bool = False,
                 resume: bool = False):
        self._kind = store_writer_kind(output_path, store_format, checkpoint=checkpoint)
        if self._kind == 'batch' and resume:
            raise ValueError(
                f'resume=True needs an incremental (line-appendable) store_format, got {store_format!r}, '
                'which writes the whole run in one go at the end and cannot resume. Use --store_format '
                'jsonl for a resumable run.')
        self._format = store_format
        self._fields = store_fields
        self._output_path = output_path
        self._storage = STORAGE_MAPPING[store_format]()
        if self._kind == 'checkpoint':
            output_dir = os.path.dirname(output_path) or 'output'
            os.makedirs(output_dir, exist_ok=True)
            self._paths = _CheckpointPaths(output_dir, os.path.basename(output_path))
            self._resume_from, write_mode = self._paths.prepare(resume)
            self._f = open(self._paths.tmp, write_mode, encoding='utf-8')
        elif self._kind == 'batch':
            # The store sits beside the jsonl name the user gave: strip a trailing ``.jsonl`` and append
            # the format's own suffix (or name, for a shape-detected format like arrow).
            base = output_path[:-len('.jsonl')] if output_path.endswith('.jsonl') else output_path
            self.store_path = base + (self._storage.store_suffix or f'.{store_format}')

    @property
    def kind(self) -> str:
        return self._kind

    def skip(self, index: int) -> bool:
        """Whether batch ``index`` was already emitted by a prior (crashed) run -- resume only."""
        return self._kind == 'checkpoint' and index <= self._resume_from

    def write(self, rows: List[Dict[str, Any]]) -> None:
        if self._kind == 'incremental':
            rows = _gather_rows(rows)
            if rows is None:
                return  # not the writing rank
            # Always append: the file's existence at run start already meant "continue it", and on a
            # fresh path 'a' and 'w' create identically -- so no first-write mode juggling is needed.
            self._storage.append(rows, self._output_path)
            logger.info(f'flushed {len(rows)} rows to {self._output_path}')
        elif self._kind == 'checkpoint':
            for row in rows:
                self._f.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
            self._f.flush()

    def checkpoint(self, index: int) -> None:
        if self._kind == 'checkpoint':
            self._paths.checkpoint(index)

    def finish(self, results: List[Dict[str, Any]]) -> None:
        if self._kind == 'checkpoint':
            self._f.close()
            self._paths.finalize()
            logger.info(f'wrote {self._paths.final}')
        elif self._kind == 'batch':
            rows = _gather_rows(results)
            if rows is None:
                return  # not the writing rank under DP
            self._storage.write(rows, self.store_path, fields=self._fields)


def open_store_writer(output_path: Optional[str], store_format: str, *,
                      store_fields: Optional[Sequence[str]] = None, checkpoint: bool = False,
                      resume: bool = False) -> StoreWriter:
    """The writer for this run, chosen from the format capability and the run-level policy.

    ``checkpoint`` marks a run that wants crash protection ('dpo' or resume); a non-incremental
    format answers 'batch' and the one-write-at-the-end degradation is the caller's to warn about --
    except for ``resume``, which cannot be honoured at all and is refused here.
    """
    return StoreWriter(output_path, store_format, store_fields=store_fields, checkpoint=checkpoint,
                       resume=resume)


def store_output_complete(output_path: str) -> bool:
    """Whether a previous run already published ``output_path``: the ``final`` marker probe.

    ``final`` only ever appears through the atomic finalize, so its existence is exactly "a
    checkpointed run over this output finished".
    """
    paths = _CheckpointPaths(os.path.dirname(output_path) or 'output', os.path.basename(output_path))
    return os.path.exists(paths.final)

# Copyright (c) ModelScope Contributors. All rights reserved.
"""Turn a raw dataset into standard rows: run a per-row transform over the whole dataset via HF
``map``, validate what comes out, and drop rows that cannot be salvaged.

This is the execution layer. It is deliberately split from ``format/``: a :class:`FormatConverter`
knows how to rewrite *one* row of *one* input shape into standard ``messages``; this layer owns
everything that is the same regardless of that shape -- the batched ``map`` call, per-row error
isolation, and the standard-row checks. The default :meth:`Preprocessor.preprocess` just delegates to
an auto-detected converter, so a plain dataset needs no subclass at all; a dataset with a quirk
subclasses this and overrides :meth:`preprocess` (tweak the row, then call ``super()``), and a dataset
whose rows are built from scratch overrides it and never touches a converter.

Versus legacy ``RowPreprocessor``: the column-name normalisation that legacy did up front with
``safe_rename_columns`` now lives inside the converter (its ``aliases``), and the multi-field
``_patch_arrow_writer`` is gone -- an earlier audit showed only ``images``/``rejected_images`` ever
needed a forced Arrow schema, so it will return, scoped to those columns, when multimodal datasets are
migrated. Text datasets need none of it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from datasets import Dataset as HfDataset
from datasets import IterableDataset as HfIterableDataset

from swift.utils import get_logger

logger = get_logger()


class Preprocessor:
    """Run a row transform over a dataset, dropping rows it cannot convert.

    Args:
        format_name: Pin a specific :class:`FormatConverter` instead of auto-detecting one. Only the
            default converter-based :meth:`preprocess` reads it; a subclass that builds rows itself
            ignores it.
        columns: Caller-supplied column renames applied before format detection -- legacy's
            ``--columns`` -- for a dataset whose field names no format would otherwise recognise.
        strict: Re-raise on a bad row instead of dropping it. A default that a per-call argument
            may override.
        traceback_limit: How many dropped-row tracebacks to log before going quiet, so one broken
            dataset cannot flood the logs.
    """

    def __init__(self,
                 *,
                 format_name: Optional[str] = None,
                 columns: Optional[Dict[str, str]] = None,
                 strict: bool = False,
                 traceback_limit: int = 10) -> None:
        self.format_name = format_name
        self.columns = columns or {}
        self.strict = strict
        self.traceback_limit = traceback_limit
        # Filled in on the first row (lazily, so a subclass that overrides `preprocess` and needs no
        # converter never triggers detection) and per worker process under `num_proc > 1`.
        self._converter = None
        self._feature_columns: List[str] = []
        self._traceback_counter = 0

    # -- row transform (override point) ----------------------------------------------------------

    def preprocess(self, row: Dict[str, Any]) -> Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]:
        """Convert one raw row to a standard row (or ``None`` to drop it, or a list to fan out).

        The default delegates to the format converter detected for this dataset. Subclasses either
        tweak ``row`` and call ``super().preprocess(row)``, or build the standard row themselves.
        """
        if self._converter is None:
            from ..format import get_converter
            self._converter = get_converter(
                self._feature_columns, format_name=self.format_name, aliases=self.columns)
        return self._converter.convert(row)

    # -- dataset-level orchestration -------------------------------------------------------------

    def __call__(self,
                 dataset,
                 *,
                 num_proc: int = 1,
                 load_from_cache_file: bool = True,
                 strict: Optional[bool] = None,
                 batch_size: Optional[int] = None):
        """Map :meth:`preprocess` over ``dataset``, returning only successfully converted rows."""
        if strict is None:
            strict = self.strict
        dataset = self.resolve_features(dataset)
        self._feature_columns = list(dataset.features)

        map_kwargs: Dict[str, Any] = {'batched': True}
        if isinstance(dataset, HfDataset):
            map_kwargs['batch_size'] = batch_size if batch_size is not None else 1000
            map_kwargs['num_proc'] = num_proc
            map_kwargs['load_from_cache_file'] = load_from_cache_file
        else:
            map_kwargs['batch_size'] = batch_size if batch_size is not None else 16

        mapped = dataset.map(
            self.batched_preprocess,
            fn_kwargs={'strict': strict},
            remove_columns=self._feature_columns,
            **map_kwargs)
        if isinstance(mapped, HfDataset) and len(mapped) != len(dataset):
            logger.info(f'Dataset filtered, origin length: {len(dataset)}, filtered length: {len(mapped)}')
        return mapped

    def batched_preprocess(self, batch: Dict[str, List[Any]], *, strict: bool) -> Dict[str, List[Any]]:
        """The function handed to ``dataset.map``: transform a column-batch, isolating per-row errors."""
        new_rows: List[Dict[str, Any]] = []
        for row in self.batched_to_rows(batch):
            try:
                out = self.preprocess(row)
                out = [] if out is None else ([out] if isinstance(out, dict) else out)
                for standard_row in out:
                    self.check_messages(standard_row)
                    self.cast_mm_data(standard_row)
                new_rows += out
            except Exception:  # noqa
                if strict:
                    logger.warning('Encountered a malformed row; pass `strict=False` to skip such rows.')
                    raise
                if self.traceback_limit is not None and self._traceback_counter < self.traceback_limit:
                    import traceback
                    logger.info(traceback.format_exc())
                    logger.warning('👆 There are errors in the dataset. This row will be dropped.')
                    self._traceback_counter += 1
        return self.rows_to_batched(new_rows)

    # -- helpers ---------------------------------------------------------------------------------

    @staticmethod
    def resolve_features(dataset):
        """An :class:`IterableDataset` may not know its columns yet; force them so ``map`` can run."""
        if dataset.features is None:
            assert isinstance(dataset, HfIterableDataset)
            dataset = dataset._resolve_features()
        return dataset

    @staticmethod
    def batched_to_rows(batch: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Column-major batch (what ``map(batched=True)`` passes) to a list of row dicts."""
        keys = list(batch)
        length = len(batch[keys[0]]) if keys else 0
        return [{key: batch[key][i] for key in keys} for i in range(length)]

    @staticmethod
    def rows_to_batched(rows: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        """Rows back to a column-major batch, unioning keys and padding absent ones with ``None``.

        Rows from one dataset can carry different keys (one turn has images, the next does not), so
        the column set is the union and every missing cell becomes ``None`` to keep columns aligned.
        An empty result is returned as an empty ``messages`` column, i.e. zero rows.
        """
        if not rows:
            return {'messages': []}
        keys: List[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        return {key: [row.get(key) for row in rows] for key in keys}

    @staticmethod
    def check_messages(row: Dict[str, Any]) -> None:
        """Validate a standard row's ``messages``: known roles, non-null content, no stray keys."""
        messages = row.get('messages')
        if messages is None:
            return
        assert len(messages) > 0, f'messages: {messages}'
        allowed_message_keys = {'role', 'content', 'loss', 'loss_scale'}
        allowed_roles = {'system', 'user', 'assistant', 'tool_call', 'tool_response', 'tool'}
        for message in messages:
            for key in set(message.keys()) - allowed_message_keys:
                message.pop(key)
            assert message['role'] in allowed_roles, f'message: {message}'
            assert message['content'] is not None, f'message: {message}'

    @staticmethod
    def cast_mm_data(row: Dict[str, Any]) -> None:
        """Normalise multimodal columns to the list-of-dict / list-of-str layout the encoder expects."""
        for key in ['images', 'rejected_images']:
            images = row.get(key)
            if images is None:
                continue
            if isinstance(images, str) or (isinstance(images, list) and images and isinstance(images[0], str)):
                images = [images] if isinstance(images, str) else images
                row[key] = [{'bytes': None, 'path': image} for image in images]
            elif isinstance(images, dict):
                row[key] = [images]
        for key in ['videos', 'audios']:
            mm_data = row.get(key)
            if isinstance(mm_data, str):
                row[key] = [mm_data]

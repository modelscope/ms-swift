# Copyright (c) ModelScope Contributors. All rights reserved.
"""Turn a raw dataset into standard rows: run a per-row transform over the whole dataset via HF
``map``, validate what comes out, and drop rows that cannot be salvaged.

This is the execution layer. It is deliberately split from ``format_converter/``: a
:class:`FormatConverter`
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
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union

import numpy as np
from datasets import Dataset as HfDataset
from datasets import IterableDataset as HfIterableDataset

from swift.utils import get_logger

__all__ = ['MessagesRepairPreprocessor', 'Preprocessor']

logger = get_logger()


class Preprocessor:
    """Run a row transform over a dataset, dropping rows it cannot convert.

    :attr:`format_name` and :attr:`columns` are readable as class attributes, so a dataset declares
    its quirks by subclassing rather than by registering a pre-built instance. That matters beyond
    style: a loader's ``preprocessor`` is a class attribute shared by every load, and this object
    carries per-load mutable state (the detected converter, the traceback counter), so a shared
    instance would leak one load's state into the next -- which is exactly what legacy did by putting
    ``preprocess_func=SomePreprocessor(...)`` in its declarations.

    Args:
        format_name: Pin a specific :class:`FormatConverter` instead of auto-detecting one. Only the
            default converter-based :meth:`preprocess` reads it; a subclass that builds rows itself
            ignores it.
        columns: Column renames applied before format detection -- legacy's ``--columns`` -- for a
            dataset whose field names no format would otherwise recognise. Merged over the class
            attribute of the same name, and winning on conflict.
        converter_kwargs: Passed to the converter's constructor, for a dataset whose dialogue column
            spells things unusually enough to need configuring rather than renaming (``content_key``
            when a message's text lives under ``'text'``, say). Merged over the class attribute.
        strict: Re-raise on a bad row instead of dropping it. A default that a per-call argument
            may override.
        traceback_limit: How many dropped-row tracebacks to log before going quiet, so one broken
            dataset cannot flood the logs.
        seed: Seeds :attr:`random_state`, for the datasets that have to pick among several candidate
            answers. Fixed by default so a run is reproducible.
    """

    # Declarations, overridable per subclass; see the constructor for the per-instance forms.
    format_name: Optional[str] = None
    columns: Dict[str, str] = {}
    converter_kwargs: Dict[str, Any] = {}

    # Columns whose Arrow type must be pinned rather than inferred. Inference only looks at the rows
    # in hand, so two shards of one dataset -- or two datasets about to be concatenated -- can end up
    # with types that will not align. Measured on datasets 4.7.0, all three groups below fail:
    # `messages` with `loss_scale` in one part and without in the other; `images` whose `bytes` is
    # all-None in one part and real bytes in the other; `objects` with an integer `bbox` in one part
    # and float in the other. The failure surfaces at concatenation, i.e. far from its cause.
    MESSAGE_COLUMNS = ('messages', 'rejected_messages', 'positive_messages', 'negative_messages')
    MEDIA_COLUMNS = ('images', 'rejected_images')
    FREEFORM_COLUMNS = ('objects', 'chat_template_kwargs')

    def __init__(self,
                 *,
                 format_name: Optional[str] = None,
                 columns: Optional[Dict[str, str]] = None,
                 converter_kwargs: Optional[Dict[str, Any]] = None,
                 strict: bool = False,
                 traceback_limit: int = 10,
                 seed: int = 42) -> None:
        self.format_name = format_name or type(self).format_name
        self.columns = {**type(self).columns, **(columns or {})}
        self.converter_kwargs = {**type(self).converter_kwargs, **(converter_kwargs or {})}
        self.strict = strict
        self.traceback_limit = traceback_limit
        self.random_state = np.random.RandomState(seed)
        # Filled in on the first row (lazily, so a subclass that overrides `preprocess` and needs no
        # converter never triggers detection) and per worker process under `num_proc > 1`.
        self._converter = None
        self._feature_columns: List[str] = []
        self._traceback_counter = 0

    @property
    def converter(self):
        """The :class:`FormatConverter` for this dataset, built on first use.

        Lazy on purpose: a subclass that builds standard rows itself never pays for detection, and
        under ``num_proc > 1`` each worker builds its own instead of one being shipped across the
        process boundary.
        """
        if self._converter is None:
            from ..format_converter import get_converter
            self._converter = get_converter(
                self._feature_columns,
                format_name=self.format_name,
                aliases=self.columns,
                **self.converter_kwargs)
        return self._converter

    # -- row transform (override point) ----------------------------------------------------------

    def preprocess(self, row: Dict[str, Any]) -> Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]:
        """Convert one raw row to a standard row (or ``None`` to drop it, or a list to fan out).

        The row arrives exactly as the dataset stores it: no renaming has happened yet, because
        :attr:`columns` and the format's own aliases are both applied inside the converter, below.
        Subclasses therefore see raw field names -- a simple contract, but the opposite of legacy,
        which renamed columns dataset-wide *before* calling its ``preprocess``.

        Subclasses either tweak ``row`` and call ``super().preprocess(row)``, or build the standard
        row themselves and never involve a converter. A subclass that needs to *read* a standard
        field before conversion calls :meth:`standardise` first.
        """
        return self.converter.convert(row)

    def standardise(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """A copy of ``row`` with aliased column names replaced by their standard ones.

        Legacy renamed the whole dataset up front, so its ``preprocess`` bodies were written against
        standard names (``row['query']``) and never had to know what the dataset actually called the
        column. Dev hands ``preprocess`` the raw row instead; calling this first reproduces legacy's
        view. It reuses the converter's own alias table rather than a second copy, so the rename is
        by construction the same one conversion will do.

        One trap, and the reason legacy's single up-front rename differed: conversion applies the
        aliases *again*. Re-reading an already standard row is harmless, since a standard name present
        is never overwritten -- but putting an **aliased** name back on the row afterwards is not. A
        row given a fresh ``target`` column after standardising gets it promoted to ``response`` by
        that second pass, inventing an assistant turn. Set such columns on the row conversion
        *returns* instead.
        """
        return self.converter.apply_aliases(row)

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
        dataset = self.prepare_dataset(dataset)
        dataset = self.resolve_features(dataset)
        self._feature_columns = list(dataset.features)

        map_kwargs: Dict[str, Any] = {'batched': True}
        if isinstance(dataset, HfDataset):
            map_kwargs['batch_size'] = batch_size if batch_size is not None else 1000
            map_kwargs['num_proc'] = num_proc
            map_kwargs['load_from_cache_file'] = load_from_cache_file
            if not dataset.cache_files:
                map_kwargs['cache_file_name'] = self.map_cache_path(dataset)
        else:
            map_kwargs['batch_size'] = batch_size if batch_size is not None else 16

        with self.pin_features(), self.serialised():
            mapped = dataset.map(
                self.batched_preprocess,
                fn_kwargs={'strict': strict},
                remove_columns=self._feature_columns,
                **map_kwargs)
        if isinstance(mapped, HfDataset) and len(mapped) != len(dataset):
            logger.info(f'Dataset filtered, origin length: {len(dataset)}, filtered length: {len(mapped)}')
        return mapped

    def prepare_dataset(self, dataset):
        """Hook for work that has to happen once for the whole dataset, before any row is seen.

        Fetching the media archive a multimodal dataset's rows only reference by relative path belongs
        here: it is one download for the dataset, and the local path it returns is what
        :meth:`preprocess` needs to turn those references into usable file paths. Whatever the hook
        stores on ``self`` reaches the workers of a ``num_proc > 1`` run, since the instance is
        pickled after this point.

        Returning the dataset (rather than mutating it) lets the hook narrow it too, which is worth it
        when the media for most rows is missing: dropping them here beats carrying them through
        conversion only to discard them.
        """
        return dataset

    @contextmanager
    def pin_features(self):
        """Pin the Arrow type of the unstable columns for the duration of one ``map``.

        Done by overriding the features the writer is constructed with, which keeps it to the single
        write pass -- casting the columns afterwards would mean reading and rewriting the whole table.

        Versus legacy's ``_patch_arrow_writer``: only columns the output **actually has** are pinned.
        Legacy assigned all of them unconditionally, so every dataset -- a plain text one included --
        gained an all-null ``images``, ``objects`` and ``rejected_messages`` column it had no use for.
        """
        from datasets import Value
        from datasets.arrow_writer import ArrowWriter
        try:
            from datasets.features import Json
            from datasets.features import List as ListFeature
        except ImportError:  # datasets < 4 inferred these columns from a fixed schema already
            yield
            return

        pinned = {
            **{name: ListFeature(Json()) for name in self.MESSAGE_COLUMNS},
            **{name: ListFeature({
                'bytes': Value('binary'),
                'path': Value('string')
            })
               for name in self.MEDIA_COLUMNS},
            **{name: Json() for name in self.FREEFORM_COLUMNS},
        }
        original_init = ArrowWriter.__init__

        def patched_init(writer, schema=None, features=None, *args, **kwargs):
            if features is not None:
                for name, feature in pinned.items():
                    if name in features:
                        features[name] = feature
            return original_init(writer, schema, features, *args, **kwargs)

        ArrowWriter.__init__ = patched_init
        try:
            yield
        finally:
            ArrowWriter.__init__ = original_init

    def batched_preprocess(self, batch: Dict[str, List[Any]], *, strict: bool) -> Dict[str, List[Any]]:
        """The function handed to ``dataset.map``: transform a column-batch, isolating per-row errors."""
        new_rows: List[Dict[str, Any]] = []
        for row in self.batched_to_rows(batch):
            try:
                out = self.preprocess(row)
                out = [] if out is None else ([out] if isinstance(out, dict) else out)
                for standard_row in out:
                    self.check_messages(standard_row)
                    self.check_objects(standard_row)
                    self.cast_mm_data(standard_row)
                new_rows += out
            except Exception as e:  # noqa
                if strict:
                    logger.warning('Encountered a malformed row; pass `strict=False` to skip such rows.')
                    raise
                from swift.template import MaxLengthError
                if isinstance(e, MaxLengthError):
                    # Not a malformed row -- it encoded fine, it just does not fit. Dropping it is the
                    # intended outcome, so it neither warrants a traceback nor should it spend the
                    # traceback budget that real data errors need.
                    continue
                if self.traceback_limit is not None and self._traceback_counter < self.traceback_limit:
                    import traceback
                    logger.info(traceback.format_exc())
                    logger.warning('👆 There are errors in the dataset. This row will be dropped.')
                    self._traceback_counter += 1
        return self.rows_to_batched(new_rows)

    # -- helpers ---------------------------------------------------------------------------------

    @staticmethod
    def serialised(key: str = 'dataset_preprocess'):
        """Let one rank run the pass and write its cache first, then the rest read it.

        Not ``sticky``: the same key is used once per dataset per run, so each round needs its own
        ordering rather than being satisfied by a flag an earlier round left set. See
        :meth:`DatasetLoader.serialised` for why this is not ``safe_ddp_context``.
        """
        from twinkle.utils import processing_lock
        return processing_lock(key)

    @staticmethod
    def map_cache_path(dataset) -> str:
        """Where to cache the ``map`` of a dataset that has no cache files of its own.

        A dataset read from the hub keeps its Arrow files on disk, so ``map`` caches its result beside
        them and a second run reuses it. One built in memory (a ``from_list``, a synthesised split) has
        nowhere to put that, so its result is written to a temporary file and recomputed on every
        launch -- expensive for exactly the datasets a user iterates on. Keyed by the fingerprint
        ``datasets`` already computes, which covers both the rows and the transform.
        """
        from modelscope.hub.utils.utils import get_cache_dir
        directory = os.path.join(get_cache_dir(), 'datasets', 'map_cache')
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, f'{dataset._fingerprint}.arrow')

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
    def pop_first(row: Dict[str, Any], *keys: str) -> Any:
        """Pop the first of ``keys`` the row actually has, or ``None``.

        For a field a dataset spells inconsistently (``content`` here, ``text`` there) that a
        subclass must consume itself rather than leave to the converter's aliases.
        """
        for key in keys:
            if key in row:
                return row.pop(key)
        return None

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
    def check_objects(row: Dict[str, Any]) -> None:
        """Validate and normalise the bounding boxes of a grounding row's ``objects``.

        Two ways a box can be wrong that nothing downstream would report. A box of the wrong length:
        the template walks it as ``zip(bbox[::2], bbox[1::2])``, so a 3-number box yields silently
        wrong coordinates rather than an error. And a box whose corners arrive swapped (``x1 > x2``),
        which still looks like a box and normalises into a negative-area region.

        Legacy's ``_check_objects`` did this too, alongside pinning the key order of ``objects`` --
        which dev has no need for, the column being pinned as ``Json()`` and so tolerating rows whose
        keys differ in order or in presence.
        """
        objects = row.get('objects')
        if not objects:
            return
        for bbox in objects.get('bbox') or []:
            assert len(bbox) in {2, 4}, f'bbox must hold a point or a box, got len {len(bbox)}: {bbox}'
            if len(bbox) == 2:
                continue
            if bbox[0] > bbox[2]:
                bbox[0], bbox[2] = bbox[2], bbox[0]
            if bbox[1] > bbox[3]:
                bbox[1], bbox[3] = bbox[3], bbox[1]

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


class MessagesRepairPreprocessor(Preprocessor):
    """A dialogue dataset whose ``messages`` column has to be fixed before it can be converted.

    Two recurring problems, both needing the dialogue in hand before conversion starts: the column is
    a string that is only *almost* a Python literal (a dump that lost its commas), and rows that turn
    out to be unusable once the dialogue is read (a leaked system prompt, too few distinct tools).

    Legacy passed these in as a ``repair_messages=`` callable on ``MessagesPreprocessor``, which meant
    the repair for a dataset sat in the argument list of its declaration. Here it is an override
    point, so each repair is a named method on the loader's own preprocessor.
    """

    def repair(self, messages: Any) -> Optional[Any]:
        """Return the repaired dialogue, or ``None`` / empty to drop the row.

        The value arrives exactly as the column holds it, *unparsed*: repairing a string that is not
        yet a valid literal is the main reason this hook exists, so parsing first would destroy the
        very input it needs. The base parses whatever comes back, so a repair that only patches string
        damage need not parse; one that inspects turns calls :meth:`FormatConverter.parse_literal`
        itself, exactly as legacy's repair callables did.
        """
        raise NotImplementedError

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row = self.standardise(row)
        messages = self.converter.parse_literal(self.repair(row.get('messages')))
        if not messages:
            return None
        row['messages'] = messages
        return super().preprocess(row)

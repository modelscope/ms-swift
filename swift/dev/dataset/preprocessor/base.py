# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union

import numpy as np
from datasets import Dataset as HfDataset
from datasets import IterableDataset as HfIterableDataset
from twinkle.dataloader.random_state import derive_seed

from swift.dev.utils import get_logger
from .._diagnostics import note_row_error

__all__ = ['MessagesRepairPreprocessor', 'Preprocessor']

logger = get_logger()


class Preprocessor:
    """Run a row transform over a dataset, dropping rows it cannot convert.

    ``format_name`` and ``columns`` are readable as class attributes, so a dataset declares its quirks
    by subclassing rather than registering a pre-built instance: a loader's ``preprocessor`` is shared
    by every load, while this object carries per-load mutable state (detected converter, traceback
    counter) -- a shared instance, legacy's ``preprocess_func=SomePreprocessor(...)``, leaks across loads.
    """

    # Declarations, overridable per subclass; see the constructor for the per-instance forms.
    format_name: Optional[str] = None
    columns: Dict[str, str] = {}
    converter_kwargs: Dict[str, Any] = {}

    # Columns whose Arrow type must be pinned rather than inferred: inference sees only the rows in
    # hand, so two shards of one dataset can infer types that only fail to align at concatenation.
    MESSAGE_COLUMNS = ('messages', 'rejected_messages', 'positive_messages', 'negative_messages')
    # Media as {bytes, path} structs, the shape cast_mm_data normalises images to: an all-None-bytes
    # shard infers a different type than one with real bytes.
    MEDIA_COLUMNS = ('images', 'rejected_images')
    # Videos/audios stay plain path-string lists; pinned for the same reason. No rejected_* variant:
    # the preference schema carries rejected_images only (the four keys cast_mm_data handles).
    MEDIA_PATH_COLUMNS = ('videos', 'audios')
    FREEFORM_COLUMNS = ('objects', 'chat_template_kwargs')

    def __init__(self,
                 *,
                 format_name: Optional[str] = None,
                 columns: Optional[Dict[str, str]] = None,
                 converter_kwargs: Optional[Dict[str, Any]] = None,
                 strict: bool = False,
                 traceback_limit: int = 10,
                 seed: int = 42) -> None:
        """Build one transform; ``format_name``/``columns``/``converter_kwargs`` merge over the class
        attributes and win on conflict, and respectively pin a converter, rename columns before
        detection (legacy ``--columns``) and reach the converter's constructor. ``strict`` re-raises
        on a bad row; ``traceback_limit`` bounds dropped-row tracebacks; ``seed`` seeds
        :attr:`random_state` so picks among candidate answers are reproducible.
        """
        self.format_name = format_name or type(self).format_name
        self.columns = {**type(self).columns, **(columns or {})}
        self.converter_kwargs = {**type(self).converter_kwargs, **(converter_kwargs or {})}
        self.strict = strict
        self.traceback_limit = traceback_limit
        self.seed = int(seed)
        self.random_state = np.random.RandomState(self.seed)
        # Filled in on the first row, so a subclass overriding `preprocess` never triggers detection;
        # per worker process under `num_proc > 1`.
        self._converter = None
        self._feature_columns: List[str] = []
        self._traceback_counter = 0

    @property
    def converter(self):
        """The :class:`FormatConverter` for this dataset, built on first use.

        Lazy so a subclass that builds standard rows itself never pays for detection, and under
        ``num_proc > 1`` each worker builds its own rather than shipping one across the process
        boundary.
        """
        if self._converter is None:
            from ..format_converter import get_converter
            self._converter = get_converter(
                self._feature_columns,
                format_name=self.format_name,
                aliases=self.columns,
                random_state=self.random_state,
                **self.converter_kwargs)
        return self._converter

    def preprocess(self, row: Dict[str, Any]) -> Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]:
        """Convert one raw row to a standard row (``None`` drops it, a list fans out).

        The row arrives exactly as the dataset stores it -- renaming happens inside the converter,
        the opposite of legacy's dataset-wide pre-rename. Subclasses tweak ``row`` and call ``super()``
        or build the standard row themselves; to *read* a standard field first call
        :meth:`standardise`.
        """
        return self.converter.convert(row)

    def standardise(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """A copy of ``row`` with aliased column names replaced by standard ones, via the converter's
        own alias table -- by construction the rename conversion itself will do.

        The trap: conversion applies the aliases *again*. A standard name present is never overwritten,
        but a fresh **aliased** name put back afterwards is promoted -- ``target`` becomes ``response``,
        inventing an assistant turn. Set such columns on the row conversion *returns*.
        """
        return self.converter.apply_aliases(row)

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
            # num_proc <= 1 maps in-process: a one-worker pool is pure spawn overhead, and a spawned
            # worker rebuilds an UNPATCHED ArrowWriter (see _pin_message_columns_for_pool), which
            # fails to serialise mixed-type message content (a tool_call dict beside plain text).
            map_kwargs['num_proc'] = num_proc if num_proc >= 2 else None
            if num_proc >= 2:
                dataset = self._pin_message_columns_for_pool(dataset)
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
                with_indices=True,
                **map_kwargs)
        if isinstance(mapped, HfDataset) and len(mapped) != len(dataset):
            logger.info(f'Dataset filtered, origin length: {len(dataset)}, filtered length: {len(mapped)}')
        return mapped

    def prepare_dataset(self, dataset):
        """Hook for work that must happen once per dataset, before any row is seen.

        Fetching the media archive that rows only reference by relative path belongs here, and
        whatever it stores on ``self`` reaches the workers of a ``num_proc > 1`` run, the instance
        being pickled after this point. Returning the dataset (rather than mutating it) lets the hook
        narrow it too.
        """
        return dataset

    @contextmanager
    def pin_features(self):
        """Pin the Arrow type of the unstable columns for one ``map``, by overriding the features the
        writer is constructed with -- a single write pass, versus rereading and rewriting the whole
        table to cast afterwards.

        Unlike legacy's ``_patch_arrow_writer``, only columns the output actually has are pinned:
        legacy assigned all of them, giving every dataset all-null ``images``/``objects`` columns.
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
            **{name: ListFeature(Value('string')) for name in self.MEDIA_PATH_COLUMNS},
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

    def _pin_message_columns_for_pool(self, dataset):
        """Carry the message-column pin into pooled map workers as DATA, not as a class patch.

        ``pin_features`` patches ``ArrowWriter.__init__`` in THIS process only. datasets' pool
        workers on a spawn start method re-import the module and rebuild an unpatched writer, so
        mixed-type message content (a ``tool_call`` dict beside plain text) fails to serialise --
        and with ``strict=False`` the row is silently DROPPED. The worker's writer is seeded from
        the input shard's features (``writer_features = shard.features``), and those features are
        pickled to the worker as data, so casting the message columns to ``List[Json]`` here makes
        the pin survive any start method. Forked workers already inherit the patched class, so
        only a spawned pool pays the cast pass.
        """
        try:
            from datasets.features import Json
            from datasets.features import List as ListFeature
        except ImportError:  # datasets < 4 inferred these columns from a fixed schema already
            return dataset
        import multiprocess
        if multiprocess.get_start_method() == 'fork':
            return dataset
        features = dataset.features.copy()  # Features.copy() keeps the class (a plain dict breaks cast)
        changed = False
        for name in self.MESSAGE_COLUMNS:
            feature = features.get(name)
            if feature is None or (isinstance(feature, ListFeature) and isinstance(feature.feature, Json)):
                continue
            features[name] = ListFeature(Json())
            changed = True
        return dataset.cast(features) if changed else dataset

    def batched_preprocess(self, batch: Dict[str, List[Any]], indices: List[int], *,
                           strict: bool) -> Dict[str, List[Any]]:
        """The function handed to ``dataset.map``: transform a column-batch, isolating per-row errors."""
        new_rows: List[Dict[str, Any]] = []
        rows = self.batched_to_rows(batch)
        for row, index in zip(rows, indices):
            try:
                self.random_state.seed(derive_seed(self.seed, int(index)) % 2**32)
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
                    # It encoded fine, it just does not fit: dropping it is the intended outcome, so
                    # it spends no traceback budget that real data errors need.
                    continue
                note_row_error(self, '👆 There are errors in the dataset. This row will be dropped.')
        return self.rows_to_batched(new_rows)

    @staticmethod
    def serialised(key: str = 'dataset_preprocess'):
        """Let one rank run the pass and write its cache first, then the rest read it.

        Not ``sticky``: the same key is used once per dataset per run, so each round needs its own
        ordering. See :meth:`DatasetLoader.serialised` for why this is not ``safe_ddp_context``.
        """
        from twinkle.utils import processing_lock
        return processing_lock(key)

    @staticmethod
    def map_cache_path(dataset) -> str:
        """Where to cache the ``map`` of a dataset that has no cache files of its own.

        A hub dataset keeps its Arrow files on disk, so ``map`` caches beside them; one built in
        memory has nowhere to put that, so the result is keyed by the fingerprint ``datasets``
        already computes (rows + transform) and reused on a second launch.
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

        Rows can carry different keys (one turn has images, the next does not), so the column set is
        the union and missing cells become ``None``; an empty result is zero rows via an empty
        ``messages`` column.
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

        For a field a dataset spells inconsistently that a subclass must consume itself, rather than
        leave to the converter's aliases.
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

        A box of the wrong length walks ``zip(bbox[::2], bbox[1::2])`` into silently wrong
        coordinates, and swapped corners (``x1 > x2``) normalise into a negative-area region. Legacy
        also pinned the key order of ``objects``, unneeded now the column is pinned as ``Json()``.
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
    """A dialogue dataset whose ``messages`` column must be fixed before it can be converted.

    Legacy passed a ``repair_messages=`` callable on ``MessagesPreprocessor``, putting each repair in
    an argument list; here it is an override point -- a named method on the loader's own preprocessor.
    """

    def repair(self, messages: Any) -> Optional[Any]:
        """Return the repaired dialogue, or ``None`` / empty to drop the row.

        The value arrives unparsed: repairing an almost-literal string is the point, so parsing first
        would destroy the input. The base parses whatever comes back; a repair that inspects turns
        calls :meth:`FormatConverter.parse_literal` itself, exactly as legacy's repair callables did.
        """
        raise NotImplementedError

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row = self.standardise(row)
        messages = self.converter.parse_literal(self.repair(row.get('messages')))
        if not messages:
            return None
        row['messages'] = messages
        return super().preprocess(row)

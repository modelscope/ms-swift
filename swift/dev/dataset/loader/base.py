# Copyright (c) ModelScope Contributors. All rights reserved.
"""Which dataset a name refers to, and how to turn it into rows."""
from __future__ import annotations

import importlib
import json
import os
import platform
import re
from dataclasses import dataclass, field
from datasets import Dataset as HfDataset
from datasets import IterableDataset as HfIterableDataset
from datasets import concatenate_datasets, interleave_datasets, load_dataset as hf_load_dataset
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Type, Union

import numpy as np

from swift.dev.utils import get_logger

__all__ = [
    'DATASET_MAPPING', 'DATASET_TYPE', 'DatasetInfo', 'DatasetLoader', 'SubsetMeta', 'get_dataset_loader',
    'load_dataset', 'match_dataset_type', 'register_dataset', 'register_dataset_info'
]

logger = get_logger()

DATASET_TYPE = Union[HfDataset, HfIterableDataset]

# dataset_type -> the loader class that owns it.
DATASET_MAPPING: Dict[str, Type['DatasetLoader']] = {}
# (hub, id) -> dataset_type, where hub is 'ms' | 'hf' | 'path'. Lazily built and invalidated on every
# registration. Legacy kept the same reverse index in `dataset_syntax._dataset_meta_mapping` but built
# it exactly once with no invalidation hook, so anything registered afterwards -- every entry from a
# user's `--custom_dataset_info` file, since those are registered after the built-in ones -- was
# missing from it and only ever found by the looser basename fallback.
_ID_MAPPING: Dict[Tuple[str, str], str] = {}


@dataclass
class SubsetMeta:
    """One selectable subset of a dataset, and what makes it differ from its siblings.

    A subset is not merely a name passed through to the hub: sibling subsets of the same dataset
    routinely need *different* preprocessing (legacy's `stsb` registers four subsets whose scores map
    to four different label sets), so `preprocessor` and `split` are overridable per subset. Both
    default to ``None`` meaning "inherit from the owning loader" -- see :meth:`resolve`.

    Args:
        subset: Subset name as the hub knows it. ``'default'`` for a dataset with no real subsets.
        name: Name users select on the command line. Defaults to :attr:`subset`; set it only when the
            hub's name is not the one worth typing.
        split: Splits to load for this subset. ``None`` inherits the loader's.
        preprocessor: Row preprocessor for this subset. ``None`` inherits the loader's.
        columns: Column renames for this subset. ``None`` inherits the loader's. Per-subset because
            sibling subsets of one dataset are often dumps of different sources that spell the same
            field differently (legacy's ``medical_zh`` names it ``instruction`` in zh and ``input``
            in en).
        is_weak_subset: Exclude this subset when the user asks for ``all``. For subsets that are
            noisy, tiny, or a near-duplicate of another -- they should be reachable by name but not
            silently mixed into a bulk request.
    """

    subset: str = 'default'
    name: Optional[str] = None
    split: Optional[List[str]] = None
    preprocessor: Optional[Any] = None
    columns: Optional[Dict[str, str]] = None
    is_weak_subset: bool = False

    def __post_init__(self) -> None:
        if self.name is None:
            self.name = self.subset

    def resolve(self, loader_cls: Type['DatasetLoader']) -> 'SubsetMeta':
        """A copy with every inherited (``None``) field filled in from ``loader_cls``.

        Returns a copy rather than mutating: the same :class:`SubsetMeta` object lives on the loader
        class as a shared class attribute, so filling fields in place would leak one load's resolved
        values into the next.
        """
        return SubsetMeta(
            subset=self.subset,
            name=self.name,
            split=list(self.split if self.split is not None else loader_cls.split),
            preprocessor=self.preprocessor if self.preprocessor is not None else loader_cls.preprocessor,
            columns=self.columns if self.columns is not None else loader_cls.columns,
            is_weak_subset=self.is_weak_subset,
        )


@dataclass
class DatasetInfo:
    """What one concrete load request resolved to.

    This is the dataset-side counterpart of ``ModelInfo``, and like it, it describes *what is being
    loaded* -- not *how*. Options that only affect the mechanics of loading (``num_proc``,
    ``load_from_cache_file``, ``streaming``, ``strict``) stay as keyword arguments threaded through
    the load call; putting them here would turn one object into both a description and a settings
    bag, and every consumer would have to know which half it was looking at.

    It merges what legacy split across ``DatasetSyntax`` (the parse of the command-line string) and
    the loose arguments passed alongside it. The parse itself does not live here -- a later ``syntax``
    module owns turning ``'hf::id:sub1/sub2#5000'`` into one of these.

    Args:
        dataset: The dataset as the user wrote it: a hub id, a local file, or a local directory.
        dataset_type: Family id, once the registry has matched one. ``None`` for an unregistered
            dataset, which is the common case and not an error -- see :func:`get_dataset_loader`.
        source: Where rows come from. ``'ms'`` / ``'hf'`` for a hub id, ``'path'`` for a local file,
            ``'repo'`` for a local directory holding a downloaded snapshot. Resolved once here so
            downstream code branches on an enum instead of re-running ``os.path`` probes.
        subsets: Subset names the user asked for. Empty means "the loader's default selection".
        sample_count: Row budget from the ``#N`` suffix. ``None`` for the whole dataset. May exceed
            the dataset's length, which means oversampling with repeats.
        revision: Hub revision to pin, when one was declared or requested.
    """

    dataset: str = ''
    dataset_type: Optional[str] = None
    source: str = 'ms'
    subsets: List[str] = field(default_factory=list)
    sample_count: Optional[int] = None
    revision: Optional[str] = None


def register_dataset(loader_cls: Type['DatasetLoader'] = None, *, exist_ok: bool = False):
    """Register a dataset family, keyed by its ``dataset_type``. Usable bare or with keywords.

    Everything is read off the class, so a dataset is declared in one place. Legacy instead paired a
    ``DatasetMeta`` literal with a separately-defined preprocessor and keyed the registry on the
    tuple ``(ms_dataset_id, hf_dataset_id, dataset_path)`` -- a composite key that no lookup could
    use directly, which is why a second reverse index had to be built and kept in sync.
    """

    def _register(cls: Type['DatasetLoader']) -> Type['DatasetLoader']:
        dataset_type = cls.dataset_type
        assert dataset_type, f'{cls.__name__} must set `dataset_type`.'
        if not exist_ok and dataset_type in DATASET_MAPPING:
            raise ValueError(f'dataset_type `{dataset_type}` is already registered '
                             f'by {DATASET_MAPPING[dataset_type].__name__}.')
        DATASET_MAPPING[dataset_type] = cls
        _ID_MAPPING.clear()  # invalidate the reverse-lookup cache
        return cls

    return _register if loader_cls is None else _register(loader_cls)


def register_dataset_info(dataset_info: Union[str, List[Dict[str, Any]], None] = None,
                          *,
                          exist_ok: bool = False) -> List[Type['DatasetLoader']]:
    """Register datasets declared as data rather than as code, synthesising a loader class for each.

    Most datasets differ from each other only in *what* they are -- ids, subsets, splits, which raw
    column means ``query`` -- and not in *how* they are read. Those need no Python at all, so they
    live in ``dataset_info.json`` and arrive here. This is also the entry point for a user's
    ``--custom_dataset_info`` file, which is the same shape.

    Each entry becomes a real :class:`DatasetLoader` subclass, so a JSON-declared dataset and a
    hand-written one are indistinguishable downstream -- there is no second code path that only
    understands dicts, which is where legacy's ``_preprocess_d_info`` shape-shifting lived.

    Recognised keys, all optional except that one of the two ids must be present:
    ``ms_dataset_id`` / ``hf_dataset_id`` / ``dataset_path`` / ``dataset_type`` / ``subsets`` /
    ``split`` / ``columns`` / ``tags`` / ``help`` / ``huge_dataset``. A ``subsets`` entry is either a
    plain name or an object with the same ``subset`` / ``split`` / ``columns`` keys.

    Args:
        dataset_info: A path to a JSON file, a JSON string, an already-parsed list, or ``None`` for
            the built-in ``dataset_info.json`` next to this module.
        exist_ok: Allow re-registering a ``dataset_type`` that is already taken.

    Returns:
        The synthesised loader classes, in declaration order.
    """
    base_dir = None
    if dataset_info is None:
        dataset_info = os.path.join(os.path.dirname(__file__), 'dataset_info.json')
    if isinstance(dataset_info, str):
        path = os.path.abspath(os.path.expanduser(dataset_info))
        if os.path.isfile(path):
            # Relative `dataset_path` entries are resolved against the file that declared them, so a
            # custom dataset_info file can sit next to the data it points at and stay portable.
            base_dir = os.path.dirname(path)
            with open(path, 'r', encoding='utf-8') as f:
                dataset_info = json.load(f)
        else:
            dataset_info = json.loads(dataset_info)

    registered: List[Type['DatasetLoader']] = []
    for entry in dataset_info:
        registered.append(DatasetLoader.from_dict(entry, base_dir=base_dir, exist_ok=exist_ok))
    if registered:
        logger.info(f'Successfully registered {len(registered)} datasets from dataset_info.')
    return registered


def get_dataset_loader(dataset_type: Optional[str]) -> Type['DatasetLoader']:
    """The loader for a family, or the plain :class:`DatasetLoader` when nothing matched.

    Unlike the model registry, a miss here is normal and must not raise: any hub id and any local
    jsonl is a legitimate dataset, and the whole point of the auto-detecting preprocessor is to read
    one that nobody registered. Registration only adds knowledge (which subsets exist, which
    preprocessor untangles this dataset's field names); its absence degrades to "infer everything",
    which is what legacy expressed by returning a bare ``DatasetMeta()`` from a failed lookup.
    """
    if not dataset_type:
        return DatasetLoader
    return DATASET_MAPPING.get(dataset_type, DatasetLoader)


def match_dataset_type(dataset: str, *, use_hf: bool = False) -> Optional[str]:
    """Match what the user typed against registered datasets; ``None`` when nothing does.

    Two passes, mirroring legacy's behaviour:

    1. Exact match on ``(hub, id)``, so the same name may legitimately mean different things on
       ModelScope and on HuggingFace.
    2. Basename match, so a local directory holding an already-downloaded snapshot resolves to the
       family it came from.

    The basename pass is deliberately second and deliberately looser: it exists for local snapshots,
    but two hubs can host different datasets whose ids end in the same word, so it must never
    override an exact hit.
    """
    if not _ID_MAPPING:
        for dataset_type, cls in DATASET_MAPPING.items():
            for ms_id, hf_id in cls.iter_ids():
                if ms_id:
                    _ID_MAPPING[('ms', ms_id)] = dataset_type
                if hf_id:
                    _ID_MAPPING[('hf', hf_id)] = dataset_type
            for path in cls.dataset_paths:
                _ID_MAPPING[('path', path)] = dataset_type

    hub = 'path' if os.path.exists(dataset) else ('hf' if use_hf else 'ms')
    dataset_type = _ID_MAPPING.get((hub, dataset))
    if dataset_type is not None:
        return dataset_type

    name = DatasetLoader.dataset_name(dataset)
    for (_, registered_id), dataset_type in _ID_MAPPING.items():
        if DatasetLoader.dataset_name(registered_id) == name:
            return dataset_type
    return None


class DatasetLoader:
    """The plain ``datasets.load_dataset`` path, driven by declarations.

    The happy path lives here so a family declares only what differs: an unregistered dataset is
    loaded by this class as-is, and a registered one usually adds no more than its ids and a
    preprocessor. Anything beyond that overrides a ``build_*`` / ``process_*`` hook.

    Note this class is not abstract and is itself the fallback -- see :func:`get_dataset_loader`.
    Legacy had a ``BaseDatasetLoader(ABC)`` whose only concrete implementation was the one the
    fallback used anyway, so the abstract layer bought nothing and made "no registration" look like
    an error case.
    """

    # -- registration ---------------------------------------------------------------------------
    dataset_type: Optional[str] = None
    # Hub ids this family covers: a bare id when both hubs agree, or a (ms_id, hf_id) pair when they
    # differ or only one hosts it -- use `None` for the absent side, e.g. `('swift/x', None)`.
    datasets: Sequence[Union[str, Tuple[Optional[str], Optional[str]]]] = ()
    # Local paths this family covers, for datasets that ship with swift or are declared by a user's
    # dataset_info file rather than living on a hub.
    dataset_paths: Sequence[str] = ()
    # Subsets, as names or full SubsetMeta objects. The default single 'default' subset is what a
    # dataset with no subsets on the hub needs, so most families leave this alone.
    subsets: Sequence[Union[str, SubsetMeta]] = ('default', )
    # Splits to load, applying to every subset that does not override them.
    split: Sequence[str] = ('train', )
    # Row preprocessor: a class, an instance, or a `'module:ClassName'` string resolved lazily.
    # `None` selects auto-detection, which is the right default -- it reads the standard field-name
    # layouts, and only a dataset with genuinely odd field names needs to name a preprocessor.
    preprocessor: Optional[Any] = None
    # Column renames this dataset needs: raw name -> standard name. Declared here rather than baked
    # into a Preprocessor subclass so a dataset whose only quirk is its field names needs no code at
    # all -- which is what lets the whole `dataset_info.json` batch be pure data.
    columns: Dict[str, str] = {}
    ms_revision: Optional[str] = None
    hf_revision: Optional[str] = None
    tags: Sequence[str] = ()
    # One-line note shown when listing datasets, e.g. a licence restriction or a manual-download step.
    help: Optional[str] = None
    # Too large to materialise: consumers should default to streaming rather than a full local copy.
    huge_dataset: bool = False

    # Temporary directories handed to `datasets`, one per prefix, kept alive for the process. See
    # :meth:`use_swift_cache_for_temp_files`.
    _temp_dirs: Dict[str, Any] = {}

    def __init__(self, dataset_info: DatasetInfo, **kwargs):
        # `dataset_info` describes one concrete load request, so it arrives per instance rather than
        # being a per-family constant.
        self._dataset_info = dataset_info
        self._kwargs = kwargs

    @property
    def dataset_info(self) -> DatasetInfo:
        return self._dataset_info

    @classmethod
    def from_dict(cls,
                  entry: Dict[str, Any],
                  *,
                  base_dir: Optional[str] = None,
                  exist_ok: bool = False) -> Type['DatasetLoader']:
        """Build and register a loader subclass from one ``dataset_info`` entry.

        Called per entry by :func:`register_dataset_info`. Subclassing ``cls`` rather than
        :class:`DatasetLoader` means a family can expose its own JSON-declarable variant by calling
        this on itself.
        """
        entry = dict(entry)
        ms_id = entry.pop('ms_dataset_id', None)
        hf_id = entry.pop('hf_dataset_id', None)
        dataset_path = entry.pop('dataset_path', None)
        if dataset_path is not None:
            if base_dir is not None and not os.path.isabs(dataset_path):
                dataset_path = os.path.join(base_dir, dataset_path)
            dataset_path = os.path.abspath(os.path.expanduser(dataset_path))
        assert ms_id or hf_id or dataset_path, f'A dataset_info entry needs an id or a path: {entry}'

        # The trailing component of the id is the name worth typing, and is what legacy's basename
        # fallback already resolved to -- so it is the natural registry key. An entry may pin one
        # explicitly when two hubs host same-named datasets.
        dataset_type = entry.pop('dataset_type', None) or cls.dataset_name(ms_id or hf_id or dataset_path)

        attrs: Dict[str, Any] = {'dataset_type': dataset_type}
        if ms_id or hf_id:
            attrs['datasets'] = [(ms_id, hf_id)]
        if dataset_path:
            attrs['dataset_paths'] = [dataset_path]
        if 'subsets' in entry:
            attrs['subsets'] = [
                SubsetMeta(**subset) if isinstance(subset, dict) else SubsetMeta(subset=subset)
                for subset in entry.pop('subsets')
            ]
        for key in ('split', 'columns', 'tags', 'help', 'huge_dataset', 'ms_revision', 'hf_revision'):
            if key in entry:
                attrs[key] = entry.pop(key)
        assert not entry, f'Unrecognised dataset_info keys for `{dataset_type}`: {sorted(entry)}'

        # A class name is cosmetic (it shows up in logs and tracebacks) but must still be a valid
        # identifier, and a few dataset names start with a digit.
        class_name = re.sub(r'\W', '_', dataset_type)
        if not class_name[:1].isalpha():
            class_name = f'Dataset_{class_name}'
        loader_cls = type(class_name, (cls, ), attrs)
        return register_dataset(loader_cls, exist_ok=exist_ok)

    # -- declaration accessors -------------------------------------------------------------------

    @classmethod
    def iter_ids(cls) -> Iterator[Tuple[Optional[str], Optional[str]]]:
        """Yield ``(ms_id, hf_id)`` for every entry, expanding the bare-string shorthand.

        A bare string means both hubs use the same id; a pair is written only when they differ or
        when one hub does not host the dataset at all.
        """
        for entry in cls.datasets:
            if isinstance(entry, str):
                yield entry, entry
            else:
                yield entry[0], entry[1]

    @classmethod
    def resolve_subsets(cls, names: Sequence[str] = ()) -> List[SubsetMeta]:
        """The subsets to load, with inherited fields filled in.

        An empty ``names`` selects every non-weak subset; ``['all']`` is the explicit spelling of the
        same request. Naming a weak subset directly still loads it -- weakness only governs bulk
        selection.
        """
        declared = [SubsetMeta(subset=s) if isinstance(s, str) else s for s in cls.subsets]
        if not names or list(names) == ['all']:
            selected = [s for s in declared if not s.is_weak_subset]
        else:
            by_name = {s.name: s for s in declared}
            # An unregistered subset name is passed through to the hub rather than rejected: this
            # class is also the fallback for datasets nobody declared, whose subsets we cannot know.
            selected = [by_name.get(name) or SubsetMeta(subset=name) for name in names]
        return [s.resolve(cls) for s in selected]

    @classmethod
    def resolve_id(cls, *, use_hf: bool = False) -> Optional[str]:
        """The hub id to fetch, for the requested hub, or ``None`` when that hub lacks it."""
        for ms_id, hf_id in cls.iter_ids():
            chosen = hf_id if use_hf else ms_id
            if chosen:
                return chosen
        return None

    # -- load hooks ------------------------------------------------------------------------------

    def build_preprocessor(self, subset: SubsetMeta) -> Any:
        """Instantiate the row preprocessor for ``subset``.

        Resolves a class or a ``'module:ClassName'`` string to an instance and passes an instance
        through untouched. A family whose preprocessor needs constructor arguments beyond ``columns``
        overrides this instead of trying to encode those arguments in a declaration -- the arguments
        are code, so they belong in code.

        Column renames from the declaration and from the caller are merged, with the caller winning:
        the declaration is what the framework believes about this dataset, ``--columns`` is what the
        user knows about the copy in front of them.
        """
        columns = {**(subset.columns or {}), **(self._kwargs.get('columns') or {})}
        preprocessor = subset.preprocessor
        if preprocessor is None:
            # No declaration: the auto-detecting base preprocessor reads the standard layouts.
            from ..preprocessor import Preprocessor
            preprocessor = Preprocessor
        elif isinstance(preprocessor, str):
            preprocessor = self.import_cls(preprocessor)
        if isinstance(preprocessor, type):
            return preprocessor(columns=columns)
        # An instance was declared: it already carries its own configuration, so it is used as-is.
        return preprocessor

    def build_dataset(self, subset: SubsetMeta, split: str, **kwargs) -> DATASET_TYPE:
        """Fetch one (subset, split) as rows, before any preprocessing.

        Dispatches on :attr:`DatasetInfo.source`: a local file is read by its extension, a local
        directory and any hub id go through the hub. A family that has to reach past this -- a manual
        file download, an archive to unpack -- overrides this hook, and should keep the lock.

        The fetch is serialised across ranks: one downloads and writes the cache while the others wait,
        then read it. ``sticky`` because the key names the *result* -- this (dataset, subset, split) --
        and fetching is idempotent, so a rank arriving after the work is done proceeds rather than
        waiting for a fresh round.
        """
        info = self._dataset_info
        with self.serialised(f'{info.dataset}/{subset.subset}/{split}', sticky=True):
            if info.source == 'path' and not os.path.isdir(info.dataset):
                extension = os.path.splitext(info.dataset)[1].lstrip('.') or 'json'
                extension = {'jsonl': 'json', 'txt': 'text'}.get(extension, extension)
                if extension == 'csv':
                    # Without this, pandas reads an empty cell as NaN, and a float NaN reaches the
                    # template where a string was meant. An empty field means an empty string.
                    kwargs['na_filter'] = False
                return hf_load_dataset(extension, data_files=info.dataset, split=split, **kwargs)
            if info.source == 'path':
                # A downloaded snapshot directory. `datasets` reads a folder as a dataset by itself, so
                # this goes through the HF loader whichever hub the copy came from.
                self.hide_dataset_infos(info.dataset)
                return self.load_from_hub(info.dataset, subset.subset, split, use_hf=True, **kwargs)
            use_hf = info.source == 'hf'
            dataset_id = self.resolve_id(use_hf=use_hf) or info.dataset
            return self.load_from_hub(
                dataset_id, subset.subset, split, use_hf=use_hf, revision=info.revision, **kwargs)

    @staticmethod
    def serialised(key: str, sticky: bool = False):
        """Let one rank do the work on ``key`` first, then the rest -- every rank still runs the body.

        Wraps ``twinkle.utils.processing_lock``, which orders ranks through its own coordination store
        (global master, then node masters, then everyone else) and falls back to a file lock when there
        is no store. Deliberately not ``swift.utils.safe_ddp_context``: that orders ranks with
        ``dist.barrier()``, which puts a dataset pass -- minutes to hours on a large corpus -- under the
        collective watchdog timeout, and leaves the waiters hanging if the writing rank dies.

        Args:
            key: Names the resource being written.
            sticky: The key names a *result* rather than a round, so a late rank proceeds instead of
                waiting for a fresh flag. For idempotent, content-addressed work such as a download;
                leave it off for work that repeats under the same key, such as preprocessing.
        """
        from twinkle.utils import processing_lock
        return processing_lock(key, sticky=sticky)

    @staticmethod
    def load_from_hub(dataset_id: str, subset: str, split: str, *, use_hf: bool, **kwargs) -> DATASET_TYPE:
        """Load from whichever hub ``use_hf`` selects, through dev's hub layer.

        Not a bare ``datasets.load_dataset``: that reaches HuggingFace only, while a ModelScope id has
        to go through ``MsDataset`` -- which is also where login, the differing revision names
        (``master`` on ModelScope, ``main`` on HuggingFace) and ``trust_remote_code`` are handled.
        ``swift.dev.utils.get_hub`` already owns those differences, so this only unwraps the result.
        """
        from swift.dev.utils import get_hub
        hub = get_hub(use_hf)
        # Both hubs accept the union of these and ignore what they have no use for (`token` on the HF
        # side, `num_proc` on the ModelScope side), so one call site serves both.
        dataset = hub.load_dataset(dataset_id, subset, split, **kwargs)
        # `MsDataset` returns a wrapper around the real thing; a streaming request answered with a
        # materialised dataset is converted rather than silently changing kind under the caller.
        if hasattr(dataset, '_hf_ds'):
            dataset = dataset._hf_ds
            if kwargs.get('streaming') and isinstance(dataset, HfDataset):
                dataset = dataset.to_iterable_dataset()
        return dataset

    @staticmethod
    def hide_dataset_infos(directory: str) -> None:
        """Move a snapshot's ``dataset_infos.json`` aside so ``datasets`` does not read it.

        A dataset downloaded from ModelScope carries this extra file; ``datasets`` treats it as
        authoritative metadata for the folder and can then disagree with the actual data files.
        """
        path = os.path.join(directory, 'dataset_infos.json')
        if not os.path.isfile(path):
            return
        try:
            os.rename(path, f'{path}_bak')
        except OSError:
            # Another process renamed it first, which is the outcome wanted either way.
            pass

    # -- orchestration ---------------------------------------------------------------------------

    def load(self) -> Optional[DATASET_TYPE]:
        """Load every requested (subset, split), preprocess each, and concatenate the lot.

        This is the per-family entry the top-level :func:`load_dataset` calls once per dataset string.
        Loading concerns (``num_proc``, ``strict`` ...) travel in ``self._kwargs``, set at
        construction, so this signature stays empty and every hook it calls reads them from there.
        """
        info = self._dataset_info
        kwargs = self._kwargs
        # A token is only sent when there is one: passing `token=None` explicitly would override the
        # credentials the hub client has already cached for the user.
        hub_kwargs = {'token': kwargs['hub_token']} if kwargs.get('hub_token') else {}
        parts: List[DATASET_TYPE] = []
        for subset in self.resolve_subsets(info.subsets):
            preprocessor = self.build_preprocessor(subset)
            for split in subset.split:
                raw = self.build_dataset(
                    subset,
                    split,
                    num_proc=kwargs.get('num_proc', 1),
                    streaming=kwargs.get('streaming', False),
                    download_mode=kwargs.get('download_mode', 'reuse_dataset_if_exists'),
                    **hub_kwargs)
                processed = preprocessor(
                    raw,
                    num_proc=kwargs.get('num_proc', 1),
                    load_from_cache_file=kwargs.get('load_from_cache_file', True),
                    strict=kwargs.get('strict', False))
                parts.append(processed)
        return self.concat_datasets(parts)

    def post_process(self,
                     dataset: Optional[DATASET_TYPE],
                     *,
                     split_dataset_ratio: float = 0.,
                     shuffle: bool = False,
                     seed: Optional[int] = 42) -> Tuple[Optional[DATASET_TYPE], Optional[DATASET_TYPE]]:
        """Apply the ``#N`` row budget, then carve off a validation split. Returns ``(train, val)``.

        Sampling comes before the split so the ratio applies to the sampled size, matching legacy and
        matching the intent of ``dataset#1000`` -- take 1000 rows, *then* hold out a fraction.

        ``shuffle`` decides both which rows a ``#N`` budget keeps and whether the validation rows are
        drawn from across the dataset or taken as the contiguous tail. Off by default: ``dataset#500``
        then means the first 500 rows, which is reproducible and matches legacy's default.
        """
        if dataset is None:
            return None, None
        sample_count = self._dataset_info.sample_count
        if isinstance(dataset, HfIterableDataset):
            return self.split_streaming(dataset, sample_count, split_dataset_ratio)
        if sample_count is not None:
            dataset = self.sample_dataset(dataset, sample_count, shuffle, seed)
        if split_dataset_ratio <= 0:
            return dataset, None
        if split_dataset_ratio >= 1:
            return None, dataset
        # Writes an indices cache of its own, so it is ordered like every other write.
        with self.serialised('dataset_split'):
            split = dataset.train_test_split(test_size=split_dataset_ratio, shuffle=shuffle, seed=seed)
        return split['train'], split['test']

    # -- helpers ---------------------------------------------------------------------------------

    @staticmethod
    def import_cls(spec: str) -> type:
        """Resolve ``'swift.dev.dataset.format.openai:OpenAIPreprocessor'`` lazily.

        Deferred so a declaration costs nothing at import time, and so a preprocessor may live in a
        module that is expensive or optional to import.
        """
        module_name, _, attr = spec.partition(':')
        return getattr(importlib.import_module(module_name), attr)

    @staticmethod
    def dataset_name(dataset: str) -> str:
        """Trailing component of an id or path, with hub cache decorations stripped.

        Hub caches bury the real name inside ``.../datasets--org--name/snapshots/<sha>``, so a plain
        ``rsplit('/')`` on a cached snapshot path would return a commit sha.
        """
        dataset = dataset.rstrip('/')
        match = re.search('/datasets--(?:.+?--)?(.+?)/snapshots/', dataset)
        if match is not None:
            return match.group(1)
        if platform.system().lower() == 'windows':
            dataset = dataset.replace('\\', '/')
        return dataset.rsplit('/', 1)[-1]

    @staticmethod
    def concat_datasets(datasets: List[DATASET_TYPE]) -> Optional[DATASET_TYPE]:
        """Concatenate, short-circuiting the 0- and 1-element cases.

        The single-dataset short-circuit is not just an optimisation: ``concatenate_datasets`` on one
        element still rebuilds the Arrow table, which would drop the cheap zero-copy path that the
        overwhelmingly common single-dataset load takes.
        """
        if not datasets:
            return None
        if len(datasets) == 1:
            return datasets[0]
        return concatenate_datasets(datasets)

    @staticmethod
    def interleave_datasets(datasets: List[DATASET_TYPE], **kwargs) -> Optional[DATASET_TYPE]:
        """Interleave with per-dataset probabilities; same short-circuits as :meth:`concat_datasets`."""
        if not datasets:
            return None
        if len(datasets) == 1:
            return datasets[0]
        return interleave_datasets(datasets, **kwargs)

    @staticmethod
    def sample_dataset(dataset: DATASET_TYPE,
                       sample_count: int,
                       shuffle: bool = False,
                       seed: Optional[int] = None) -> DATASET_TYPE:
        """Take ``sample_count`` rows, oversampling with whole repeats when it exceeds the length.

        A request larger than the dataset is not an error: the surplus is made of full copies plus a
        remainder, so ``dataset#Ntimeslength`` repeats every row evenly.

        ``shuffle`` governs only which rows the remainder is: off, it is the first rows of the dataset,
        so ``dataset#500`` means "the first 500" and is reproducible without a seed; on, it is a random
        draw. The whole-copy part is never shuffled either way -- it holds every row regardless, and
        the training sampler will shuffle epochs anyway.
        """
        length = len(dataset)
        if sample_count is None or length == 0:
            return dataset
        indices = np.tile(np.arange(length), sample_count // length)
        remainder = sample_count % length
        if remainder > 0:
            if shuffle:
                drawn = np.random.RandomState(seed).permutation(length)[:remainder]
            else:
                drawn = np.arange(remainder)
            indices = np.concatenate([indices, drawn])
        return dataset.select(indices)

    @staticmethod
    def split_sample_count(dataset: str) -> Tuple[str, Optional[int]]:
        """Split a trailing ``#N`` row budget off a dataset string: ``'id#500'`` -> ``('id', 500)``."""
        name, sep, count = dataset.rpartition('#')
        if sep and count.isdigit():
            return name, int(count)
        return dataset, None

    @staticmethod
    def load_cached_datasets(cached_dataset: Sequence[str],
                             cached_val_dataset: Sequence[str],
                             *,
                             max_length: Optional[int] = None,
                             truncation_strategy: Optional[str] = None,
                             data_seed: Optional[int] = None,
                             shuffle: bool = False) -> Tuple[List[DATASET_TYPE], List[DATASET_TYPE]]:
        """Load pre-encoded splits written by ``swift export --to_cached_dataset`` from disk.

        Reimplements legacy ``swift.pipelines.utils.get_cached_dataset`` on dev primitives so the dev
        dataset path no longer reaches into ``swift.pipelines``. Each path may carry a trailing ``#N``
        row budget (:meth:`split_sample_count`); the saved table is read with ``load_from_disk``, its
        3.x ``length`` column renamed to ``lengths``, and -- only under
        ``truncation_strategy='delete'`` -- rows longer than ``max_length`` are dropped before the
        optional ``#N`` subsample (:meth:`sample_dataset`).

        Returns ``(train, val)`` as lists (possibly empty), not concatenated: the per-split builder
        merges each with the freshly-encoded split, so a cache-only run still produces a loader.
        """
        from datasets import load_from_disk
        train_datasets: List[DATASET_TYPE] = []
        val_datasets: List[DATASET_TYPE] = []
        for paths, out in ((cached_dataset, train_datasets), (cached_val_dataset, val_datasets)):
            for path in (paths or []):
                # An existing path is a directory to read as-is; otherwise a trailing ``#N`` is a
                # row budget, matching the ``dataset#N`` syntax used everywhere else.
                if os.path.exists(path):
                    sample_count = None
                else:
                    path, sample_count = DatasetLoader.split_sample_count(path)
                dataset = load_from_disk(path)
                # ms-swift 3.x wrote the encoded token count as ``length``; dev reads ``lengths``.
                if 'length' in dataset.column_names and 'lengths' not in dataset.column_names:
                    dataset = dataset.rename_column('length', 'lengths')
                if truncation_strategy == 'delete' and max_length is not None:
                    lengths = dataset['lengths']
                    # ``lengths`` is a per-row token count, but a packed cache stores a list of the
                    # counts it packed -- take the longest so the filter is on the real sequence
                    # length. A row MeasurePreprocessor could not encode carries an empty list and is
                    # treated as length 0 (kept, then substituted at access time).
                    if lengths and isinstance(lengths[0], list):
                        arr = np.fromiter((max(x) if x else 0 for x in lengths), dtype=np.int64, count=len(lengths))
                    else:
                        arr = np.asarray(lengths, dtype=np.int64)
                    keep = arr <= max_length
                    if not bool(keep.all()):
                        dataset = dataset.select(np.flatnonzero(keep))
                if sample_count is not None:
                    dataset = DatasetLoader.sample_dataset(dataset, sample_count, shuffle, data_seed)
                out.append(dataset)
        return train_datasets, val_datasets

    @staticmethod
    def parse_legacy_syntax(entry: str) -> Tuple[str, List[str], Optional[int], Optional[bool]]:
        """Pull legacy's one-string dataset DSL apart into the plain fields dev takes.

        Legacy packed four things into a single command-line token: ``hf::org/name:sub1/sub2#500`` is
        hub ``hf``, id ``org/name``, subsets ``sub1`` and ``sub2``, and a 500-row budget. Dev's own
        interface takes those as ordinary arguments -- ``subsets=['sub1', 'sub2']``, ``use_hf=True`` --
        which is the form worth writing; this exists so the strings already in scripts and docs keep
        working, and nothing else in dev needs to know the syntax.

        Returns ``(dataset, subsets, sample_count, use_hf)``, where ``use_hf`` is ``None`` when the
        string pins no hub and ``subsets`` is empty when it names none.

        An existing path is returned untouched: a Windows drive letter and an ordinary filename can
        both contain the characters this splits on, so a path is never parsed. The check is repeated
        after the row budget comes off, since a path may be what is left.
        """
        if os.path.exists(entry):
            return entry, [], None, None

        use_hf: Optional[bool] = None
        hub, sep, rest = entry.partition('::')
        if sep:
            use_hf = {'hf': True, 'ms': False}.get(hub.lower())
            if use_hf is not None:
                entry = rest

        entry, sample_count = DatasetLoader.split_sample_count(entry)
        if os.path.exists(entry):
            return entry, [], sample_count, use_hf

        entry, sep, subsets = entry.partition(':')
        return entry, subsets.split('/') if sep and subsets else [], sample_count, use_hf

    @staticmethod
    def shuffle_dataset(dataset: DATASET_TYPE, seed: Optional[int] = None, buffer_size: int = 1000) -> DATASET_TYPE:
        """Shuffle either kind of dataset: a materialised one globally, a stream through a buffer.

        A stream cannot be permuted, only reordered within a window, so ``buffer_size`` is the whole
        extent of the shuffle there and does nothing for a materialised dataset.
        """
        if isinstance(dataset, HfIterableDataset):
            return dataset.shuffle(seed=seed, buffer_size=buffer_size)
        # A permutation of a materialised dataset is written as an indices cache file.
        with DatasetLoader.serialised('dataset_shuffle'):
            return dataset.shuffle(seed=seed)

    @staticmethod
    def split_streaming(dataset: DATASET_TYPE, sample_count: Optional[int],
                        split_dataset_ratio: float) -> Tuple[Optional[DATASET_TYPE], Optional[DATASET_TYPE]]:
        """``(train, val)`` for a stream, where a split can only be taken off the front.

        A stream has no length, so a *fraction* of it only means something once a row budget bounds it:
        given ``#N``, the first ``N * ratio`` rows become the validation set and the remainder the
        training set. Without a budget only the degenerate ratios have an answer -- 0, no validation
        set, and 1, the whole stream is one.
        """
        if sample_count is None:
            if split_dataset_ratio <= 0:
                return dataset, None
            if split_dataset_ratio >= 1:
                return None, dataset
            raise ValueError('A streaming dataset can only be split into train/val when a `#N` row '
                             'budget bounds it, e.g. `my-dataset#10000`.')
        dataset = dataset.take(sample_count)
        val_count = int(sample_count * split_dataset_ratio)
        if val_count == 0:
            return dataset, None
        return dataset.skip(val_count), dataset.take(val_count)

    @classmethod
    def use_swift_cache_for_temp_files(cls) -> None:
        """Point ``datasets``' temporary Arrow files at swift's cache directory.

        A dataset held in memory -- a fresh ``map`` result, a synthesised split -- writes its Arrow
        file to a temporary directory, which defaults to ``/tmp``: routinely a small tmpfs in a
        container, where a real dataset fills it and the run dies far from the cause.

        Legacy did the same, but as a side effect of importing its package, so merely importing the
        dataset code changed ``datasets``' global behaviour for the process. Here the caller that is
        about to create those files asks for it.
        """
        import tempfile

        import datasets.arrow_dataset
        import datasets.config
        import datasets.fingerprint
        from modelscope.hub.utils.utils import get_cache_dir

        def temporary_cache_files_directory(prefix: Optional[str] = None) -> str:
            prefix = prefix or datasets.config.TEMP_CACHE_DIR_PREFIX
            if prefix not in cls._temp_dirs:
                root = os.path.join(get_cache_dir(), 'tmp')
                os.makedirs(root, exist_ok=True)
                # Held in a class-level pool rather than used as a context manager: callers treat the
                # path as valid for the rest of the process, so the directory has to outlive this call
                # and is cleaned up when the interpreter drops the object at exit.
                cls._temp_dirs[prefix] = tempfile.TemporaryDirectory(
                    prefix=prefix, dir=root, ignore_cleanup_errors=True)
                logger.info(f'Created dataset tmp_dir: {cls._temp_dirs[prefix].name}')
            return cls._temp_dirs[prefix].name

        datasets.fingerprint.get_temporary_cache_files_directory = temporary_cache_files_directory
        datasets.arrow_dataset.get_temporary_cache_files_directory = temporary_cache_files_directory

    @staticmethod
    def detect_source(dataset: str, use_hf: bool) -> str:
        """Where rows come from: a local file/dir is ``'path'``, otherwise the chosen hub."""
        return 'path' if os.path.exists(dataset) else ('hf' if use_hf else 'ms')


def load_dataset(
    datasets: Union[str, List[str]],
    *,
    split_dataset_ratio: float = 0.,
    seed: Optional[int] = 42,
    num_proc: int = 1,
    load_from_cache_file: bool = True,
    shuffle: bool = False,
    streaming: bool = False,
    interleave_prob: Optional[List[float]] = None,
    stopping_strategy: str = 'first_exhausted',
    shuffle_buffer_size: int = 1000,
    use_hf: Optional[bool] = None,
    hub_token: Optional[str] = None,
    strict: bool = False,
    download_mode: str = 'reuse_dataset_if_exists',
    columns: Optional[Dict[str, str]] = None,
    subsets: Optional[Sequence[str]] = None,
    model_name: Optional[Union[str, Sequence[str]]] = None,
    model_author: Optional[Union[str, Sequence[str]]] = None,
) -> Tuple[Optional[DATASET_TYPE], Optional[DATASET_TYPE]]:
    """Load and preprocess one or more datasets into standard ``(train, val)`` splits.

    A name is matched against the registry to pick a :class:`DatasetLoader`; an unmatched name is a
    normal case (a bare hub id or a local file) and loads through the base loader.

    How the parts are then put together: by default every dataset's train part is concatenated (and
    likewise every val part), so the result is all of the data. Passing ``interleave_prob`` instead
    samples between them with those probabilities, which is what mixing corpora of very different
    sizes needs -- ``stopping_strategy`` then decides whether the mixture ends with the smallest
    dataset or keeps drawing until every one is spent. ``shuffle`` applies last, to the combined
    result, and in streaming mode is a ``shuffle_buffer_size`` window rather than a permutation.

    ``subsets``, ``use_hf`` and a ``#N`` row budget are ordinary arguments here. Legacy instead
    encoded them in each dataset string (``hf::org/name:sub#500``); such strings still work, being
    unpacked by :meth:`DatasetLoader.parse_legacy_syntax`, and what a string carries wins over the
    argument for that one dataset.
    """
    if isinstance(datasets, str):
        datasets = [datasets]
    DatasetLoader.use_swift_cache_for_temp_files()
    if streaming:
        # `num_proc` distributes a `map` over shards of a materialised table; a stream is consumed by
        # one process and `datasets` rejects the pair.
        num_proc = None

    train_parts: List[DATASET_TYPE] = []
    val_parts: List[DATASET_TYPE] = []
    for entry in datasets:
        dataset, entry_subsets, sample_count, entry_use_hf = DatasetLoader.parse_legacy_syntax(entry)
        if entry_use_hf is None:
            entry_use_hf = use_hf
        dataset_type = match_dataset_type(dataset, use_hf=bool(entry_use_hf))
        loader_cls = get_dataset_loader(dataset_type)
        info = DatasetInfo(
            dataset=dataset,
            dataset_type=dataset_type,
            source=DatasetLoader.detect_source(dataset, entry_use_hf),
            subsets=entry_subsets or list(subsets or []),
            sample_count=sample_count,
            # The two hubs number their revisions separately, so a family declares one for each and
            # the chosen hub decides which is meant.
            revision=loader_cls.hf_revision if entry_use_hf else loader_cls.ms_revision)
        loader = loader_cls(
            info,
            num_proc=num_proc,
            load_from_cache_file=load_from_cache_file,
            strict=strict,
            download_mode=download_mode,
            columns=columns,
            streaming=streaming,
            hub_token=hub_token,
            model_name=model_name,
            model_author=model_author)
        train, val = loader.post_process(
            loader.load(), split_dataset_ratio=split_dataset_ratio, shuffle=shuffle, seed=seed)
        if train is not None:
            train_parts.append(train)
        if val is not None:
            val_parts.append(val)

    if interleave_prob is None:
        train_dataset = DatasetLoader.concat_datasets(train_parts)
        val_dataset = DatasetLoader.concat_datasets(val_parts)
    else:
        interleave_kwargs = {
            'probabilities': interleave_prob,
            'seed': seed,
            'stopping_strategy': stopping_strategy,
        }
        train_dataset = DatasetLoader.interleave_datasets(train_parts, **interleave_kwargs)
        val_dataset = DatasetLoader.interleave_datasets(val_parts, **interleave_kwargs)

    if shuffle:
        if train_dataset is not None:
            train_dataset = DatasetLoader.shuffle_dataset(train_dataset, seed, shuffle_buffer_size)
        if val_dataset is not None:
            val_dataset = DatasetLoader.shuffle_dataset(val_dataset, seed, shuffle_buffer_size)
    return train_dataset, val_dataset

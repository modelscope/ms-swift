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

from swift.utils import get_logger

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

        Dispatches on :attr:`DatasetInfo.source`: a local file is loaded by its extension, everything
        else goes to the hub. A family that has to reach past this -- a manual file download, an
        archive to unpack -- overrides this hook.
        """
        info = self._dataset_info
        if info.source == 'path':
            extension = os.path.splitext(info.dataset)[1].lstrip('.') or 'json'
            extension = {'jsonl': 'json', 'txt': 'text'}.get(extension, extension)
            return hf_load_dataset(extension, data_files=info.dataset, split=split, **kwargs)
        dataset_id = self.resolve_id(use_hf=info.source == 'hf') or info.dataset
        return hf_load_dataset(dataset_id, subset.subset, split=split, revision=info.revision, **kwargs)

    # -- orchestration ---------------------------------------------------------------------------

    def load(self) -> Optional[DATASET_TYPE]:
        """Load every requested (subset, split), preprocess each, and concatenate the lot.

        This is the per-family entry the top-level :func:`load_dataset` calls once per dataset string.
        Loading concerns (``num_proc``, ``strict`` ...) travel in ``self._kwargs``, set at
        construction, so this signature stays empty and every hook it calls reads them from there.
        """
        info = self._dataset_info
        kwargs = self._kwargs
        parts: List[DATASET_TYPE] = []
        for subset in self.resolve_subsets(info.subsets):
            preprocessor = self.build_preprocessor(subset)
            for split in subset.split:
                raw = self.build_dataset(
                    subset,
                    split,
                    num_proc=kwargs.get('num_proc', 1),
                    streaming=kwargs.get('streaming', False),
                    download_mode=kwargs.get('download_mode', 'reuse_dataset_if_exists'))
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
                     seed: Optional[int] = 42) -> Tuple[Optional[DATASET_TYPE], Optional[DATASET_TYPE]]:
        """Apply the ``#N`` row budget, then carve off a validation split. Returns ``(train, val)``.

        Sampling comes before the split so the ratio applies to the sampled size, matching legacy and
        matching the intent of ``dataset#1000`` -- take 1000 rows, *then* hold out a fraction.
        """
        if dataset is None:
            return None, None
        info = self._dataset_info
        if info.sample_count is not None:
            dataset = self.sample_dataset(dataset, info.sample_count, seed)
        if split_dataset_ratio > 0:
            split = dataset.train_test_split(test_size=split_dataset_ratio, seed=seed)
            return split['train'], split['test']
        return dataset, None

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
    def sample_dataset(dataset: DATASET_TYPE, sample_count: int, seed: Optional[int] = None) -> DATASET_TYPE:
        """Take ``sample_count`` rows, oversampling with whole repeats when it exceeds the length.

        A request larger than the dataset is not an error: the surplus is made of full copies plus a
        random remainder, so ``dataset#Ntimeslength`` repeats every row evenly.
        """
        length = len(dataset)
        if sample_count is None or length == 0:
            return dataset
        random_state = np.random.RandomState(seed)
        indices = np.tile(np.arange(length), sample_count // length)
        remainder = sample_count % length
        if remainder > 0:
            indices = np.concatenate([indices, random_state.permutation(length)[:remainder]])
        return dataset.select(indices)

    @staticmethod
    def split_sample_count(dataset: str) -> Tuple[str, Optional[int]]:
        """Split a trailing ``#N`` row budget off a dataset string: ``'id#500'`` -> ``('id', 500)``."""
        name, sep, count = dataset.rpartition('#')
        if sep and count.isdigit():
            return name, int(count)
        return dataset, None

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
    use_hf: Optional[bool] = None,
    strict: bool = False,
    download_mode: str = 'reuse_dataset_if_exists',
    columns: Optional[Dict[str, str]] = None,
    streaming: bool = False,
    subsets: Optional[Sequence[str]] = None,
) -> Tuple[Optional[DATASET_TYPE], Optional[DATASET_TYPE]]:
    """Load and preprocess one or more datasets into standard ``(train, val)`` splits.

    Each entry may carry a trailing ``#N`` row budget (``'AI-ModelScope/alpaca-gpt4-data-zh#500'``).
    A name is matched against the registry to pick a :class:`DatasetLoader`; an unmatched name is a
    normal case (a bare hub id or a local file) and loads through the base loader. Every dataset's
    train part -- and every val part, when ``split_dataset_ratio > 0`` -- is concatenated.

    This is the dev-layer counterpart of legacy's ``load_dataset``. What legacy parsed with a full
    ``DatasetSyntax`` DSL (``hub::id:sub1/sub2#N``) is, for now, a plain name plus a ``#N`` suffix and
    a ``subsets`` argument; the richer syntax is a later, separate module.
    """
    if isinstance(datasets, str):
        datasets = [datasets]

    train_parts: List[DATASET_TYPE] = []
    val_parts: List[DATASET_TYPE] = []
    for entry in datasets:
        dataset, sample_count = DatasetLoader.split_sample_count(entry)
        dataset_type = match_dataset_type(dataset, use_hf=use_hf)
        loader_cls = get_dataset_loader(dataset_type)
        info = DatasetInfo(
            dataset=dataset,
            dataset_type=dataset_type,
            source=DatasetLoader.detect_source(dataset, use_hf),
            subsets=list(subsets or []),
            sample_count=sample_count)
        loader = loader_cls(
            info,
            num_proc=num_proc,
            load_from_cache_file=load_from_cache_file,
            strict=strict,
            download_mode=download_mode,
            columns=columns,
            streaming=streaming)
        train, val = loader.post_process(
            loader.load(), split_dataset_ratio=split_dataset_ratio, seed=seed)
        if train is not None:
            train_parts.append(train)
        if val is not None:
            val_parts.append(val)
    return DatasetLoader.concat_datasets(train_parts), DatasetLoader.concat_datasets(val_parts)

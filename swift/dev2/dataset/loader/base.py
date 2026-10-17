# Copyright (c) ModelScope Contributors. All rights reserved.
# Which dataset a name refers to, and how to turn it into rows.
from __future__ import annotations

import importlib
import json
import os
import platform
import re
from dataclasses import dataclass, field
from datasets import Dataset as HfDataset
from datasets import IterableDataset as HfIterableDataset
from datasets import concatenate_datasets, interleave_datasets
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Type, Union

import numpy as np

from swift.dev.utils.tags import validate_tags
from swift.dev.utils import get_logger
from swift.dev.dataset.storage import STORAGE_MAPPING, detect_storage_format

__all__ = [
    'DATASET_MAPPING', 'DATASET_TYPE', 'DatasetInfo', 'DatasetLoader', 'SubsetMeta', 'get_dataset_loader',
    'load_dataset', 'match_dataset_type', 'register_dataset', 'register_dataset_info'
]

logger = get_logger()

DATASET_TYPE = Union[HfDataset, HfIterableDataset]
DATASET_MAPPING: Dict[str, Type['DatasetLoader']] = {}
_ID_MAPPING: Dict[Tuple[str, str], str] = {}


@dataclass
class SubsetMeta:
    """One selectable subset of a dataset, and what makes it differ from its siblings.

    Sibling subsets routinely need *different* preprocessing (legacy's `stsb` maps four subsets to
    four label sets), so `preprocessor` and `split` are overridable per subset; both default to
    ``None`` meaning "inherit from the owning loader" -- see ``resolve``.
    """

    subset: str = 'default'
    # Command-line name; defaults to `subset`. Set only when the hub's name is not worth typing.
    name: Optional[str] = None
    # Splits to load; None inherits the loader's.
    split: Optional[List[str]] = None
    # Row preprocessor; None inherits. Sibling subsets are often dumps of different sources that
    # spell the same field differently (legacy `medical_zh`: `instruction` in zh, `input` in en).
    preprocessor: Optional[Any] = None
    # Column renames; None inherits.
    columns: Optional[Dict[str, str]] = None
    # Exclude from a bulk `all` request: reachable by name, not silently mixed in.
    is_weak_subset: bool = False

    def __post_init__(self) -> None:
        if self.name is None:
            self.name = self.subset

    def resolve(self, loader_cls: Type['DatasetLoader']) -> 'SubsetMeta':
        """A copy with every inherited (``None``) field filled in from ``loader_cls``.

        A copy, not a mutation: the same object lives on the loader class as a shared attribute,
        so filling it in place would leak one load's resolved values into the next.
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
    """What one concrete load request resolved to -- the dataset-side ``ModelInfo``.

    Describes *what is being loaded*, not *how*: loading mechanics (``num_proc``,
    ``load_from_cache_file``, ``streaming``, ``strict``) stay keyword arguments threaded through the
    load call. Merges what legacy split between ``DatasetSyntax`` (the command-line parse) and loose
    arguments; the parse itself lives in ``parse_dataset_syntax``.
    """

    dataset: str = ''
    # Family id once the registry matched; None (unregistered) is the common case, not an error.
    dataset_type: Optional[str] = None
    # 'ms'/'hf' for a hub id, 'path' for a local file, 'repo' for a local snapshot directory.
    source: str = 'ms'
    # Subset names asked for; empty means the loader's default selection.
    subsets: List[str] = field(default_factory=list)
    # Row budget from the `#N` suffix; None for the whole dataset. May exceed the length
    # (oversampling with repeats).
    sample_count: Optional[int] = None
    # Hub revision to pin, when declared or requested.
    revision: Optional[str] = None


def register_dataset(loader_cls: Type['DatasetLoader'] = None, *, exist_ok: bool = False):
    """Register a dataset family, keyed by its ``dataset_type``. Usable bare or with keywords.

    Everything is read off the class, so a dataset is declared in one place. Legacy instead paired
    a ``DatasetMeta`` literal with a separately-defined preprocessor and keyed the registry on the
    tuple ``(ms_dataset_id, hf_dataset_id, dataset_path)`` -- a key no lookup could use directly.
    """

    def _register(cls: Type['DatasetLoader']) -> Type['DatasetLoader']:
        dataset_type = cls.dataset_type
        assert dataset_type, f'{cls.__name__} must set `dataset_type`.'
        validate_tags(cls.tags, context=f'dataset_type `{dataset_type}`')
        if not exist_ok and dataset_type in DATASET_MAPPING:
            raise ValueError(f'dataset_type `{dataset_type}` is already registered '
                             f'by {DATASET_MAPPING[dataset_type].__name__}.')
        DATASET_MAPPING[dataset_type] = cls
        _ID_MAPPING.clear()
        return cls

    return _register if loader_cls is None else _register(loader_cls)


def register_dataset_info(dataset_info: Union[str, List[Dict[str, Any]], None] = None,
                          *,
                          exist_ok: bool = False) -> List[Type['DatasetLoader']]:
    """Register datasets declared as data rather than as code, synthesising a loader class per entry.

    Most datasets differ only in *what* they are (ids, subsets, splits, which raw column means
    ``query``), so they live in ``dataset_info.json`` -- or a user's ``--custom_dataset_info`` file
    of the same shape. Each entry becomes a real ``DatasetLoader`` subclass, so a JSON-declared
    dataset and a hand-written one are indistinguishable downstream.
    """
    # Keys (all optional except one id): the dataset ids/path, dataset_type, subsets (names or
    # objects with subset/split/columns), split, columns, tags, help, huge_dataset.
    base_dir = None
    if dataset_info is None:
        dataset_info = os.path.join(os.path.dirname(__file__), 'dataset_info.json')
    if isinstance(dataset_info, str):
        path = os.path.abspath(os.path.expanduser(dataset_info))
        if os.path.isfile(path):
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
    """The loader for a family, or the plain ``DatasetLoader`` when nothing matched.

    Unlike the model registry, a miss is normal and must not raise: any hub id and any local jsonl
    is a legitimate dataset. Registration only adds knowledge (which subsets exist, which
    preprocessor untangles this dataset's fields); its absence degrades to "infer everything".
    """
    if not dataset_type:
        return DatasetLoader
    return DATASET_MAPPING.get(dataset_type, DatasetLoader)


def match_dataset_type(dataset: str, *, use_hf: bool = False) -> Optional[str]:
    """Match what the user typed against registered datasets; ``None`` when nothing does.

    Two passes: an exact ``(hub, id)`` match first, so the same name may mean different things on
    ModelScope and HuggingFace; then a basename match, so a local directory holding a downloaded
    snapshot resolves to the family it came from. The basename pass is deliberately second and
    looser -- two hubs can host different datasets whose ids end in the same word.
    """
    if dataset in DATASET_MAPPING:
        return dataset
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
    loaded by this class as-is, and a registered one usually adds no more than ids and a
    preprocessor. Not abstract -- this class is itself the fallback (see ``get_dataset_loader``).
    """
    # -- registration ---------------------------------------------------------------------------
    dataset_type: Optional[str] = None
    # Hub ids this family covers: a bare id when both hubs agree, or a (ms_id, hf_id) pair when
    # they differ or only one hosts it -- `None` for the absent side, e.g. `('swift/x', None)`.
    datasets: Sequence[Union[str, Tuple[Optional[str], Optional[str]]]] = ()
    # Local paths this family covers: datasets shipping with swift, or a user's dataset_info entry.
    dataset_paths: Sequence[str] = ()
    # Subsets, as names or full SubsetMeta objects. The single 'default' subset is what a dataset
    # with no subsets on the hub needs, so most families leave this alone.
    subsets: Sequence[Union[str, SubsetMeta]] = ('default', )
    # Splits to load, applying to every subset that does not override them.
    split: Sequence[str] = ('train', )
    # Row preprocessor: a class, an instance, or a 'module:ClassName' string resolved lazily. None
    # selects auto-detection -- only a dataset with genuinely odd field names needs to name one.
    preprocessor: Optional[Any] = None
    # Column renames this dataset needs: raw name -> standard name. Declared rather than baked into
    # a Preprocessor subclass, so a dataset whose only quirk is its field names needs no code.
    columns: Dict[str, str] = {}
    ms_revision: Optional[str] = None
    hf_revision: Optional[str] = None
    tags: Sequence[str] = ()
    # One-line note shown when listing datasets, e.g. a licence restriction or a manual-download step.
    help: Optional[str] = None
    # Too large to materialise: consumers should default to streaming rather than a full local copy.
    huge_dataset: bool = False

    # Temporary directories handed to `datasets`, one per prefix, kept alive for the process.
    _temp_dirs: Dict[str, Any] = {}

    def __init__(self, dataset_info: DatasetInfo, **kwargs):
        # `dataset_info` describes one concrete load request, so it arrives per instance rather
        # than being a per-family constant.
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

        Called per entry by ``register_dataset_info``. Subclassing ``cls`` rather than
        ``DatasetLoader`` means a family can expose its own JSON-declarable variant.
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

        # The trailing id component is the name worth typing (legacy's basename fallback resolved
        # here too) -- the natural registry key; an entry may pin one when hubs host same names.
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

        # A class name is cosmetic (logs, tracebacks) but must be an identifier; some dataset
        # names start with a digit.
        class_name = re.sub(r'\W', '_', dataset_type)
        if not class_name[:1].isalpha():
            class_name = f'Dataset_{class_name}'
        loader_cls = type(class_name, (cls, ), attrs)
        return register_dataset(loader_cls, exist_ok=exist_ok)

    @classmethod
    def iter_ids(cls) -> Iterator[Tuple[Optional[str], Optional[str]]]:
        """Yield ``(ms_id, hf_id)`` for every entry, expanding the bare-string shorthand.

        A bare string means both hubs use the same id; a pair is written only when they differ or
        one hub does not host the dataset.
        """
        for entry in cls.datasets:
            if isinstance(entry, str):
                yield entry, entry
            else:
                yield entry[0], entry[1]

    @classmethod
    def resolve_subsets(cls, names: Sequence[str] = ()) -> List[SubsetMeta]:
        """The subsets to load, with inherited fields filled in.

        An empty ``names`` selects every non-weak subset; ``['all']`` spells the same request.
        Naming a weak subset directly still loads it -- weakness only governs bulk selection.
        """
        declared = [SubsetMeta(subset=s) if isinstance(s, str) else s for s in cls.subsets]
        if not names or list(names) == ['all']:
            selected = [s for s in declared if not s.is_weak_subset]
        else:
            by_name = {s.name: s for s in declared}
            # An unregistered subset name passes through to the hub rather than being rejected:
            # this class is also the fallback for datasets nobody declared.
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

        Resolves a class or a ``'module:ClassName'`` string to an instance; an instance passes
        through. One needing constructor arguments beyond ``columns`` overrides this -- those
        arguments are code. Declaration and caller renames merge, caller winning.
        """
        columns = {**(subset.columns or {}), **(self._kwargs.get('columns') or {})}
        preprocessor = subset.preprocessor
        if preprocessor is None:
            # No declaration: the auto-detecting base preprocessor reads the standard layouts.
            from swift.dev.dataset.preprocessor import Preprocessor  # TODO: point at dev2 once migrated
            preprocessor = Preprocessor
        elif isinstance(preprocessor, str):
            preprocessor = self.import_cls(preprocessor)
        if isinstance(preprocessor, type):
            return preprocessor(columns=columns, seed=self._kwargs.get('seed', 42))
        return preprocessor

    def build_dataset(self, subset: SubsetMeta, split: str, **kwargs) -> DATASET_TYPE:
        """Fetch one (subset, split) as rows, before any preprocessing.

        Dispatches on ``DatasetInfo.source``. A local path is routed by the unified storage layer
        (``detect_storage_format``: a file by extension, a ``save_to_disk`` dir as arrow, the lake
        formats ``datasets`` cannot route alone); an ordinary snapshot dir falls through to the hub
        folder scanner, as does any hub id. A family reaching past this overrides the hook.
        """
        info = self._dataset_info
        # The fetch is serialised across ranks, sticky because the key names the *result* and
        # fetching is idempotent: a rank arriving after the work is done proceeds, not waits.
        with self.serialised(f'{info.dataset}/{subset.subset}/{split}', sticky=True):
            if info.source == 'path':
                storage_format = detect_storage_format(info.dataset)
                if storage_format is not None:
                    return STORAGE_MAPPING[storage_format]().read(info.dataset, split=split, **kwargs)
                self.hide_dataset_infos(info.dataset)
                return self.load_from_hub(info.dataset, subset.subset, split, use_hf=True, **kwargs)
            use_hf = info.source == 'hf'
            dataset_id = self.resolve_id(use_hf=use_hf) or info.dataset
            return self.load_from_hub(
                dataset_id, subset.subset, split, use_hf=use_hf, revision=info.revision, **kwargs)

    @staticmethod
    def serialised(key: str, sticky: bool = False):
        """Let one rank do the work on ``key`` first, then the rest -- every rank runs the body.

        Wraps ``twinkle.utils.processing_lock`` (global master, node masters, then everyone; a
        file lock when there is no store). Deliberately not a ``dist.barrier()`` round: that puts
        a dataset pass under the collective watchdog timeout, and leaves waiters hanging if the
        writing rank dies.
        """  # sticky: the key names a *result*, so a late rank proceeds instead of waiting for a
        # fresh flag; use it for idempotent content-addressed work (a download), not work repeating
        # under the same key.

        from twinkle.utils import processing_lock
        return processing_lock(key, sticky=sticky)

    @staticmethod
    def load_from_hub(dataset_id: str, subset: str, split: str, *, use_hf: bool, **kwargs) -> DATASET_TYPE:
        """Load from whichever hub ``use_hf`` selects, through dev's hub layer.

        Not a bare ``datasets.load_dataset``: that reaches HuggingFace only, while a ModelScope id
        must go through ``MsDataset`` -- where login, the differing revision names (``master`` vs
        ``main``) and ``trust_remote_code`` are handled. ``swift.dev.utils.get_hub`` owns those
        differences, so this only unwraps the result.
        """
        from swift.dev.utils import get_hub
        hub = get_hub(use_hf)
        dataset = hub.load_dataset(dataset_id, subset, split, **kwargs)
        if hasattr(dataset, '_hf_ds'):
            dataset = dataset._hf_ds
            if kwargs.get('streaming') and isinstance(dataset, HfDataset):
                dataset = dataset.to_iterable_dataset()
        return dataset

    @staticmethod
    def hide_dataset_infos(directory: str) -> None:
        """Move a snapshot's ``dataset_infos.json`` aside so ``datasets`` does not read it.

        A ModelScope-downloaded dataset carries this extra file; ``datasets`` treats it as
        authoritative folder metadata and can then disagree with the actual data files.
        """
        path = os.path.join(directory, 'dataset_infos.json')
        if not os.path.isfile(path):
            return
        try:
            os.rename(path, f'{path}_bak')
        except OSError:
            pass

    # -- orchestration ---------------------------------------------------------------------------

    def load(self) -> Optional[DATASET_TYPE]:
        """Load every requested (subset, split), preprocess each, and concatenate the lot.

        The per-family entry the top-level ``load_dataset`` calls once per dataset string. Loading
        concerns (``num_proc``, ``strict`` ...) travel in ``self._kwargs``, set at construction, so
        this signature stays empty and every hook reads them from there.
        """
        info = self._dataset_info
        kwargs = self._kwargs
        # A token is only sent when there is one: passing `token=None` explicitly would override
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

        Sampling comes before the split so the ratio applies to the sampled size, matching legacy
        and the intent of ``dataset#1000``. ``shuffle`` decides both which rows a ``#N`` budget
        keeps and whether validation rows are drawn across the dataset or taken as the contiguous
        tail; off by default, so ``dataset#500`` means the first 500 rows, reproducibly.
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

        Deferred so a declaration costs nothing at import time, and so a preprocessor may live in
        a module that is expensive or optional to import.
        """
        module_name, _, attr = spec.partition(':')
        return getattr(importlib.import_module(module_name), attr)

    @staticmethod
    def dataset_name(dataset: str) -> str:
        """Trailing component of an id or path, with hub cache decorations stripped.

        Hub caches bury the real name inside ``.../datasets--org--name/snapshots/<sha>``, so a
        plain ``rsplit('/')`` on a cached snapshot path would return a commit sha.
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

        The single-dataset short-circuit is not just an optimisation: ``concatenate_datasets`` on
        one element still rebuilds the Arrow table, dropping the cheap zero-copy path that the
        overwhelmingly common single-dataset load takes.
        """
        if not datasets:
            return None
        if len(datasets) == 1:
            return datasets[0]
        return concatenate_datasets(datasets)

    @staticmethod
    def interleave_datasets(datasets: List[DATASET_TYPE], **kwargs) -> Optional[DATASET_TYPE]:
        """Interleave with per-dataset probabilities; same short-circuits as ``concat_datasets``."""
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

        A larger-than-length request is not an error: the surplus is full copies plus a remainder.
        ``shuffle`` governs only the remainder -- off, the first rows (reproducible without a seed);
        on, a random draw. The whole-copy part is never shuffled: it holds every row regardless.
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

        The per-path mechanics (backend detection, the ``#N`` budget, the 3.x `length`->`lengths`
        rename, the delete filter) live in ``load_dataset_store``; this loops it over the train
        and val path lists, returning lists (possibly empty) that the per-split builder merges with
        the freshly-encoded split, so a cache-only run still produces a loader.
        """
        from swift.dev.dataset.storage import load_dataset_store  # TODO: point at dev2 once migrated
        train_datasets: List[DATASET_TYPE] = []
        val_datasets: List[DATASET_TYPE] = []
        for paths, out in ((cached_dataset, train_datasets), (cached_val_dataset, val_datasets)):
            for path in (paths or []):
                out.append(
                    load_dataset_store(
                        path,
                        max_length=max_length,
                        truncation_strategy=truncation_strategy,
                        data_seed=data_seed,
                        shuffle=shuffle))
        return train_datasets, val_datasets

    @staticmethod
    def parse_dataset_syntax(entry: str) -> Tuple[str, List[str], Optional[int], Optional[bool]]:
        """Pull the one-string dataset syntax apart into the plain loader fields.

        ``hf://org/name:sub1/sub2#500`` selects hub ``hf``, id ``org/name``, subsets ``sub1``/``sub2``,
        a 500-row budget. Returns ``(dataset, subsets, sample_count, use_hf)`` (``None``/empty when
        the string pins/names none). An existing path returns untouched -- its drive letter or file
        name can contain the split characters -- re-checked after the budget comes off.
        """
        if os.path.exists(entry):
            return entry, [], None, None

        use_hf: Optional[bool] = None
        hub, sep, rest = entry.partition('://')
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

        A stream cannot be permuted, only reordered within a window, so ``buffer_size`` is the
        whole extent of the shuffle there and does nothing for a materialised dataset.
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

        A stream has no length, so a *fraction* only means something once a row budget bounds it:
        given ``#N``, the first ``N * ratio`` rows become validation and the remainder training.
        Without a budget only the degenerate ratios have an answer -- 0, no validation set, and 1,
        the whole stream is one.
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

        An in-memory dataset (a fresh ``map`` result, a synthesised split) writes its Arrow file to
        a temp directory defaulting to ``/tmp`` -- routinely a small tmpfs in a container, where a
        real dataset fills it and the run dies far from the cause. Legacy did the same as an import
        side effect; here the caller about to create those files asks for it.
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
                # Class-level pool, not a context manager: the path must outlive this call; the
                # interpreter cleans the dir up when the object drops at exit.
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

    A name matched against the registry picks a ``DatasetLoader``; an unmatched name is normal
    (a bare hub id, a local file) and loads through the base loader. Train parts concatenate by
    default (likewise val); ``interleave_prob`` samples between them instead, ``shuffle`` applies
    last (a buffer window when streaming), and ``hf://org/name:sub#500`` wins per dataset.
    """
    if isinstance(datasets, str):
        datasets = [datasets]
    DatasetLoader.use_swift_cache_for_temp_files()
    if streaming:
        # `num_proc` distributes a `map` over shards of a materialised table; a stream is consumed
        # by one process and `datasets` rejects the pair.
        num_proc = None

    train_parts: List[DATASET_TYPE] = []
    val_parts: List[DATASET_TYPE] = []
    for entry in datasets:
        dataset, entry_subsets, sample_count, entry_use_hf = DatasetLoader.parse_dataset_syntax(entry)
        if entry_use_hf is None:
            entry_use_hf = use_hf
        dataset_type = match_dataset_type(dataset, use_hf=bool(entry_use_hf))
        loader_cls = get_dataset_loader(dataset_type)
        # A path-only dataset (a local `dataset_path`, no hub ids) is selected by NAME; resolve the
        # name to the path before detect_source() calls it a hub id. Hub-id loaders stay untouched.
        load_id = dataset
        if loader_cls.dataset_paths and not loader_cls.datasets and not os.path.exists(dataset):
            load_id = loader_cls.dataset_paths[0]
        info = DatasetInfo(
            dataset=load_id,
            dataset_type=dataset_type,
            source=DatasetLoader.detect_source(load_id, entry_use_hf),
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
            model_author=model_author,
            seed=seed)
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

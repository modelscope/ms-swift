import os
import platform
import re
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

from .dataset_meta import DATASET_MAPPING, DatasetMeta

_dataset_meta_mapping = None


@dataclass
class DatasetSyntax:
    dataset: str
    subsets: List[str] = field(default_factory=list)
    dataset_sample: Optional[int] = None
    use_hf: Optional[bool] = None

    def __post_init__(self):
        if os.path.isfile(self.dataset):
            self.dataset_type = 'path'
        else:  # dataset_id or dataset_dir
            self.dataset_type = 'repo'

    def get_raw(self):
        subsets = '/'.join(self.subsets)
        dataset_sample = '' if self.dataset_sample is None else f'#{self.dataset_sample}'
        return f'{self.dataset}{subsets}{dataset_sample}'

    @staticmethod
    def _safe_split(s: str,
                    sep: str,
                    use_0: bool,
                    split_mode: Literal['left', 'right'] = 'left') -> Tuple[Optional[str], Optional[str]]:
        """
        use_0: When the length of the part is 1, is it considered as part0 or part1.
        split_mode: use split or rsplit
        """
        if s is None or len(s) == 0:
            return None, None
        if split_mode == 'left':
            part = s.split(sep, 1)
        else:
            part = s.rsplit(sep, 1)
        if len(part) == 1:
            if use_0:
                part = part[0], None
            else:
                part = None, part[0]
        else:
            assert len(part) == 2
        return part

    @classmethod
    def parse(cls, dataset: str) -> 'DatasetSyntax':
        """Parse the dataset from the command line"""
        # hf/ms::dataset_id or dataset_path:subset1/subset2/subset3#dataset_sample
        if os.path.exists(dataset):
            use_hf = None
        else:
            use_hf, dataset = cls._safe_split(dataset, '::', False)
            if isinstance(use_hf, str):
                use_hf = use_hf.lower()
            use_hf = {'hf': True, 'ms': False}.get(use_hf)
        if os.path.exists(dataset):
            other, dataset_sample = dataset, None
        else:
            other, dataset_sample = cls._safe_split(dataset, '#', True, 'right')
        if os.path.exists(other):
            dataset, subsets = other, None
        else:
            dataset, subsets = cls._safe_split(other, ':', True)

        if subsets is not None:
            subsets = [subset.strip() for subset in subsets.split('/')]
        if dataset_sample is not None:
            dataset_sample = int(dataset_sample)
        return cls(dataset.strip(), subsets or [], dataset_sample, use_hf)

    def get_dataset_meta(self, use_hf: bool):
        dataset_meta_mapping = self._get_dataset_meta_mapping()
        dataset_type = self.dataset_type
        if dataset_type == 'path':
            dataset_meta = dataset_meta_mapping.get((dataset_type, self.dataset))
        else:
            dataset_type = 'repo' if os.path.isdir(self.dataset) else {True: 'hf', False: 'ms'}[use_hf]
            dataset_meta = dataset_meta_mapping.get((dataset_type, self.dataset))
        return dataset_meta or self._get_matched_dataset_meta(dataset_meta_mapping) or DatasetMeta()

    @staticmethod
    def _get_dataset_meta_mapping() -> Dict[Tuple[str, str], DatasetMeta]:
        global _dataset_meta_mapping
        if _dataset_meta_mapping is not None:
            return _dataset_meta_mapping
        _dataset_meta_mapping = {}
        for dataset_meta in DATASET_MAPPING.values():
            if dataset_meta.dataset_path is not None:
                dataset_type = 'repo' if os.path.isdir(dataset_meta.dataset_path) else 'path'
                _dataset_meta_mapping[(dataset_type, dataset_meta.dataset_path)] = dataset_meta
            if dataset_meta.ms_dataset_id is not None:
                _dataset_meta_mapping[('ms', dataset_meta.ms_dataset_id)] = dataset_meta
            if dataset_meta.hf_dataset_id is not None:
                _dataset_meta_mapping[('hf', dataset_meta.hf_dataset_id)] = dataset_meta
        return _dataset_meta_mapping

    @staticmethod
    def get_dataset_name(dataset_id: str) -> str:
        # compat hf hub
        dataset_id = dataset_id.rstrip('/')
        match_ = re.search('/datasets--.+?--(.+?)/snapshots/', dataset_id)
        if match_ is not None:
            return match_.group(1)

        dataset_name = dataset_id.rsplit('/', 1)[-1]
        if platform.system().lower() == 'windows':
            dataset_name = dataset_name.rsplit('\\', 1)[-1]
        return dataset_name

    def _get_matched_dataset_meta(self, dataset_meta_mapping):
        suffix_dataset_meta_mapping = {}
        for dataset_name, dataset_meta in dataset_meta_mapping.items():
            dataset_name = self.get_dataset_name(dataset_name[1])
            suffix_dataset_meta_mapping[dataset_name] = dataset_meta
        dataset_name = self.get_dataset_name(self.dataset)
        dataset_meta = suffix_dataset_meta_mapping.get(dataset_name)
        return dataset_meta

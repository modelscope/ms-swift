# Copyright (c) ModelScope Contributors. All rights reserved.
"""Loaders for the ``open_jev_*`` decision-distillation corpora.

One family per typed-decision model family: the corpus row shape is model-specific, so each
loader pairs the hub repo with the matching ``ScoringPreprocessor`` subclass, which fans a raw
record out into the standard decision row (``messages`` + ``questions``/``kinds``/``options``/
``target_probs``). The ``_v11`` families read the smaller Open-Jev-v1.1 release, which publishes
a single subset.
"""
from .base import DatasetLoader, register_dataset
from ..preprocessor import (
    ClefPreprocessor,
    JevPreprocessor,
    KevPreprocessor,
    LayaPreprocessor,
    OmniJevPreprocessor,
    StartLuxPreprocessor,
)

_OPEN_JEV_SUBSETS = [
    'release-v2-redistributable',
    'browser-drone-expansion-v1-redistributable',
    'citation-control-v1',
    'entity-alignment-control-v1',
    'amount-extraction-control-v1',
    'email-selection-control-v1',
    'phone-extraction-control-v1',
    'context-retention-control-v1',
    'sponsor-segment-control-v1',
    'silent-failure-control-v1',
    'ir-control-v1',
    'mailroom-control-v1',
]


@register_dataset
class OpenJevJevLoader(DatasetLoader):
    dataset_type = 'open_jev_jev'
    datasets = [('ZefanCai/Open-Jev', None)]
    subsets = list(_OPEN_JEV_SUBSETS)
    preprocessor = JevPreprocessor
    tags = ['decision', 'jev']


@register_dataset
class OpenJevJevV11Loader(DatasetLoader):
    dataset_type = 'open_jev_jev_v11'
    datasets = [('ZefanCai/Open-Jev-v1.1', None)]
    subsets = ['community-hard-mix-v2-redistributable']
    preprocessor = JevPreprocessor
    tags = ['decision', 'jev']


@register_dataset
class OpenJevClefLoader(DatasetLoader):
    dataset_type = 'open_jev_clef'
    datasets = [('ZefanCai/Open-Jev', None)]
    subsets = list(_OPEN_JEV_SUBSETS)
    preprocessor = ClefPreprocessor
    tags = ['decision', 'clef']


@register_dataset
class OpenJevClefV11Loader(DatasetLoader):
    dataset_type = 'open_jev_clef_v11'
    datasets = [('ZefanCai/Open-Jev-v1.1', None)]
    subsets = ['community-hard-mix-v2-redistributable']
    preprocessor = ClefPreprocessor
    tags = ['decision', 'clef']


@register_dataset
class OpenJevOmniejvLoader(DatasetLoader):
    dataset_type = 'open_jev_omnijev'
    datasets = [('ZefanCai/Open-Jev', None)]
    subsets = list(_OPEN_JEV_SUBSETS)
    preprocessor = OmniJevPreprocessor
    tags = ['decision', 'omnijev']


@register_dataset
class OpenJevOmniejvV11Loader(DatasetLoader):
    dataset_type = 'open_jev_omnijev_v11'
    datasets = [('ZefanCai/Open-Jev-v1.1', None)]
    subsets = ['community-hard-mix-v2-redistributable']
    preprocessor = OmniJevPreprocessor
    tags = ['decision', 'omnijev']


@register_dataset
class OpenJevKevLoader(DatasetLoader):
    dataset_type = 'open_jev_kev'
    datasets = [('ZefanCai/Open-Jev', None)]
    subsets = list(_OPEN_JEV_SUBSETS)
    preprocessor = KevPreprocessor
    tags = ['decision', 'kev']


@register_dataset
class OpenJevKevV11Loader(DatasetLoader):
    dataset_type = 'open_jev_kev_v11'
    datasets = [('ZefanCai/Open-Jev-v1.1', None)]
    subsets = ['community-hard-mix-v2-redistributable']
    preprocessor = KevPreprocessor
    tags = ['decision', 'kev']


@register_dataset
class OpenJevLayaLoader(DatasetLoader):
    dataset_type = 'open_jev_laya'
    datasets = [('ZefanCai/Open-Jev', None)]
    subsets = list(_OPEN_JEV_SUBSETS)
    preprocessor = LayaPreprocessor
    tags = ['decision', 'laya']


@register_dataset
class OpenJevLayaV11Loader(DatasetLoader):
    dataset_type = 'open_jev_laya_v11'
    datasets = [('ZefanCai/Open-Jev-v1.1', None)]
    subsets = ['community-hard-mix-v2-redistributable']
    preprocessor = LayaPreprocessor
    tags = ['decision', 'laya']


@register_dataset
class OpenJevStartluxLoader(DatasetLoader):
    dataset_type = 'open_jev_startlux'
    datasets = [('ZefanCai/Open-Jev', None)]
    subsets = list(_OPEN_JEV_SUBSETS)
    preprocessor = StartLuxPreprocessor
    tags = ['decision', 'startlux']


@register_dataset
class OpenJevStartluxV11Loader(DatasetLoader):
    dataset_type = 'open_jev_startlux_v11'
    datasets = [('ZefanCai/Open-Jev-v1.1', None)]
    subsets = ['community-hard-mix-v2-redistributable']
    preprocessor = StartLuxPreprocessor
    tags = ['decision', 'startlux']

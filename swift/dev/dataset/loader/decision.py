# Copyright (c) ModelScope Contributors. All rights reserved.
"""Loaders for the decision corpora: the ``open_jev_*`` hub releases and the offline mini samples.

One family per typed-decision model family: the corpus row shape is model-specific, so each
loader pairs the data source with the matching ``ScoringPreprocessor`` subclass, which fans a raw
record out into the standard decision row (``messages`` + ``questions``/``kinds``/``options``/
``target_probs``). The ``_v11`` families read the smaller Open-Jev-v1.1 release, which publishes
a single subset.

The ``<family>_decision`` loaders read the OFFLINE mini corpora shipped with swift under
``swift/dev/dataset/decision_data`` (a mirror of ``examples/train/decision/data``): a few rows per
family, loadable with ``--dataset jev_decision`` (no ``--external_plugins``, no network) for smoke
runs and the example scripts.
"""
import os

from .base import DatasetLoader, register_dataset
from ..preprocessor import (
    ClefPreprocessor,
    JevPreprocessor,
    KevPreprocessor,
    LayaPreprocessor,
    OmniJevPreprocessor,
    StartLuxPreprocessor,
    Thx02Preprocessor,
)

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'decision_data')


@register_dataset
class JevDecisionLoader(DatasetLoader):
    dataset_type = 'jev_decision'
    dataset_paths = [os.path.join(_DATA_DIR, 'jev.jsonl')]
    preprocessor = JevPreprocessor
    tags = ['decision', 'jev']


@register_dataset
class ClefDecisionLoader(DatasetLoader):
    dataset_type = 'clef_decision'
    dataset_paths = [os.path.join(_DATA_DIR, 'clef.jsonl')]
    preprocessor = ClefPreprocessor
    tags = ['decision', 'clef']


@register_dataset
class KevDecisionLoader(DatasetLoader):
    dataset_type = 'kev_decision'
    dataset_paths = [os.path.join(_DATA_DIR, 'kev.jsonl')]
    preprocessor = KevPreprocessor
    tags = ['decision', 'kev']


@register_dataset
class LayaDecisionLoader(DatasetLoader):
    dataset_type = 'laya_decision'
    dataset_paths = [os.path.join(_DATA_DIR, 'laya.jsonl')]
    preprocessor = LayaPreprocessor
    tags = ['decision', 'laya']


@register_dataset
class OmniJevDecisionLoader(DatasetLoader):
    dataset_type = 'omnijev_decision'
    dataset_paths = [os.path.join(_DATA_DIR, 'omnijev.jsonl')]
    preprocessor = OmniJevPreprocessor
    tags = ['decision', 'omnijev']


@register_dataset
class StartLuxDecisionLoader(DatasetLoader):
    dataset_type = 'startlux_decision'
    dataset_paths = [os.path.join(_DATA_DIR, 'startlux.jsonl')]
    preprocessor = StartLuxPreprocessor
    tags = ['decision', 'startlux']


@register_dataset
class Thx02DecisionLoader(DatasetLoader):
    # thx02 is not in the controlled tag vocabulary; the family tag is deliberately omitted.
    dataset_type = 'thx02_decision'
    dataset_paths = [os.path.join(_DATA_DIR, 'thx02.jsonl')]
    preprocessor = Thx02Preprocessor
    tags = ['decision']


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

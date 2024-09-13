# Copyright (c) Alibaba, Inc. and its affiliates.
import os

import datasets.fingerprint

from swift.llm.dataset.media import MediaCache, MediaTag
from swift.llm.dataset.preprocess import (AlpacaPreprocessor, ClsPreprocessor, ComposePreprocessor,
                                          ConversationsPreprocessor,
                                          ListPreprocessor, PreprocessFunc, RenameColumnsPreprocessor,
                                          SmartPreprocessor,
                                          TextGenerationPreprocessor, preprocess_sharegpt)
from swift.utils.torch_utils import _find_local_mac


def _update_fingerprint_mac(*args, **kwargs):
    mac = _find_local_mac().replace(':', '')
    fp = datasets.fingerprint._update_fingerprint(*args, **kwargs)
    fp += '-' + mac
    if len(fp) > 64:
        fp = fp[:64]
    return fp


datasets.fingerprint._update_fingerprint = datasets.fingerprint.update_fingerprint
datasets.fingerprint.update_fingerprint = _update_fingerprint_mac
datasets.arrow_dataset.update_fingerprint = _update_fingerprint_mac


def partialed_map(self, *args, **kwargs):
    if 'num_proc' not in kwargs:
        num_proc = os.environ.get('DATASET_MAP_NPROC')
        kwargs['num_proc'] = int(num_proc) if num_proc else num_proc
    return self._origin_map(*args, **kwargs)


datasets.Dataset._origin_map = datasets.Dataset.map
datasets.Dataset.map = partialed_map

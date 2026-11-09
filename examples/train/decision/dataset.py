# Copyright (c) ModelScope Contributors. All rights reserved.
import os

from swift.dataset import DatasetMeta, register_dataset
from swift.dataset.preprocessor.decision import (ClefPreprocessor, JevPreprocessor, KevPreprocessor, LayaPreprocessor,
                                                 OmniJevPreprocessor, StartLuxPreprocessor, Thx02Preprocessor)

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

register_dataset(
    DatasetMeta(
        dataset_name='clef_decision',
        dataset_path=os.path.join(_DATA_DIR, 'clef.jsonl'),
        preprocess_func=ClefPreprocessor(),
    ),
    exist_ok=True)

register_dataset(
    DatasetMeta(
        dataset_name='jev_decision',
        dataset_path=os.path.join(_DATA_DIR, 'jev.jsonl'),
        preprocess_func=JevPreprocessor(),
    ),
    exist_ok=True)

register_dataset(
    DatasetMeta(
        dataset_name='omnijev_decision',
        dataset_path=os.path.join(_DATA_DIR, 'omnijev.jsonl'),
        preprocess_func=OmniJevPreprocessor(),
    ),
    exist_ok=True)

register_dataset(
    DatasetMeta(
        dataset_name='kev_decision',
        dataset_path=os.path.join(_DATA_DIR, 'kev.jsonl'),
        preprocess_func=KevPreprocessor(),
    ),
    exist_ok=True)

register_dataset(
    DatasetMeta(
        dataset_name='thx02_decision',
        dataset_path=os.path.join(_DATA_DIR, 'thx02.jsonl'),
        preprocess_func=Thx02Preprocessor(),
    ),
    exist_ok=True)

register_dataset(
    DatasetMeta(
        dataset_name='laya_decision',
        dataset_path=os.path.join(_DATA_DIR, 'laya.jsonl'),
        preprocess_func=LayaPreprocessor(),
    ),
    exist_ok=True)

register_dataset(
    DatasetMeta(
        dataset_name='startlux_decision',
        dataset_path=os.path.join(_DATA_DIR, 'startlux.jsonl'),
        preprocess_func=StartLuxPreprocessor(),
    ),
    exist_ok=True)

if __name__ == '__main__':
    # Quick self-check: load each dataset and print the normalized decision row contract.
    from swift.dataset import load_dataset
    for name in ('clef_decision', 'jev_decision', 'omnijev_decision', 'kev_decision', 'thx02_decision', 'laya_decision',
                 'startlux_decision'):
        ds = load_dataset([name], num_proc=1)[0]
        print(f'==== {name}: {len(ds)} rows ====')
        row = ds[0]
        for k in ('messages', 'questions', 'kinds', 'options', 'target_probs', 'images'):
            if k in row:
                print(f'  {k}: {row[k]}')

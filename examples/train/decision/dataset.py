# Copyright (c) ModelScope Contributors. All rights reserved.
"""External plugin: register the three typed-decision mini datasets used by these examples.

Each model has its own `ScoringPreprocessor` subclass (see `swift/dataset/preprocessor/decision.py`)
because the three corpora differ in raw shape:
  * Clef    -- ONE record -> ONE row carrying ALL its questions (joint decode); native
               `{state, questions: {qid: {type, instructions, criteria}}, answers}` schema.
  * JEV     -- ONE record -> ONE row PER question (s1_pass scores one question per forward); unified
               `{state, questions: [{kind, question, options, gold}]}` schema.
  * OmniJev -- ONE record -> ONE row PER question, multimodal state (image / pre-composed video
               mosaic); unified option blocks with region / abstain support.

Pass this file to `swift sft --external_plugins examples/train/decision/dataset.py` and reference a
dataset by its registered name (`clef_decision` / `jev_decision` / `omnijev_decision`).
"""
import os

from swift.dataset import DatasetMeta, register_dataset
from swift.dataset.preprocessor.decision import ClefPreprocessor, JevPreprocessor, OmniJevPreprocessor

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

if __name__ == '__main__':
    # Quick self-check: load each dataset and print the normalized decision row contract.
    from swift.dataset import load_dataset
    for name in ('clef_decision', 'jev_decision', 'omnijev_decision', 'generic_decision'):
        ds = load_dataset([name], num_proc=1)[0]
        print(f'==== {name}: {len(ds)} rows ====')
        row = ds[0]
        for k in ('messages', 'questions', 'kinds', 'options', 'target_probs', 'images'):
            if k in row:
                print(f'  {k}: {row[k]}')

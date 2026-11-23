# Copyright (c) ModelScope Contributors. All rights reserved.
"""End-to-end text-dataset tests: a real file on disk all the way to a collated batch of tokens.

The other dataset tests each stop short of this seam. ``test_api.py`` mocks ``load_dataset`` and the
template, so it checks wiring; ``test_swift_dataset.py`` starts from rows already in memory, so it
never touches the loader; ``test_format_converter.py`` starts from rows in memory too. What is left
unproven is the whole chain at once -- the one a user actually runs:

    a file -> load_dataset (detect source, read by extension, auto-detect the format, preprocess)
           -> standard ``messages`` rows -> template.encode -> integer ``input_ids`` -> a padded batch.

If this runs, a plain text dataset is usable end to end. The loader-level assertions (no model needed)
pin the load mechanics; the encode/batch assertions need the local model and are skipped without it.
"""
import json
import os

import pytest
from datasets import Dataset as HfDataset

MODEL = os.environ.get('SWIFT_TEST_MODEL', '/mnt/workspace/yzhao/tastelikefeet/Qwen3.5-4B-CM-v2')
MODEL_TYPE = os.environ.get('SWIFT_TEST_MODEL_TYPE', 'qwen3_5')

needs_model = pytest.mark.skipif(not os.path.isdir(MODEL), reason=f'no local model at {MODEL}')

# Response-format rows (auto-detected): varied-length text so the collated batch has to pad.
ROWS = [
    {'query': 'hello there, how are you doing today', 'response': 'fine'},
    {'query': 'hi', 'response': 'ok'},
    {'query': 'give me a somewhat longer answer please', 'response': 'sure, here is a longer answer for you'},
    {'query': 'and one more', 'response': 'done'},
]


def write_jsonl(path, rows):
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    return str(path)


def write_csv(path, rows):
    import csv
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


@pytest.fixture(scope='module')
def processor():
    from swift.model import get_model_processor
    return get_model_processor(MODEL, load_model=False, model_type=MODEL_TYPE)[1]


def build_template(processor, max_length=8192):
    from swift.template import get_template
    template = get_template(processor, template_type='qwen2_5', max_length=max_length)
    template.set_mode('train')
    return template


# ---- the load mechanics: a real file -> standard messages rows (no model needed) --------------


def test_load_jsonl_file_reaches_standard_messages(tmp_path):
    from swift.dev.dataset import load_dataset
    path = write_jsonl(tmp_path / 'a3_text.jsonl', ROWS)
    train, val = load_dataset([path])
    assert val is None, 'no split ratio was asked for'
    assert isinstance(train, HfDataset) and len(train) == len(ROWS)
    first = train[0]['messages']
    assert first == [{'role': 'user', 'content': ROWS[0]['query']},
                     {'role': 'assistant', 'content': ROWS[0]['response']}]


def test_load_csv_file_reaches_standard_messages(tmp_path):
    from swift.dev.dataset import load_dataset
    path = write_csv(tmp_path / 'a3_text.csv', ROWS)
    train, _ = load_dataset([path])
    assert len(train) == len(ROWS)
    assert train[0]['messages'][0]['content'] == ROWS[0]['query']


def test_split_ratio_carves_off_a_validation_set(tmp_path):
    from swift.dev.dataset import load_dataset
    path = write_jsonl(tmp_path / 'a3_split.jsonl', ROWS * 5)
    train, val = load_dataset([path], split_dataset_ratio=0.2)
    assert train is not None and val is not None
    assert len(train) + len(val) == len(ROWS) * 5
    assert len(val) == len(ROWS), 'one fifth of twenty rows'


def test_row_budget_suffix_caps_the_load(tmp_path):
    from swift.dev.dataset import load_dataset
    path = write_jsonl(tmp_path / 'a3_budget.jsonl', ROWS * 5)
    train, _ = load_dataset([f'{path}#3'])
    assert len(train) == 3


def test_caller_column_renames_reach_the_converter(tmp_path):
    """A file whose fields no format recognises, renamed by the caller into the response format."""
    from swift.dev.dataset import load_dataset
    rows = [{'q': r['query'], 'a': r['response']} for r in ROWS]
    path = write_jsonl(tmp_path / 'a3_renamed.jsonl', rows)
    train, _ = load_dataset([path], columns={'q': 'query', 'a': 'response'})
    assert train[0]['messages'][0]['content'] == ROWS[0]['query']


# ---- the full chain: file -> messages -> input_ids -> a padded batch (needs the model) ---------


@needs_model
def test_loaded_rows_encode_to_input_ids(processor, tmp_path):
    from swift.dev.dataset import SwiftDataset, load_dataset
    path = write_jsonl(tmp_path / 'a3_encode.jsonl', ROWS)
    train, _ = load_dataset([path])
    ds = SwiftDataset(train, build_template(processor), load_from_cache_file=False)
    assert len(ds) == len(ROWS)
    row = ds[0]
    assert row['input_ids'] and all(isinstance(token, int) for token in row['input_ids'])
    assert row['labels'] and len(row['labels']) == len(row['input_ids'])
    # Measuring reads the encoded lengths without changing the row count.
    assert len(ds.lengths) == len(ROWS)
    assert all(length > 0 for length in ds.lengths)


@needs_model
def test_dataloader_serves_a_padded_batch(processor, tmp_path):
    """The last link: a real DataLoader over the encoded dataset, collated by the real template.

    The rows differ in length, so a correct collate must pad every sequence to the batch maximum --
    rectangular tensors, an attention mask that marks the padding, and labels that ignore it.
    """
    from torch.utils.data import DataLoader

    from swift.dev.dataset import SwiftDataset, load_dataset
    path = write_jsonl(tmp_path / 'a3_batch.jsonl', ROWS)
    train, _ = load_dataset([path])
    template = build_template(processor)
    ds = SwiftDataset(train, template, load_from_cache_file=False)

    loader = DataLoader(ds, batch_size=len(ROWS), shuffle=False, collate_fn=template.data_collator)
    batch = next(iter(loader))

    n = len(ROWS)
    assert batch['input_ids'].shape[0] == n
    assert batch['input_ids'].shape == batch['labels'].shape == batch['attention_mask'].shape
    # The longest row is unpadded, so the batch width equals its length; shorter rows are padded out.
    widths = {len(ds[i]['input_ids']) for i in range(n)}
    assert batch['input_ids'].shape[1] == max(widths)
    assert len(widths) > 1, 'rows must differ in length, or the padding is never exercised'
    # Every real token is attended to and every pad is not, so the mask sums to the unpadded total.
    assert batch['attention_mask'].sum().item() == sum(len(ds[i]['input_ids']) for i in range(n))

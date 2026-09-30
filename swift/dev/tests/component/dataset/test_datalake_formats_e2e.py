# Copyright (c) ModelScope Contributors. All rights reserved.
"""End-to-end tests for the columnar data-lake formats the loader resolves itself.

``parquet`` and ``arrow`` already load through ``datasets`` -- a single file by its extension, a
directory by folder auto-detection -- so they appear here only as regression guards for the supported
set. The two formats that needed new loader code are the point of this file:

- ``orc``: ``datasets`` ships no builder at all, so the loader reads it through ``pyarrow.orc`` -- a
  single file, or a directory of shards concatenated in sorted order -- and hands the rows to the same
  preprocessing / encode chain every other dataset goes through.
- ``lance``: ``datasets`` has a builder, but it is only reached by name and the folder scanner never
  selects it for a ``.lance`` directory, so the loader routes to it explicitly. The builder needs
  ``pylance``; when that is missing the loader fails loudly with an install hint rather than surfacing
  an opaque builder error.

Every format is driven end to end through ``swift.dev.dataset.load_dataset`` to standard ``messages``
rows, and orc -- the self-written path -- is carried all the way to encoded ``input_ids`` to prove the
rows it produces are ordinary dataset rows, not a special case downstream has to know about.
"""
import os

import pyarrow as pa
import pyarrow.orc as orc
import pyarrow.parquet as pq
import pytest

MODEL = os.environ.get('SWIFT_TEST_MODEL', '/mnt/workspace/yzhao/tastelikefeet/Qwen3.5-4B-CM-v2')
MODEL_TYPE = os.environ.get('SWIFT_TEST_MODEL_TYPE', 'qwen3_5')

needs_model = pytest.mark.skipif(not os.path.isdir(MODEL), reason=f'no local model at {MODEL}')

# Response-format rows: the loader auto-detects query/response and converts them to standard messages.
ROWS = [
    {'query': 'hello there', 'response': 'hi'},
    {'query': 'how are you doing', 'response': 'fine thanks'},
    {'query': 'one more please', 'response': 'done'},
]
TABLE = pa.table({'query': [r['query'] for r in ROWS], 'response': [r['response'] for r in ROWS]})


# ---- writers: one per format, single file and a directory of two shards ------------------------


def write_parquet(path, table):
    pq.write_table(table, str(path))
    return str(path)


def write_arrow(path, table):
    _write_arrow_ipc(table, path)
    return str(path)


def write_orc(path, table):
    orc.write_table(table, str(path))
    return str(path)


def _write_arrow_ipc(table, path):
    # Feather V2 *is* the Arrow IPC file format; write it directly rather than via the deprecated
    # pyarrow.feather alias.
    with pa.OSFile(str(path), 'wb') as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)


_WRITERS = {'parquet': pq.write_table, 'arrow': _write_arrow_ipc, 'orc': orc.write_table}


def write_shards(directory, ext, table):
    """Split ``table`` into two shards under ``directory``, so a directory load has to concatenate."""
    directory = str(directory)
    os.makedirs(directory, exist_ok=True)
    writer = _WRITERS[ext]
    writer(table.slice(0, 2), os.path.join(directory, f'part0.{ext}'))
    writer(table.slice(2, 1), os.path.join(directory, f'part1.{ext}'))
    return directory


def assert_standard_messages(dataset):
    """The loaded rows reached the standard ``messages`` form, in dataset order."""
    assert len(dataset) == len(ROWS)
    assert dataset.column_names == ['messages']
    for i, row in enumerate(ROWS):
        assert dataset[i]['messages'] == [{'role': 'user', 'content': row['query']},
                                          {'role': 'assistant', 'content': row['response']}]


# ---- format detection: only orc and lance are intercepted -------------------------------------


def test_resolve_lake_format_flags_only_orc_and_lance(tmp_path):
    """The resolver intercepts exactly the two formats `datasets` cannot route on its own."""
    from swift.dev.dataset.loader.base import DatasetLoader

    orc_file = write_orc(tmp_path / 'a.orc', TABLE)
    orc_dir = write_shards(tmp_path / 'orc_dir', 'orc', TABLE)
    parquet_file = write_parquet(tmp_path / 'a.parquet', TABLE)
    parquet_dir = write_shards(tmp_path / 'parquet_dir', 'parquet', TABLE)

    assert DatasetLoader.resolve_lake_format(orc_file) == 'orc'
    assert DatasetLoader.resolve_lake_format(orc_dir) == 'orc'
    # parquet resolves through the ordinary dispatch, so it must NOT be intercepted.
    assert DatasetLoader.resolve_lake_format(parquet_file) is None
    assert DatasetLoader.resolve_lake_format(parquet_dir) is None

    # A single .lance file, a directory of .lance files, and a lance *dataset* directory (named
    # `something.lance`, holding version/data subfolders rather than loose .lance files) all count.
    lance_file = tmp_path / 'a.lance'
    lance_file.write_bytes(b'')
    assert DatasetLoader.resolve_lake_format(str(lance_file)) == 'lance'
    lance_shard_dir = tmp_path / 'shards'
    lance_shard_dir.mkdir()
    (lance_shard_dir / 'part0.lance').write_bytes(b'')
    assert DatasetLoader.resolve_lake_format(str(lance_shard_dir)) == 'lance'
    lance_dataset_dir = tmp_path / 'mydata.lance'
    (lance_dataset_dir / '_versions').mkdir(parents=True)
    assert DatasetLoader.resolve_lake_format(str(lance_dataset_dir)) == 'lance'


# ---- orc: the self-written reader, single file / directory / streaming --------------------------


def test_orc_single_file_loads_to_messages(tmp_path):
    from swift.dev.dataset import load_dataset
    path = write_orc(tmp_path / 'single.orc', TABLE)
    train, val = load_dataset([path])
    assert val is None
    assert_standard_messages(train)


def test_orc_directory_of_shards_concatenates_in_order(tmp_path):
    """Two shards become one dataset, read in sorted filename order so the row order is stable."""
    from swift.dev.dataset import load_dataset
    directory = write_shards(tmp_path / 'orc_shards', 'orc', TABLE)
    train, _ = load_dataset([directory])
    assert_standard_messages(train)


def test_orc_streaming_yields_rows(tmp_path):
    """A streaming request still works: the table is read, then served as an iterable."""
    from swift.dev.dataset import load_dataset
    path = write_orc(tmp_path / 'stream.orc', TABLE)
    train, _ = load_dataset([path], streaming=True)
    rows = list(train)
    assert len(rows) == len(ROWS)
    assert rows[0]['messages'][0]['content'] == ROWS[0]['query']


def test_orc_directory_without_shards_fails_loudly(tmp_path):
    """A directory the resolver tagged orc but that holds no .orc file is an error, not an empty set."""
    from swift.dev.dataset.loader.base import DatasetLoader
    empty = tmp_path / 'empty.orc'
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match=r'\.orc'):
        DatasetLoader.load_orc(str(empty))


# ---- parquet / arrow: regression guards for the already-supported formats -----------------------


@pytest.mark.parametrize('ext', ['parquet', 'arrow'])
def test_columnar_formats_still_load_single_and_dir(tmp_path, ext):
    """parquet and arrow keep loading through `datasets`; the new dispatch must not disturb them."""
    from swift.dev.dataset import load_dataset
    writer = {'parquet': write_parquet, 'arrow': write_arrow}[ext]
    single = writer(tmp_path / f'single.{ext}', TABLE)
    assert_standard_messages(load_dataset([single])[0])
    directory = write_shards(tmp_path / f'{ext}_dir', ext, TABLE)
    assert_standard_messages(load_dataset([directory])[0])


# ---- the whole chain: an orc file all the way to encoded tokens (needs the model) ---------------


@needs_model
def test_orc_rows_encode_to_input_ids(tmp_path):
    """Rows read by the orc path are ordinary dataset rows: they encode like any other."""
    from swift.dev.dataset import SwiftDataset, load_dataset
    from swift.model import get_model_processor
    from swift.template import get_template

    path = write_orc(tmp_path / 'encode.orc', TABLE)
    train, _ = load_dataset([path])
    processor = get_model_processor(MODEL, load_model=False, model_type=MODEL_TYPE)[1]
    template = get_template(processor, template_type='qwen2_5', max_length=8192)
    template.set_mode('train')

    ds = SwiftDataset(train, template, load_from_cache_file=False)
    assert len(ds) == len(ROWS)
    row = ds[0]
    assert row['input_ids'] and all(isinstance(token, int) for token in row['input_ids'])
    assert len(row['labels']) == len(row['input_ids'])
    assert len(ds.lengths) == len(ROWS) and all(length > 0 for length in ds.lengths)


# ---- lance: routed to `datasets`' builder; needs pylance, fails loudly without it ---------------


def test_lance_without_pylance_fails_loudly(tmp_path):
    """With no pylance, a lance path must raise a clear install hint, not an opaque builder error."""
    import importlib.util
    if importlib.util.find_spec('lance') is not None:
        pytest.skip('pylance is installed; the missing-dependency path is not exercised here')
    from swift.dev.dataset import load_dataset
    lance_dir = tmp_path / 'mydata.lance'
    (lance_dir / '_versions').mkdir(parents=True)
    with pytest.raises(ImportError, match='pylance'):
        load_dataset([str(lance_dir)])


def test_lance_dataset_roundtrip(tmp_path):
    """Where pylance is present, a real lance dataset loads end to end through the same chain.

    Skipped without pylance. Written against the canonical ``lance.write_dataset`` API, which produces
    a ``.lance`` dataset directory -- the shape the loader's directory routing targets.
    """
    lance = pytest.importorskip('lance')
    from swift.dev.dataset import load_dataset
    path = str(tmp_path / 'data.lance')
    lance.write_dataset(TABLE, path)
    train, _ = load_dataset([path])
    assert_standard_messages(train)

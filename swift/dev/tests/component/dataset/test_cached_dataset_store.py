# Copyright (c) ModelScope Contributors. All rights reserved.
"""End-to-end tests for ``export_cached_dataset`` and the store it writes through.

``test_store.py`` pins the serialiser's contract on hand-built rows; this file drives the REAL recipe
end to end so the seams around it are proven too -- the ones a primitive-level test cannot see:

* ``export_cached_dataset`` really loads a processor (``load_model=False``, tokenizer only, CPU), builds
  a real template, and runs the SAME ``swift.dev.builders.dataset._encode`` the eager training path uses;
* ``store_encoded=False`` keeps the raw row plus a ``lengths`` column (``MeasurePreprocessor``), while
  ``store_encoded=True`` materialises real tokens (``EncodePreprocessor``) -- asserted by decoding the
  written ``input_ids`` back to the source text and checking the ``labels`` loss mask, so a template or
  tokenizer regression shows up here rather than only at train time;
* ``store_format`` really switches container (arrow directory vs a single ``.jsonl`` file), ``store_fields``
  really restricts the persisted columns, and a bad field name is refused rather than dropped;
* the write is consumable by the read half: pointing ``DatasetConfig.cached_dataset`` at the exported
  ``train`` directory and loading it through ``_load_cached_datasets`` (the exact call ``build_dataset``
  makes) returns the materialised rows;
* the guards the recipe owns fire: no dataset, streaming, a resolved-lazy mode, and multimodal +
  ``store_encoded`` (tokens without the vision tensors would train broken -- see cached_dataset.py).

Everything here is CPU-only and offline apart from the one-time tokenizer snapshot: the text model is
fetched with ``load_model=False`` (no weights), and a box that cannot fetch it skips rather than fails,
mirroring ``test_template_contract`` / ``test_chat_template_parity``.
"""
import os

import pytest

from swift.dev.tests.tiny import TinyData

#: A small TEXT-only checkpoint: ``store_encoded=True`` materialises tokens, which the recipe refuses for
#: a multimodal model, so the happy path needs a plain LM. Fetched tokenizer-only (``load_model=False``).
MODEL = os.environ.get('SWIFT_TEST_TEXT_MODEL', 'Qwen/Qwen2.5-0.5B-Instruct')
MODEL_TYPE = os.environ.get('SWIFT_TEST_TEXT_MODEL_TYPE', 'qwen2')
TEMPLATE = os.environ.get('SWIFT_TEST_TEXT_TEMPLATE', 'qwen2_5')

#: A MULTIMODAL checkpoint, used only to prove the ``store_encoded`` guard fires. Skipped when absent.
MM_MODEL = os.environ.get('SWIFT_TEST_MODEL', '/mnt/workspace/yzhao/tastelikefeet/Qwen3.5-4B-CM-v2')
MM_MODEL_TYPE = os.environ.get('SWIFT_TEST_MODEL_TYPE', 'qwen3_5')


@pytest.fixture(scope='module')
def tokenizer():
    """The real tokenizer (no weights). Requesting it is what gates the module on fetchability."""
    from swift.model import get_model_processor
    try:
        _, proc = get_model_processor(MODEL, load_model=False, model_type=MODEL_TYPE)
    except Exception as exc:  # noqa: BLE001 -- an unfetchable tokenizer is not a recipe defect
        pytest.skip(f'{MODEL}: tokenizer not fetchable ({type(exc).__name__})')
    return proc


@pytest.fixture
def sft_file(tmp_path):
    return TinyData.sft(str(tmp_path / 'sft.jsonl'), n=6)


def _export(tmp_path,
            dataset_file,
            *,
            store_encoded=False,
            store_format='arrow',
            store_fields=None,
            split_dataset_ratio=0.0,
            template_mode='train',
            lazy_tokenize=None,
            streaming=False):
    """Drive the real ``export_cached_dataset`` recipe over a local dataset file."""
    from swift.dev.config import DatasetConfig, ModelConfig, TemplateConfig
    from swift.dev.recipe.cached_dataset import export_cached_dataset

    model_config = ModelConfig(model=MODEL, model_type=MODEL_TYPE)
    template_config = TemplateConfig(template=TEMPLATE, max_length=2048)
    dataset_config = DatasetConfig(
        dataset=[dataset_file],
        dataset_num_proc=1,
        dataset_shuffle=False,
        split_dataset_ratio=split_dataset_ratio,
        lazy_tokenize=lazy_tokenize,
        streaming=streaming,
    )
    out_dir = str(tmp_path / 'cached')
    return export_cached_dataset(
        model_config,
        template_config,
        dataset_config,
        output_dir=out_dir,
        template_mode=template_mode,
        store_format=store_format,
        store_encoded=store_encoded,
        store_fields=store_fields,
    )


def _read(path):
    """Read a written store back through the shared reader."""
    from swift.dev.dataset.store import load_dataset_store
    return load_dataset_store(path)


# --- happy path: the two encoding modes ------------------------------------------------


def test_default_store_keeps_raw_row_plus_lengths(tokenizer, sft_file, tmp_path):
    """``store_encoded=False`` (the default) mirrors the eager training path: ``MeasurePreprocessor``
    keeps the raw ``messages`` and adds only ``lengths`` -- the tokens are NOT materialised, so they are
    still computed per batch at train time."""
    train_dir, val_dir = _export(tmp_path, sft_file)
    assert val_dir is None  # split_dataset_ratio defaults to 0
    assert os.path.isdir(train_dir)  # arrow -> a save_to_disk directory

    ds = _read(train_dir)
    assert len(ds) == 6
    assert 'messages' in ds.column_names and 'lengths' in ds.column_names
    assert 'input_ids' not in ds.column_names  # not materialised
    # lengths is measured per row and stays List[int] (a real token count, so > 0).
    assert all(isinstance(x, list) and x and isinstance(x[0], int) for x in ds['lengths'])


def test_store_encoded_materialises_real_tokens(tokenizer, sft_file, tmp_path):
    """``store_encoded=True`` runs the full ``EncodePreprocessor``: rows carry ``input_ids`` / ``labels``
    and train with no re-encoding. Proven end to end by decoding the written tokens back to the source
    text and checking the loss mask, so a template/tokenizer regression surfaces here."""
    train_dir, _ = _export(tmp_path, sft_file, store_encoded=True)
    ds = _read(train_dir)
    assert len(ds) == 6
    for col in ('input_ids', 'labels', 'lengths'):
        assert col in ds.column_names, col

    row = ds[0]
    assert isinstance(row['input_ids'], list) and row['input_ids']
    assert len(row['labels']) == len(row['input_ids'])
    # The tokens are a real encoding of the source row: decoding them reproduces the prompt text.
    decoded = tokenizer.decode(row['input_ids'])
    assert '2+2' in decoded  # TinyData.PROMPTS[0] == 'What is 2+2?'
    # labels are a real loss mask: the prompt is masked (-100) and the response is learned (some != -100).
    assert -100 in row['labels']
    assert any(label != -100 for label in row['labels'])
    # lengths is the encoded token count, kept List[int] on the materialised path too.
    assert row['lengths'] == [len(row['input_ids'])]


# --- container + column allow-list ----------------------------------------------------


def test_jsonl_format_writes_a_single_file(tokenizer, sft_file, tmp_path):
    """``store_format='jsonl'`` writes one ``.jsonl`` file (path carries the extension a reader detects
    the backend from) instead of an arrow directory, and it reads back identically."""
    train_dir, _ = _export(tmp_path, sft_file, store_encoded=True, store_format='jsonl')
    assert train_dir.endswith('.jsonl')
    assert os.path.isfile(train_dir) and not os.path.isdir(train_dir)

    ds = _read(train_dir)
    assert len(ds) == 6
    assert {'input_ids', 'labels', 'lengths'} <= set(ds.column_names)


def test_store_fields_restricts_persisted_columns(tokenizer, sft_file, tmp_path):
    """``store_fields`` is an allow-list: only the named columns are written, everything else dropped."""
    train_dir, _ = _export(tmp_path, sft_file, store_encoded=True, store_fields=['input_ids', 'labels', 'lengths'])
    ds = _read(train_dir)
    assert set(ds.column_names) == {'input_ids', 'labels', 'lengths'}


def test_store_fields_rejects_unknown_name(tokenizer, sft_file, tmp_path):
    """A misspelt / non-store field is refused rather than silently dropped -- the point of the guard."""
    with pytest.raises(ValueError, match='unrecognised'):
        _export(tmp_path, sft_file, store_encoded=True, store_fields=['input_ids', 'no_such_field'])


# --- val split -------------------------------------------------------------------------


def test_val_split_written_when_ratio_positive(tokenizer, sft_file, tmp_path):
    """``split_dataset_ratio > 0`` writes a second ``val`` store; both subdirectories are standalone
    ``load_from_disk`` targets, matching the layout ``cached_dataset`` / ``cached_val_dataset`` expect."""
    train_dir, val_dir = _export(tmp_path, sft_file, store_encoded=True, split_dataset_ratio=1 / 6)
    assert val_dir is not None
    assert os.path.isdir(train_dir) and os.path.isdir(val_dir)
    assert len(_read(train_dir)) == 5 and len(_read(val_dir)) == 1


# --- consumer read-back (the read half of --cached_dataset) ---------------------------


def test_exported_store_is_consumable_by_build_dataset_path(tokenizer, sft_file, tmp_path):
    """Full write -> consume loop: point ``DatasetConfig.cached_dataset`` at the exported ``train`` dir
    and load it through ``_load_cached_datasets`` -- the exact call ``build_dataset`` makes -- and the
    materialised rows come back with their tokens intact."""
    from swift.dev.builders.dataset import _load_cached_datasets
    from swift.dev.config import DatasetConfig, TemplateConfig

    train_dir, _ = _export(tmp_path, sft_file, store_encoded=True)
    dataset_config = DatasetConfig(cached_dataset=[train_dir])
    template_config = TemplateConfig(template=TEMPLATE, max_length=2048)

    train_datasets, val_datasets = _load_cached_datasets(dataset_config, template_config)
    assert val_datasets == []
    assert len(train_datasets) == 1
    loaded = train_datasets[0]
    assert len(loaded) == 6
    assert {'input_ids', 'labels', 'lengths'} <= set(loaded.column_names)
    assert loaded[0]['input_ids']  # tokens survived the round trip


def test_consumer_delete_filter_applies_to_exported_store(tokenizer, sft_file, tmp_path):
    """The reader's ``truncation_strategy='delete'`` length filter works on a real exported store: a
    ``max_length`` below every row's length drops them all, proving ``lengths`` is wired to the filter."""
    from swift.dev.builders.dataset import _load_cached_datasets
    from swift.dev.config import DatasetConfig, TemplateConfig

    train_dir, _ = _export(tmp_path, sft_file, store_encoded=True)
    dataset_config = DatasetConfig(cached_dataset=[train_dir])
    template_config = TemplateConfig(template=TEMPLATE, max_length=1, truncation_strategy='delete')

    train_datasets, _ = _load_cached_datasets(dataset_config, template_config)
    assert len(train_datasets[0]) == 0  # every row exceeds max_length=1


# --- recipe guards ---------------------------------------------------------------------


def test_no_dataset_raises(tmp_path):
    """``cached_dataset`` is the OUTPUT of this step, not its input: an empty source is refused early,
    before any model is touched."""
    from swift.dev.config import DatasetConfig, ModelConfig, TemplateConfig
    from swift.dev.recipe.cached_dataset import export_cached_dataset

    with pytest.raises(ValueError, match='needs DatasetConfig.dataset'):
        export_cached_dataset(
            ModelConfig(model=MODEL, model_type=MODEL_TYPE),
            TemplateConfig(template=TEMPLATE),
            DatasetConfig(),
            output_dir=str(tmp_path / 'cached'),
        )


def test_streaming_raises(tmp_path, sft_file):
    """An ``IterableDataset`` has no ``save_to_disk``, so streaming is refused before loading anything."""
    with pytest.raises(ValueError, match='does not support streaming'):
        _export(tmp_path, sft_file, streaming=True)


def test_lazy_mode_raises(tokenizer, sft_file, tmp_path):
    """A resolved-lazy mode would hand back a ``LazyLLMDataset`` wrapper with no ``save_to_disk``, which
    defeats the whole point (nothing precomputed), so the recipe refuses it and points at eager."""
    with pytest.raises(ValueError, match='requires eager encoding'):
        _export(tmp_path, sft_file, lazy_tokenize=True)


@pytest.mark.skipif(not os.path.isdir(MM_MODEL), reason=f'no local multimodal model at {MM_MODEL}')
def test_multimodal_store_encoded_raises(tmp_path):
    """``store_encoded=True`` materialises tokens but not the vision tensors, and the forward's
    ``_not_encoded`` guard is all-or-nothing, so a token-only multimodal row would reach the model as
    image placeholders with no ``pixel_values``. The recipe refuses the combo rather than write a row
    that trains broken."""
    from swift.dev.config import DatasetConfig, ModelConfig, TemplateConfig
    from swift.dev.recipe.cached_dataset import export_cached_dataset

    dataset_file = TinyData.sft(str(tmp_path / 'sft.jsonl'), n=2)
    with pytest.raises(ValueError, match='vision tensors'):
        export_cached_dataset(
            ModelConfig(model=MM_MODEL, model_type=MM_MODEL_TYPE),
            TemplateConfig(template='qwen2_5', max_length=2048),
            DatasetConfig(dataset=[dataset_file], dataset_num_proc=1, lazy_tokenize=False),
            output_dir=str(tmp_path / 'cached'),
            store_encoded=True,
        )

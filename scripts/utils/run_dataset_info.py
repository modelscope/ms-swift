# Copyright (c) ModelScope Contributors. All rights reserved.
"""Regenerate the ``datasets`` and ``tags`` blocks of ``docs/v5/**/get-started/models-and-datasets.mdx``
from the dev dataset registry.

Reads ``swift.dev.dataset.loader.DATASET_MAPPING`` (not legacy's ``DATASET_MAPPING``), emits one row
per dataset family, and replaces only the text between each block's markers; the surrounding prose
and the ``models`` block are preserved (the latter is owned by ``run_model_info.py``).

Dataset size and per-token statistics are inherited from the current ``datasets`` block rather than
recomputed: producing them means downloading and tokenizing every dataset, which is a separate,
network-bound maintenance pass. A family that is new (absent from the current table) and not flagged
``huge_dataset`` is emitted with ``-`` placeholders and reported at the end, so the gap is visible
instead of silent.

Run from the repository root (CI does this after every merge to ``main``)::

    python scripts/utils/run_dataset_info.py
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from doc_common import (DOC_PATHS, PROSE, build_tags_legend, extract_block, hub_link, read_doc, replace_block,
                        write_doc)

from swift.dev.dataset.loader import DATASET_MAPPING

_DATASET_HEADER = ('| Dataset ID | Subset Name | Dataset Size | Statistic (token) | Tags | HF Dataset ID |\n'
                   '| ---------- | ----------- | ------------ | ----------------- | ---- | ------------- |\n')

# Per-language column legend, mirroring the model half's style.
_LEGEND: Dict[str, List[str]] = {
    'zh': [
        '下表介绍了 ms-swift 接入的数据集的相关信息:',
        '- Dataset ID: ModelScope 数据集 id（链接）',
        '- Subset Name: 子数据集名称',
        '- Dataset Size: 数据集大小',
        '- Statistic (token): 以 token 数统计的数据集规模，对调整 `max_length` 有帮助；使用 qwen2.5 的 tokenizer 分词，'
        '不同 tokenizer 的统计量不同',
        '- Tags: 数据集的 tags（取值见文末的 Tags 说明）',
        '- HF Dataset ID: HuggingFace 数据集 id（链接）',
    ],
    'en': [
        'The table below lists the datasets supported by ms-swift:',
        '- Dataset ID: the ModelScope dataset id (linked)',
        '- Subset Name: the subset names',
        '- Dataset Size: the number of rows',
        '- Statistic (token): token-count statistics, useful for tuning `max_length`; tokenized with '
        'the qwen2.5 tokenizer, so the numbers differ across tokenizers',
        '- Tags: the dataset tags (see the Tags Legend at the end of this page)',
        '- HF Dataset ID: the HuggingFace dataset id (linked)',
    ],
}


def _subset_names(loader_cls) -> str:
    """The user-facing subset names, ``<br/>``-joined for the table cell (MDX needs the self-closing form)."""
    names = [s if isinstance(s, str) else (s.name or s.subset) for s in loader_cls.subsets]
    return '<br/>'.join(names) or 'default'


def _cache_mapping(block: str) -> Dict[str, Tuple[str, str]]:
    """Parse ``dataset_id -> (size, statistic)`` from the current ``datasets`` block.

    Lets an unchanged dataset keep the size/statistics already published, so regenerating the table
    does not need to re-download and re-tokenize everything. Reads the same six-column layout this
    script emits.
    """
    cache: Dict[str, Tuple[str, str]] = {}
    for line in block.split('\n'):
        if not line.startswith('|'):
            continue
        items = line.split('|')
        if len(items) < 7 or 'Dataset ID' in items[1] or set(items[1]) <= set('- '):
            continue
        key_cell = items[1] if items[1].strip() != '-' else items[6]
        match = re.search(r'\[(.+?)\]', key_cell)
        if match is None:
            continue
        size, stat = items[3].strip(), items[4].strip()
        if size == '-':
            # The placeholder a previous run emitted for an unknown size; inheriting it would turn
            # every unknown into "huge dataset" on the next run.
            continue
        cache[match.group(1)] = (size, stat)
    return cache


def _build_rows(cache: Dict[str, Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Return ``([(sort_key, row)], missing)`` for every registered dataset family.

    A registry-only family has no hub id and is selected on the CLI by its ``dataset_type``, so that is
    what its Dataset ID cell shows.
    """
    rows: List[Tuple[str, str]] = []
    missing: List[str] = []
    for dataset_type in DATASET_MAPPING:
        loader_cls = DATASET_MAPPING[dataset_type]
        ms_id, hf_id = next(loader_cls.iter_ids(), (None, None))
        dataset_id = ms_id or hf_id
        tags = ', '.join(loader_cls.tags) or '-'
        subsets = _subset_names(loader_cls)
        if loader_cls.huge_dataset:
            size, stat = 'huge dataset', '-'
        elif dataset_id in cache:
            size, stat = cache[dataset_id]
        else:
            size, stat = '-', '-'
            if dataset_id:
                missing.append(dataset_id)
        row = (f'|{hub_link(ms_id, "ms", "dataset") if ms_id else dataset_type}|{subsets}|{size}|{stat}|{tags}|'
               f'{hub_link(hf_id, "hf", "dataset")}|')
        rows.append((dataset_id or dataset_type, row))
    rows.sort(key=lambda item: item[0])
    return rows, missing


def _render_block(lang: str, cache: Dict[str, Tuple[str, str]]) -> str:
    """The body of the ``datasets`` block: a summary, the column legend, then the table."""
    p = PROSE[lang]
    rows, missing = _build_rows(cache)
    if lang == 'zh':
        summary = f'共 {len(rows)} 个数据集。'
    else:
        summary = f'{len(rows)} datasets in total.'
    parts = [
        f'### {p["dataset_list"]}',
        '',
        summary,
        '',
        *_LEGEND[lang],
        '',
        _DATASET_HEADER.rstrip('\n'),
        '\n'.join(row for _, row in rows),
    ]
    print(f'[{lang}] datasets: {len(rows)}')
    if missing:
        print(f'[{lang}] WARNING: {len(missing)} dataset(s) have no cached size/statistic and were '
              f'emitted as "-": {missing}')
    return '\n'.join(parts) + '\n'


def main() -> None:
    for lang, path in DOC_PATHS:
        text = read_doc(path)
        text = replace_block(text, 'datasets', _render_block(lang, _cache_mapping(extract_block(text, 'datasets'))))
        text = replace_block(text, 'tags', build_tags_legend(lang))
        write_doc(path, text)
        print(f'[{lang}] wrote {path}')


if __name__ == '__main__':
    main()

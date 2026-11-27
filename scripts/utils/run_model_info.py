# Copyright (c) ModelScope Contributors. All rights reserved.
"""Regenerate the ``models`` block of ``docs/v5/**/get-started/models-and-datasets.mdx`` from the dev model registry.

Reads ``swift.dev.model.loader.MODEL_MAPPING`` (not legacy's ``MODEL_MAPPING``), emits one row per
checkpoint id grouped by family, split into large-language and multimodal tables, and replaces only
the text between the block's markers; the surrounding prose is preserved. The ``datasets`` and
``tags`` blocks of the same files are owned by ``run_dataset_info.py``.

Run from the repository root (CI does this after every merge to ``main``)::

    python scripts/utils/run_model_info.py
"""
from __future__ import annotations

from itertools import chain
from typing import Dict, Iterator, List, Optional, Set, Tuple

from doc_common import (DOC_PATHS, PROSE, read_doc, replace_block, hub_link, write_doc)

from swift.dev.model.loader import MODEL_MAPPING

# Quantization markers whose checkpoints cannot be converted to Megatron weights, so a family that
# would otherwise support Megatron is marked down for these specific ids.
_QUANT_WORDS = ('gptq', 'awq', 'bnb', 'aqlm', 'int4', 'int8', 'nf4')

_MODEL_HEADER = ('| Model ID | Model Type | Default Template | Requires | Support Megatron | Tags | HF Model ID |\n'
                 '| -------- | ---------- | ---------------- | -------- | ---------------- | ---- | ----------- |\n')

_CHECK = '&#x2714;'
_CROSS = '&#x2718;'

# Per-language column legend. dev's ``ModelLoader`` has no agent template (that lived on legacy's
# template registry), so the column legacy carried is dropped here rather than emitted empty.
_LEGEND: Dict[str, List[str]] = {
    'zh': [
        '下表介绍了 ms-swift 接入的模型的相关信息:',
        '- Model ID: ModelScope 模型 id（链接）',
        '- Model Type: 模型类型',
        '- Default Template: 默认对话模板',
        '- Requires: 使用该模型的额外依赖（**如果训练时遇到 transformers 版本不匹配的问题，可以查看模型 '
        '`config.json` 中的 `transformers_version` 字段**）',
        '- Support Megatron: 是否支持 Megatron 训练',
        '- Tags: 模型的 tags（取值见文末的 Tags 说明）',
        '- HF Model ID: HuggingFace 模型 id（链接）',
    ],
    'en': [
        'The table below lists the models supported by ms-swift:',
        '- Model ID: the ModelScope model id (linked)',
        '- Model Type: the model type',
        '- Default Template: the default chat template',
        '- Requires: extra dependencies for this model (**if you hit a transformers version mismatch '
        'during training, check the `transformers_version` field in the model `config.json`**)',
        '- Support Megatron: whether Megatron training is supported',
        '- Tags: the model tags (see the Tags Legend at the end of this page)',
        '- HF Model ID: the HuggingFace model id (linked)',
    ],
}


def _iter_ids(models) -> Iterator[Tuple[Optional[str], Optional[str]]]:
    """Expand ``ModelLoader.models`` into ``(ms_id, hf_id)`` pairs.

    Mirrors the registry's own shorthand: a bare string means both hubs use the same id, a pair is
    written only when they differ. Kept local rather than importing the registry's private
    ``_iter_ids`` so this script depends only on the loader's public class attributes.
    """
    for entry in models:
        if isinstance(entry, str):
            yield entry, entry
        else:
            yield entry[0], entry[1]


def _mcore_supported_types() -> Optional[Set[str]]:
    """The set of dev ``model_type``s Megatron can serve, or ``None`` when the bridge is unavailable.

    A family supports Megatron when it declares its own ``mcore_model_type`` or when its
    ``model_type`` is covered by some Megatron conversion. ``None`` signals the bridge is not
    installed, so the caller marks everything unsupported and says so rather than guessing.
    """
    try:
        from mcore_bridge.model import MODEL_MAPPING as MCORE_MODEL_MAPPING
    except Exception:  # noqa: BLE001 - the bridge is an optional dependency
        return None
    return set(chain.from_iterable(meta.model_types for meta in MCORE_MODEL_MAPPING.values()))


def _support_megatron(loader_cls, ms_id: Optional[str], supported: Optional[Set[str]]) -> str:
    if supported is None:
        return _CROSS
    ok = loader_cls.mcore_model_type is not None or loader_cls.model_type in supported
    if ok and ms_id and any(word in ms_id.lower() for word in _QUANT_WORDS):
        ok = False
    return _CHECK if ok else _CROSS


def _build_rows(supported: Optional[Set[str]]) -> Tuple[List[str], List[str], int, int]:
    """Return ``(llm_rows, mllm_rows, llm_megatron_count, mllm_megatron_count)``.

    Families are visited in sorted ``model_type`` order so every checkpoint of one family lands
    contiguously -- that is the family grouping the table shows.
    """
    llm_rows: List[str] = []
    mllm_rows: List[str] = []
    llm_mg = mllm_mg = 0
    for model_type in sorted(MODEL_MAPPING):
        loader_cls = MODEL_MAPPING[model_type]
        template = loader_cls.template or '-'
        # Code spans: a bound like ``transformers>=4.54,<4.56`` would otherwise open a JSX tag in MDX.
        requires = ', '.join(f'`{req}`' for req in loader_cls.requires) or '-'
        tags = ', '.join(loader_cls.tags) or '-'
        for ms_id, hf_id in _iter_ids(loader_cls.models):
            support = _support_megatron(loader_cls, ms_id, supported)
            row = (f'|{hub_link(ms_id, "ms", "model")}|{model_type}|{template}|{requires}|'
                   f'{support}|{tags}|{hub_link(hf_id, "hf", "model")}|\n')
            if loader_cls.is_multimodal:
                mllm_rows.append(row)
                mllm_mg += support == _CHECK
            else:
                llm_rows.append(row)
                llm_mg += support == _CHECK
    return llm_rows, mllm_rows, llm_mg, mllm_mg


def _render_block(lang: str, supported: Optional[Set[str]]) -> str:
    """The body of the ``models`` block: a summary, the column legend, then the two family tables."""
    p = PROSE[lang]
    llm_rows, mllm_rows, llm_mg, mllm_mg = _build_rows(supported)
    total = len(llm_rows) + len(mllm_rows)
    if lang == 'zh':
        summary = (f'共 {len(MODEL_MAPPING)} 个模型系列、{total} 个模型，其中 {llm_mg + mllm_mg} 个支持 Megatron 训练。'
                   if supported is not None else f'共 {len(MODEL_MAPPING)} 个模型系列、{total} 个模型。')
    else:
        summary = (f'{len(MODEL_MAPPING)} model families and {total} models in total, '
                   f'{llm_mg + mllm_mg} of which support Megatron training.'
                   if supported is not None else f'{len(MODEL_MAPPING)} model families and {total} models in total.')

    parts = [
        f'### {p["list"]}',
        '',
        summary,
        '',
        *_LEGEND[lang],
        '',
        f'#### {p["llm"]} ({len(llm_rows)})',
        '',
        _MODEL_HEADER.rstrip('\n'),
        ''.join(llm_rows).rstrip('\n'),
        '',
        f'#### {p["mllm"]} ({len(mllm_rows)})',
        '',
        _MODEL_HEADER.rstrip('\n'),
        ''.join(mllm_rows).rstrip('\n'),
    ]
    print(f'[{lang}] LLM: {len(llm_rows)} (Megatron {llm_mg}), MLLM: {len(mllm_rows)} (Megatron {mllm_mg})')
    return '\n'.join(parts) + '\n'


def main() -> None:
    supported = _mcore_supported_types()
    if supported is None:
        print('WARNING: mcore_bridge is unavailable; every model is marked as not supporting Megatron.')
    for lang, path in DOC_PATHS:
        write_doc(path, replace_block(read_doc(path), 'models', _render_block(lang, supported)))
        print(f'[{lang}] wrote {path}')


if __name__ == '__main__':
    main()

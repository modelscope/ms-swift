# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared plumbing for the supported-models-and-datasets doc generators.

``run_model_info.py`` fills the ``models`` block and ``run_dataset_info.py`` the ``datasets`` and
``tags`` blocks of ``docs/v5/**/get-started/models-and-datasets.mdx``. A block is the text between a
pair of markers::

    {/* BEGIN AUTO-GENERATED: models */}
    ...
    {/* END AUTO-GENERATED: models */}

Everything outside the markers is hand-written and preserved verbatim, so prose and tables evolve
independently. MDX rejects HTML comments, hence the JSX form. The markers are the only thing the
two scripts share besides the file paths, hub-link format and tags legend -- one definition of each
is what stops them from drifting apart.

CI regenerates the blocks after every merge to ``main``
(``.github/workflows/docs_supported_tables.yaml``).
"""
from __future__ import annotations

import re
from typing import Dict, Match, Optional, Tuple

from swift.dev.utils.tags import CONTROLLED_TAGS

# (language, path). The two files carry identical tables; only the prose differs by language.
DOC_PATHS: Tuple[Tuple[str, str], ...] = (
    ('zh', 'docs/v5/source_zh/get-started/models-and-datasets.mdx'),
    ('en', 'docs/v5/source_en/get-started/models-and-datasets.mdx'),
)

# Headings per language, as the generators emit them inside the blocks. ``list`` heads the tables of
# a block; ``tags_legend`` the controlled-vocabulary legend that closes the dataset block.
PROSE: Dict[str, Dict[str, str]] = {
    'zh': {
        'llm': '大语言模型',
        'mllm': '多模态大模型',
        'list': '模型列表',
        'dataset_list': '数据集列表',
        'tags_legend': 'Tags 说明',
        'tags_legend_intro': ('`tags` 取自受控词表（单一真源：`swift/dev/utils/tags.py`）。注册模型或数据集时使用词表之外的 '
                              'tag 会直接报错，因此下表列出的即当前允许的全部 tag：'),
    },
    'en': {
        'llm': 'Large language models',
        'mllm': 'Multimodal large models',
        'list': 'Model list',
        'dataset_list': 'Dataset list',
        'tags_legend': 'Tags Legend',
        'tags_legend_intro': ('`tags` come from a controlled vocabulary (single source of truth: '
                              '`swift/dev/utils/tags.py`). Registering a model or dataset with a tag outside it '
                              'raises, so the list below is the complete set of allowed tags:'),
    },
}

_HUB_URL = {
    ('ms', 'model'): 'https://modelscope.cn/models/{}',
    ('hf', 'model'): 'https://huggingface.co/{}',
    ('ms', 'dataset'): 'https://modelscope.cn/datasets/{}',
    ('hf', 'dataset'): 'https://huggingface.co/datasets/{}',
}

# ``%s`` rather than ``str.format``: the braces are part of the marker text, not template fields.
_BEGIN = '{/* BEGIN AUTO-GENERATED: %s */}'
_END = '{/* END AUTO-GENERATED: %s */}'


def get_url_suffix(raw_id: str) -> str:
    """Strip a ``:revision`` suffix so the link points at the id, not a pinned revision."""
    return raw_id.split(':')[0] if ':' in raw_id else raw_id


def hub_link(raw_id: Optional[str], hub: str, kind: str) -> str:
    """A markdown link to a ModelScope/HuggingFace model or dataset, or ``'-'`` when absent."""
    if not raw_id:
        return '-'
    url = _HUB_URL[(hub, kind)].format(get_url_suffix(raw_id))
    return f'[{raw_id}]({url})'


def read_doc(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def write_doc(path: str, text: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


def _find_block(text: str, name: str) -> Match[str]:
    pattern = re.compile(re.escape(_BEGIN % name) + r'\n(.*?)' + re.escape(_END % name), re.DOTALL)
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise ValueError(f'expected exactly one {name!r} block delimited by "{_BEGIN % name}" and '
                         f'"{_END % name}" in the doc, found {len(matches)}.')
    return matches[0]


def extract_block(text: str, name: str) -> str:
    """The current body of block ``name`` (markers excluded), so a generator can read what it last emitted."""
    return _find_block(text, name).group(1)


def replace_block(text: str, name: str, body: str) -> str:
    """Return ``text`` with block ``name``'s body replaced by ``body``; everything else is untouched.

    Fails loudly when the marker pair is missing or duplicated: silently appending a second copy or
    dropping the tables would only show up as a wrong page.
    """
    match = _find_block(text, name)
    begin, end = _BEGIN % name, _END % name
    return f'{text[:match.start()]}{begin}\n{body.rstrip()}\n{end}{text[match.end():]}'


def build_tags_legend(lang: str) -> str:
    """The controlled-vocabulary legend, rendered straight from :data:`CONTROLLED_TAGS`.

    Derived from the same frozenset the registration guards check, so the documented list can never
    drift from the enforced one.
    """
    p = PROSE[lang]
    tags = ', '.join(f'`{tag}`' for tag in sorted(CONTROLLED_TAGS))
    return '\n'.join([f'### {p["tags_legend"]}', '', p['tags_legend_intro'], '', tags])

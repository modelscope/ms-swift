# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared plumbing for the supported-models-and-datasets doc generators.

``run_model_info.py`` and ``run_dataset_info.py`` each regenerate one half of the same bilingual
markdown file and must preserve the other half verbatim, so they share a single definition of the
file paths, the section headings that delimit the two halves, the hub-link format, and the tags
legend. Centralising those is what stops the two scripts from drifting apart: a heading renamed in
one but not the other would silently duplicate or drop a whole section on the next run.

Ownership split, delimited by the ``## Datasets`` / ``## 数据集`` heading:

- ``run_model_info`` regenerates everything *above* that heading (title, table of contents, tags
  legend, and the model tables) and preserves the dataset half.
- ``run_dataset_info`` regenerates that heading and everything *below* it, preserving the model half.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from swift.dev.utils.tags import CONTROLLED_TAGS

# (language, path). The two files are generated together and carry identical tables; only the
# surrounding prose differs by language.
DOC_PATHS: List[Tuple[str, str]] = [
    ('zh', 'docs/source/Instruction/Supported-models-and-datasets.md'),
    ('en', 'docs/source_en/Instruction/Supported-models-and-datasets.md'),
]

# Headings per language. ``datasets`` doubles as the boundary between the two halves, so it must stay
# in sync with what the generators emit and with the heading already present in the committed docs.
PROSE: Dict[str, Dict[str, str]] = {
    'zh': {
        'title': '支持的模型和数据集',
        'toc': '目录',
        'tags_legend': 'Tags 说明',
        'models': '模型',
        'llm': '大语言模型',
        'mllm': '多模态大模型',
        'datasets': '数据集',
        'tags_legend_intro': ('`tags` 取自受控词表（单一真源：`swift/dev/tags.py`）。注册模型或数据集时使用词表之外的 '
                              'tag 会直接报错，因此下表列出的即当前允许的全部 tag：'),
    },
    'en': {
        'title': 'Supported Models and Datasets',
        'toc': 'Table of Contents',
        'tags_legend': 'Tags Legend',
        'models': 'Models',
        'llm': 'Large language models',
        'mllm': 'Multimodal large models',
        'datasets': 'Datasets',
        'tags_legend_intro': ('`tags` come from a controlled vocabulary (single source of truth: '
                              '`swift/dev/tags.py`). Registering a model or dataset with a tag outside it '
                              'raises, so the list below is the complete set of allowed tags:'),
    },
}

_HUB_URL = {
    ('ms', 'model'): 'https://modelscope.cn/models/{}',
    ('hf', 'model'): 'https://huggingface.co/{}',
    ('ms', 'dataset'): 'https://modelscope.cn/datasets/{}',
    ('hf', 'dataset'): 'https://huggingface.co/datasets/{}',
}


def get_url_suffix(raw_id: str) -> str:
    """Strip a ``:revision`` suffix so the link points at the id, not a pinned revision."""
    return raw_id.split(':')[0] if ':' in raw_id else raw_id


def hub_link(raw_id: Optional[str], hub: str, kind: str) -> str:
    """A markdown link to a ModelScope/HuggingFace model or dataset, or ``'-'`` when absent."""
    if not raw_id:
        return '-'
    url = _HUB_URL[(hub, kind)].format(get_url_suffix(raw_id))
    return f'[{raw_id}]({url})'


def gh_anchor(heading: str) -> str:
    """The in-page anchor GitHub derives from a heading: lower-case, punctuation dropped, spaces to hyphens.

    ``\\w`` is Unicode-aware here, so CJK headings keep their characters (``Tags 说明`` -> ``tags-说明``).
    """
    text = heading.strip().lower()
    text = re.sub(r'[^\w\s-]', '', text, flags=re.UNICODE)
    return re.sub(r'\s+', '-', text)


def read_doc(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def write_doc(path: str, text: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


def split_head_tail(text: str, lang: str) -> Tuple[str, str]:
    """Split a doc into the model half (head) and the dataset half (tail), at the datasets heading.

    The tail begins at the ``## <datasets>`` line so it can be handed back unchanged by the model
    generator, or replaced wholesale by the dataset generator. When the heading is absent (a freshly
    bootstrapped file) the whole text is head and the tail is empty.
    """
    heading = f'## {PROSE[lang]["datasets"]}'
    match = re.search(rf'^{re.escape(heading)}\s*$', text, flags=re.MULTILINE)
    if match is None:
        return text, ''
    return text[:match.start()], text[match.start():]


def build_toc(lang: str) -> str:
    """The top-of-page anchor menu, listing every section the two generators emit."""
    p = PROSE[lang]
    lines = [
        f'## {p["toc"]}',
        '',
        f'- [{p["models"]}](#{gh_anchor(p["models"])})',
        f'  - [{p["llm"]}](#{gh_anchor(p["llm"])})',
        f'  - [{p["mllm"]}](#{gh_anchor(p["mllm"])})',
        f'- [{p["datasets"]}](#{gh_anchor(p["datasets"])})',
        f'- [{p["tags_legend"]}](#{gh_anchor(p["tags_legend"])})',
    ]
    return '\n'.join(lines)


def build_tags_legend(lang: str) -> str:
    """The controlled-vocabulary legend, rendered straight from :data:`CONTROLLED_TAGS`.

    Derived from the same frozenset the registration guards check, so the documented list can never
    drift from the enforced one.
    """
    p = PROSE[lang]
    tags = ', '.join(f'`{tag}`' for tag in sorted(CONTROLLED_TAGS))
    return '\n'.join([f'## {p["tags_legend"]}', '', p['tags_legend_intro'], '', tags])

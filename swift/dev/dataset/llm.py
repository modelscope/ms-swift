# Copyright (c) ModelScope Contributors. All rights reserved.
"""Pilot pure-text dataset registrations, mirroring the model side's ``loader/llm.py``.

A handful of simple families, chosen to exercise each way a dataset plugs into the pipeline:

* :class:`AlpacaZhLoader` / :class:`LongAlpacaLoader` -- auto-detected alpaca format, with a
  :class:`Preprocessor` subclass that only tweaks a field before delegating to the converter.
* :class:`RuozhibaLoader` -- several subsets and a preprocessor that builds the standard row from
  scratch, never touching a converter.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

from .base import DatasetLoader, register_dataset
from .preprocessor import Preprocessor


class AlpacaZhPreprocessor(Preprocessor):
    """Alpaca, minus the ``'输入：'`` lead-in some rows prepend to the ``input`` half of the turn."""

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        input_ = row.get('input')
        if isinstance(input_, str) and input_.startswith('输入：'):
            row['input'] = input_[len('输入：'):]
        return super().preprocess(row)


@register_dataset
class AlpacaZhLoader(DatasetLoader):
    dataset_type = 'alpaca-zh'
    datasets = [('AI-ModelScope/alpaca-gpt4-data-zh', 'llm-wizard/alpaca-gpt4-data-zh')]
    preprocessor = AlpacaZhPreprocessor
    tags = ['chat', 'general', '🔥']


class LongAlpacaPreprocessor(Preprocessor):
    """Alpaca, minus the ``'Answer: '`` prefix this dataset puts on every answer."""

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        output = row.get('output')
        prefix = 'Answer: '
        if isinstance(output, str) and output.startswith(prefix):
            row['output'] = output[len(prefix):].strip()
        return super().preprocess(row)


@register_dataset
class LongAlpacaLoader(DatasetLoader):
    dataset_type = 'long-alpaca-12k'
    datasets = [('AI-ModelScope/LongAlpaca-12k', 'Yukang/LongAlpaca-12k')]
    preprocessor = LongAlpacaPreprocessor
    tags = ['long-sequence', 'QA']


class RuozhibaPreprocessor(Preprocessor):
    """A pretrain dataset: each row is one completion, emitted as a lone assistant turn.

    Builds the standard row itself (no format converter applies): it stitches ``title``/``content``
    and an optional ``abs``, strips a leading list-item number, and drops rows left empty.
    """

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        title = row['title'] if row.get('title') is not None else row.get('content')
        abstract = row.get('abs')
        if abstract and abstract != title:
            title = f'{title}，{abstract}'
        match = re.search(r'\d+[\.,\s,\、](.+)', title)
        if match:
            title = match.group(1)
        if title:
            return {'messages': [{'role': 'assistant', 'content': title}]}


@register_dataset
class RuozhibaLoader(DatasetLoader):
    dataset_type = 'ruozhiba'
    datasets = ['AI-ModelScope/ruozhiba']
    subsets = ['post-annual', 'title-good', 'title-norm']
    preprocessor = RuozhibaPreprocessor
    tags = ['pretrain', '🔥']

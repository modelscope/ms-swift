# Copyright (c) ModelScope Contributors. All rights reserved.
"""Dataset one-string syntax tests (C3-DS05): ``hf://org/name:sub#N`` and friends.

Pure CPU, hand-computed expectations -- the parser's splits are checked against the syntax contract,
not against another framework call. Existing paths must win over the split characters.
"""
import os

import pytest

from swift.dev.dataset.loader.base import DatasetLoader


@pytest.mark.parametrize('entry,expected', [
    ('hf://org/name:sub1/sub2#500', ('org/name', ['sub1', 'sub2'], 500, True)),
    ('ms://org/name#10', ('org/name', [], 10, False)),
    ('HF://Org/Name', ('Org/Name', [], None, True)),
    ('org/name:sub', ('org/name', ['sub'], None, None)),
    ('org/name', ('org/name', [], None, None)),
    ('id:', ('id', [], None, None)),
    ('id#0', ('id', [], 0, None)),
    ('hf://org/name:sub#1e3', ('org/name', ['sub#1e3'], None, True)),  # not digits -> stays in the subset name
    ('id#-5', ('id#-5', [], None, None)),
])
def test_parse_dataset_syntax_matrix(entry, expected):
    assert DatasetLoader.parse_dataset_syntax(entry) == expected


def test_parse_dataset_syntax_unknown_hub_prefix_is_refused():
    with pytest.raises(ValueError, match='Unknown hub prefix'):
        DatasetLoader.parse_dataset_syntax('ftp://x/y')


def test_parse_dataset_syntax_existing_path_wins_over_split_characters(tmp_path):
    p = tmp_path / 'weird:name#3.jsonl'
    p.write_text('{}\n')
    assert DatasetLoader.parse_dataset_syntax(str(p)) == (str(p), [], None, None)


def test_parse_dataset_syntax_budget_then_path_recheck(tmp_path):
    p = tmp_path / 'data.jsonl'
    p.write_text('{}\n')
    assert DatasetLoader.parse_dataset_syntax(f'{p}#5') == (str(p), [], 5, None)


@pytest.mark.parametrize('entry,expected', [
    ('id#500', ('id', 500)),
    ('id#1e3', ('id#1e3', None)),
    ('id#-5', ('id#-5', None)),
    ('id#', ('id#', None)),
    ('id', ('id', None)),
    ('a#b#7', ('a#b', 7)),  # rpartition: only the last #N is the budget
])
def test_split_sample_count(entry, expected):
    assert DatasetLoader.split_sample_count(entry) == expected
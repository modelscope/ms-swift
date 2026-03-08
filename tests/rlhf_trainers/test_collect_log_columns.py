import logging

from swift.rlhf_trainers import utils as rlhf_utils


def test_collect_log_columns_empty_config():
    rows = [{'a': 1}]
    assert rlhf_utils.collect_log_columns(rows, []) == {}


def test_collect_log_columns_empty_rows():
    assert rlhf_utils.collect_log_columns([], ['a']) == {}


def test_collect_log_columns_all_present():
    rows = [{'a': 1, 'b': 'x'}, {'a': 2, 'b': 'y'}]
    result = rlhf_utils.collect_log_columns(rows, ['a', 'b'])
    assert result == {'a': [1, 2], 'b': ['x', 'y']}


def test_collect_log_columns_missing_warns_once_per_column():
    rows = [{'a': 1}, {'b': 2}]
    warned = set()
    logger = rlhf_utils.get_logger()
    records = []

    class ListHandler(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = ListHandler()
    handler.setLevel(logging.WARNING)
    logger.addHandler(handler)
    try:
        result1 = rlhf_utils.collect_log_columns(rows, ['a', 'b'], warned_columns=warned)
        result2 = rlhf_utils.collect_log_columns(rows, ['a', 'b'], warned_columns=warned)
    finally:
        logger.removeHandler(handler)

    assert result1 == {'a': [1, None], 'b': [None, 2]}
    assert result2 == {'a': [1, None], 'b': [None, 2]}
    assert len([r for r in records if 'log_completions_extra_columns' in r.getMessage()]) == 2


def test_collect_log_columns_keeps_complex_types():
    d = {'k': 'v'}
    values = [1, 2, 3]
    rows = [{'meta': d, 'trace': values}]
    result = rlhf_utils.collect_log_columns(rows, ['meta', 'trace'])
    assert result['meta'][0] is d
    assert result['trace'][0] is values

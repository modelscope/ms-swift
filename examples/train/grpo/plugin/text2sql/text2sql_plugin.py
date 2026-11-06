# Copyright (c) ModelScope Contributors. All rights reserved.
"""Execution-based text-to-SQL rewards for GRPO on SQLite databases (e.g. BIRD, Spider).

Each dataset row needs the reference query in ``solution`` and the database in ``db_path``
(absolute path to a ``.sqlite`` file) or ``db_id`` (resolved as
``$TEXT2SQL_DB_ROOT/<db_id>/<db_id>.sqlite``, the BIRD / Spider layout).

``text2sql_exec`` gives 1.0 when the predicted query returns the same set of rows as the reference,
the execution-accuracy definition of BIRD and Spider. Because the reward comes from running
model-written SQL, it is guarded against the ways a policy can collect reward without solving the
task:

* the database is opened read-only and an SQLite authorizer only allows reads, so a rollout cannot
  write, ATTACH another file, run PRAGMAs or load extensions (and cannot change the data that later
  rollouts are scored on);
* one statement per query: ``SELECT ...; DROP TABLE t`` is rejected instead of half-executed;
* a wall-clock timeout interrupts runaway queries (recursive CTEs, huge cross joins) inside SQLite;
* a row cap stops "return everything" queries from exhausting memory;
* a reference query that fails is logged and its rollouts get ``None`` (no reward, NaN in the
  trainer) instead of 0, so broken labels give no learning signal instead of a misleading one;
  with ``TEXT2SQL_SKIP_EMPTY_GOLD=1`` the same is done for references that return no rows, because
  any query returning nothing (e.g. ``WHERE 1 = 0``) would match them.
"""
import os
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

from swift.rewards import ORM, orms
from swift.utils import get_logger

logger = get_logger()

_ALLOWED_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
_DENIED_FUNCTIONS = {'load_extension', 'readfile', 'writefile', 'edit', 'fts3_tokenizer'}
_FENCE_RE = re.compile(r'```[ \t]*(?:sql|sqlite)?[ \t]*\n?(.*?)```', re.IGNORECASE | re.DOTALL)
_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)


def extract_sql(completion: str) -> str:
    """The last fenced code block after any ``<think>`` section, or the whole answer."""
    text = _THINK_RE.sub('', completion or '')
    blocks = _FENCE_RE.findall(text)
    sql = blocks[-1] if blocks else text
    return sql.strip().rstrip(';').strip()


def _authorizer(action, arg1, arg2, db_name, trigger):
    if action not in _ALLOWED_ACTIONS:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION and arg2 and arg2.lower() in _DENIED_FUNCTIONS:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def execute_sql(db_path: str, sql: str, timeout: float, max_rows: int) -> Tuple[Optional[List[tuple]], Optional[str]]:
    """Run one read-only statement. Returns ``(rows, None)`` or ``(None, error)``."""
    if not sql:
        return None, 'empty query'
    path = Path(db_path)
    if not path.is_file():
        return None, f'database not found: {db_path}'
    deadline = time.monotonic() + timeout
    conn = None
    try:
        conn = sqlite3.connect(f'{path.resolve().as_uri()}?mode=ro', uri=True, check_same_thread=False)
        conn.text_factory = lambda b: b.decode('utf-8', errors='replace')
        conn.set_authorizer(_authorizer)
        conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 1000)
        rows = conn.execute(sql).fetchmany(max_rows + 1)
        if len(rows) > max_rows:
            return None, f'result exceeds {max_rows} rows'
        return [tuple(r) for r in rows], None
    except sqlite3.OperationalError as e:
        return None, 'timeout' if 'interrupted' in str(e) else f'OperationalError: {e}'
    except Exception as e:  # sqlite3.Warning (several statements), DatabaseError, ...
        return None, f'{type(e).__name__}: {e}'
    finally:
        if conn is not None:
            conn.close()


class Text2SQLExecAccuracy(ORM):
    """1.0 if the predicted SQL returns the same set of rows as ``solution`` on the sample's database."""

    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)
        self.db_root = os.environ.get('TEXT2SQL_DB_ROOT', '')
        self.timeout = float(os.environ.get('TEXT2SQL_TIMEOUT', 30))
        self.max_rows = int(os.environ.get('TEXT2SQL_MAX_ROWS', 100000))
        self.skip_empty_gold = os.environ.get('TEXT2SQL_SKIP_EMPTY_GOLD', '0') == '1'
        self.pool = ThreadPoolExecutor(max_workers=int(os.environ.get('TEXT2SQL_WORKERS', 16)))
        self._gold_cache = {}

    def _db_path(self, db_path: Optional[str], db_id: Optional[str]) -> str:
        if db_path:
            return db_path
        if db_id and self.db_root:
            return str(Path(self.db_root) / db_id / f'{db_id}.sqlite')
        raise ValueError('text2sql_exec needs a `db_path` column, or a `db_id` column and TEXT2SQL_DB_ROOT.')

    def _gold(self, db_path: str, sql: str):
        key = (db_path, sql)
        if key not in self._gold_cache:
            if len(self._gold_cache) >= 10000:  # every rollout of a prompt reuses its entry; keep memory bounded
                self._gold_cache.clear()
            rows, error = execute_sql(db_path, sql, self.timeout, self.max_rows)
            if error:
                logger.warning(f'text2sql_exec: reference SQL failed on {db_path}: {error}')
            self._gold_cache[key] = (rows, error)
        return self._gold_cache[key]

    def _score(self, completion: str, solution: str, db_path: str) -> Optional[float]:
        gold_rows, gold_error = self._gold(db_path, solution)
        if gold_error or (self.skip_empty_gold and not gold_rows):
            return None
        rows, error = execute_sql(db_path, extract_sql(completion), self.timeout, self.max_rows)
        return float(error is None and set(rows) == set(gold_rows))

    def __call__(self, completions, solution, db_path=None, db_id=None, **kwargs) -> List[Optional[float]]:
        n = len(completions)
        db_paths = [self._db_path((db_path or [None] * n)[i], (db_id or [None] * n)[i]) for i in range(n)]
        return list(self.pool.map(self._score, completions, solution, db_paths))


class Text2SQLFormat(ORM):
    """1.0 if the answer (outside ``<think>``) ends with a fenced SQL block that starts with SELECT or WITH."""

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        for completion in completions:
            blocks = _FENCE_RE.findall(_THINK_RE.sub('', completion or ''))
            ok = bool(blocks) and re.match(r'\s*(select|with)\b', blocks[-1], re.IGNORECASE) is not None
            rewards.append(1.0 if ok else 0.0)
        return rewards


orms['text2sql_exec'] = Text2SQLExecAccuracy
orms['text2sql_format'] = Text2SQLFormat

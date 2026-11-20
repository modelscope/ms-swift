# Copyright (c) ModelScope Contributors. All rights reserved.
"""Convert BIRD (or any BIRD-style JSON / JSONL) into a GRPO dataset for the text2sql plugin.

Each input record needs ``db_id``, ``question``, ``SQL`` and optionally ``evidence``. The prompt
holds the database's CREATE TABLE statements, the question and BIRD's external knowledge, and asks
for the final query in a ```sql block. Each output row is::

    {"messages": [{"role": "user", "content": "<prompt>"}], "solution": "<reference SQL>", "db_id": "..."}

Example (BIRD train, about 9k questions; databases from https://bird-bench.github.io/)::

    python prepare_bird.py \
        --input train.json --db_root train/train_databases --output bird_train.jsonl
"""
import argparse
import json
import sqlite3
from pathlib import Path


def schema_prompt(db_path: Path) -> str:
    conn = sqlite3.connect(f'{db_path.resolve().as_uri()}?mode=ro', uri=True)
    try:
        rows = conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table'").fetchall()
        return '\n\n'.join(sql for name, sql in rows if sql and name != 'sqlite_sequence')
    finally:
        conn.close()


def build_prompt(schema: str, question: str, evidence: str = None) -> str:
    knowledge = ' and understanding External Knowledge' if evidence else ''
    lines = [
        schema, '', f'-- Using valid SQLite{knowledge}, answer the following questions for the tables provided above.',
        f'-- {question}'
    ]
    if evidence:
        lines.append(f'-- External Knowledge: {evidence}')
    lines += [
        '', 'Generate the SQLite for the above question after thinking step by step.',
        'Write the final query in a single ```sql code block at the end of your answer.'
    ]
    return '\n'.join(lines)


def load_records(path: Path):
    text = path.read_text(encoding='utf-8')
    if path.suffix == '.jsonl':
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', required=True, help='BIRD-style .json or .jsonl file')
    parser.add_argument('--db_root', required=True, help='Directory with <db_id>/<db_id>.sqlite')
    parser.add_argument('--output', required=True, help='Output .jsonl for --dataset')
    args = parser.parse_args()

    schemas = {}
    written = skipped = 0
    with open(args.output, 'w', encoding='utf-8') as f:
        for record in load_records(Path(args.input)):
            db_id = record['db_id']
            db_path = Path(args.db_root) / db_id / f'{db_id}.sqlite'
            if not db_path.is_file():
                skipped += 1
                continue
            if db_id not in schemas:
                schemas[db_id] = schema_prompt(db_path)
            prompt = build_prompt(schemas[db_id], record['question'], record.get('evidence') or None)
            row = {'messages': [{'role': 'user', 'content': prompt}], 'solution': record['SQL'], 'db_id': db_id}
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            written += 1
    print(f'wrote {written} rows to {args.output}'
          + (f', skipped {skipped} with a missing database' if skipped else ''))


if __name__ == '__main__':
    main()

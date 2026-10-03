"""Compile per-question training inputs; keep gold symbols outside messages."""
import hashlib
import json
from pathlib import Path

from decision_schema import compile_row


def main():
    root = Path(__file__).parent
    out = root / 'decision_data'
    out.mkdir(exist_ok=True)
    counts = {}
    for split in ('train', 'validation', 'calibration', 'test'):
        rows = []
        for line in (root / 'prepared' / f'{split}.jsonl').read_text().splitlines():
            case = json.loads(line)
            for field, question in case['questions'].items():
                row = {'state': case['state'], 'questions': {field: question},
                       'targets': {field: case['targets'][field]}}
                compiled = compile_row(row)
                assert compiled.messages == compile_row(row, include_targets=False).messages
                rows.append({'messages': compiled.messages,
                             'decision_targets': [compiled.targets[field]],
                             'case_id': case['id'], 'field': field})
        path = out / f'{split}.jsonl'
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        counts[split] = {'decisions': len(rows), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (out / 'manifest.json').write_text(json.dumps(counts, indent=2))
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main()

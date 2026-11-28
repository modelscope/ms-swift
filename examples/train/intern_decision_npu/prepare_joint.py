"""Compile joint decisions and check the pinned official input contract."""
import argparse
import hashlib
import importlib.util
import json
import sys
from decision_schema import compile_row
from pathlib import Path


def prepare(prepared, output, official_schema):
    spec = importlib.util.spec_from_file_location('official_decision_schema', official_schema)
    official = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = official
    spec.loader.exec_module(official)
    if output.exists():
        raise FileExistsError('Use a fresh output directory; do not overwrite frozen data')
    splits = {}
    seen_ids, seen_inputs = set(), set()
    manifest = {
        'protocol': 'joint fields; full-vocabulary causal CE; targets excluded from messages',
        'official_schema_sha256': hashlib.sha256(official_schema.read_bytes()).hexdigest(),
        'training_source': 'public Typed Decisions training split, not the unpublished official training corpus',
        'splits': {},
    }
    for split in ('train', 'validation', 'calibration', 'test'):
        source = prepared / f"{split}.jsonl"
        cases = [json.loads(line) for line in source.read_text().splitlines()]
        rows, ids, inputs = [], set(), set()
        for case in cases:
            compiled = compile_row(case)
            reference = official.compile_row(case)
            if compiled.messages != reference.messages or compiled.targets != reference.targets:
                raise ValueError(f"Official compiler mismatch in {split}")
            if compiled.messages != compile_row(case, include_targets=False).messages:
                raise ValueError('Targets leaked into input messages')
            if len(compiled.targets) != len(compiled.fields):
                raise ValueError('All fields must have one supervised answer')
            identity = hashlib.sha256(json.dumps(compiled.messages, ensure_ascii=False).encode()).hexdigest()
            if case['id'] in ids or identity in inputs:
                raise ValueError(f"Duplicate case in {split}")
            ids.add(case['id'])
            inputs.add(identity)
            rows.append({
                'messages': compiled.messages,
                'decision_targets': [compiled.targets[f] for f in compiled.fields],
                'case_id': case['id'],
                'fields': list(compiled.fields)
            })
        if ids & seen_ids or inputs & seen_inputs:
            raise ValueError('Cross-split input or case overlap')
        seen_ids |= ids
        seen_inputs |= inputs
        payload = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)
        splits[split] = payload
        manifest['splits'][split] = {
            'cases': len(rows),
            'decisions': sum(len(r['decision_targets']) for r in rows),
            'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'sha256': hashlib.sha256(payload.encode()).hexdigest(),
        }
    output.mkdir(parents=True)
    for split, payload in splits.items():
        (output / f"{split}.jsonl").write_text(payload)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--official-schema', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.prepared, args.output, args.official_schema), indent=2))


if __name__ == '__main__':
    main()

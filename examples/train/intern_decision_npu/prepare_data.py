"""Prepare a case-disjoint decision-training pilot; never tokenize test labels as input."""
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'prepared'
OUT.mkdir(exist_ok=True)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def convert(raw):
    state, questions, gold = [json.loads(raw[k]) for k in ('state', 'questions', 'gold')]
    assert len(questions) == raw['n_questions'] == 5
    targets = {}
    for field, q in questions.items():
        assert set(q) <= {'type', 'instructions', 'criteria'}
        if q['type'] == 'choice':
            keys = list(q['criteria'])
            output_keys = keys
        elif q['type'] == 'score':
            keys = [str(i) for i in range(len(q['criteria']))]
            output_keys = keys
        else:
            assert q['type'] == 'noul'
            keys, output_keys = ['false', 'true'], ['no', 'yes']
        p = [float(gold[field]['probabilities'][k]) for k in keys]
        assert all(math.isfinite(v) and 0 <= v <= 1 for v in p)
        assert abs(sum(p)-1) <= 1e-4
        p = [v/sum(p) for v in p]
        # Rounded probabilities can tie. Preserve the published hard label instead
        # of silently selecting a different first-argmax answer.
        label_index = keys.index(str(gold[field]['label']).lower() if q['type'] == 'noul' else str(gold[field]['label']))
        assert max(p)-p[label_index] <= 2e-6
        targets[field] = {'label': output_keys[label_index],
                          'soft_probabilities': dict(zip(output_keys, p))}
    return {'id': raw['id'], 'workflow': raw['workflow'], 'state': state,
            'questions': questions, 'images': [], 'targets': targets}


raw = {s: pq.read_table(ROOT/'raw'/f'{s}.parquet').to_pylist() for s in ('train', 'test')}
assert len(raw['train']) == 1200 and len(raw['test']) == 400
converted = {s: [convert(r) for r in rr] for s, rr in raw.items()}
by_workflow = defaultdict(list)
for row in converted['train']:
    by_workflow[row['workflow']].append(row)
splits = {'train': [], 'validation': [], 'calibration': [], 'test': converted['test']}
for workflow, rows in sorted(by_workflow.items()):
    assert len(rows) == 300
    ordered = sorted(rows, key=lambda r: sha(('20261003:'+workflow+':'+r['id']).encode()))
    splits['train'] += ordered[:240]
    splits['validation'] += ordered[240:270]
    splits['calibration'] += ordered[270:]
seen_ids, seen_states, seen_inputs = set(), set(), set()
audit = {'dataset_revision': json.loads((ROOT/'sources/typed-meta.json').read_text())['sha'],
         'source_license': 'apache-2.0 (dataset card)',
         'split_policy': 'per workflow 240 train / 30 validation / 30 calibration cases; official test unchanged',
         'seed_key': '20261003', 'supervision': 'normalized teacher probabilities and original published hard labels; not human gold',
         'model_input_allowlist': ['state', 'questions', 'images'],
         'excluded_input_fields': ['id', 'workflow', 'targets', 'gold', 'factors', 'label_agreement'],
         'tokenization': 'not yet performed with target model tokenizer', 'splits': {}}
for split, rows in splits.items():
    ids = {r['id'] for r in rows}
    states = {sha(canonical(r['state']).encode()) for r in rows}
    inputs = {sha(canonical({'state': r['state'], 'questions': r['questions']}).encode()) for r in rows}
    assert len(ids) == len(rows)
    assert not ids & seen_ids and not states & seen_states and not inputs & seen_inputs
    seen_ids |= ids; seen_states |= states; seen_inputs |= inputs
    path = OUT/f'{split}.jsonl'
    payload = ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in rows)
    path.write_text(payload)
    audit['splits'][split] = {'cases': len(rows), 'decisions': sum(len(r['questions']) for r in rows),
                              'workflows': dict(Counter(r['workflow'] for r in rows)),
                              'question_types': dict(Counter(q['type'] for r in rows for q in r['questions'].values())),
                              'sha256': sha(path.read_bytes()), 'ids': sorted(ids)}
# Confirm these are the same test cases previously evaluated, even if IDs differ.
prior = json.loads((ROOT.parent/'new-model-20261001/laya-comparison-20261003/laya_suites_multi.json').read_text())['suites']['typed_decisions']['cases']
assert len(prior) == len(converted['test'])
for a, b in zip(prior, converted['test']):
    assert a['state'] == b['state'] and a['questions'] == b['questions']
    for field, g in a['gold'].items():
        keys = list(b['targets'][field]['soft_probabilities'])
        assert keys[g['idx']] == b['targets'][field]['label']
audit['previous_test_verified'] = {'cases': 400, 'decisions': 2000, 'state_question_label_match': True}
audit['exact_cross_split_overlap'] = {'id': 0, 'state': 0, 'state_questions': 0}
audit['semantic_near_duplicate_audit'] = 'not performed; exact matching does not prove semantic independence'
audit['raw_sha256'] = {s: sha((ROOT/'raw'/f'{s}.parquet').read_bytes()) for s in raw}
(ROOT/'data-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))
print(json.dumps({k: {x:v[x] for x in ['cases','decisions','workflows']} for k,v in audit['splits'].items()}, ensure_ascii=False, indent=2))
print('Exact split audits and prior test identity checks passed.')

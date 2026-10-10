"""Frozen Laya-suite evaluation of trained checkpoints; reuse audited typed test."""
import argparse
import hashlib
import json
import math
import numpy as np
import time
import torch
import torch_npu  # noqa: F401
from collections import defaultdict
from decision_schema import _options, compile_row
from pathlib import Path

from swift.model import get_model_processor


def normalize_question(question):
    q = {k: question[k] for k in ('type', 'instructions', 'criteria') if k in question}
    assert set(question) <= {'type', 'instructions', 'criteria'}, set(question)
    if not isinstance(q['instructions'], str):
        q['instructions'] = json.dumps(q['instructions'])
    if q['type'] == 'choice' and isinstance(q['criteria'], list):
        q['criteria'] = dict.fromkeys(q['criteria'], '')
    # Match Laya's rendering of structured criterion values and missing descriptions.

    def render(value):
        if value is None:
            return ''
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

    c = q.get('criteria')
    if isinstance(c, dict):
        q['criteria'] = {str(k): render(v) for k, v in c.items()}
    elif isinstance(c, list):
        q['criteria'] = [render(v) for v in c]
    return q


def metrics(records):
    if not records:
        return {'n': 0}
    g = np.array([r['gold_index'] for r in records])
    pred = np.array([r['prediction_index'] for r in records])
    correct = pred == g
    confidence = np.array([max(r['probabilities']) for r in records])
    f1 = []
    for k in sorted(set(g) | set(pred)):
        tp = int(((pred == k) & (g == k)).sum())
        fp = int(((pred == k) & (g != k)).sum())
        fn = int(((pred != k) & (g == k)).sum())
        f1.append(2 * tp / max(1, 2 * tp + fp + fn))
    ece = 0.0
    edges = np.linspace(0, 1, 16)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            ece += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    return {
        'n':
        len(records),
        'correct':
        int(correct.sum()),
        'accuracy':
        float(correct.mean()),
        'macro_f1':
        float(np.mean(f1)),
        'ece':
        float(ece),
        'brier':
        float(
            np.mean([sum((p - (i == r['gold_index']))**2 for i, p in enumerate(r['probabilities'])) for r in records])),
        'nll':
        float(np.mean([-math.log(max(r['probabilities'][r['gold_index']], 1e-12)) for r in records]))
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--typed-data', required=True)
    ap.add_argument('--typed-result', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--batch-size', type=int, default=8)
    a = ap.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    suites = json.loads(Path(a.dataset).read_text())['suites']
    typed_cases = [json.loads(s) for s in Path(a.typed_data).read_text().splitlines()]
    report_path = Path(a.typed_result)
    typed_report = json.loads(report_path.read_text())
    assert typed_report['checkpoint'] == a.checkpoint
    assert typed_report['data_sha256'] == hashlib.sha256(Path(a.typed_data).read_bytes()).hexdigest()
    typed_predictions = [json.loads(s) for s in report_path.with_suffix('.predictions.jsonl').read_text().splitlines()]
    typed_predictions = {(r['case_id'], r['field']): r for r in typed_predictions}
    assert len(typed_predictions) == typed_report['decisions'] == 2000
    assert len(typed_cases) == len(suites['typed_decisions']['cases']) == 400
    jobs, records, excluded = [], [], []
    for name, suite in sorted(suites.items()):
        for ci, case in enumerate(suite['cases']):
            for field, question in case['questions'].items():
                q = normalize_question(question)
                options = _options(q)
                gold = case['gold'][field]
                assert 0 <= gold['idx'] < len(options)
                if 'keys' in gold:
                    assert [str(k) for k in gold['keys']
                            ] == (['false', 'true'] if q['type'] == 'noul' else [k for k, _ in options])
                meta = {
                    'suite': name,
                    'case_index': ci,
                    'field': field,
                    'type': q['type'],
                    'gold_index': gold['idx'],
                    'option_keys': [k for k, _ in options]
                }
                if len(options) > 62:
                    excluded.append({**meta, 'reason': 'native_candidate_limit_62', 'option_count': len(options)})
                    continue
                clean = {'state': case['state'], 'questions': {field: q}}
                compiled = compile_row(clean, include_targets=False)
                assert compiled.targets is None and len(compiled.fields) == 1
                if name == 'typed_decisions':
                    original = typed_cases[ci]
                    prior = compile_row({
                        'state': original['state'],
                        'questions': {
                            field: original['questions'][field]
                        }
                    },
                                        include_targets=False)
                    assert compiled.messages == prior.messages
                    r = typed_predictions[(original['id'], field)]
                    assert r['gold_index'] == gold['idx']
                    assert len(r['probabilities']) == len(options)
                    records.append({
                        **meta, 'prediction_index': r['prediction_index'],
                        'probabilities': r['probabilities'],
                        'reused_typed_test': True
                    })
                else:
                    jobs.append((meta, compiled))
    assert len(records) == 2000 and len(jobs) == 15416 and len(excluded) == 500
    assert sum(r['prediction_index'] == r['gold_index'] for r in records) == typed_report['correct']
    manifest = {
        'checkpoint': a.checkpoint,
        'dataset_sha256': hashlib.sha256(Path(a.dataset).read_bytes()).hexdigest(),
        'source_suites': len(suites),
        'source_decisions': len(records) + len(jobs) + len(excluded),
        'fresh_decisions': len(jobs),
        'reused_typed_decisions': len(records),
        'excluded_decisions': len(excluded),
        'protocol': 'one question, original options/order/gold, no truncation, native template, candidate argmax',
        'backend': 'common MS-SWIFT HF NPU BF16 SDPA',
        'batch_size': a.batch_size,
        'padding_multiple': 128,
        'temperature': 1.0,
        'typed_protocol_exactly_matched': True
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (out / 'excluded.json').write_text(json.dumps(excluded, indent=2))
    torch.manual_seed(42)
    torch.npu.set_device(0)
    model, processor = get_model_processor(
        a.checkpoint,
        model_type='qwen3_5',
        torch_dtype=torch.bfloat16,
        device_map='npu:0',
        attn_impl='sdpa',
        new_special_tokens=['<decision>'])
    model.eval()
    tok = processor.tokenizer
    tok.padding_side = 'right'
    marker = tok.convert_tokens_to_ids('<decision>')

    def infer(items):
        texts = [
            tok.apply_chat_template(c.messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
            for _, c in items
        ]
        batch = tok(
            texts, add_special_tokens=False, padding=True, pad_to_multiple_of=128, return_tensors='pt').to('npu:0')
        assert batch['input_ids'].shape[1] <= 8192
        positions = []
        for row in batch['input_ids']:
            indices = (row == marker).nonzero().flatten()
            assert indices.numel() == 1
            positions.append(int(indices[0]) - 1)
        keep = sorted(set(positions))
        with torch.inference_mode():
            logits = model(**batch, use_cache=False, logits_to_keep=torch.tensor(keep, device='npu:0')).logits
        results = []
        for i, (meta, compiled) in enumerate(items):
            ids = [tok.encode(s, add_special_tokens=False) for s in compiled.symbols[meta['field']]]
            assert all(len(v) == 1 for v in ids)
            scores = logits[i, keep.index(positions[i]), [v[0] for v in ids]].float().cpu()
            probs = scores.softmax(-1).tolist()
            assert all(math.isfinite(v) for v in probs)
            results.append({
                **meta, 'prediction_index': int(scores.argmax()),
                'probabilities': probs,
                'input_tokens': int(batch['attention_mask'][i].sum())
            })
        return results

    smoke = []
    seen = set()
    for job in jobs:
        if job[0]['type'] not in seen:
            smoke.append(job)
            seen.add(job[0]['type'])
    singles = [infer([j])[0] for j in smoke]
    batched = infer(smoke)
    agreement = all(x['prediction_index'] == y['prediction_index'] for x, y in zip(singles, batched))
    delta = max(abs(a - b) for x, y in zip(singles, batched) for a, b in zip(x['probabilities'], y['probabilities']))
    (out / 'batch-smoke.json').write_text(
        json.dumps({
            'top1_agreement': agreement,
            'max_probability_difference': delta,
            'types': sorted(seen)
        }, indent=2))
    assert agreement and delta < 0.01
    started = time.monotonic()
    with (out / 'predictions.jsonl').open('x') as stream:
        for r in records:
            stream.write(json.dumps(r, ensure_ascii=False) + '\n')
        for offset in range(0, len(jobs), a.batch_size):
            chunk = infer(jobs[offset:offset + a.batch_size])
            for r in chunk:
                stream.write(json.dumps(r, ensure_ascii=False) + '\n')
                records.append(r)
            if offset % (a.batch_size * 25) == 0 or offset + a.batch_size >= len(jobs):
                stream.flush()
                progress = {
                    'fresh_done': min(offset + a.batch_size, len(jobs)),
                    'fresh_total': len(jobs),
                    'reused_typed': 2000,
                    'suite': chunk[-1]['suite'],
                    'elapsed_seconds': time.monotonic() - started
                }
                tmp = out / 'progress.tmp'
                tmp.write_text(json.dumps(progress))
                tmp.replace(out / 'progress.json')
                print(json.dumps(progress), flush=True)
    by_suite = defaultdict(list)
    for r in records:
        by_suite[r['suite']].append(r)
    assert len(records) == 17416 and len(by_suite) == 49
    assert len({(r['suite'], r['case_index'], r['field']) for r in records}) == len(records)
    result = {
        'manifest': manifest,
        'fresh_elapsed_seconds': time.monotonic() - started,
        'overall': metrics(records),
        'suites': {
            k: metrics(v)
            for k, v in by_suite.items()
        }
    }
    (out / 'summary.json').write_text(json.dumps(result, indent=2))
    print('COMPLETE', json.dumps(result['overall']), flush=True)


if __name__ == '__main__':
    main()

"""Fixed per-question candidate scoring, isolated from training/selection."""
import argparse
import hashlib
import json
import time
import torch
import torch_npu  # noqa: F401
from collections import defaultdict
from decision_schema import _options, compile_row
from pathlib import Path

from swift.model import get_model_processor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--limit', type=int, default=0, help='Smoke check only; zero means full evaluation')
    a = ap.parse_args()
    output = Path(a.output)
    if output.exists():
        raise RuntimeError('Use a new evaluation output')
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
    marker_id = tok.convert_tokens_to_ids('<decision>')
    jobs = []
    for line in Path(a.data).read_text().splitlines():
        case = json.loads(line)
        for field, question in case['questions'].items():
            clean = {'state': case['state'], 'questions': {field: question}}
            c = compile_row(clean, include_targets=False)
            text = tok.apply_chat_template(
                c.messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
            options = _options(question)
            symbols = c.symbols[field]
            candidate_ids = [tok.encode(s, add_special_tokens=False) for s in symbols]
            assert all(len(v) == 1 for v in candidate_ids)
            gold = str(case['targets'][field]['label'])
            keys = [k for k, _ in options]
            if question['type'] == 'noul':
                gold = 'yes' if gold.lower() in {'yes', 'true', '1'} else 'no'
            assert gold in keys
            jobs.append((text, [v[0] for v in candidate_ids], keys.index(gold), {
                'case_id': case['id'],
                'field': field,
                'workflow': case['workflow']
            }))
    if a.limit:
        jobs = jobs[:a.limit]
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    start = time.monotonic()
    with output.with_suffix('.predictions.jsonl').open('x') as stream, torch.inference_mode():
        for offset in range(0, len(jobs), a.batch_size):
            batch_jobs = jobs[offset:offset + a.batch_size]
            batch = tok([j[0] for j in batch_jobs],
                        add_special_tokens=False,
                        padding=True,
                        pad_to_multiple_of=128,
                        return_tensors='pt').to('npu:0')
            assert batch['input_ids'].shape[1] <= 8192
            positions = []
            for row in batch['input_ids']:
                indices = (row == marker_id).nonzero().flatten()
                assert indices.numel() == 1
                positions.append(int(indices[0]) - 1)
            keep = sorted(set(positions))
            logits = model(**batch, use_cache=False, logits_to_keep=torch.tensor(keep, device='npu:0')).logits
            for i, job in enumerate(batch_jobs):
                scores = logits[i, keep.index(positions[i]), job[1]].float().cpu()
                probs = scores.softmax(-1).tolist()
                record = {
                    **job[3], 'gold_index': job[2],
                    'prediction_index': int(scores.argmax()),
                    'probabilities': probs
                }
                records.append(record)
                stream.write(json.dumps(record) + '\n')
            if offset % 100 == 0:
                stream.flush()
                print(f'evaluated {len(records)}/{len(jobs)}', flush=True)
    by_workflow = defaultdict(list)
    for r in records:
        by_workflow[r['workflow']].append(r)

    def stats(rs):
        correct = sum(r['prediction_index'] == r['gold_index'] for r in rs)
        return {'decisions': len(rs), 'correct': correct, 'accuracy': correct / len(rs)}

    report = {
        **stats(records), 'by_workflow': {
            k: stats(v)
            for k, v in by_workflow.items()
        },
        'checkpoint': a.checkpoint,
        'data_sha256': hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),
        'elapsed_seconds': time.monotonic() - start,
        'protocol': 'one question, original state/options/order, candidate softmax, no truncation',
        'backend': 'MS-SWIFT HF NPU BF16 SDPA',
        'batch_size': a.batch_size,
        'padding_multiple': 128,
        'smoke_limit': a.limit
    }
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

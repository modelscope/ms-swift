"""Independent joint-field candidate evaluation, one complete case per forward."""
import argparse
import hashlib
import json
import time
import torch
import torch_npu  # noqa: F401
from decision_schema import compile_row
from pathlib import Path

from swift.model import get_model_processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument(
        '--data', type=Path, required=True, help='Original prepared case JSONL, not tokenized training rows')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0, help='Nonzero is smoke only')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = [json.loads(line) for line in args.data.read_text().splitlines()]
    if args.limit:
        cases = cases[:args.limit]
    assert cases and len({case['id'] for case in cases}) == len(cases)
    torch.manual_seed(42)
    torch.npu.set_device(0)
    model, processor = get_model_processor(
        args.checkpoint,
        model_type='qwen3_5',
        torch_dtype=torch.bfloat16,
        device_map='npu:0',
        attn_impl='sdpa',
        new_special_tokens=['<decision>'])
    model.eval()
    tokenizer = processor.tokenizer
    marker = tokenizer.convert_tokens_to_ids('<decision>')
    correct = total = 0
    started = time.monotonic()
    with (args.output / 'predictions.jsonl').open('x') as stream, torch.inference_mode():
        for case in cases:
            compiled = compile_row(case, include_targets=False)
            gold = compile_row(case, include_targets=True).targets
            assert compiled.targets is None
            text = tokenizer.apply_chat_template(
                compiled.messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
            inputs = tokenizer(text, add_special_tokens=False, return_tensors='pt').to('npu:0')
            assert inputs['input_ids'].shape[1] <= 8192
            positions = (inputs['input_ids'][0] == marker).nonzero().flatten()
            assert len(positions) == len(compiled.fields) and bool((positions > 0).all())
            scores = model(**inputs, use_cache=False, logits_to_keep=positions - 1).logits[0]
            for index, field in enumerate(compiled.fields):
                symbols = compiled.symbols[field]
                ids = [tokenizer.encode(symbol, add_special_tokens=False) for symbol in symbols]
                assert all(len(token_ids) == 1 for token_ids in ids)
                logits = scores[index, [token_ids[0] for token_ids in ids]].float().cpu()
                assert bool(torch.isfinite(logits).all())
                prediction = int(logits.argmax())
                expected = symbols.index(gold[field])
                record = {
                    'case_id': case['id'],
                    'field': field,
                    'gold_index': expected,
                    'prediction_index': prediction,
                    'symbols': symbols,
                    'probabilities': logits.softmax(-1).tolist()
                }
                stream.write(json.dumps(record) + '\n')
                correct += prediction == expected
                total += 1
            stream.flush()
    assert total == sum(len(case['questions']) for case in cases)
    result = {
        'cases': len(cases),
        'decisions': total,
        'correct': correct,
        'accuracy': correct / total,
        'protocol': 'joint fields, original order, one complete case per forward, no padding or truncation',
        'checkpoint': args.checkpoint,
        'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
        'backend': 'common MS-SWIFT HF NPU BF16 SDPA',
        'temperature': 1.0,
        'elapsed_seconds': time.monotonic() - started,
        'smoke_limit': args.limit
    }
    (args.output / 'metrics.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

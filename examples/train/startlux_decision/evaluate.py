# Copyright (c) ModelScope Contributors. All rights reserved.
"""Compare an adapter with its original checkpoint on disjoint labeled cases."""
import argparse
import contextlib
import hashlib
import json
import torch
from decision_training import audit_splits, candidate_loss, collate, encode_records, install_head
from pathlib import Path
from peft import PeftModel
from transformers import AutoConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--train', required=True, help='Used only to reject overlapping cases')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config = AutoConfig.from_pretrained(args.model)
    from swift.model import get_model_processor
    model_type = 'qwen3_5_moe' if 'Moe' in config.architectures[0] else 'qwen3_5'
    model, processor = get_model_processor(
        args.model, model_type=model_type, torch_dtype=torch.bfloat16, device_map='auto', attn_impl='sdpa')
    tokenizer = getattr(processor, 'tokenizer', processor)
    install_head(model, args.model, tokenizer)
    rows, identities = encode_records(args.data, tokenizer, with_gold=True)
    _, train_ids = encode_records(args.train, tokenizer)
    audit_splits(train_ids, identities)
    model = PeftModel.from_pretrained(model, args.adapter).eval()
    device = model.get_input_embeddings().weight.device
    totals = {mode: {'correct': 0, 'total': 0, 'loss_sum': 0.} for mode in ('base', 'adapter')}
    with torch.inference_mode(), (output / 'predictions.jsonl').open('w') as stream:
        for index, row in enumerate(rows):
            batch = collate([row], tokenizer.pad_token_id)
            targets = batch.pop('labels').to(device)
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            record = {'case_id': row['case_id'], 'question_id': row['question_id'], 'gold': row['gold']}
            for mode in totals:
                context = model.disable_adapter() if mode == 'base' else contextlib.nullcontext()
                with context:
                    logits = model(**batch).logits
                loss = candidate_loss(logits, batch['attention_mask'], targets).item()
                selected = logits[0, -1].float().masked_fill(targets[0] < 0, -torch.inf)
                if not torch.isfinite(selected[targets[0] >= 0]).all():
                    raise RuntimeError('Non-finite candidate logits')
                prediction = int(selected.argmax())
                totals[mode]['correct'] += prediction == row['gold']
                totals[mode]['total'] += 1
                totals[mode]['loss_sum'] += loss
                record[mode] = {'prediction': prediction, 'probabilities_t1': selected.softmax(-1).tolist()}
            stream.write(json.dumps(record) + '\n')
            if (index + 1) % 100 == 0:
                print(json.dumps({'decisions': index + 1, 'total': len(rows)}), flush=True)
    for result in totals.values():
        result['accuracy'] = result['correct'] / result['total']
        result['mean_soft_ce'] = result.pop('loss_sum') / result['total']
    report = {
        'results':
        totals,
        'accuracy_gain_percentage_points':
        100 * (totals['adapter']['accuracy'] - totals['base']['accuracy']),
        'data_sha256':
        hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        'adapter_sha256':
        hashlib.sha256((Path(args.adapter) / 'adapter_model.safetensors').read_bytes()).hexdigest(),
        'arguments':
        vars(args),
        'scope':
        'Hard-label decision accuracy; not seven-suite macro accuracy. '
        'Probabilities use T=1; re-fit calibration on a separate calibration split.'
    }
    (output / 'metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report['results']), flush=True)


if __name__ == '__main__':
    main()

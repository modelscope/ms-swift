import copy
import decision_plugin  # noqa: F401
import json
import numpy as np
from collections.abc import Mapping
from pathlib import Path

from swift.model import get_processor
from swift.template import get_template

processor = get_processor('/models/Qwen3.5-4B', new_special_tokens=['<decision>'])
tokenizer = processor.tokenizer
template = get_template(
    processor,
    template_type='intern_decision_training',
    max_length=8192,
    truncation_strategy='raise',
    enable_thinking=False,
    remove_unused_columns=False)
template.set_mode('train')
report = {}
for split in ('train', 'validation', 'calibration', 'test'):
    rows = [json.loads(s) for s in Path(f'/data/joint/{split}.jsonl').read_text().splitlines()]
    lengths = []
    for i, row in enumerate(rows):
        encoded = template.encode(copy.deepcopy(row))
        labels = encoded['labels']
        pos = [j for j, x in enumerate(labels) if x != -100]
        assert len(pos) == len(row['decision_targets']) and min(pos) > 0
        assert all(encoded['input_ids'][p] == tokenizer.convert_tokens_to_ids('<decision>') for p in pos)
        assert [tokenizer.decode([labels[p]]) for p in pos] == row['decision_targets']
        assert '_extra_kwargs' not in encoded
        lengths.append(len(encoded['input_ids']))
        if i < 10:
            hf_ids = tokenizer.apply_chat_template(
                row['messages'], tokenize=True, add_generation_prompt=False, enable_thinking=False)
            if isinstance(hf_ids, Mapping):
                hf_ids = hf_ids['input_ids']
            mismatch = (split, i, 'HF prompt mismatch', tokenizer.decode(encoded['input_ids']),
                        tokenizer.decode(hf_ids))
            assert encoded['input_ids'] == hf_ids, mismatch
            altered = copy.deepcopy(row)
            altered['decision_targets'] = ['B' if target == 'A' else 'A' for target in row['decision_targets']]
            check = template.encode(altered)
            assert check['input_ids'] == encoded['input_ids']
            assert check['labels'] != labels
    report[split] = {
        'examples': len(rows),
        'supervised_tokens': sum(len(r['decision_targets']) for r in rows),
        'min': min(lengths),
        'max': max(lengths),
        'p50': float(np.percentile(lengths, 50)),
        'p95': float(np.percentile(lengths, 95)),
        'rejected': 0
    }
    print(split, report[split], flush=True)
report['marker_id'] = tokenizer.convert_tokens_to_ids('<decision>')
report['tokenizer_length'] = len(tokenizer)
report['prompt_matches_hf_checked_per_split'] = 10
Path('/workspace/template-audit.json').write_text(json.dumps(report, indent=2))

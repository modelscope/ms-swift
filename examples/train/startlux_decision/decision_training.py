# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared StartLux training contract: official prompts and candidate-only soft CE."""
import hashlib
import json
import math
import torch
from pathlib import Path
from torch import nn
from torch.nn import functional as F


class CandidateHead(nn.Module):
    """Keep the original checkpoint weight/name; project legal letters in FP32."""

    def __init__(self, head, letter_ids):
        super().__init__()
        if head.bias is not None or len(set(letter_ids)) != 26:
            raise ValueError('Expected a bias-free head and 26 distinct letters')
        self.weight = head.weight
        self.weight.requires_grad_(False)
        self.register_buffer('letter_ids', torch.tensor(letter_ids), persistent=False)

    def forward(self, hidden):
        with torch.autocast(hidden.device.type, enabled=False):
            weight = self.weight.index_select(0, self.letter_ids.to(self.weight.device))
            return F.linear(hidden.float(), weight.float())


def install_head(model, checkpoint, tokenizer):
    from startlux_decision import jevfmt
    config = json.loads((Path(checkpoint) / 'decision_config.json').read_text())
    letters = jevfmt.check_tokenizer(tokenizer)
    if letters != config['letter_token_ids']:
        raise ValueError('Checkpoint and tokenizer letters differ')
    model.set_output_embeddings(CandidateHead(model.get_output_embeddings(), letters))
    model.config.use_cache = False
    if hasattr(model.config, 'text_config'):
        model.config.text_config.use_cache = False


def contains_media(value):
    """Reject media blocks without rejecting ordinary JSON business evidence."""
    if isinstance(value, dict):
        media_keys = {'image', 'image_url', 'video', 'video_url', 'images', 'videos'}
        return (bool(media_keys.intersection(value))
                or value.get('type') in ('image', 'image_url', 'video', 'video_url')
                or any(contains_media(item) for item in value.values()))
    if isinstance(value, list):
        return any(contains_media(item) for item in value)
    return False


def encode_records(path, tokenizer, max_length=8192, with_gold=False):
    from startlux_decision import jevfmt
    features, identities = [], set()
    for line_number, text in enumerate(Path(path).read_text().splitlines(), 1):
        if not text.strip():
            continue
        record = json.loads(text)
        if record.get('images') or record.get('videos') or contains_media(record['state']):
            raise ValueError('This training recipe is text-only')
        identity = hashlib.sha256(
            json.dumps([record['state'], record['questions']], sort_keys=True,
                       ensure_ascii=False).encode()).hexdigest()
        if identity in identities:
            raise ValueError(f'Duplicate case at {path}:{line_number}')
        identities.add(identity)
        if set(record['targets']) - set(record['questions']):
            raise ValueError('Target references an unknown question')
        for key, question in record['questions'].items():
            if key not in record['targets']:
                continue
            row = jevfmt.from_systemone(record['state'], question, qid=key)
            order = [option['id'] for option in row['options']]
            if not 2 <= len(order) <= 26:
                raise ValueError('Split wide choices before training; no silent filtering')
            target = record['targets'][key]
            if set(target) - set(order):
                raise ValueError('Target references an unknown option')
            values = [float(target.get(option, 0)) for option in order]
            if any(not math.isfinite(v) or v < 0 for v in values) or math.fsum(values) <= 0:
                raise ValueError('Targets must have finite nonnegative mass')
            ids, rendered_order = jevfmt.render_ids(row, tokenizer, order, max_length=max_length)
            if rendered_order != order:
                raise ValueError('Renderer changed candidate order')
            features.append({
                'input_ids': ids,
                'labels': [v / math.fsum(values) for v in values] + [-1.] * (26 - len(values))
            })
            if with_gold:
                gold = str(record['gold'][key])
                if gold not in order:
                    raise ValueError('Gold label is not a legal candidate')
                features[-1]['gold'] = order.index(gold)
                features[-1]['case_id'] = identity
                features[-1]['question_id'] = key
    if not features:
        raise ValueError('No supervised questions')
    return features, identities


def collate(features, pad_token_id):
    length = max(len(row['input_ids']) for row in features)
    ids = torch.full((len(features), length), pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(features):
        size = len(row['input_ids'])
        ids[i, :size] = torch.tensor(row['input_ids'])
        mask[i, :size] = 1
    return {
        'input_ids': ids,
        'attention_mask': mask,
        'labels': torch.tensor([r['labels'] for r in features], dtype=torch.float32),
        'use_cache': False
    }


def candidate_loss(logits, attention_mask, targets):
    if logits.ndim != 3 or logits.shape[-1] != 26:
        raise ValueError('Expected [batch, sequence, 26] candidate logits')
    positions = attention_mask.sum(-1) - 1
    selected = logits[torch.arange(logits.shape[0], device=logits.device), positions].float()
    valid = targets >= 0
    log_probs = F.log_softmax(selected.masked_fill(~valid, -torch.inf), -1).masked_fill(~valid, 0)
    return -(targets.clamp_min(0) * log_probs).sum(-1)


def audit_splits(train, validation):
    overlap = train & validation
    if overlap:
        raise ValueError(f'Train/validation overlap: {len(overlap)} cases')


def copy_metadata(checkpoint, output):
    """Retain tokenizer/protocol; old fitted temperatures require recalibration."""
    import shutil
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name in [
            'decision_config.json', 'chat_template.jinja', 'preprocessor_config.json', 'processor_config.json',
            'tokenizer_config.json', 'tokenizer.json'
    ]:
        source = Path(checkpoint) / name
        if source.exists() and not (output / name).exists():
            shutil.copy2(source, output / name)


def audit_trainable(model):
    names = {name: parameter.numel() for name, parameter in model.named_parameters() if parameter.requires_grad}
    if not names or any('lora_' not in name or 'visual' in name or 'lm_head' in name for name in names):
        raise ValueError('Only language-backbone LoRA parameters may be trainable')
    return {'parameters': sum(names.values()), 'tensors': names}

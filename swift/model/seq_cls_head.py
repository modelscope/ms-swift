# Copyright (c) ModelScope Contributors. All rights reserved.
"""Independent sequence heads with an ordered, checkpoint-stable label layout."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def normalize_seq_cls_heads(heads):
    if not isinstance(heads, list) or not heads:
        raise ValueError('seq_cls_heads must be a nonempty list of head configurations.')
    result = []
    names = set()
    for head in heads:
        if not isinstance(head, dict) or set(head) - {'name', 'num_labels', 'problem_type', 'weight'}:
            raise ValueError(f'Invalid sequence head: {head!r}')
        name, size = head.get('name'), head.get('num_labels')
        if not isinstance(name, str) or not name or name in names:
            raise ValueError('Each sequence head needs a unique nonempty name.')
        if type(size) is not int or size < 1:
            raise ValueError('Each sequence head needs a positive integer num_labels.')
        problem = head.get('problem_type', 'regression' if size == 1 else 'single_label_classification')
        if problem not in {'regression', 'single_label_classification', 'multi_label_classification'}:
            raise ValueError(f'Invalid problem_type for {name}: {problem}')
        weight = float(head.get('weight', 1.0))
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError('Head weights must be finite and positive.')
        names.add(name)
        result.append({'name': name, 'num_labels': size, 'problem_type': problem, 'weight': weight})
    return result


def label_width(head):
    return 1 if head['problem_type'] == 'single_label_classification' else head['num_labels']


def encode_seq_cls_labels(labels, heads):
    if not isinstance(labels, dict) or set(labels) != {head['name'] for head in heads}:
        raise ValueError('Multi-head labels must contain exactly one entry for every configured head.')
    packed = []
    for head in heads:
        value = labels[head['name']]
        size, problem = head['num_labels'], head['problem_type']
        if problem == 'single_label_classification':
            if type(value) is not int or not 0 <= value < size:
                raise ValueError(f"Invalid class label for {head['name']}: {value}")
            values = [value]
        elif problem == 'multi_label_classification':
            if not isinstance(value, list) or any(type(i) is not int or not 0 <= i < size for i in value):
                raise ValueError(f"Invalid active class indices for {head['name']}: {value}")
            values = [float(i in value) for i in range(size)]
        else:
            values = value if isinstance(value, list) else [value]
            if len(values) != size or any(not math.isfinite(float(v)) for v in values):
                raise ValueError(f"Invalid regression target for {head['name']}: {value}")
        packed.extend(float(v) for v in values)
    return packed


class MultiHeadSequenceClassifier(nn.Module):

    def __init__(self, hidden_size, heads, *, dtype=None, initializer_range=0.02):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(hidden_size, h['num_labels'], bias=False, dtype=dtype) for h in heads])
        for head in self.heads:
            if head.weight.device.type == 'meta':
                head.to_empty(device='cpu')
            nn.init.normal_(head.weight, mean=0.0, std=initializer_range)

    @property
    def weight(self):
        # The existing pooling patch and PEFT inspect the score module's dtype/device.
        return self.heads[0].weight

    def forward(self, hidden_states):
        return torch.cat([head(hidden_states) for head in self.heads], dim=-1)


def multi_head_loss(logits, labels, heads):
    expected = sum(label_width(h) for h in heads)
    if labels.ndim != 2 or labels.shape != (logits.shape[0], expected):
        raise ValueError(f'Multi-head labels must have shape ({logits.shape[0]}, {expected}).')
    loss = logits.new_zeros(())
    logit_offset = label_offset = 0
    for head in heads:
        size, width = head['num_labels'], label_width(head)
        prediction = logits[:, logit_offset:logit_offset + size]
        target = labels[:, label_offset:label_offset + width].to(prediction.device)
        problem = head['problem_type']
        if problem == 'single_label_classification':
            value = F.cross_entropy(prediction, target[:, 0].long())
        elif problem == 'multi_label_classification':
            value = F.binary_cross_entropy_with_logits(prediction, target.to(prediction.dtype))
        else:
            value = F.mse_loss(prediction, target.to(prediction.dtype))
        loss = loss + head['weight'] * value
        logit_offset += size
        label_offset += width
    return loss


def multi_head_metrics(logits, labels, heads):
    metrics = {}
    logit_offset = label_offset = 0
    for head in heads:
        size, width = head['num_labels'], label_width(head)
        prediction = logits[:, logit_offset:logit_offset + size]
        target = labels[:, label_offset:label_offset + width].to(prediction.device)
        if head['problem_type'] == 'single_label_classification':
            values = prediction.argmax(-1) == target[:, 0]
            metric = 'acc'
        elif head['problem_type'] == 'multi_label_classification':
            values = ((prediction >= 0) == target.bool()).all(-1)
            metric = 'acc'
        else:
            values = (prediction - target).float().square().mean(-1)
            metric = 'mse'
        metrics[f"{head['name']}_{metric}"] = values.float()
        logit_offset += size
        label_offset += width
    return metrics

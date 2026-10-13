# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import tempfile
import torch
import unittest
from pathlib import Path
from torch.nn import functional as F
from types import SimpleNamespace

from swift.metrics.acc import AccMetrics
from swift.model.seq_cls_head import (MultiHeadSequenceClassifier, encode_seq_cls_labels, multi_head_loss,
                                      multi_head_metrics, normalize_seq_cls_heads)
from swift.template.base import Template


class TestSequenceHeads(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(10314)
        self.heads = normalize_seq_cls_heads([
            {
                'name': 'category',
                'num_labels': 3
            },
            {
                'name': 'tags',
                'num_labels': 2,
                'problem_type': 'multi_label_classification',
                'weight': 0.5
            },
            {
                'name': 'quality',
                'num_labels': 1,
                'weight': 0.25
            },
        ])

    def test_weighted_values_and_gradients(self):
        logits = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([[0, 1, 0, .2], [1, 0, 1, .4], [2, 1, 1, .6], [0, 0, 0, .8]], dtype=torch.float64)
        expected = (
            F.cross_entropy(logits[:, :3], labels[:, 0].long())
            + .5 * F.binary_cross_entropy_with_logits(logits[:, 3:5], labels[:, 1:3])
            + .25 * F.mse_loss(logits[:, 5:], labels[:, 3:]))
        actual = multi_head_loss(logits, labels, self.heads)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(torch.autograd.grad(actual, logits)[0], torch.autograd.grad(expected, logits)[0])
        self.assertTrue(torch.autograd.gradcheck(lambda x: multi_head_loss(x, labels, self.heads), (logits, )))

    def test_independent_heads_and_backbone_gradients(self):
        hidden = torch.randn(4, 8, requires_grad=True)
        score = MultiHeadSequenceClassifier(8, self.heads)
        labels = torch.tensor([[0, 1, 0, .2]] * 4)
        multi_head_loss(score(hidden), labels, self.heads).backward()
        self.assertGreater(hidden.grad.abs().sum(), 0)
        for head in score.heads:
            self.assertTrue(torch.isfinite(head.weight.grad).all())
            self.assertGreater(head.weight.grad.abs().sum(), 0)
        before = [h.weight.detach().clone() for h in score.heads]
        with torch.no_grad():
            score.heads[0].weight.add_(1)
        for i in [1, 2]:
            torch.testing.assert_close(score.heads[i].weight, before[i], rtol=0, atol=0)

    def test_labels_decode_and_metrics(self):
        self.assertEqual(
            encode_seq_cls_labels({
                'category': 2,
                'tags': [0],
                'quality': .5
            }, self.heads), [2., 1., 0., .5])
        template = object.__new__(Template)
        template.config = SimpleNamespace(seq_cls_heads=self.heads, problem_type='single_label_classification')
        logits = torch.tensor([[0., 1., 2., 2., -2., .5]])
        predictions, logprobs = template.decode_seq_cls(logits, 2)
        self.assertEqual(predictions, [{'category': 2, 'tags': [0], 'quality': .5}])
        self.assertEqual(set(logprobs[0]), {'category', 'tags', 'quality'})
        labels = torch.tensor([[2., 1., 0., .5]])
        metrics = multi_head_metrics(logits, labels, self.heads)
        self.assertEqual({
            k: v.item()
            for k, v in metrics.items()
        }, {
            'category_acc': 1.,
            'tags_acc': 1.,
            'quality_mse': 0.
        })
        metric = AccMetrics(SimpleNamespace(), SimpleNamespace(template=template))
        torch.testing.assert_close(metric.preprocess_logits_for_metrics(logits, labels), logits)

    def test_checkpoint_layout_preserves_head_order(self):
        # Transformers writes config JSON with sort_keys=True; the head list must retain its order.
        restored = normalize_seq_cls_heads(json.loads(json.dumps(self.heads, sort_keys=True)))
        self.assertEqual(restored, self.heads)
        original = MultiHeadSequenceClassifier(8, self.heads)
        replacement = MultiHeadSequenceClassifier(8, restored)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.pt'
            torch.save(original.state_dict(), path)
            replacement.load_state_dict(torch.load(path, weights_only=True))
        hidden = torch.randn(3, 8)
        torch.testing.assert_close(original(hidden), replacement(hidden), rtol=0, atol=0)

    def test_invalid_configuration_and_targets(self):
        for heads in [[], [{
                'name': 'x',
                'num_labels': 0
        }], [{
                'name': 'x',
                'num_labels': 2
        }] * 2, [{
                'name': 'x',
                'num_labels': 2,
                'weight': float('nan')
        }]]:
            with self.subTest(heads=heads), self.assertRaises(ValueError):
                normalize_seq_cls_heads(heads)
        for label in [{
                'category': 3,
                'tags': [],
                'quality': 0
        }, {
                'category': 1,
                'tags': [2],
                'quality': 0
        }, {
                'category': 1,
                'tags': []
        }, {
                'category': 1,
                'tags': [],
                'quality': float('inf')
        }]:
            with self.subTest(label=label), self.assertRaises(ValueError):
                encode_seq_cls_labels(label, self.heads)


if __name__ == '__main__':
    unittest.main()

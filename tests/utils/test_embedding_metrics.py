# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import numpy as np
import os
import tempfile
import unittest
from transformers import EvalPrediction

from swift.metrics.embedding import InfonceMetrics
from swift.utils import JsonlWriter, check_json_format

# Two queries, each made of an anchor, a positive and a negative sample.
DIM = 8
SENTENCES = np.arange(6 * DIM, dtype=np.float32).reshape(6, DIM) / 100
LABELS = np.array([1, 0, 0, 1, 0, 0], dtype=np.int64)


class TestInfonceMetrics(unittest.TestCase):

    def setUp(self):
        self.metric = InfonceMetrics(None, None)
        self.metrics = self.metric.compute_metrics(EvalPrediction(predictions=SENTENCES, label_ids=LABELS))

    def test_metrics_are_python_floats(self):
        for name, value in self.metrics.items():
            with self.subTest(name=name):
                self.assertIsInstance(value, float)
                self.assertNotIsInstance(value, np.generic)

    def test_metrics_can_be_logged_as_json(self):
        # The eval metrics are written to logging.jsonl; a numpy scalar made the writer raise.
        self.assertEqual(json.loads(json.dumps(check_json_format(self.metrics))), json.loads(json.dumps(self.metrics)))

    def test_metrics_can_be_written_to_the_jsonl_log(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fpath = os.path.join(tmp_dir, 'logging.jsonl')
            JsonlWriter(fpath).append({'eval_mean_neg': self.metrics['mean_neg']})
            with open(fpath) as f:
                self.assertIn('eval_mean_neg', json.loads(f.read()))


if __name__ == '__main__':
    unittest.main()

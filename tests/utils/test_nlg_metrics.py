import unittest
from types import SimpleNamespace
from unittest.mock import patch

from swift.metrics.nlg import compute_rouge_bleu


# Keep aggregation tests independent from the optional jieba dependency.
@patch.dict('sys.modules', {'jieba': SimpleNamespace(cut=str.split)})
class TestNlgMetrics(unittest.TestCase):

    def test_exact_match_scores_full_points(self):
        scores = compute_rouge_bleu(['the cat is here'], ['the cat is here'])

        self.assertEqual(scores, {
            'rouge-1': 100.0,
            'rouge-2': 100.0,
            'rouge-l': 100.0,
            'bleu-4': 100.0,
        })

    def test_empty_prediction_counts_towards_mean(self):
        exact_match = 'the cat is here'
        scores = compute_rouge_bleu([exact_match, ''], [exact_match, 'a different reference'])

        self.assertEqual(scores, {
            'rouge-1': 50.0,
            'rouge-2': 50.0,
            'rouge-l': 50.0,
            'bleu-4': 50.0,
        })


if __name__ == '__main__':
    unittest.main()

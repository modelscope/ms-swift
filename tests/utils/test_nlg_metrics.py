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

    def test_period_only_prediction_counts_towards_mean(self):
        exact_match = 'the cat is here'
        for prediction in ['.', '...', '. .']:
            with self.subTest(prediction=prediction):
                scores = compute_rouge_bleu([exact_match, prediction], [exact_match, 'a different reference'])
                self.assertEqual(scores, {'rouge-1': 50.0, 'rouge-2': 50.0, 'rouge-l': 50.0, 'bleu-4': 50.0})

    def test_period_only_reference(self):
        scores = compute_rouge_bleu(['the cat is here'], ['.'])
        self.assertEqual(scores, {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0, 'bleu-4': 0.0})

    def test_bleu_keeps_period_tokens(self):
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
        scores = compute_rouge_bleu(['.'], ['.'])
        expected_bleu = sentence_bleu([['.']], ['.'], smoothing_function=SmoothingFunction().method3)
        self.assertEqual(scores['bleu-4'], round(expected_bleu * 100, 6))
        self.assertGreater(scores['bleu-4'], 0)
        for key in ['rouge-1', 'rouge-2', 'rouge-l']:
            self.assertEqual(scores[key], 0)

    def test_ordinary_punctuation_keeps_rouge_scores(self):
        from rouge.rouge import Rouge
        text = 'the cat is here.'
        scores = compute_rouge_bleu([text], [text])
        reference = Rouge().get_scores(text, text)[0]
        for key, value in reference.items():
            self.assertEqual(scores[key], round(value['f'] * 100, 6))

    def test_spaced_periods_keep_existing_rouge_behavior(self):
        from rouge.rouge import Rouge
        text = '. .'
        scores = compute_rouge_bleu([text], [text])
        reference = Rouge().get_scores(text, text)[0]
        for key, value in reference.items():
            self.assertEqual(scores[key], round(value['f'] * 100, 6))

    def test_unrelated_rouge_errors_are_not_suppressed(self):
        with patch('rouge.rouge.Rouge.get_scores', side_effect=ValueError('unrelated failure')):
            with self.assertRaisesRegex(ValueError, 'unrelated failure'):
                compute_rouge_bleu(['the cat is here'], ['the cat is here'])


if __name__ == '__main__':
    unittest.main()

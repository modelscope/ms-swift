# Copyright (c) ModelScope Contributors. All rights reserved.
import unittest
from unittest.mock import patch

from swift.rewards.orm import ReactORM


class TestToolbenchReward(unittest.TestCase):

    def test_plain_text_action_inputs(self):
        reward = ReactORM()
        reference = 'Action: search\nAction Input: New York weather'
        predictions = [
            reference,
            'Action: search\nAction Input: New York weather today',
            'Action: search\nAction Input: unrelated words',
            'Action: other_tool\nAction Input: New York weather',
        ]
        expected = [1.0, 1.0, 0.0, 0.0]
        self.assertEqual(reward(predictions, [reference] * len(predictions)), expected)
        requests = [{'messages': [{'role': 'assistant', 'content': text}]} for text in predictions]
        self.assertEqual(reward(requests, [reference] * len(predictions)), expected)

    def test_normalized_rouge_thresholds(self):
        for score, expected in [(None, False), (0.0, False), (0.09, False), (0.1, False), (0.19, False), (0.2, True),
                                (0.5, True), (1.0, True)]:
            with self.subTest(score=score), patch.object(ReactORM, 'evaluate_rougel', return_value=score):
                self.assertEqual(
                    ReactORM.evaluate_action_reward(['search'], ['search'], ['query'], ['query']), expected)

    def test_json_and_invalid_input_controls(self):
        cases = [
            ('{"city": "New York"}', '{"city": "New York"}', True),
            ('{"city": "New York"}', '{"city": "London"}', False),
            ('{"city": "New York", "unit": "C"}', '{"city": "New York"}', False),
            ('{}', '{}', True),
            ('{}', '{"city": "New York"}', False),
            ('New York', '{"city": "New York"}', False),
            ('[1, 2]', '[1, 2]', False),
            ('', '', False),
        ]
        for prediction, reference, expected in cases:
            with self.subTest(prediction=prediction, reference=reference):
                self.assertEqual(
                    ReactORM.evaluate_action_reward(['search'], ['search'], [prediction], [reference]), expected)


if __name__ == '__main__':
    unittest.main()

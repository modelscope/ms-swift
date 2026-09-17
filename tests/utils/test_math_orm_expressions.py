# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import unittest
from unittest.mock import patch

from swift.infer_engine import InferRequest
from swift.rewards.orm import MathORM


class TestMathORMExpressions(unittest.TestCase):

    def test_boxed_extraction_preserves_groups(self):
        cases = [
            (r'Answer: \boxed{\frac{1}{23}}.', r'\frac{1}{23}'),
            (r'\boxed{x^{12} + \sqrt{3}}', r'x^{12} + \sqrt{3}'),
            (r'\boxed{\{1, 2\}}', r'\{1, 2\}'),
            ('\\boxed{\n\\frac{1}{2}\n}', r'\frac{1}{2}'),
            (r'\boxed{42} then \boxed{43}', '42'),
            ('42', '42'),
            (r'\boxed{\frac{1}{2}', r'\boxed{\frac{1}{2}'),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(MathORM.extract_boxed_result(text), expected)

    def test_distinct_nested_answers_do_not_get_exact_match_reward(self):
        pairs = [
            (r'\boxed{\frac{1}{2}}', r'\boxed{\frac{1}{3}}'),
            (r'\boxed{\frac{1}{23}}', r'\boxed{\frac{12}{3}}'),
            (r'\frac{1}{23}', r'\frac{12}{3}'),
            (r'x^{12}', r'x^{1}2'),
        ]
        with patch.dict(os.environ, {'USE_OPENCOMPASS_EVALUATOR': 'False'}):
            reward = MathORM()
        # A missing optional parser must not turn different expressions into an exact match.
        with patch.object(MathORM, 'parse_expression', return_value=None):
            for prediction, solution in pairs:
                with self.subTest(prediction=prediction, solution=solution):
                    request = InferRequest(messages=[{'role': 'assistant', 'content': prediction}])
                    self.assertEqual(reward([request], [solution]), [0.0])
            for prediction, solution in [(r'\boxed{\frac{1}{2}}', r'\frac{1}{2}'), ('{42}', '42')]:
                with self.subTest(prediction=prediction, solution=solution):
                    request = InferRequest(messages=[{'role': 'assistant', 'content': prediction}])
                    self.assertEqual(reward([request], [solution]), [1.0])

    def test_latex_parser_receives_intact_expressions(self):
        from sympy import Rational
        with patch.object(MathORM, 'parse_expression', side_effect=[Rational(1, 2), Rational(2, 4)]) as parse:
            self.assertTrue(MathORM.compare_consecutive(r'\(\frac{1}{2}\)', r'\[\frac{2}{4}\]'))
        self.assertEqual([call.args[0] for call in parse.call_args_list], [r'\frac{1}{2}', r'\frac{2}{4}'])


if __name__ == '__main__':
    unittest.main()

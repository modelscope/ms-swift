import unittest


class TestMathAccuracy(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            from swift.plugin.orm import MathAccuracy
            cls.math_accuracy = MathAccuracy()
            cls.available = True
        except (ImportError, AssertionError) as e:
            print(f'Warning: MathAccuracy not available: {e}')
            cls.available = False

    def setUp(self):
        if not self.available:
            self.skipTest('MathAccuracy not available (math_verify not installed)')

    def test_pure_latex_format(self):
        completions = ['The answer is \\boxed{42}']
        solutions = ['\\boxed{42}']

        rewards = self.math_accuracy(completions, solutions)

        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)

    def test_latex_in_long_text(self):
        completions = ['After careful calculation, the final answer is \\boxed{100}']
        solutions = ['\\boxed{100}']

        rewards = self.math_accuracy(completions, solutions)

        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)

    def test_multiple_steps_with_boxed(self):
        completions = [
            'Let me solve step by step:\n'
            '1. First we have x = 2\n'
            '2. Then y = 3x = 6\n'
            '3. Finally z = x + y = 8\n'
            '\nFinal answer: \\boxed{8}'
        ]
        solutions = ['\\boxed{8}']

        rewards = self.math_accuracy(completions, solutions)

        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)

    def test_wrong_answer_no_tag(self):
        completions = ['The answer is \\boxed{42}']
        solutions = ['\\boxed{100}']

        rewards = self.math_accuracy(completions, solutions)

        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)

    def test_batch_processing_no_tag(self):
        completions = ['\\boxed{42}', '\\boxed{100}', '\\boxed{8}']
        solutions = ['\\boxed{42}', '\\boxed{100}', '\\boxed{8}']

        rewards = self.math_accuracy(completions, solutions)

        self.assertEqual(len(rewards), 3)
        self.assertEqual(rewards[0], 1.0)
        self.assertEqual(rewards[1], 1.0)
        self.assertEqual(rewards[2], 1.0)

    def test_answer_tag_with_plain_number(self):
        completions = ['<answer>84</answer>']
        solutions = ['\\boxed{84}']

        rewards = self.math_accuracy(completions, solutions)

        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)

    def test_answer_tag_with_latex(self):
        completions = ['<answer>\\boxed{100}</answer>']
        solutions = ['\\boxed{100}']

        rewards = self.math_accuracy(completions, solutions)

        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)

    def test_long_text_with_answer_tag(self):
        completions = [
            'Let me solve:\n'
            'Step 1: Calculate x = 10\n'
            'Step 2: Calculate y = 20\n'
            'Step 3: Sum = 30\n'
            '\n<answer>54</answer>'
        ]
        solutions = ['\\boxed{54}']

        rewards = self.math_accuracy(completions, solutions)

        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)

    def test_answer_tag_with_complex_expression(self):
        completions = ['<answer>\\frac{1}{2}</answer>']
        solutions = ['\\boxed{\\frac{1}{2}}']

        rewards = self.math_accuracy(completions, solutions)

        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)

    def test_solution_with_answer_tag(self):
        completions = ['<answer>84</answer>']
        solutions = ['<answer>\\boxed{84}</answer>']

        rewards = self.math_accuracy(completions, solutions)

        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)

    def test_answer_tag_wrong_answer(self):
        completions = ['<answer>42</answer>']
        solutions = ['\\boxed{100}']

        rewards = self.math_accuracy(completions, solutions)

        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)

    def test_mixed_batch_with_and_without_tags(self):
        completions = [
            '\\boxed{42}',
            '<answer>100</answer>',
            'The answer is \\boxed{8}',
        ]
        solutions = [
            '\\boxed{42}',
            '\\boxed{100}',
            '\\boxed{8}',
        ]

        rewards = self.math_accuracy(completions, solutions)

        self.assertEqual(len(rewards), 3)
        self.assertEqual(rewards[0], 1.0)
        self.assertEqual(rewards[1], 1.0)
        self.assertEqual(rewards[2], 1.0)

    def test_empty_solution(self):
        completions = ['<answer>42</answer>']
        solutions = ['']

        rewards = self.math_accuracy(completions, solutions)

        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)

    def test_malformed_latex(self):
        completions = ['\\boxed{42']
        solutions = ['\\boxed{42}']

        rewards = self.math_accuracy(completions, solutions)

        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)

    def test_answer_tag_with_extra_whitespace(self):
        completions = ['<answer>  84  </answer>']
        solutions = ['\\boxed{84}']

        rewards = self.math_accuracy(completions, solutions)

        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)

    def test_multiple_answer_tags(self):
        completions = ['<answer>42</answer> Some text <answer>100</answer>']
        solutions = ['\\boxed{42}']

        rewards = self.math_accuracy(completions, solutions)

        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)

    def test_real_world_example_from_user(self):
        completions = [
            'We are given a geometric sequence $\\{a_n\\}$ with:\n\n'
            '- $a_3 = 2$\n- $a_5 = 6$\n\n'
            'We are to find $a_9$.\n\n---\n\n'
            '### Step 1: Recall the formula\n\n'
            '$$a_n = a_1 \\cdot r^{n-1}$$\n\n---\n\n'
            '### Step 2: Use the given terms\n\n'
            '$$a_3 = a_1 \\cdot r^2 = 2$$\n'
            '$$a_5 = a_1 \\cdot r^4 = 6$$\n\n'
            'Divide equation (2) by equation (1):\n'
            '$$r^2 = 3$$\n\n---\n\n'
            '### Step 3: Find $a_9$\n\n'
            '$$a_9 = a_1 \\cdot r^8 = \\frac{2}{3} \\cdot 81 = 54$$\n\n'
            '### ✅ Final Answer:\n\n'
            '<answer>54</answer>'
        ]
        solutions = ['\\boxed{54}']

        rewards = self.math_accuracy(completions, solutions)

        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)

    def test_equivalent_fractions(self):
        completions = ['<answer>0.5</answer>']
        solutions = ['\\boxed{\\frac{1}{2}}']

        rewards = self.math_accuracy(completions, solutions)

        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)

    def test_different_forms_same_answer(self):
        completions = ['<answer>2</answer>']
        solutions = ['\\boxed{\\sqrt{4}}']

        rewards = self.math_accuracy(completions, solutions)

        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)

    def test_latex_inline_math_delimiters(self):
        completions = ['<answer>84</answer>', '<answer>3</answer>']
        solutions = ['\n\n\\[\n\\boxed{84}\n\\]', 'Therefore, the value of \\(a^2 - a + 2\\) is \\(\\boxed{3}\\).']

        rewards = self.math_accuracy(completions, solutions)

        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)
        self.assertEqual(rewards[1], 1.0)

    def test_latex_display_math_delimiters(self):
        completions = ['<answer>100</answer>']
        solutions = ['\\[\\boxed{100}\\]']

        rewards = self.math_accuracy(completions, solutions)

        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)

    def test_mixed_latex_delimiters(self):
        completions = ['<answer>\\(x = 42\\)</answer>']
        solutions = ['\\[\\boxed{x = 42}\\]']

        rewards = self.math_accuracy(completions, solutions)

        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)


if __name__ == '__main__':
    unittest.main()

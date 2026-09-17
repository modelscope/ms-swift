# Copyright (c) ModelScope Contributors. All rights reserved.
import unittest

from swift.infer_engine.protocol import ChatCompletionResponse, ChatCompletionResponseChoice, ChatMessage, UsageInfo
from swift.rewards.rm_plugin import GenRMPlugin


def _response(contents):
    return ChatCompletionResponse(
        model='reward-model',
        choices=[
            ChatCompletionResponseChoice(
                index=i, message=ChatMessage(role='assistant', content=content), finish_reason='stop')
            for i, content in enumerate(contents)
        ],
        usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2))


class TestGenRMRewardScores(unittest.TestCase):

    def test_valid_rewards(self):
        for output, expected in [('Reward: 0', 0.), ('Reward: 1.0', 1.), ('Reward: 0.85', 0.85),
                                 ('Reasoning.\nReward: 0.25.', 0.25), ('**Reward: 0.5**', 0.5), ('Reward: 1e-1', 0.1),
                                 ('Reward: +0.75', 0.75)]:
            with self.subTest(output=output):
                self.assertEqual(GenRMPlugin.extract_reward(output), expected)

    def test_custom_reward_scales(self):
        for value, expected in [('10', 10.), ('100', 100.), ('1.5', 1.5), ('1.0001', 1.0001), ('-0.2', -0.2), ('2', 2.),
                                ('0.5e2', 50.), ('-2.5e+1', -25.)]:
            with self.subTest(value=value):
                self.assertEqual(GenRMPlugin.extract_reward(f'Reward: {value}'), expected)

    def test_cjk_reward_suffixes(self):
        for output, expected in [('Reward: 0.85分', 0.85), ('Reward: 10分', 10.), ('Reward: -2.5分', -2.5),
                                 ('Reward: 1e-1分', 0.1)]:
            with self.subTest(output=output):
                self.assertEqual(GenRMPlugin.extract_reward(output), expected)

    def test_invalid_reward_text(self):
        for output in [
                '', 'No score', 'Reward: nan', 'Reward: inf', 'Reward: -inf', 'Reward: 1e309', 'Reward: -1e309',
                'Reward: 0.5e', 'Reward: 0.5.2'
        ]:
            with self.subTest(output=output):
                self.assertIsNone(GenRMPlugin.extract_reward(output))

    def test_invalid_choices_do_not_change_average(self):
        plugin = GenRMPlugin.__new__(GenRMPlugin)
        results = [
            _response(['Reward: 0.2', 'Reward: 10', 'Reward: 0.6', 'Reward: 1.5']),
            _response(['Reward: 100', 'Reward: -0.2']),
            _response(['Reward: 1e-1']),
            _response(['Reward: -2', 'Reward: 10', 'Reward: 1e309', 'No score']),
            _response(['Reward: nan', 'Reward: -1e309'])
        ]
        rewards = plugin.compute_rewards(results)
        self.assertEqual(len(rewards), len(results))
        for actual, expected in zip(rewards, [3.075, 49.9, 0.1, 4., 0.]):
            self.assertAlmostEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()

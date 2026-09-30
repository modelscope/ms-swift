# Copyright (c) ModelScope Contributors. All rights reserved.
import math
import torch
import unittest
from types import SimpleNamespace

from swift.infer_engine.protocol import (ChatCompletionResponse, ChatCompletionResponseChoice, ChatMessage,
                                         RolloutOutput, UsageInfo)
from swift.rewards.orm import CosineReward, SoftOverlong
from swift.rl_core.data import GRPOSample
from swift.rl_core.grpo_algorithm import score_completions


class TestMultiturnLengthRewards(unittest.TestCase):

    def setUp(self):
        args = SimpleNamespace(
            cosine_min_len_value_wrong=-0.5,
            cosine_max_len_value_wrong=0.0,
            cosine_min_len_value_correct=1.0,
            cosine_max_len_value_correct=0.5,
            cosine_max_len=8,
            soft_max_length=8,
            soft_cache_length=2)

        def accuracy(completions, solution, **kwargs):
            return [float(c == s) for c, s in zip(completions, solution)]

        self.cosine = CosineReward(args, accuracy_orm=accuracy)
        self.soft = SoftOverlong(args)

    def test_length_rewards_do_not_depend_on_turn_partition(self):
        for length in [0, 2, 6, 7, 8]:
            flat = list(range(length))
            for ids in [flat, [flat], [flat[:2], [], flat[2:]]]:
                for correct in [False, True]:
                    with self.subTest(length=length, ids=ids, correct=correct):
                        completion = 'correct' if correct else 'wrong'
                        wave = 0.25 * math.cos(length * math.pi / 8)
                        expected_cosine = 0.75 + wave if correct else -0.25 - wave
                        self.assertAlmostEqual(
                            self.cosine([completion], ['correct'], response_token_ids=[ids])[0], expected_cosine)
                with self.subTest(length=length, ids=ids, reward='soft_overlong'):
                    self.assertEqual(self.soft(['answer'], response_token_ids=[ids]), [min(-(length - 6) / 2, 0)])

    def test_rollout_output_to_grpo_reward_dispatch(self):
        for token_ids in [list(range(7)), [list(range(3)), list(range(3, 7))]]:
            with self.subTest(token_ids=token_ids):
                response = ChatCompletionResponse(
                    model='test',
                    choices=[ChatCompletionResponseChoice(0, ChatMessage('assistant', 'correct'), 'stop')],
                    usage=UsageInfo(1, 7, 8))
                output = RolloutOutput(
                    response=response,
                    messages=[{
                        'role': 'assistant',
                        'content': 'correct'
                    }],
                    response_token_ids=token_ids)
                sample = GRPOSample(messages=[], extra={'solution': 'correct'})
                sample.apply_rollout_output(rollout_output=output)
                rewards = score_completions([sample], [self.cosine, self.soft], None, False, torch.device('cpu'))
                expected = torch.tensor([[0.75 + 0.25 * math.cos(7 * math.pi / 8), -0.5]])
                torch.testing.assert_close(rewards, expected)
                self.assertEqual(sample.response_token_ids, output.response_token_ids)


if __name__ == '__main__':
    unittest.main()

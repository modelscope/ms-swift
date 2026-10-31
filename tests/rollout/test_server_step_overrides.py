# Copyright (c) ModelScope Contributors. All rights reserved.
import asyncio
import unittest
from copy import deepcopy

from swift.infer_engine.protocol import (ChatCompletionResponse, ChatCompletionResponseChoice, ChatMessage,
                                         RequestConfig, RolloutInferRequest, UsageInfo)
from swift.rollout.multi_turn import MultiTurnScheduler


class PrefixTokenizer:

    def encode(self, text, add_special_tokens=False):
        return {'<prefix>': [9, 8], '<next>': [7]}[text]

    def decode(self, token_ids, **kwargs):
        return 'decoded:' + ','.join(map(str, token_ids))


class PrefixTemplate:

    def _get_response_prefix(self, inputs):
        return inputs.chat_template_kwargs.get('response_prefix', '')


class Engine:

    def __init__(self, responses):
        self.responses = iter(responses)
        self.history = []

    async def infer_async(self, request, config, **kwargs):
        self.history.append(deepcopy(request.messages))
        ids = next(self.responses)
        choice = ChatCompletionResponseChoice(
            0,
            ChatMessage('assistant', 'answer'),
            None,
            logprobs={'content': [{
                'logprob': -0.2
            } for _ in ids]},
            token_ids=ids)
        return ChatCompletionResponse('test-model', [choice], UsageInfo(0, len(ids), len(ids)))


class OverrideScheduler(MultiTurnScheduler):

    def __init__(self, overrides, *, next_prefix=None, continuation=False, **kwargs):
        super().__init__(**kwargs)
        self.overrides = overrides
        self.next_prefix = next_prefix
        self.continuation = continuation

    def step(self, request, choice, turn):
        if not self.continuation:
            request.messages.append({'role': 'user', 'content': 'observation'})
        if self.next_prefix is not None:
            request.chat_template_kwargs['response_prefix'] = self.next_prefix
        return {'infer_request': request, **self.overrides[turn - 1]}


def run_scheduler(overrides, *, responses=None, prefix='<prefix>', next_prefix=None, continuation=False):
    responses = responses or [[11, 12], [13]]
    engine = Engine(responses)
    scheduler = OverrideScheduler(
        overrides,
        next_prefix=next_prefix,
        continuation=continuation,
        infer_engine=engine,
        tokenizer=PrefixTokenizer(),
        template=PrefixTemplate(),
        max_turns=len(responses))
    request = RolloutInferRequest(
        messages=[{
            'role': 'user',
            'content': 'question'
        }], chat_template_kwargs={'response_prefix': prefix})
    result = asyncio.run(scheduler.run(request, RequestConfig(n=1)))
    return result, engine.history


class TestServerStepOverrides(unittest.TestCase):

    def test_raw_ids_with_optional_mask(self):
        for include_mask in (False, True):
            with self.subTest(include_mask=include_mask):
                override = {'response_token_ids': [11, 12]}
                if include_mask:
                    override['response_loss_mask'] = [1, 1]
                original = deepcopy(override)
                result, history = run_scheduler([override])
                self.assertEqual(result.response_token_ids, [[9, 8, 11, 12], [9, 8, 13]])
                self.assertEqual(result.response_loss_mask, [[0, 0, 1, 1], [0, 0, 1]])
                self.assertEqual(result.rollout_logprobs, [[-0.2, -0.2], [-0.2]])
                self.assertEqual(history[1][1]['content'], [9, 8, 11, 12])
                self.assertEqual(override, original)

    def test_masked_observation_retains_alignment(self):
        result, history = run_scheduler([{'response_token_ids': [11, 12, 6], 'response_loss_mask': [1, 1, 0]}])
        self.assertEqual(result.response_token_ids[0], [9, 8, 11, 12, 6])
        self.assertEqual(result.response_loss_mask[0], [0, 0, 1, 1, 0])
        self.assertEqual(result.rollout_logprobs, [[-0.2, -0.2], [-0.2]])
        self.assertEqual(history[1][1]['content'], [9, 8, 11, 12, 6])

    def test_explicit_masked_prefix_is_not_duplicated(self):
        result, _ = run_scheduler([{'response_token_ids': [9, 8, 11, 12], 'response_loss_mask': [0, 0, 1, 1]}])
        self.assertEqual(result.response_token_ids[0], [9, 8, 11, 12])
        self.assertEqual(result.response_loss_mask[0], [0, 0, 1, 1])

    def test_sampled_prefix_repetition_is_preserved(self):
        result, _ = run_scheduler([{
            'response_token_ids': [9, 8, 11],
            'response_loss_mask': [1, 1, 1]
        }],
                                  responses=[[9, 8, 11], [13]])
        self.assertEqual(result.response_token_ids[0], [9, 8, 9, 8, 11])
        self.assertEqual(result.response_loss_mask[0], [0, 0, 1, 1, 1])
        self.assertEqual(result.rollout_logprobs[0], [-0.2, -0.2, -0.2])

    def test_step_changes_only_the_next_turn_prefix(self):
        result, history = run_scheduler([{
            'response_token_ids': [11, 12],
            'response_loss_mask': [1, 1]
        }],
                                        next_prefix='<next>')
        self.assertEqual(result.response_token_ids, [[9, 8, 11, 12], [7, 13]])
        self.assertEqual(result.response_loss_mask, [[0, 0, 1, 1], [0, 1]])
        self.assertEqual(history[1][1]['content'], [9, 8, 11, 12])

    def test_continuation_override_does_not_repeat_prefix(self):
        result, _ = run_scheduler([
            {
                'response_token_ids': [11],
                'response_loss_mask': [1]
            },
            {
                'response_token_ids': [12],
                'response_loss_mask': [1]
            },
        ],
                                  responses=[[11], [12], [13]],
                                  continuation=True)
        self.assertEqual(result.response_token_ids, [[9, 8, 11, 12, 13]])
        self.assertEqual(result.response_loss_mask, [[0, 0, 1, 1, 1]])
        self.assertEqual(result.rollout_logprobs, [[-0.2, -0.2, -0.2]])

    def test_no_prefix_or_no_override(self):
        for overrides, prefix, expected_ids, expected_mask in (
            ([{
                'response_token_ids': [11, 12]
            }], '', [11, 12], [1, 1]),
            ([{}], '<prefix>', [9, 8, 11, 12], [0, 0, 1, 1]),
        ):
            with self.subTest(overrides=overrides, prefix=prefix):
                result, _ = run_scheduler(overrides, prefix=prefix)
                self.assertEqual(result.response_token_ids[0], expected_ids)
                self.assertEqual(result.response_loss_mask[0], expected_mask)


if __name__ == '__main__':
    unittest.main()

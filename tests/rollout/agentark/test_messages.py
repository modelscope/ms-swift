from __future__ import annotations

import unittest

from ._fakes import IMAGE_URL


class MessageHelperTests(unittest.TestCase):

    def test_action_extraction_matches_agentark_api_agent_semantics(self):
        from swift.rollout.agentark.messages import extract_action

        cases = [
            (
                '<think>reasoning</think>\n<tool_call>{"name":"ExecutePlan","arguments":{"plan":"R1"}}</tool_call>',
                '<tool_call>{"name":"ExecutePlan","arguments":{"plan":"R1"}}</tool_call>',
            ),
            ('prefix<params> L1, U2 </params>suffix', '<params>L1, U2</params>'),
            ('explanation<code>\nvar x = 1;\n</code>', 'var x = 1;'),
            ('raw assistant action', 'raw assistant action'),
            ('', None),
        ]
        for assistant, expected in cases:
            with self.subTest(assistant=assistant):
                self.assertEqual(extract_action(assistant), expected)

    def test_full_transcript_response_keeps_only_new_environment_delta(self):
        from swift.rollout.agentark.messages import new_environment_messages

        conversation = [
            {
                'role': 'system',
                'content': 'system'
            },
            {
                'role': 'user',
                'content': 'initial'
            },
            {
                'role': 'assistant',
                'content': 'ExecutePlan R1'
            },
        ]
        returned = [
            *conversation,
            {
                'role': 'assistant',
                'content': 'ExecutePlan R1'
            },
            {
                'role':
                'user',
                'content': [
                    {
                        'type': 'text',
                        'text': 'new state'
                    },
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': IMAGE_URL
                        }
                    },
                ],
            },
        ]

        delta = new_environment_messages(conversation, returned)

        self.assertEqual(len(delta), 1)
        self.assertEqual(delta[0]['role'], 'user')
        self.assertEqual(delta[0]['content'][1]['image_url']['url'], IMAGE_URL)


class BuiltinRegistrationTests(unittest.TestCase):

    def test_agentark_env_and_scheduler_are_registered(self):
        from swift.rollout.agentark.env import AgentArkEnv
        from swift.rollout.gym_env import envs
        from swift.rollout.multi_turn import AgentArkScheduler, multi_turns

        self.assertIs(envs['agentark'], AgentArkEnv)
        self.assertIs(multi_turns['agentark_scheduler'], AgentArkScheduler)


if __name__ == '__main__':
    unittest.main()

import unittest

from swift.ui.llm_infer.llm_infer import LLMInfer

TOOLBENCH = 'I should use a search tool.\nAction: search\nAction Input: {"query": "berlin"}'


def _messages(*pairs):
    return [{'role': role, 'content': content} for role, content in pairs]


def _roles(messages):
    return [message['role'] for message in messages]


class TestMarkToolbenchObservation(unittest.TestCase):

    def test_the_observation_after_a_toolbench_turn_becomes_a_tool_response(self):
        messages = _messages(('system', 'sys'), ('user', 'What is the weather in Berlin?'), ('assistant', TOOLBENCH),
                             ('user', 'sunny'))
        LLMInfer.mark_toolbench_observation(messages)
        self.assertEqual(_roles(messages), ['system', 'user', 'assistant', 'tool'])

    def test_only_the_last_assistant_turn_is_classified(self):
        messages = _messages(('system', 'sys'), ('user', 'q1'), ('assistant', 'hello'), ('tool', 'obs'), ('user', 'q2'),
                             ('assistant', TOOLBENCH), ('user', 'sunny'))
        LLMInfer.mark_toolbench_observation(messages)
        self.assertEqual(_roles(messages), ['system', 'user', 'assistant', 'tool', 'user', 'assistant', 'tool'])

    def test_only_the_message_right_after_the_assistant_turn_is_relabelled(self):
        messages = _messages(('user', 'q'), ('assistant', TOOLBENCH), ('user', 'obs'), ('user', 'extra'))
        LLMInfer.mark_toolbench_observation(messages)
        self.assertEqual(_roles(messages), ['user', 'assistant', 'tool', 'user'])

    def test_an_assistant_turn_that_ends_the_conversation_is_untouched(self):
        messages = _messages(('user', 'q'), ('assistant', TOOLBENCH))
        LLMInfer.mark_toolbench_observation(messages)
        self.assertEqual(_roles(messages), ['user', 'assistant'])

    def test_a_react_turn_is_not_relabelled(self):
        messages = _messages(('user', 'q'), ('assistant', 'Action: search\nObservation:'), ('user', 'obs'))
        LLMInfer.mark_toolbench_observation(messages)
        self.assertEqual(_roles(messages), ['user', 'assistant', 'user'])

    def test_a_conversation_without_an_assistant_turn_is_untouched(self):
        messages = _messages(('system', 'sys'), ('user', 'q'))
        LLMInfer.mark_toolbench_observation(messages)
        self.assertEqual(_roles(messages), ['system', 'user'])

    def test_empty_conversation(self):
        messages = []
        LLMInfer.mark_toolbench_observation(messages)
        self.assertEqual(messages, [])


if __name__ == '__main__':
    unittest.main()

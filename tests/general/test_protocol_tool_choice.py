import unittest

from swift.infer_engine.protocol import ChatCompletionRequest


def _request(tool_choice):
    tools = [
        {
            'type': 'function',
            'function': {
                'name': 'get_weather'
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'get_time'
            }
        },
    ]
    return ChatCompletionRequest(
        model='test', messages=[{
            'role': 'user',
            'content': 'hi'
        }], tools=tools, tool_choice=tool_choice)


class TestChatCompletionRequestToolChoice(unittest.TestCase):

    def test_selected_tool_keeps_only_that_tool(self):
        request = _request({'type': 'function', 'function': {'name': 'get_time'}})
        self.assertEqual([tool['function']['name'] for tool in request.tools], ['get_time'])

    def test_unknown_tool_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "Tool choice 'no_such_tool' not found in tools."):
            _request({'type': 'function', 'function': {'name': 'no_such_tool'}})

    def test_string_tool_choice_keeps_every_tool(self):
        self.assertEqual(len(_request('auto').tools), 2)


if __name__ == '__main__':
    unittest.main()

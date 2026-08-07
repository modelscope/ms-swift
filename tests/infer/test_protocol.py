import unittest

from swift.infer_engine.protocol import ChatCompletionRequest


class TestChatCompletionRequest(unittest.TestCase):

    @staticmethod
    def _tool(name):
        return {
            'type': 'function',
            'function': {
                'name': name,
                'description': f'Call {name}',
                'parameters': {},
            },
        }

    def test_rejects_unknown_explicit_tool_choice(self):
        with self.assertRaisesRegex(ValueError, "Tool choice 'missing' not found in tools"):
            ChatCompletionRequest(
                model='test',
                messages=[{
                    'role': 'user',
                    'content': 'hello',
                }],
                tools=[self._tool('weather')],
                tool_choice={
                    'type': 'function',
                    'function': {
                        'name': 'missing',
                    },
                },
            )

    def test_keeps_only_explicit_tool_choice(self):
        weather = self._tool('weather')
        request = ChatCompletionRequest(
            model='test',
            messages=[{
                'role': 'user',
                'content': 'hello',
            }],
            tools=[weather, self._tool('calculator')],
            tool_choice={
                'type': 'function',
                'function': {
                    'name': 'weather',
                },
            },
        )

        self.assertEqual(request.tools, [weather])


if __name__ == '__main__':
    unittest.main()

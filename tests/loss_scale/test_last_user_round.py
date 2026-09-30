# Copyright (c) ModelScope Contributors. All rights reserved.

from swift.template import get_last_user_round


def test_get_last_user_round_accepts_tool_message():
    messages = [
        {
            'role': 'user',
            'content': 'question 1'
        },
        {
            'role': 'assistant',
            'content': 'answer 1'
        },
        {
            'role': 'tool',
            'content': 'tool result'
        },
        {
            'role': 'assistant',
            'content': 'final answer'
        },
    ]

    assert get_last_user_round(messages) == 2


def test_get_last_user_round_can_exclude_tool_messages():
    messages = [
        {
            'role': 'user',
            'content': 'question 1'
        },
        {
            'role': 'assistant',
            'content': 'answer 1'
        },
        {
            'role': 'tool',
            'content': 'tool result'
        },
        {
            'role': 'assistant',
            'content': 'final answer'
        },
    ]

    assert get_last_user_round(messages, include_tool=False) == 0
    assert get_last_user_round(messages, include_tool=True) == 2

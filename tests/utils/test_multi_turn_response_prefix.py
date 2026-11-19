from swift.rlhf_trainers.utils import replace_assistant_response_with_ids


def test_non_thinking_prefix_is_added_to_every_generated_assistant_turn():
    messages = [
        {
            'role': 'user',
            'content': 'Q1'
        },
        {
            'role': 'assistant',
            'content': 'A'
        },
        {
            'role': 'user',
            'content': 'Q2'
        },
        {
            'role': 'assistant',
            'content': 'B'
        },
    ]

    result = replace_assistant_response_with_ids(
        messages,
        completion_ids=[[11], [12]],
        loss_mask=[[1], [1]],
        non_thinking_prefix_ids=[9, 8],
    )

    assert result[1]['content'] == {'loss_scale': [0, 0, 1], 'token_ids': [9, 8, 11]}
    assert result[3]['content'] == {'loss_scale': [0, 0, 1], 'token_ids': [9, 8, 12]}


def test_each_turn_handles_an_existing_masked_prefix_independently():
    messages = [
        {
            'role': 'user',
            'content': 'Q1'
        },
        {
            'role': 'assistant',
            'content': 'A'
        },
        {
            'role': 'user',
            'content': 'Q2'
        },
        {
            'role': 'assistant',
            'content': 'B'
        },
    ]

    result = replace_assistant_response_with_ids(
        messages,
        completion_ids=[[9, 8, 11], [12]],
        loss_mask=[[0, 0, 1], [1]],
        non_thinking_prefix_ids=[9, 8],
    )

    assert result[1]['content'] == {'loss_scale': [0, 0, 1], 'token_ids': [9, 8, 11]}
    assert result[3]['content'] == {'loss_scale': [0, 0, 1], 'token_ids': [9, 8, 12]}


def test_sampled_prefix_values_are_preserved_as_actions():
    messages = [
        {
            'role': 'user',
            'content': 'Q1'
        },
        {
            'role': 'assistant',
            'content': 'A'
        },
    ]

    result = replace_assistant_response_with_ids(
        messages,
        completion_ids=[[11, 9, 8]],
        loss_mask=[[1, 1, 1]],
        non_thinking_prefix_ids=[9, 8],
    )

    assert result[1]['content'] == {
        'loss_scale': [0, 0, 1, 1, 1],
        'token_ids': [9, 8, 11, 9, 8],
    }

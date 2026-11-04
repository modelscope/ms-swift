# Copyright (c) ModelScope Contributors. All rights reserved.

from swift.loss_scale import get_loss_scale


def test_ignore_think_prefix():
    loss_scale = get_loss_scale('ignore_think_prefix')
    contexts, weights = loss_scale.get_loss_scale('<think>\nreasoning\n</think>\nanswer')

    assert contexts == ['<think>\n', 'reasoning\n</think>\nanswer']
    assert weights == [0.0, 1.0]

    contexts, weights = loss_scale.get_loss_scale('<think>reasoning</think>')
    assert contexts == ['<think>', 'reasoning</think>']
    assert weights == [0.0, 1.0]


def test_ignore_think_prefix_only_matches_response_start():
    loss_scale = get_loss_scale('ignore_think_prefix')
    response = 'answer<think>\nreasoning\n</think>'
    contexts, weights = loss_scale.get_loss_scale(response)

    assert contexts == [response]
    assert weights == [1.0]

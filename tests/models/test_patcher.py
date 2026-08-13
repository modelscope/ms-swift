# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest
import torch

from swift.model.patcher import revert_padding_free


@pytest.mark.parametrize(
    'inputs', [{
        'position_ids': torch.tensor([[0, 1, 2, 0, 0, 1, 2]])
    }, {
        'cu_seq_lens_q': torch.tensor([0, 3, 4, 7], dtype=torch.int32)
    }],
    ids=['position_ids', 'cu_seq_lens_q'])
def test_revert_padding_free_preserves_single_token_sequence(inputs):
    token_states = torch.arange(1, 8, dtype=torch.float32).view(1, 7, 1)

    outputs = revert_padding_free({'last_hidden_state': token_states}, inputs, padding_side='right')

    expected = torch.tensor([[[1.], [2.], [3.]], [[4.], [0.], [0.]], [[5.], [6.], [7.]]])
    torch.testing.assert_close(outputs['last_hidden_state'], expected)

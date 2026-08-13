# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest
import torch

from swift.model.patcher import revert_padding_free
from swift.utils import get_cu_seqlens_from_position_ids


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


def test_revert_padding_free_uses_text_position_boundaries_for_mrope():
    text_position_ids = torch.tensor([[0, 1, 2, 0, 0, 1, 2]])
    cu_seqlens = get_cu_seqlens_from_position_ids(text_position_ids)
    mrope_position_ids = torch.full((3, 1, 7), 42)
    token_states = torch.arange(1, 8, dtype=torch.float32).view(1, 7, 1)

    outputs = revert_padding_free({'last_hidden_state': token_states}, {
        'position_ids': mrope_position_ids,
        'cu_seq_lens_q': cu_seqlens,
    },
                                  padding_side='left')

    expected = torch.tensor([[[1.], [2.], [3.]], [[0.], [0.], [4.]], [[5.], [6.], [7.]]])
    torch.testing.assert_close(outputs['last_hidden_state'], expected)


def test_revert_padding_free_preserves_consecutive_single_token_sequences():
    token_states = torch.arange(1, 5, dtype=torch.float32).view(1, 4, 1)

    outputs = revert_padding_free({'last_hidden_state': token_states}, {'position_ids': torch.tensor([[0, 0, 0, 1]])},
                                  padding_side='right')

    expected = torch.tensor([[[1.], [0.]], [[2.], [0.]], [[3.], [4.]]])
    torch.testing.assert_close(outputs['last_hidden_state'], expected)

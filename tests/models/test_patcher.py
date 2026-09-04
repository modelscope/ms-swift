import torch

from swift.model.patcher import select_last_packed_states


def test_select_last_packed_states_with_cu_seq_lens():
    hidden_states = torch.arange(16, dtype=torch.float32).view(1, 8, 2).requires_grad_()
    outputs = {'logits': hidden_states}
    inputs = {'cu_seq_lens_q': torch.tensor([0, 3, 7, 8], dtype=torch.int32)}

    result = select_last_packed_states(outputs, inputs)['logits']

    torch.testing.assert_close(result[:, 0], hidden_states[0, [2, 6, 7]])
    assert result.shape == (3, 1, 2)
    result.sum().backward()
    expected_grad = torch.zeros_like(hidden_states)
    expected_grad[0, [2, 6, 7]] = 1
    torch.testing.assert_close(hidden_states.grad, expected_grad)


def test_select_last_packed_states_with_position_ids():
    hidden_states = torch.arange(12, dtype=torch.float32).view(1, 6, 2)
    outputs = {'last_hidden_state': hidden_states}
    inputs = {'position_ids': torch.tensor([[0, 1, 2, 0, 1, 0]])}

    result = select_last_packed_states(outputs, inputs)['last_hidden_state']

    torch.testing.assert_close(result[:, 0], hidden_states[0, [2, 4, 5]])
    assert result.shape == (3, 1, 2)

import pytest
import torch

from swift.sequence_parallel.zigzag_ring_attn import get_half_lse


def _make_cu_seqlens(lengths, dtype=torch.int32, device='cpu'):
    lengths = torch.tensor(lengths, dtype=dtype, device=device)
    return torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))


def _reference_get_half_lse(lse, cu_seqlens, front):
    output = torch.empty((lse.shape[0], lse.shape[1] // 2), dtype=lse.dtype, device=lse.device)
    for i in range(len(cu_seqlens) - 1):
        start = int(cu_seqlens[i].item())
        end = int(cu_seqlens[i + 1].item())
        half_length = (end - start) // 2
        destination_start = start // 2
        source_start = start if front else start + half_length
        output[:, destination_start:destination_start + half_length] = lse[:, source_start:source_start + half_length]
    return output


@pytest.mark.parametrize('lengths',
                         ([8], [2, 4, 6], [2, 4, 6, 8, 2, 4, 6], [2, 4, 2, 4, 6, 8, 2, 4, 6], [0, 4, 0, 8, 2, 6, 0, 4]))
@pytest.mark.parametrize('dtype', (torch.float32, torch.bfloat16))
@pytest.mark.parametrize('cu_dtype', (torch.int32, torch.int64))
@pytest.mark.parametrize('front', (True, False))
def test_get_half_lse_matches_reference(lengths, dtype, cu_dtype, front):
    total_length = sum(lengths)
    lse = torch.randn((3, total_length), dtype=dtype, requires_grad=True)
    cu_seqlens = _make_cu_seqlens(lengths, dtype=cu_dtype)

    actual = get_half_lse(lse, cu_seqlens, front=front)
    expected = _reference_get_half_lse(lse.detach(), cu_seqlens, front)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (3, total_length // 2)
    assert actual.dtype == lse.dtype
    assert actual.device == lse.device


def test_get_half_lse_backward_matches_reference():
    lengths = [2, 4, 6, 8, 2, 4, 6, 8, 2]
    cu_seqlens = _make_cu_seqlens(lengths, dtype=torch.int64)
    lse_actual = torch.randn((2, sum(lengths)), requires_grad=True)
    lse_expected = lse_actual.detach().clone().requires_grad_(True)
    output_grad = torch.randn((2, sum(lengths) // 2))

    actual = get_half_lse(lse_actual, cu_seqlens, front=False)
    expected = _reference_get_half_lse(lse_expected, cu_seqlens, front=False)
    actual.backward(output_grad)
    expected.backward(output_grad)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(lse_actual.grad, lse_expected.grad, rtol=0, atol=0)


def test_get_half_lse_is_scripted():
    assert isinstance(get_half_lse, torch.jit.ScriptFunction)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is not available')
@pytest.mark.parametrize('dtype', (torch.float32, torch.bfloat16))
@pytest.mark.parametrize('front', (True, False))
def test_get_half_lse_cuda_forward_matches_reference(dtype, front):
    lengths = [2, 4, 6, 8, 2, 4, 6, 8, 2]
    cu_seqlens = _make_cu_seqlens(lengths, dtype=torch.int32, device='cuda')
    lse = torch.randn((3, sum(lengths)), dtype=dtype, device='cuda', requires_grad=True)

    actual = get_half_lse(lse, cu_seqlens, front=front)
    expected = _reference_get_half_lse(lse.detach(), cu_seqlens, front)
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.device == lse.device

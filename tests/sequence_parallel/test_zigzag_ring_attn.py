import pytest
import torch

from swift.sequence_parallel.zigzag_ring_attn import get_half_index, get_half_lse


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


def _reference_get_half_index(cu_seqlens, front):
    """The original per-document bool-mask implementation, kept as the oracle."""
    index = torch.zeros((cu_seqlens[-1].item(), ), dtype=torch.bool)
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        if front:
            end = (start + end) // 2
        else:
            start = (start + end) // 2
        index[start:end] = True
    return index


@pytest.mark.parametrize(
    'lengths',
    ([8], [2, 4, 6], [2, 4, 6, 8, 2, 4, 6], [2, 4, 2, 4, 6, 8, 2, 4, 6], [0, 4, 0, 8, 2, 6, 0, 4], [3, 5, 7, 9]))
@pytest.mark.parametrize('cu_dtype', (torch.int32, torch.int64))
@pytest.mark.parametrize('front', (True, False))
def test_get_half_index_gather_matches_reference(lengths, cu_dtype, front):
    cu_seqlens = _make_cu_seqlens(lengths, dtype=cu_dtype)
    source = torch.randn((sum(lengths), 2, 4))

    index = get_half_index(cu_seqlens, front=front)
    reference = _reference_get_half_index(cu_seqlens, front)

    torch.testing.assert_close(source[index], source[reference], rtol=0, atol=0)


@pytest.mark.parametrize('front', (True, False))
def test_get_half_index_accumulation_matches_reference(front):
    lengths = [2, 4, 6, 8, 2, 4, 6, 8, 2]
    cu_seqlens = _make_cu_seqlens(lengths, dtype=torch.int64)

    index = get_half_index(cu_seqlens, front=front)
    reference = _reference_get_half_index(cu_seqlens, front)
    gradient = torch.randn((sum(lengths) // 2, 2, 4))

    actual = torch.zeros((sum(lengths), 2, 4))
    expected = torch.zeros_like(actual)
    actual[index] += gradient
    expected[reference] += gradient

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_get_half_index_halves_are_complementary():
    lengths = [2, 4, 6, 8, 2, 4, 6, 8, 2]
    cu_seqlens = _make_cu_seqlens(lengths, dtype=torch.int32)

    front_index = get_half_index(cu_seqlens, front=True)
    tail_index = get_half_index(cu_seqlens, front=False)
    covered = torch.cat((front_index, tail_index)).sort().values

    torch.testing.assert_close(covered, torch.arange(sum(lengths)), rtol=0, atol=0)


def test_get_half_index_returns_slice_for_a_single_sequence():
    cu_seqlens = _make_cu_seqlens([8], dtype=torch.int32)

    # The autograd function branches on this type via `ctx.is_half_index_tensor`.
    assert isinstance(get_half_index(cu_seqlens, front=True), slice)
    assert isinstance(get_half_index(cu_seqlens, front=False), slice)


@pytest.mark.parametrize('cu_dtype', (torch.int32, torch.int64))
def test_get_half_index_is_built_on_the_input_device(cu_dtype):
    cu_seqlens = _make_cu_seqlens([2, 4, 6, 8], dtype=cu_dtype)

    index = get_half_index(cu_seqlens, front=False)

    assert index.device == cu_seqlens.device
    assert index.dtype == torch.long


def test_get_half_index_host_readbacks_do_not_grow_with_document_count(monkeypatch):
    original = {
        'item': torch.Tensor.item,
        'int': torch.Tensor.__int__,
        'index': torch.Tensor.__index__,
    }
    counts = {}

    def make_counter(name):

        def counter(self):
            counts[name] += 1
            return original[name](self)

        return counter

    for name in original:
        monkeypatch.setattr(torch.Tensor, f'__{name}__' if name != 'item' else 'item', make_counter(name))

    readbacks = {}
    for num_documents in (4, 128):
        cu_seqlens = _make_cu_seqlens([8] * num_documents, dtype=torch.int32)
        counts.update({name: 0 for name in original})
        get_half_index(cu_seqlens, front=False)
        readbacks[num_documents] = sum(counts.values())

    # The old loop read two boundaries per document; only a single length readback is left.
    assert readbacks[4] == readbacks[128]
    assert readbacks[128] <= 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is not available')
@pytest.mark.parametrize('front', (True, False))
def test_get_half_index_cuda_matches_reference(front):
    lengths = [2, 4, 6, 8, 2, 4, 6, 8, 2]
    cu_seqlens = _make_cu_seqlens(lengths, dtype=torch.int32, device='cuda')
    source = torch.randn((sum(lengths), 2, 4), device='cuda')

    index = get_half_index(cu_seqlens, front=front)
    reference = _reference_get_half_index(cu_seqlens.cpu(), front).to(cu_seqlens.device)
    torch.cuda.synchronize()

    assert index.device == cu_seqlens.device
    torch.testing.assert_close(source[index], source[reference], rtol=0, atol=0)

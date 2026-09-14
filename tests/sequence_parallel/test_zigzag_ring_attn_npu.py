import os
import pytest
import statistics
import time
import torch

from swift.sequence_parallel.zigzag_ring_attn_npu import _get_second_half_lse


def _npu_available():
    try:
        import torch_npu  # noqa: F401
        return hasattr(torch, 'npu') and torch.npu.is_available()
    except Exception:
        return False


def _reference_normalize_lse(softmax_lse, total_len):
    lse = softmax_lse
    if lse.dim() == 3 and lse.shape[0] == 1:
        lse = lse.squeeze(0)
    if lse.dim() != 2:
        raise RuntimeError(f'Unexpected softmax_lse shape: {tuple(softmax_lse.shape)}')
    if lse.shape[1] != total_len:
        lse = lse.transpose(0, 1).contiguous()
    if lse.shape[1] != total_len:
        raise RuntimeError(f'Unexpected softmax_lse shape: {tuple(softmax_lse.shape)}')
    return lse


def _reference_second_half_lse(softmax_lse, cu_seqlens):
    total_len = int(cu_seqlens[-1].item())
    lse = _reference_normalize_lse(softmax_lse, total_len)
    second_half_lse = torch.empty((lse.shape[0], lse.shape[1] // 2), dtype=lse.dtype, device=lse.device)
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        new_start, new_end = start // 2, end // 2
        start += (end - start) // 2
        second_half_lse[:, new_start:new_end] = lse[:, start:end]
    return second_half_lse


def _make_cu_seqlens(lengths, dtype=torch.int32):
    lengths = torch.tensor(lengths, dtype=dtype)
    return torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))


@pytest.mark.parametrize('lengths', ([8], [2, 4, 8, 6], [0, 4, 0, 8]))
@pytest.mark.parametrize('dtype', (torch.float32, torch.bfloat16))
@pytest.mark.parametrize('cu_dtype', (torch.int32, torch.int64))
def test_second_half_lse_matches_reference(lengths, dtype, cu_dtype):
    num_heads = 3
    total_len = sum(lengths)
    cu_seqlens = _make_cu_seqlens(lengths, dtype=cu_dtype)
    lse = torch.randn((num_heads, total_len), dtype=dtype)

    actual = _get_second_half_lse(lse, cu_seqlens)
    expected = _reference_second_half_lse(lse, cu_seqlens)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (num_heads, total_len // 2)
    assert actual.dtype == lse.dtype
    assert actual.device == lse.device


@pytest.mark.parametrize('shape_kind', ('tokens_heads', 'batched_heads_tokens'))
def test_second_half_lse_normalizes_supported_layouts(shape_kind):
    lengths = [4, 8]
    num_heads = 3
    total_len = sum(lengths)
    cu_seqlens = _make_cu_seqlens(lengths)
    base_lse = torch.arange(num_heads * total_len, dtype=torch.float32).reshape(num_heads, total_len)
    if shape_kind == 'tokens_heads':
        lse = base_lse.transpose(0, 1)
    else:
        lse = base_lse.unsqueeze(0)

    actual = _get_second_half_lse(lse, cu_seqlens)
    expected = _reference_second_half_lse(lse, cu_seqlens)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (num_heads, total_len // 2)


def test_second_half_lse_preserves_gradient():
    lengths = [2, 6, 4]
    cu_seqlens = _make_cu_seqlens(lengths)
    lse_actual = torch.randn((2, sum(lengths)), dtype=torch.float32, requires_grad=True)
    lse_expected = lse_actual.detach().clone().requires_grad_(True)
    output_grad = torch.randn((2, sum(lengths) // 2), dtype=torch.float32)

    actual = _get_second_half_lse(lse_actual, cu_seqlens)
    expected = _reference_second_half_lse(lse_expected, cu_seqlens)
    actual.backward(output_grad)
    expected.backward(output_grad)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(lse_actual.grad, lse_expected.grad, rtol=0, atol=0)


def test_second_half_lse_rejects_invalid_rank():
    cu_seqlens = _make_cu_seqlens([4])
    with pytest.raises(RuntimeError, match='Unexpected softmax_lse shape'):
        _get_second_half_lse(torch.randn(2, 1, 4, 1), cu_seqlens)


def _measure_npu(fn, lse, cu_seqlens, warmup=10, repeats=30):
    for _ in range(warmup):
        fn(lse, cu_seqlens)
    torch.npu.synchronize()
    samples = []
    for _ in range(repeats):
        torch.npu.synchronize()
        start = time.perf_counter()
        fn(lse, cu_seqlens)
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1e6)
    return statistics.median(samples), statistics.quantiles(samples, n=20)[18]


@pytest.mark.skipif(not _npu_available(), reason='Ascend NPU is not available')
@pytest.mark.parametrize('lengths', ([8], [2, 4, 8, 6], [0, 4, 0, 8]))
@pytest.mark.parametrize('dtype', (torch.float32, torch.bfloat16))
def test_second_half_lse_matches_reference_on_npu(lengths, dtype):
    device = torch.device('npu:0')
    total_len = sum(lengths)
    cu_seqlens = _make_cu_seqlens(lengths).to(device)
    lse = torch.randn((3, total_len), dtype=dtype, device=device, requires_grad=True)
    expected_lse = lse.detach().clone().requires_grad_(True)

    actual = _get_second_half_lse(lse, cu_seqlens)
    expected = _reference_second_half_lse(expected_lse, cu_seqlens)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    output_grad = torch.randn_like(actual)
    actual.backward(output_grad)
    expected.backward(output_grad)
    torch.testing.assert_close(lse.grad, expected_lse.grad, rtol=0, atol=0)
    torch.npu.synchronize()


def test_second_half_lse_benchmark():
    if os.environ.get('SWIFT_RUN_NPU_BENCHMARK') != '1':
        pytest.skip('Set SWIFT_RUN_NPU_BENCHMARK=1 to run the NPU benchmark')
    if not _npu_available():
        pytest.skip('Ascend NPU is not available')

    total_len = 4096
    device = torch.device('npu:0')
    for num_sequences in (1, 8, 32, 128, 512, 1024):
        sequence_len = total_len // num_sequences
        cu_seqlens = torch.arange(0, total_len + 1, sequence_len, dtype=torch.int32, device=device)
        lse = torch.randn((8, total_len), dtype=torch.bfloat16, device=device)
        baseline_median, baseline_p95 = _measure_npu(_reference_second_half_lse, lse, cu_seqlens)
        optimized_median, optimized_p95 = _measure_npu(_get_second_half_lse, lse, cu_seqlens)
        print(f'{num_sequences} sequences x {sequence_len} tokens: '
              f'baseline={baseline_median:.1f}/{baseline_p95:.1f} us, '
              f'optimized={optimized_median:.1f}/{optimized_p95:.1f} us, '
              f'speedup={baseline_median / optimized_median:.2f}x')

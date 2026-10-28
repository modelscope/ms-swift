# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import unittest
from torch.utils._python_dispatch import TorchDispatchMode

from swift.rlhf_trainers.utils import selective_log_softmax


def naive(logits, index, temperature=1.0):
    """The operation `selective_log_softmax` is defined to be equivalent to."""
    if temperature != 1.0:
        logits = logits / temperature
    squeeze = index.ndim == logits.ndim - 1
    if squeeze:
        index = index.unsqueeze(-1)
    res = torch.gather(logits.log_softmax(-1), dim=-1, index=index)
    return res.squeeze(-1) if squeeze else res


class CloneRecorder(TorchDispatchMode):

    def __init__(self):
        super().__init__()
        self.sizes = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if func == torch.ops.aten.clone.default and args[0].is_floating_point():
            self.sizes.append(args[0].numel())
        return result


class TestSelectiveLogSoftmax(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.vocab_size = 512
        self.shapes_2d = [(8192, self.vocab_size), (1, self.vocab_size), (7, self.vocab_size)]
        self.shapes_3d = [(2, 64, self.vocab_size), (1, 33, self.vocab_size)]
        self.chunk_sizes = [1, 3, 1024, 10**6]

    def _make(self, shape, dtype, k=None):
        logits = torch.randn(shape, dtype=dtype)
        index_shape = shape[:-1] if k is None else (*shape[:-1], k)
        index = torch.randint(0, shape[-1], index_shape)
        return logits, index

    def test_matches_naive_low_precision(self):
        # The dtype used in training: this path must be bit-for-bit identical.
        for dtype in (torch.bfloat16, torch.float16):
            for shape in self.shapes_2d + self.shapes_3d:
                logits, index = self._make(shape, dtype)
                expected = naive(logits, index)
                for chunk_size in self.chunk_sizes:
                    with self.subTest(dtype=dtype, shape=shape, chunk_size=chunk_size):
                        self.assertTrue(
                            torch.equal(selective_log_softmax(logits, index, chunk_size=chunk_size), expected))

    def test_matches_naive_high_precision(self):
        # log-softmax is taken as `x - logsumexp(x)` here, so allow last-bit drift.
        for shape in self.shapes_2d + self.shapes_3d:
            logits, index = self._make(shape, torch.float32)
            expected = naive(logits, index)
            for chunk_size in self.chunk_sizes:
                with self.subTest(shape=shape, chunk_size=chunk_size):
                    torch.testing.assert_close(
                        selective_log_softmax(logits, index, chunk_size=chunk_size), expected, rtol=1e-6, atol=1e-6)

    def test_chunk_size_does_not_change_result(self):
        logits, index = self._make((4096, self.vocab_size), torch.bfloat16)
        reference = selective_log_softmax(logits, index, chunk_size=1)
        for chunk_size in self.chunk_sizes:
            with self.subTest(chunk_size=chunk_size):
                self.assertTrue(torch.equal(selective_log_softmax(logits, index, chunk_size=chunk_size), reference))

    def test_topk_index(self):
        # both precision branches must keep the K dimension: `index` of shape (..., K) is how
        # callers gather top-k log-probabilities.
        for dtype in (torch.bfloat16, torch.float32):
            for k in (1, 4):
                with self.subTest(dtype=dtype, k=k):
                    logits, index = self._make((128, self.vocab_size), dtype, k=k)
                    got = selective_log_softmax(logits, index)
                    self.assertEqual(got.shape, index.shape)
                    if dtype == torch.bfloat16:
                        self.assertTrue(torch.equal(got, naive(logits, index)))
                    else:
                        torch.testing.assert_close(got, naive(logits, index), rtol=1e-6, atol=1e-6)

    def test_temperature(self):
        logits, index = self._make((256, self.vocab_size), torch.bfloat16)
        for temperature in (0.5, 2.0):
            with self.subTest(temperature=temperature):
                torch.testing.assert_close(
                    selective_log_softmax(logits, index, temperature=temperature),
                    naive(logits, index, temperature=temperature),
                    rtol=0,
                    atol=0)
        with self.assertRaises(ValueError):
            selective_log_softmax(logits, index, temperature=0.0)

    def test_row_mask(self):
        logits, index = self._make((64, self.vocab_size), torch.bfloat16)
        row_mask = torch.randint(0, 2, index.shape)
        expected = naive(logits, index).masked_fill(row_mask == 0, 0.0)
        self.assertTrue(torch.equal(selective_log_softmax(logits, index, row_mask=row_mask), expected))

    def test_output_shape_and_dtype(self):
        logits, index = self._make((3, 17, self.vocab_size), torch.bfloat16)
        out = selective_log_softmax(logits, index)
        self.assertEqual(out.shape, index.shape)
        self.assertEqual(out.dtype, logits.dtype)

    def test_gradient_flows(self):
        logits = torch.randn(32, self.vocab_size, dtype=torch.float32, requires_grad=True)
        index = torch.randint(0, self.vocab_size, (32, ))
        selective_log_softmax(logits, index).sum().backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_strided_logits_do_not_copy_all_rows(self):
        for dtype in (torch.float32, torch.bfloat16):
            base = torch.randn(2, 17, self.vocab_size, dtype=dtype)
            # GRPO drops the final token before computing completion log-probabilities.
            for logits in (base[:, :-1, :], base.transpose(0, 1)):
                with self.subTest(dtype=dtype, stride=logits.stride()):
                    index = torch.randint(0, self.vocab_size, logits.shape[:-1])
                    with CloneRecorder() as recorder:
                        result = selective_log_softmax(logits, index, chunk_size=3)
                    self.assertLessEqual(max(recorder.sizes, default=0), 3 * self.vocab_size)
                    torch.testing.assert_close(result, naive(logits, index), rtol=1e-6, atol=1e-6)

    def test_strided_values_and_gradients_with_topk_and_broadcast_mask(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
            for temperature in (1.0, 0.7):
                for topk in (False, True):
                    with self.subTest(dtype=dtype, temperature=temperature, topk=topk):
                        base = torch.randn(2, 3, 5, 31, dtype=dtype, requires_grad=True)
                        reference = base.detach().clone().requires_grad_()
                        logits = base.transpose(0, 1)[:, :, :-1, :]
                        ref_logits = reference.transpose(0, 1)[:, :, :-1, :]
                        shape = (*logits.shape[:-1], 4) if topk else logits.shape[:-1]
                        index = torch.randint(0, logits.shape[-1], shape)
                        row_mask = torch.tensor([1, 0, 1, 1])
                        if topk:
                            row_mask = row_mask.unsqueeze(-1)
                        result = selective_log_softmax(
                            logits, index, temperature=temperature, row_mask=row_mask, chunk_size=3)
                        expected = naive(ref_logits, index, temperature).masked_fill(row_mask == 0, 0.)
                        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)
                        weights = torch.randn_like(result)
                        result.backward(weights)
                        expected.backward(weights)
                        torch.testing.assert_close(base.grad, reference.grad, rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    unittest.main()

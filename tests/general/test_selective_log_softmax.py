# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import unittest

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


if __name__ == '__main__':
    unittest.main()

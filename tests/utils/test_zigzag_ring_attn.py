import sys
import torch
import unittest

from unittest.mock import patch

from swift.sequence_parallel import zigzag_ring_attn as zra


class _FakeTorchNPU:

    def __init__(self):
        self.calls = []
        self.grad_calls = []

    def npu_fusion_attention(self, **kwargs):
        self.calls.append(kwargs)
        q = kwargs['query']
        k = kwargs['key']
        v = kwargs['value']
        groups = q.shape[1] // k.shape[1]
        k_expand = k.repeat_interleave(groups, dim=1)[:q.shape[0]]
        v_expand = v.repeat_interleave(groups, dim=1)[:q.shape[0]]
        out = q + k_expand + v_expand
        softmax_max = torch.zeros((q.shape[0], k.shape[1], groups), dtype=torch.float32, device=q.device)
        softmax_sum = torch.ones_like(softmax_max)
        attention_in = torch.full((1, ), 7.0, dtype=torch.float32, device=q.device)
        return out, softmax_max, softmax_sum, attention_in, 11, 13, 17

    def npu_fusion_attention_grad(self, **kwargs):
        self.grad_calls.append(kwargs)
        q = kwargs['query']
        k = kwargs['key']
        v = kwargs['value']
        return (
            torch.full_like(q, 2.0),
            torch.full_like(k, 3.0),
            torch.full_like(v, 4.0),
            torch.zeros((1, ), dtype=q.dtype, device=q.device),
            torch.zeros((1, ), dtype=q.dtype, device=q.device),
        )


class TestZigZagRingNPUHelpers(unittest.TestCase):

    @staticmethod
    def _full_attention_out_and_lse(q, k, v):
        scale = q.shape[-1] ** -0.5
        groups = q.shape[1] // k.shape[1]
        k_exp = k.repeat_interleave(groups, dim=1) if groups > 1 else k
        v_exp = v.repeat_interleave(groups, dim=1) if groups > 1 else v
        scores = torch.einsum('qhd,khd->hqk', q.float(), k_exp.float()) * scale
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum('hqk,khd->qhd', probs, v_exp.float())
        lse = torch.logsumexp(scores, dim=-1)
        return out, lse

    def test_cu_seqlens_to_actual_seq(self):
        cu_seqlens = torch.tensor([0, 3, 7, 9], dtype=torch.int32)
        self.assertEqual(zra._cu_seqlens_to_actual_seq(cu_seqlens), (3, 7, 9))

    def test_reshape_npu_lse_tnd(self):
        lse = torch.arange(24, dtype=torch.float32).reshape(4, 3, 2)
        reshaped = zra._reshape_npu_lse(lse, seqlen_q=4, num_heads=6)
        expected = lse.permute(1, 2, 0).reshape(6, 4)
        self.assertTrue(torch.equal(reshaped, expected))

    def test_reshape_npu_lse_tnd_repeated_slots(self):
        lse = torch.arange(8, dtype=torch.float32).reshape(4, 2, 1).expand(4, 2, 8)
        reshaped = zra._reshape_npu_lse(lse, seqlen_q=4, num_heads=2)
        expected = lse[..., 0].transpose(0, 1).contiguous()
        self.assertTrue(torch.equal(reshaped, expected))

    def test_npu_forward_maps_official_params(self):
        fake_torch_npu = _FakeTorchNPU()
        q = torch.randn(4, 4, 8)
        k = torch.randn(4, 2, 8)
        v = torch.randn(4, 2, 8)
        cu = torch.tensor([0, 4], dtype=torch.int32)

        with patch.dict(sys.modules, {'torch_npu': fake_torch_npu}):
            block_out, block_lse = zra._npu_forward(
                q,
                k,
                v,
                causal=True,
                cu_seqlens_q=cu,
                cu_seqlens_kv=cu,
                dropout_p=0.1,
                softmax_scale=None,
                deterministic=True,
                window_size=(8, 4),
            )

        self.assertEqual(block_out.shape, q.shape)
        self.assertEqual(block_lse.shape, (4, 4))
        self.assertEqual(len(fake_torch_npu.calls), 1)
        call = fake_torch_npu.calls[0]
        self.assertEqual(call['input_layout'], 'TND')
        self.assertEqual(call['head_num'], 4)
        self.assertEqual(call['actual_seq_qlen'], (4, ))
        self.assertEqual(call['actual_seq_kvlen'], (4, ))
        self.assertEqual(call['sparse_mode'], 4)
        self.assertEqual(call['pre_tockens'], 8)
        self.assertEqual(call['next_tockens'], 0)
        self.assertTrue(call['sync'])
        self.assertEqual(call.get('softmax_layout'), 'TND')
        self.assertEqual(call['atten_mask'].shape, (zra._NPU_BLOCK_MASK_SIZE, zra._NPU_BLOCK_MASK_SIZE))

    def test_npu_backward_matches_manual_attention_and_fills_prefix_buffers(self):
        torch.manual_seed(0)
        q = torch.randn(4, 2, 3, requires_grad=True)
        k = torch.randn(4, 2, 3, requires_grad=True)
        v = torch.randn(4, 2, 3, requires_grad=True)
        dout = torch.randn_like(q)
        cu = torch.tensor([0, 4], dtype=torch.int32)
        dq_buffer = torch.empty_like(q)
        dk_buffer = torch.empty((6, 2, 3), dtype=q.dtype)
        dv_buffer = torch.empty((6, 2, 3), dtype=q.dtype)

        out, lse = self._full_attention_out_and_lse(q, k, v)
        out.backward(dout)

        zra._npu_backward(
            dout.detach(),
            q.detach(),
            k.detach(),
            v.detach(),
            out.detach(),
            lse.detach(),
            causal=False,
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            dq_buffer=dq_buffer,
            dk_buffer=dk_buffer,
            dv_buffer=dv_buffer,
            dropout_p=0.,
            softmax_scale=None,
            deterministic=False,
            window_size=(-1, -1),
        )

        self.assertTrue(torch.allclose(dq_buffer, q.grad, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(dk_buffer[:4], k.grad, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(dv_buffer[:4], v.grad, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(dk_buffer[4:], torch.zeros_like(dk_buffer[4:])))
        self.assertTrue(torch.allclose(dv_buffer[4:], torch.zeros_like(dv_buffer[4:])))

    def test_manual_block_backward_uses_full_softmax_lse(self):
        torch.manual_seed(1)
        q = torch.randn(4, 2, 3)
        k_full = torch.randn(6, 2, 3)
        v_full = torch.randn(6, 2, 3)
        k_block = k_full[:3].clone()
        v_block = v_full[:3].clone()
        dout = torch.randn_like(q)
        cu_q = torch.tensor([0, 4], dtype=torch.int32)
        cu_k = torch.tensor([0, 3], dtype=torch.int32)

        out_full, lse_full = self._full_attention_out_and_lse(q, k_full, v_full)
        dq, dk, dv = zra._manual_varlen_attention_backward(
            dout,
            q,
            k_block,
            v_block,
            out_full,
            lse_full,
            cu_q,
            cu_k,
            None,
            False,
            (-1, -1),
        )

        scale = q.shape[-1] ** -0.5
        scores_block = torch.einsum('qhd,khd->hqk', q.float(), k_block.float()) * scale
        probs_block = torch.exp(scores_block - lse_full.unsqueeze(-1))
        delta = (out_full * dout).sum(dim=-1).transpose(0, 1).unsqueeze(-1)
        d_probs = torch.einsum('qhd,khd->hqk', dout.float(), v_block.float())
        d_scores = probs_block * (d_probs - delta)
        expected_dq = torch.einsum('hqk,khd->qhd', d_scores, k_block.float()) * scale
        expected_dk = torch.einsum('hqk,qhd->khd', d_scores, q.float()) * scale
        expected_dv = torch.einsum('hqk,qhd->khd', probs_block, dout.float())

        self.assertTrue(torch.allclose(dq, expected_dq, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(dk, expected_dk, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(dv, expected_dv, atol=1e-5, rtol=1e-5))

    def test_manual_varlen_lse_backward_matches_autograd(self):
        torch.manual_seed(0)
        q = torch.randn(4, 4, 3, requires_grad=True)
        k = torch.randn(4, 2, 3, requires_grad=True)
        v = torch.randn(4, 2, 3, requires_grad=True)
        dlse = torch.randn(4, 4)
        cu = torch.tensor([0, 4], dtype=torch.int32)
        scale = q.shape[-1] ** -0.5

        groups = q.shape[1] // k.shape[1]
        scores = torch.einsum('qhd,khd->hqk', q, k.repeat_interleave(groups, dim=1)) * scale
        lse = torch.logsumexp(scores, dim=-1).transpose(0, 1)
        (lse * dlse).sum().backward()

        dq, dk = zra._manual_varlen_lse_backward(
            dlse,
            q.detach(),
            k.detach(),
            cu,
            cu,
            None,
            False,
            (-1, -1),
        )

        self.assertTrue(torch.allclose(dq, q.grad, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(dk, k.grad, atol=1e-5, rtol=1e-5))


if __name__ == '__main__':
    unittest.main()

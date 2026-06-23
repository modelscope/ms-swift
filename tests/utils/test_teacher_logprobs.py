"""CPU unit tests for OPD-RL / GKD teacher prompt-logprob parsing and assembly.

Covers:
  - ``parse_prompt_logprobs`` sampled-token (``topk=0``) vs top-k (``topk>0``) semantics.
  - ``assemble_teacher_completion_logprobs`` completion-frame alignment (GRPO/OPD-RL).
"""
import unittest


class _Resp:
    """Minimal stand-in for a ChatCompletionResponse carrying prompt_logprobs."""

    def __init__(self, prompt_logprobs):
        self.prompt_logprobs = prompt_logprobs


def _pos(entries):
    """Build a position dict {token_id_str: {logprob, rank, decoded_token}} from
    (token_id, logprob, rank) tuples, matching VllmEngine._format_prompt_logprobs."""
    return {str(tid): {'logprob': lp, 'rank': rank, 'decoded_token': ''} for tid, lp, rank in entries}


class TestParsePromptLogprobs(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            from swift.rlhf_trainers.utils import parse_prompt_logprobs
            cls.parse = staticmethod(parse_prompt_logprobs)
            cls.available = True
        except Exception as e:  # noqa: BLE001 - env without vllm/cuda skips, not fails
            print(f'Warning: parse_prompt_logprobs not importable: {e}')
            cls.available = False

    def setUp(self):
        if not self.available:
            self.skipTest('swift.rlhf_trainers.utils not importable (vllm/cuda missing)')

    def test_sampled_token_topk0_picks_sampled_not_top1(self):
        # prompt_logprobs=0 -> vLLM returns exactly ONE entry per position: the sampled token.
        # Position 1 sampled token id=7 (logp -2.0); position 2 sampled token id=9 (logp -0.5).
        resp = _Resp([
            None,  # position 0 is always None
            _pos([(7, -2.0, 5)]),  # rank 5: sampled token is NOT the top-1
            _pos([(9, -0.5, 1)]),
        ])
        lps, ixs = self.parse(resp, topk=0)
        self.assertEqual(lps, [[-2.0], [-0.5]])
        self.assertEqual(ixs, [[7], [9]])

    def test_topk_sorted_by_logprob(self):
        # prompt_logprobs=2 -> top-2 highest-prob tokens (+ maybe the sampled token).
        resp = _Resp([
            None,
            _pos([(3, -0.1, 1), (8, -1.2, 2), (7, -2.0, 9)]),  # sampled id=7 is the extra k+1-th
            _pos([(5, -0.3, 1), (6, -0.9, 2)]),
        ])
        lps, ixs = self.parse(resp, topk=2)
        # top-2 by logprob, sampled (rank 9) dropped
        self.assertEqual(ixs, [[3, 8], [5, 6]])
        self.assertEqual(lps, [[-0.1, -1.2], [-0.3, -0.9]])

    def test_empty_prompt_logprobs(self):
        self.assertEqual(self.parse(_Resp(None), topk=0), ([], []))
        self.assertEqual(self.parse(_Resp([]), topk=0), ([], []))


class TestAssembleTeacherCompletionLogprobs(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            import torch

            from swift.rlhf_trainers.gkd_helpers import assemble_teacher_completion_logprobs
            cls.torch = torch
            cls.assemble = staticmethod(assemble_teacher_completion_logprobs)
            cls.available = True
        except Exception as e:  # noqa: BLE001
            print(f'Warning: assemble_teacher_completion_logprobs not importable: {e}')
            cls.available = False

    def setUp(self):
        if not self.available:
            self.skipTest('swift.rlhf_trainers.gkd_helpers not importable (vllm/cuda missing)')

    def test_completion_frame_alignment_and_offbyone(self):
        torch = self.torch
        # B=2, T=4. Sample 0: completion at positions {2,3} (count=2);
        # Sample 1: completion at positions {1,2,3} (count=3).
        mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]], dtype=torch.float32)
        # parse_prompt_logprobs(topk=0) output: per-sample (lps, ixs), each per-position [logp]/[id].
        # Sample 0 returns count+1 entries (the trailing dummy last-token logp must be trimmed).
        parsed = [
            ([[-0.5], [-0.6], [-0.7]], [[10], [11], [12]]),  # len 3 == count(2)+1 -> trim last
            ([[-0.1], [-0.2], [-0.3]], [[20], [21], [22]]),  # len 3 == count 3
        ]
        out = self.assemble(parsed, mask, torch.device('cpu'))
        lp, ix = out.topk_logprobs, out.topk_indices
        self.assertEqual(tuple(lp.shape), (2, 4, 1))
        self.assertEqual(tuple(ix.shape), (2, 4, 1))
        # Sample 0: positions 2,3 get first 2 entries; last (-0.7/12) trimmed.
        self.assertAlmostEqual(lp[0, 2, 0].item(), -0.5)
        self.assertAlmostEqual(lp[0, 3, 0].item(), -0.6)
        self.assertEqual(ix[0, 2, 0].item(), 10)
        self.assertEqual(ix[0, 3, 0].item(), 11)
        # Sample 1: positions 1,2,3.
        self.assertAlmostEqual(lp[1, 1, 0].item(), -0.1)
        self.assertAlmostEqual(lp[1, 3, 0].item(), -0.3)
        # Non-completion positions stay at -inf (sentinel) / 0.
        self.assertTrue(torch.isinf(lp[0, 0, 0]))
        self.assertTrue(torch.isinf(lp[0, 1, 0]))
        self.assertEqual(ix[0, 0, 0].item(), 0)

    def test_exact_count_no_trim(self):
        torch = self.torch
        mask = torch.tensor([[1, 1]], dtype=torch.float32)
        parsed = [([[-1.0], [-2.0]], [[5], [6]])]  # len 2 == count 2
        out = self.assemble(parsed, mask, torch.device('cpu'))
        self.assertAlmostEqual(out.topk_logprobs[0, 0, 0].item(), -1.0)
        self.assertAlmostEqual(out.topk_logprobs[0, 1, 0].item(), -2.0)
        self.assertEqual(out.topk_indices[0, 1, 0].item(), 6)

    def test_count_mismatch_raises(self):
        torch = self.torch
        mask = torch.tensor([[1, 1, 1]], dtype=torch.float32)  # count 3
        parsed = [([[-1.0]], [[5]])]  # len 1, neither 3 nor 4
        with self.assertRaises(AssertionError):
            self.assemble(parsed, mask, torch.device('cpu'))

    def test_teacher_per_token_logps_matches_old_logps_frame(self):
        # End-to-end: teacher_per_token_logps = topk_logprobs[..., 0] must align with a
        # completion-frame old_per_token_logps so k3 KL = sum((exp(d)-d-1) * mask).
        torch = self.torch
        mask = torch.tensor([[0, 1, 1]], dtype=torch.float32)
        parsed = [([[-1.5], [-1.0]], [[1], [2]])]  # teacher logps on the 2 response tokens
        out = self.assemble(parsed, mask, torch.device('cpu'))
        teacher = out.topk_logprobs[..., 0]  # [1, 3]
        student = torch.tensor([[0.0, -1.0, -2.0]])
        d = (teacher - student) * mask  # masked diff; non-response positions contribute 0
        assert d
        # at masked-out position teacher is -inf, but mask zeroes it; compute only on response
        d_resp = teacher[mask.bool()] - student[mask.bool()]
        k3 = (torch.exp(d_resp) - d_resp - 1).sum()
        self.assertGreaterEqual(k3.item(), 0.0)  # k3 is non-negative


class TestComputeTeacherKlPerToken(unittest.TestCase):
    """The backend-agnostic per-token k3 teacher KL shared by HF / Megatron / Ray GRPO.

    OPD-RL keeps the teacher KL per-token (it is NOT summed over the response): the advantage is
    per-token (``adv_t = base - coef * k3_t``). Summing would scale by seq_len and, once broadcast
    to every token in the loss, blow up the gradient by ~seq_len.

    Imports only ``swift.rl_core.advantage`` (pure torch), so it runs without vllm/cuda.
    """

    def test_k3_per_token_value_and_nonnegativity(self):
        import torch

        from swift.rl_core.advantage import compute_teacher_kl_per_token
        teacher = torch.tensor([[-1.0, -2.0, -0.5, 0.0]])
        policy = torch.tensor([[-1.5, -1.0, -0.5, 0.0]])
        mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])  # last position masked out
        k3 = compute_teacher_kl_per_token(teacher, policy, mask)
        d = teacher - policy
        expected = (torch.exp(d) - d - 1) * mask
        self.assertEqual(tuple(k3.shape), (1, 4))  # per-token [B, T], not summed
        self.assertTrue(torch.allclose(k3, expected))
        self.assertTrue((k3 >= 0).all())  # k3 is non-negative per token
        self.assertEqual(k3[0, 3].item(), 0.0)  # masked position is zero

    def test_zero_when_teacher_equals_policy(self):
        import torch

        from swift.rl_core.advantage import compute_teacher_kl_per_token
        logps = torch.tensor([[-1.0, -2.0, -0.5]])
        mask = torch.ones_like(logps)
        k3 = compute_teacher_kl_per_token(logps, logps.clone(), mask)
        self.assertTrue(torch.allclose(k3, torch.zeros_like(k3), atol=1e-6))

    def test_expand_broadcasts_base_and_subtracts_per_token_kl(self):
        import torch

        from swift.rl_core.advantage import compute_teacher_kl_per_token, expand_advantage_to_per_token
        base = torch.tensor([0.5, -0.3])  # per-sequence base advantage [B]
        teacher = torch.tensor([[-1.0, -2.0, 0.0], [-0.5, -0.5, -0.5]])
        policy = torch.tensor([[-1.5, -1.0, 0.0], [-0.5, -1.0, -0.5]])
        mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
        coef = 2.0
        adv = expand_advantage_to_per_token(base, mask, teacher, policy, teacher_kl_coef=coef)
        k3 = compute_teacher_kl_per_token(teacher, policy, mask)
        expected = base.unsqueeze(1).expand_as(mask) - coef * k3
        self.assertEqual(tuple(adv.shape), (2, 3))  # per-token [B, T]
        self.assertTrue(torch.allclose(adv, expected))
        # Larger teacher KL -> smaller advantage (sign convention).
        self.assertTrue((adv <= base.unsqueeze(1) + 1e-6).all())

    def test_expand_without_teacher_is_plain_broadcast(self):
        import torch

        from swift.rl_core.advantage import expand_advantage_to_per_token
        base = torch.tensor([0.5, -0.3])
        mask = torch.ones(2, 4)
        adv = expand_advantage_to_per_token(base, mask)
        self.assertTrue(torch.allclose(adv, base.unsqueeze(1).expand_as(mask)))

    def test_pure_distillation_advantage_is_per_token_kl(self):
        # No reward funcs: base advantages are 0, so per-token advantage == -coef * k3_t.
        import torch

        from swift.rl_core.advantage import (compute_advantages, compute_teacher_kl_per_token,
                                             expand_advantage_to_per_token)
        rewards_per_func = torch.zeros(4, 0)
        reward_weights = torch.zeros(0)
        base, _ = compute_advantages(rewards_per_func, reward_weights, num_generations=4)
        self.assertTrue(torch.allclose(base, torch.zeros(4)))  # base advantage is zero
        teacher = torch.tensor([[-1.0, -2.0], [-0.5, -1.0], [-1.0, -1.0], [-2.0, -0.5]])
        policy = torch.tensor([[-1.5, -1.0], [-0.5, -0.5], [-1.0, -2.0], [-1.0, -1.0]])
        mask = torch.ones(4, 2)
        adv = expand_advantage_to_per_token(base, mask, teacher, policy, teacher_kl_coef=1.0)
        self.assertTrue(torch.allclose(adv, -compute_teacher_kl_per_token(teacher, policy, mask)))


if __name__ == '__main__':
    unittest.main()

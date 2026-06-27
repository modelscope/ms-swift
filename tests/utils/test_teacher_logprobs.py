import torch
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from swift.rl_core.advantage import (compute_teacher_kl_per_token, compute_teacher_signed_logratio,
                                     expand_advantage_to_per_token)
from swift.rl_core.data import GRPOSample
from swift.rlhf_trainers.gkd_helpers import assemble_teacher_completion_logprobs, build_teacher_requests
from swift.rlhf_trainers.utils import parse_prompt_logprobs


class _FakeResponse:

    def __init__(self, prompt_logprobs):
        self.prompt_logprobs = prompt_logprobs


class TestParsePromptLogprobs(unittest.TestCase):

    def test_topk_zero_sampled_token(self):
        resp = _FakeResponse([
            None,
            {
                101: {
                    'logprob': -0.5,
                    'rank': 3
                }
            },
            {
                202: {
                    'logprob': -1.2,
                    'rank': 1
                }
            },
        ])
        lps, ixs = parse_prompt_logprobs(resp, topk=0)
        self.assertEqual(lps, [[-0.5], [-1.2]])
        self.assertEqual(ixs, [[101], [202]])

    def test_topk_positive(self):
        resp = _FakeResponse([
            None,
            {
                1: {
                    'logprob': -0.1,
                    'rank': 1
                },
                2: {
                    'logprob': -0.5,
                    'rank': 2
                },
                3: {
                    'logprob': -1.0,
                    'rank': 3
                },
            },
        ])
        lps, ixs = parse_prompt_logprobs(resp, topk=2)
        self.assertEqual(lps, [[-0.1, -0.5]])
        self.assertEqual(ixs, [[1, 2]])


class TestAssembleTeacherCompletionLogprobs(unittest.TestCase):

    def test_aligns_to_completion_mask(self):
        completion_mask = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.float32)
        parsed = [([[-0.3], [-0.4], [-0.5]], [[10], [11], [12]])]
        out = assemble_teacher_completion_logprobs(parsed, completion_mask, device=torch.device('cpu'))
        self.assertEqual(out.topk_logprobs.shape, (1, 5, 1))
        self.assertAlmostEqual(out.topk_logprobs[0, 2, 0].item(), -0.3, places=5)
        self.assertAlmostEqual(out.topk_logprobs[0, 4, 0].item(), -0.5, places=5)

    def test_full_sequence_match_by_response_token_ids(self):
        completion_mask = torch.tensor([[0, 0, 1, 1]], dtype=torch.float32)
        parsed = [(
            [[-0.1], [-0.2], [-0.3], [-0.4], [-0.5]],
            [[1], [2], [10], [11], [12]],
        )]
        out = assemble_teacher_completion_logprobs(
            parsed,
            completion_mask,
            device=torch.device('cpu'),
            response_token_ids=[[10, 11]],
        )
        self.assertAlmostEqual(out.topk_logprobs[0, 2, 0].item(), -0.3, places=5)
        self.assertAlmostEqual(out.topk_logprobs[0, 3, 0].item(), -0.4, places=5)


class TestTeacherKlPerToken(unittest.TestCase):

    def test_k3_non_negative(self):
        teacher = torch.tensor([[0.0, -0.5, -1.0]])
        policy = torch.tensor([[-0.2, -0.5, -0.8]])
        mask = torch.tensor([[1.0, 1.0, 1.0]])
        k3 = compute_teacher_kl_per_token(teacher, policy, mask)
        self.assertTrue((k3 >= 0).all())
        self.assertEqual(k3.shape, (1, 3))

    def test_expand_without_teacher_broadcasts(self):
        base = torch.tensor([0.5, -0.3])
        mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        out = expand_advantage_to_per_token(base, mask)
        self.assertEqual(out.shape, (2, 3))
        self.assertAlmostEqual(out[0, 0].item(), 0.5, places=5)
        self.assertAlmostEqual(out[0, 1].item(), 0.5, places=5)
        self.assertEqual(out[0, 2].item(), 0.5)

    def test_signed_logratio_has_direction(self):
        # teacher prefers token 0 over student (d>0); student over-confident on token 1 (d<0).
        teacher = torch.tensor([[-0.2, -1.5]])
        policy = torch.tensor([[-0.8, -0.3]])
        mask = torch.tensor([[1.0, 1.0]])
        signed = compute_teacher_signed_logratio(teacher, policy, mask)
        self.assertGreater(signed[0, 0].item(), 0.0)  # teacher-preferred -> raise prob
        self.assertLess(signed[0, 1].item(), 0.0)  # student over-confident -> lower prob
        self.assertEqual(signed.shape, (1, 2))

    def test_expand_with_teacher_adds_signed_logratio(self):
        # adv_t = base + coef * (teacher_logp - student_logp), the signed k1 reward.
        base = torch.tensor([1.0])
        mask = torch.tensor([[1.0, 1.0]])
        teacher = torch.tensor([[0.0, -2.0]])
        policy = torch.tensor([[-0.5, -0.5]])
        out = expand_advantage_to_per_token(
            base, mask, teacher_per_token_logps=teacher, policy_per_token_logps=policy, teacher_kl_coef=1.0)
        # token 0: teacher higher -> advantage > base; token 1: teacher lower -> advantage < base
        self.assertAlmostEqual(out[0, 0].item(), 1.0 + (0.0 - (-0.5)), places=5)
        self.assertAlmostEqual(out[0, 1].item(), 1.0 + (-2.0 - (-0.5)), places=5)
        self.assertGreater(out[0, 0].item(), 1.0)
        self.assertLess(out[0, 1].item(), 1.0)

    def test_pure_distillation_advantage_is_signed(self):
        # base_adv = 0 (no reward funcs): advantage is purely the signed teacher log-ratio,
        # so it must be able to take *both* signs (the old k3 bug made it always <= 0).
        base = torch.tensor([0.0])
        mask = torch.tensor([[1.0, 1.0]])
        teacher = torch.tensor([[0.0, -2.0]])
        policy = torch.tensor([[-0.5, -0.5]])
        out = expand_advantage_to_per_token(
            base, mask, teacher_per_token_logps=teacher, policy_per_token_logps=policy, teacher_kl_coef=1.0)
        self.assertGreater(out[0, 0].item(), 0.0)
        self.assertLess(out[0, 1].item(), 0.0)


class TestBuildTeacherRequests(unittest.TestCase):

    def test_injects_response_token_ids_when_template_given(self):
        sample = GRPOSample(
            messages=[
                {
                    'role': 'user',
                    'content': 'hi'
                },
                {
                    'role': 'assistant',
                    'content': 'old text'
                },
            ],
            response_token_ids=[[42, 43]],
        )
        template = MagicMock()
        template.get_response_prefix_ids = MagicMock(return_value=[])
        import swift.rlhf_trainers.gkd_helpers as gh
        orig = gh.replace_assistant_response_with_ids
        captured = {}

        def _capture(msgs, ids, loss_mask=None, non_thinking_prefix_ids=None):
            captured['ids'] = ids
            return orig(msgs, ids, loss_mask=loss_mask, non_thinking_prefix_ids=non_thinking_prefix_ids)

        gh.replace_assistant_response_with_ids = _capture
        try:
            build_teacher_requests([sample], template=template)
        finally:
            gh.replace_assistant_response_with_ids = orig
        self.assertEqual(captured.get('ids'), [[42, 43]])

    def test_teacher_messages_win_over_response_token_ids(self):
        # GKD OPSD view (teacher_messages) must take precedence over the GRPO response-id injection.
        teacher_view = [
            {
                'role': 'user',
                'content': 'hi'
            },
            {
                'role': 'assistant',
                'content': 'teacher view'
            },
        ]
        sample = GRPOSample(
            messages=[
                {
                    'role': 'user',
                    'content': 'hi'
                },
                {
                    'role': 'assistant',
                    'content': 'old text'
                },
            ],
            response_token_ids=[[42, 43]],
        )
        sample.teacher_messages = teacher_view
        template = MagicMock()
        template.get_response_prefix_ids = MagicMock(return_value=[])
        import swift.rlhf_trainers.gkd_helpers as gh
        orig = gh.replace_assistant_response_with_ids
        called = {'hit': False}

        def _capture(msgs, ids, loss_mask=None, non_thinking_prefix_ids=None):
            called['hit'] = True
            return orig(msgs, ids, loss_mask=loss_mask, non_thinking_prefix_ids=non_thinking_prefix_ids)

        gh.replace_assistant_response_with_ids = _capture
        try:
            reqs = build_teacher_requests([sample], template=template)
        finally:
            gh.replace_assistant_response_with_ids = orig
        self.assertFalse(called['hit'])  # response-id injection skipped
        self.assertEqual(reqs[0].messages, teacher_view)


class TestLigerAdvantageReduce(unittest.TestCase):
    """Liger expects [B]; expand_advantage_to_per_token yields [B, T]."""

    def test_masked_mean_matches_per_seq_for_constant_tokens(self):
        B, T = 3, 4
        coef_1 = torch.ones(B, T)
        mask = torch.ones(B, T)
        base = torch.tensor([0.5, -0.3, 0.1])
        adv_tok = expand_advantage_to_per_token(base, mask)
        adv_seq = (adv_tok * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
        loss_liger = ((coef_1 * adv_seq.unsqueeze(1) * mask).sum(-1) / mask.sum(-1)).sum() / B
        loss_direct = ((coef_1 * adv_tok * mask).sum(-1) / mask.sum(-1)).sum() / B
        self.assertAlmostEqual(loss_liger.item(), loss_direct.item(), places=5)

    def test_b_gt_1_wrong_if_skip_reduce(self):
        B, T = 3, 4
        coef_1 = torch.ones(B, T)
        mask = torch.ones(B, T)
        base = torch.tensor([0.5, -0.3, 0.1])
        adv_tok = expand_advantage_to_per_token(base, mask)
        wrong = coef_1 * adv_tok.unsqueeze(1)
        self.assertEqual(wrong.shape, (B, B, T))


if __name__ == '__main__':
    unittest.main()

"""CPU numerical contracts; these do not replace checkpoint training validation."""
import importlib.util
import json
import tempfile
import torch
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

source = Path(__file__).resolve().parents[2] / 'examples/train/startlux_decision/decision_training.py'
spec = importlib.util.spec_from_file_location('startlux_training_contract', source)
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)
CandidateHead = contract.CandidateHead
audit_splits = contract.audit_splits
candidate_loss = contract.candidate_loss
collate = contract.collate


class DecisionContract(unittest.TestCase):

    def test_candidate_head_matches_fp32_reference_and_freezes_weight(self):
        torch.manual_seed(4)
        head = torch.nn.Linear(8, 64, bias=False).to(torch.bfloat16)
        ids = list(range(1, 53, 2))
        candidate = CandidateHead(head, ids)
        x = torch.randn(2, 5, 8, dtype=torch.bfloat16, requires_grad=True)
        actual = candidate(x)
        expected = x.float() @ head.weight[ids].float().T
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.square().mean().backward()
        self.assertIsNone(head.weight.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertEqual(set(candidate.state_dict()), {'weight'})

    def test_soft_loss_uses_last_real_token_and_only_legal_candidates(self):
        torch.manual_seed(3)
        logits = torch.randn(2, 5, 26, requires_grad=True)
        mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1]])
        targets = torch.full((2, 26), -1.)
        targets[0, :2] = torch.tensor([.3, .7])
        targets[1, :3] = torch.tensor([0., .4, .6])
        loss = candidate_loss(logits, mask, targets)
        expected = torch.stack([
            -(targets[0, :2] * logits[0, 1, :2].log_softmax(-1)).sum(),
            -(targets[1, :3] * logits[1, 4, :3].log_softmax(-1)).sum()
        ])
        torch.testing.assert_close(loss, expected)
        loss.mean().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(logits.grad[0, 2:].abs().sum().item(), 0)
        self.assertEqual(logits.grad[:, :, 3:].abs().sum().item(), 0)

    def test_collation_does_not_append_answer_tokens(self):
        batch = collate([{
            'input_ids': [5, 7],
            'labels': [1., 0.] + [-1.] * 24
        }, {
            'input_ids': [6],
            'labels': [0., 1.] + [-1.] * 24
        }], 0)
        self.assertEqual(batch['input_ids'].tolist(), [[5, 7], [6, 0]])
        self.assertEqual(batch['attention_mask'].tolist(), [[1, 1], [1, 0]])
        self.assertFalse(batch['use_cache'])

    def test_case_split_overlap_is_rejected(self):
        audit_splits({'a'}, {'b'})
        with self.assertRaisesRegex(ValueError, 'overlap'):
            audit_splits({'a'}, {'a'})

    def test_accuracy_gold_is_not_replaced_by_soft_target_tie_break(self):
        renderer = SimpleNamespace(
            from_systemone=lambda *args, **kwargs: {'options': [{
                'id': 'a'
            }, {
                'id': 'b'
            }]},
            render_ids=lambda row, tokenizer, order, **kwargs: ([7, 9], order))
        row = {'state': 'question', 'questions': {'q': {}}, 'targets': {'q': {'a': 0.5, 'b': 0.5}}, 'gold': {'q': 'b'}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cases.jsonl'
            path.write_text(json.dumps(row) + '\n')
            with patch.dict('sys.modules', {'startlux_decision': SimpleNamespace(jevfmt=renderer)}):
                features, _ = contract.encode_records(path, None, with_gold=True)
        self.assertEqual(features[0]['gold'], 1)
        self.assertEqual(features[0]['labels'][:2], [0.5, 0.5])
        self.assertEqual(features[0]['input_ids'], [7, 9])


if __name__ == '__main__':
    unittest.main()

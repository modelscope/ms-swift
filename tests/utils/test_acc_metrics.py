import torch
import unittest

from swift.metrics import compute_acc

# Two sequences, each two prompt tokens (label -100) followed by two response tokens.
# The prediction at position i is the guess for token i + 1, so the first sequence is
# answered correctly and the second one gets its final token wrong.
PACKED_LABELS = torch.tensor([[-100, -100, 5, 6, -100, -100, 7, 8]])
PACKED_PREDS = torch.tensor([[0, 5, 6, 0, 0, 7, 9, 0]])
CU_SEQLENS = torch.tensor([0, 4, 8])

BATCH_LABELS = torch.tensor([[-100, -100, 5, 6], [-100, -100, 7, 8]])
BATCH_PREDS = torch.tensor([[0, 5, 6, 0], [0, 7, 9, 0]])

# A row whose response is cut off by --truncation_strategy right/split keeps no supervised label at all,
# so the second sequence below cannot be scored in either direction.
IGNORED_BATCH_LABELS = torch.tensor([[-100, -100, 7, 8], [-100, -100, -100, -100]])
IGNORED_BATCH_PREDS = torch.tensor([[0, 7, 9, 0], [0, 1, 2, 3]])
IGNORED_PACKED_LABELS = torch.tensor([[-100, -100, 7, 8, -100, -100, -100, -100]])
IGNORED_PACKED_PREDS = torch.tensor([[0, 7, 9, 0, 0, 1, 2, 3]])

ALL_IGNORED_LABELS = torch.full((2, 4), -100, dtype=torch.long)
ALL_IGNORED_PREDS = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])


class TestComputeAcc(unittest.TestCase):

    def test_padding_free_seq_acc_skips_ignored_labels(self):
        metrics = compute_acc(PACKED_PREDS, PACKED_LABELS, acc_strategy='seq', cu_seqlens=CU_SEQLENS)

        self.assertEqual([bool(acc) for acc in metrics['seq_acc']], [True, False])

    def test_padding_free_seq_acc_matches_the_padded_batch(self):
        packed = compute_acc(PACKED_PREDS, PACKED_LABELS, acc_strategy='seq', cu_seqlens=CU_SEQLENS)
        padded = compute_acc(BATCH_PREDS, BATCH_LABELS, acc_strategy='seq')

        self.assertEqual([bool(acc) for acc in packed['seq_acc']], [bool(acc) for acc in padded['seq_acc']])

    def test_padded_seq_acc_skips_a_sequence_without_supervised_labels(self):
        metrics = compute_acc(IGNORED_BATCH_PREDS, IGNORED_BATCH_LABELS, acc_strategy='seq')

        self.assertEqual([bool(acc) for acc in metrics['seq_acc']], [False])

    def test_padding_free_seq_acc_skips_a_sequence_without_supervised_labels(self):
        metrics = compute_acc(IGNORED_PACKED_PREDS, IGNORED_PACKED_LABELS, acc_strategy='seq', cu_seqlens=CU_SEQLENS)

        self.assertEqual([bool(acc) for acc in metrics['seq_acc']], [False])

    def test_seq_acc_reports_nothing_when_no_label_is_supervised(self):
        self.assertEqual(compute_acc(ALL_IGNORED_PREDS, ALL_IGNORED_LABELS, acc_strategy='seq'), {})

    def test_token_acc_reports_nothing_when_no_label_is_supervised(self):
        self.assertEqual(compute_acc(ALL_IGNORED_PREDS, ALL_IGNORED_LABELS, acc_strategy='token'), {})


if __name__ == '__main__':
    unittest.main()

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


class TestComputeAcc(unittest.TestCase):

    def test_padding_free_seq_acc_skips_ignored_labels(self):
        metrics = compute_acc(PACKED_PREDS, PACKED_LABELS, acc_strategy='seq', cu_seqlens=CU_SEQLENS)

        self.assertEqual([bool(acc) for acc in metrics['seq_acc']], [True, False])

    def test_padding_free_seq_acc_matches_the_padded_batch(self):
        packed = compute_acc(PACKED_PREDS, PACKED_LABELS, acc_strategy='seq', cu_seqlens=CU_SEQLENS)
        padded = compute_acc(BATCH_PREDS, BATCH_LABELS, acc_strategy='seq')

        self.assertEqual([bool(acc) for acc in packed['seq_acc']], [bool(acc) for acc in padded['seq_acc']])


if __name__ == '__main__':
    unittest.main()

import torch
import unittest
from unittest import mock

from swift.trainers.utils import pad_for_ddp_gather, pad_to_global_max_len


class TestSeq2SeqTrainerDdpPadding(unittest.TestCase):

    def test_pad_to_global_max_len(self):
        tensor = torch.tensor([[1, 2, 3], [4, 5, 0]])
        padded = pad_to_global_max_len(tensor, global_max_len=5, padding_value=0)
        self.assertEqual(padded.shape, (2, 5))
        self.assertTrue(torch.equal(padded, torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])))

    def test_pad_to_global_max_len_noop_when_already_max(self):
        tensor = torch.tensor([[1, 2], [3, 4]])
        padded = pad_to_global_max_len(tensor, global_max_len=2, padding_value=0)
        self.assertTrue(torch.equal(padded, tensor))

    def test_ddp_gather_preserves_2d_shape_after_global_padding(self):
        rank0 = pad_to_global_max_len(torch.tensor([[1, 2, 3], [4, 5, 0]]), global_max_len=5)
        rank1 = pad_to_global_max_len(torch.tensor([[6, 7, 8, 9, 10], [11, 12, 0, 0, 0]]), global_max_len=5)
        gathered = torch.cat([rank0, rank1], dim=0)
        self.assertEqual(gathered.ndim, 2)
        self.assertEqual(gathered.shape, (4, 5))

    def test_pad_for_ddp_gather_without_dist(self):
        tensor = torch.tensor([[1, 2, 3], [4, 5, 0]])
        padded = pad_for_ddp_gather(tensor, padding_value=0)
        self.assertTrue(torch.equal(padded, tensor))

    def test_pad_for_ddp_gather_with_dist(self):
        tensor = torch.tensor([[1, 2, 3], [4, 5, 0]])

        def fake_all_reduce(t, op=None):
            t.fill_(5)

        with mock.patch('swift.trainers.utils.dist.is_available', return_value=True), \
                mock.patch('swift.trainers.utils.dist.is_initialized', return_value=True), \
                mock.patch('swift.trainers.utils.dist.all_reduce', side_effect=fake_all_reduce):
            padded = pad_for_ddp_gather(tensor, padding_value=0)

        self.assertEqual(padded.shape, (2, 5))
        self.assertTrue(torch.equal(padded, torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])))


if __name__ == '__main__':
    unittest.main()

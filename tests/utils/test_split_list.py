# Copyright (c) ModelScope Contributors. All rights reserved.
import unittest

from swift.utils import split_list


class TestSplitList(unittest.TestCase):

    def test_round_robin_preserves_mixed_types(self):
        self.assertEqual(split_list([1, 'two', 3.5, 'four'], 2, contiguous=False), [[1, 3.5], ['two', 'four']])

    def test_round_robin_preserves_ragged_lists(self):
        items = [[1], [2, 3], [4, 5, 6]]
        self.assertEqual(split_list(items, 2, contiguous=False), [[[1], [4, 5, 6]], [[2, 3]]])

    def test_round_robin_preserves_item_identity(self):
        items = [(1, 2), (3, 4), (5, 6)]
        shards = split_list(items, 2, contiguous=False)
        self.assertIs(shards[0][0], items[0])
        self.assertIs(shards[0][1], items[2])
        self.assertIs(shards[1][0], items[1])

    def test_round_robin_order(self):
        self.assertEqual(split_list(list(range(5)), 3, contiguous=False), [[0, 3], [1, 4], [2]])

    def test_contiguous_order(self):
        self.assertEqual(split_list(list(range(5)), 3), [[0], [1, 2], [3, 4]])

    def test_empty_input(self):
        for contiguous in (True, False):
            with self.subTest(contiguous=contiguous):
                self.assertEqual(split_list([], 3, contiguous=contiguous), [[], [], []])

    def test_more_shards_than_items(self):
        self.assertEqual(split_list([1, 2], 4, contiguous=False), [[1], [2], [], []])


if __name__ == '__main__':
    unittest.main()

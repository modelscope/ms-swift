import unittest

from swift.utils.llm_utils import _count_startswith


class TestLlmUtils(unittest.TestCase):

    def test_count_startswith(self):
        arr = [-100, -100, 2, -100]
        self.assertTrue(_count_startswith(arr, -100, 0) == 2)
        self.assertTrue(_count_startswith(arr, -100, 1) == 1)


if __name__ == '__main__':
    unittest.main()

import unittest

from swift.utils.llm_utils import lower_bound, upper_bound


class TestLlmUtils(unittest.TestCase):

    def test_count_startswith(self):
        arr = [-100] * 1000 + list(range(1000))
        self.assertTrue(
            lower_bound(0, len(arr), lambda i: arr[i] != -100) == 1000)

    def test_count_endswith(self):
        arr = list(range(1000)) + [-100] * 1000
        self.assertTrue(
            lower_bound(0, len(arr), lambda i: arr[i] == -100) == 1000)


if __name__ == '__main__':
    unittest.main()

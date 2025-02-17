import unittest

from swift.llm import load_dataset
from swift.utils import lower_bound


class TestLlmUtils(unittest.TestCase):

    def test_count_startswith(self):
        arr = [-100] * 1000 + list(range(1000))
        self.assertTrue(lower_bound(0, len(arr), lambda i: arr[i] != -100) == 1000)

    def test_count_endswith(self):
        arr = list(range(1000)) + [-100] * 1000
        self.assertTrue(lower_bound(0, len(arr), lambda i: arr[i] == -100) == 1000)

    @unittest.skip('avoid ci error')
    def test_dataset(self):
        dataset = load_dataset(['AI-ModelScope/alpaca-gpt4-data-zh#1000', 'AI-ModelScope/alpaca-gpt4-data-en#200'],
                               num_proc=4,
                               strict=False,
                               download_mode='force_redownload')
        print(f'dataset[0]: {dataset[0]}')
        print(f'dataset[1]: {dataset[1]}')


if __name__ == '__main__':
    unittest.main()

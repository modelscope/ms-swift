import unittest

from swift.llm import load_dataset


class TestDataset(unittest.TestCase):

    def test_load_v_dataset(self):
        if not __name__ == '__main__':
            # ignore citest error in github
            return

        for ds in ['m3it#1000', 'mantis-instruct#1000', 'llava-med-zh-instruct#1000']:
            ds = load_dataset(ds)
            assert len(ds[0]) > 800


if __name__ == '__main__':
    unittest.main()

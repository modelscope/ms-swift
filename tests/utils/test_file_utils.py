import os
import shutil
import tempfile
import unittest

from swift.utils import copy_files_by_pattern


class TestFileUtils(unittest.TestCase):

    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir = self._tmp_dir.name

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_copy_files(self):
        os.makedirs(os.path.join(self.tmp_dir, 'source'))
        os.makedirs(os.path.join(self.tmp_dir, 'source', 'subfolder'))
        with open(os.path.join(self.tmp_dir, 'source', '1.txt'), 'w') as f:
            f.write('')
        with open(os.path.join(self.tmp_dir, 'source', 'subfolder', '2.txt'), 'w') as f:
            f.write('')
        copy_files_by_pattern(
            os.path.join(self.tmp_dir, 'source'), os.path.join(self.tmp_dir, 'target'), ['*.txt', 'subfolder/*.txt'])
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, 'target', '1.txt')))
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, 'target', 'subfolder', '2.txt')))


if __name__ == '__main__':
    unittest.main()

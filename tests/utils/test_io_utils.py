import os
import shutil
import tempfile
import unittest

from swift.utils import append_to_jsonl, get_logger, read_from_jsonl, write_to_jsonl

logger = get_logger()


class TestIOUtils(unittest.TestCase):

    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir = self._tmp_dir.name
        # self.tmp_dir = 'test'
        logger.info(f'self.tmp_dir: {self.tmp_dir}')

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_jsonl(self):
        fpath = os.path.join(self.tmp_dir, '1.jsonl')
        obj_list = [{'aaa': 'bbb'}, 111, [1.1]]
        write_to_jsonl(fpath, obj_list)
        new_obj = {'bbb': 'aaa'}
        obj_list.append(new_obj)
        append_to_jsonl(fpath, new_obj)
        new_obj_list = read_from_jsonl(fpath)
        self.assertTrue(new_obj_list == obj_list)

    def test_jsonl2(self):
        fpath = os.path.join(self.tmp_dir, '1.jsonl')
        obj_list = [{'aaa': 'bbb'}, 111, [1.1]]
        for obj in obj_list:
            append_to_jsonl(fpath, obj)
        new_obj_list = read_from_jsonl(fpath)
        self.assertTrue(new_obj_list == obj_list)


if __name__ == '__main__':
    unittest.main()

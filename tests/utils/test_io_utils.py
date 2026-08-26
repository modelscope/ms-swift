import os
import shutil
import tempfile
import unittest

from swift.utils import (LAST_CHECKPOINT_SYMLINK, append_to_jsonl, get_logger, read_from_jsonl,
                         update_last_checkpoint_symlink, write_to_jsonl)

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

    def _make_checkpoint(self, step: int) -> str:
        checkpoint_dir = os.path.join(self.tmp_dir, f'checkpoint-{step}')
        os.makedirs(checkpoint_dir)
        return checkpoint_dir

    def test_last_checkpoint_symlink(self):
        link_path = os.path.join(self.tmp_dir, LAST_CHECKPOINT_SYMLINK)
        self.assertEqual(update_last_checkpoint_symlink(self._make_checkpoint(2)), link_path)
        # The target is relative so that the output directory stays movable.
        self.assertEqual(os.readlink(link_path), 'checkpoint-2')
        self.assertTrue(os.path.isdir(link_path))

        update_last_checkpoint_symlink(self._make_checkpoint(4))
        self.assertEqual(os.readlink(link_path), 'checkpoint-4')
        self.assertFalse(os.path.lexists(f'{link_path}.tmp'))

    def test_last_checkpoint_symlink_skips_unusable_targets(self):
        # A checkpoint whose directory is not there yet must not be linked.
        self.assertIsNone(update_last_checkpoint_symlink(os.path.join(self.tmp_dir, 'checkpoint-2')))
        self.assertFalse(os.path.lexists(os.path.join(self.tmp_dir, LAST_CHECKPOINT_SYMLINK)))

    def test_last_checkpoint_symlink_keeps_real_directory(self):
        link_path = os.path.join(self.tmp_dir, LAST_CHECKPOINT_SYMLINK)
        os.makedirs(link_path)
        self.assertIsNone(update_last_checkpoint_symlink(self._make_checkpoint(2)))
        self.assertFalse(os.path.islink(link_path))


if __name__ == '__main__':
    unittest.main()

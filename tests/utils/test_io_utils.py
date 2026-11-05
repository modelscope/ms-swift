import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

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

    def test_empty_jsonl(self):
        fpath = os.path.join(self.tmp_dir, 'empty.jsonl')
        write_to_jsonl(fpath, [])
        self.assertEqual(read_from_jsonl(fpath), [])

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
        self.assertEqual(set(os.listdir(self.tmp_dir)), {'checkpoint-2', 'checkpoint-4', LAST_CHECKPOINT_SYMLINK})

    def test_last_checkpoint_symlink_overlapping_updates(self):
        first_checkpoint = self._make_checkpoint(1)
        second_checkpoint = self._make_checkpoint(2)
        link_path = os.path.join(self.tmp_dir, LAST_CHECKPOINT_SYMLINK)
        original_replace = os.replace
        for fail_first in (False, True):
            with self.subTest(fail_first=fail_first):

                def replace(src, dst):
                    # Complete another update while the first updater's staging link is still pending.
                    with patch('swift.utils.io_utils.os.replace', original_replace):
                        self.assertEqual(update_last_checkpoint_symlink(second_checkpoint), link_path)
                    self.assertEqual(os.readlink(dst), 'checkpoint-2')
                    if fail_first:
                        raise OSError('Simulated replacement failure')
                    original_replace(src, dst)

                with patch('swift.utils.io_utils.os.replace', side_effect=replace):
                    self.assertEqual(
                        update_last_checkpoint_symlink(first_checkpoint), None if fail_first else link_path)
                self.assertEqual(os.readlink(link_path), 'checkpoint-2' if fail_first else 'checkpoint-1')
                self.assertEqual(
                    set(os.listdir(self.tmp_dir)), {'checkpoint-1', 'checkpoint-2', LAST_CHECKPOINT_SYMLINK})

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

import os
import shutil
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Event, current_thread
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

    def _check_concurrent_checkpoint_updates(self, fail_first=False):
        link_path = update_last_checkpoint_symlink(self._make_checkpoint(0))
        first_checkpoint = self._make_checkpoint(2)
        second_checkpoint = self._make_checkpoint(1)
        first_thread = current_thread()
        first_staged = Event()
        second_staged = Event()
        first_finished = Event()
        publications = []
        original_replace = os.replace

        def replace(src, dst):
            if current_thread() is first_thread:
                first_staged.set()
                self.assertTrue(second_staged.wait(10), 'Second updater did not stage its link')
                self.assertEqual(os.readlink(dst), 'checkpoint-0')
                self.assertTrue(os.path.isdir(dst))
                if fail_first:
                    raise OSError('Simulated replacement failure')
                expected_target = 'checkpoint-2'
            else:
                second_staged.set()
                self.assertTrue(first_finished.wait(10), 'First updater did not finish')
                self.assertEqual(os.readlink(dst), 'checkpoint-0' if fail_first else 'checkpoint-2')
                self.assertTrue(os.path.isdir(dst))
                expected_target = 'checkpoint-1'
            original_replace(src, dst)
            publications.append((expected_target, os.readlink(dst)))

        def update_second():
            self.assertTrue(first_staged.wait(10), 'First updater did not stage its link')
            return update_last_checkpoint_symlink(second_checkpoint)

        # Stage both links before allowing either replacement, then publish in a fixed order.
        with patch(
                'swift.utils.io_utils.os.replace', side_effect=replace), ThreadPoolExecutor(max_workers=1) as executor:
            second = executor.submit(update_second)
            try:
                first_result = update_last_checkpoint_symlink(first_checkpoint)
            finally:
                first_finished.set()
            second_result = second.result(timeout=10)

        self.assertEqual(first_result, None if fail_first else link_path)
        self.assertEqual(second_result, link_path)
        expected_targets = ['checkpoint-1'] if fail_first else ['checkpoint-2', 'checkpoint-1']
        self.assertEqual(publications, [(target, target) for target in expected_targets])
        # Publication order, rather than the checkpoint step, determines the final target.
        self.assertEqual(os.readlink(link_path), 'checkpoint-1')
        self.assertEqual(
            set(os.listdir(self.tmp_dir)), {'checkpoint-0', 'checkpoint-1', 'checkpoint-2', LAST_CHECKPOINT_SYMLINK})

    def test_last_checkpoint_symlink_concurrent_updates(self):
        self._check_concurrent_checkpoint_updates()

    def test_last_checkpoint_symlink_failed_update_keeps_other_writer(self):
        self._check_concurrent_checkpoint_updates(fail_first=True)

    def test_last_checkpoint_symlink_creation_failure(self):
        link_path = update_last_checkpoint_symlink(self._make_checkpoint(1))
        checkpoint_dir = self._make_checkpoint(2)
        with patch('swift.utils.io_utils.os.symlink', side_effect=OSError('Symlinks are not supported')):
            self.assertIsNone(update_last_checkpoint_symlink(checkpoint_dir))
        self.assertEqual(os.readlink(link_path), 'checkpoint-1')
        self.assertTrue(os.path.isdir(link_path))
        self.assertEqual(set(os.listdir(self.tmp_dir)), {'checkpoint-1', 'checkpoint-2', LAST_CHECKPOINT_SYMLINK})

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

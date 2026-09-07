# Copyright (c) ModelScope Contributors. All rights reserved.
import types
import unittest

from swift.trainers.utils import accepts_parameter, check_dlrover_flash_checkpoint_api


class DLRover061Engine:

    def __init__(self):
        self.calls = []

    def wait_latest_checkpoint(self, timeout=None):
        self.calls.append((timeout, ))


class DLRoverMasterEngine:

    def __init__(self):
        self.calls = []

    def wait_latest_checkpoint(self, timeout=None, max_steps=None):
        self.calls.append((timeout, max_steps))


class DLRover061Checkpointer:

    def save_checkpoint_to_storage(self, step):
        pass


class DLRoverMasterCheckpointer:

    def save_checkpoint_to_storage(self, step, blocking=False):
        pass


class KwargsCheckpointer:

    def save_checkpoint_to_storage(self, step, **kwargs):
        pass


class TestFlashCheckpointCompatibility(unittest.TestCase):

    def test_accepts_parameter(self):
        self.assertFalse(accepts_parameter(DLRover061Checkpointer.save_checkpoint_to_storage, 'blocking'))
        self.assertTrue(accepts_parameter(DLRoverMasterCheckpointer.save_checkpoint_to_storage, 'blocking'))
        self.assertTrue(accepts_parameter(KwargsCheckpointer.save_checkpoint_to_storage, 'blocking'))
        self.assertTrue(accepts_parameter(DLRoverMasterCheckpointer().save_checkpoint_to_storage, 'blocking'))

    def test_dlrover_061_only_warns(self):
        with self.assertLogs('swift', level='WARNING') as logs:
            check_dlrover_flash_checkpoint_api(DLRover061Checkpointer, DLRover061Engine)
        message = '\n'.join(logs.output)
        self.assertIn('blocking, max_steps', message)
        self.assertIn('pip install git+https://github.com/intelligent-machine-learning/dlrover.git', message)

    def test_new_dlrover_api_is_silent(self):
        with self.assertNoLogs('swift', level='WARNING'):
            check_dlrover_flash_checkpoint_api(DLRoverMasterCheckpointer, DLRoverMasterEngine)


class TestWaitLatestCheckpoint(unittest.TestCase):
    """`SwiftMixin.wait_latest_checkpoint` has to call whichever signature the installed dlrover exposes."""

    @staticmethod
    def _wait(engine):
        from swift.trainers.mixin import SwiftMixin
        trainer = types.SimpleNamespace(
            flash_checkpointer=types.SimpleNamespace(async_save_engine=engine), symlink_barriers=[])
        trainer._update_last_checkpoint_symlink = lambda barrier=True: trainer.symlink_barriers.append(barrier)
        SwiftMixin.wait_latest_checkpoint(trainer, 30, 4)
        return trainer

    def test_legacy_api_is_called_without_max_steps(self):
        engine = DLRover061Engine()
        self._wait(engine)
        self.assertEqual(engine.calls, [(30, )])

    def test_new_api_is_called_with_max_steps(self):
        engine = DLRoverMasterEngine()
        self._wait(engine)
        self.assertEqual(engine.calls, [(30, 4)])

    def test_symlink_is_refreshed_without_barrier(self):
        # The asynchronous persistence only lands during the wait, so the symlink is refreshed afterwards.
        trainer = self._wait(DLRoverMasterEngine())
        self.assertEqual(trainer.symlink_barriers, [False])


if __name__ == '__main__':
    unittest.main()

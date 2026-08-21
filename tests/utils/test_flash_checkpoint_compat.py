# Copyright (c) ModelScope Contributors. All rights reserved.
import unittest

from swift.trainers.mixin import _validate_dlrover_flash_checkpoint_api


class DLRover061Engine:

    def wait_latest_checkpoint(self, timeout=None):
        pass


class DLRoverMasterEngine:

    def wait_latest_checkpoint(self, timeout=None, max_steps=None):
        pass


class DLRover061Checkpointer:

    def save_checkpoint_to_storage(self, step):
        pass


class DLRoverMasterCheckpointer:

    def save_checkpoint_to_storage(self, step, blocking=False):
        pass


class TestFlashCheckpointCompatibility(unittest.TestCase):

    def test_dlrover_061_is_rejected_before_training(self):
        with self.assertRaisesRegex(ValueError, 'blocking.*max_steps') as error:
            _validate_dlrover_flash_checkpoint_api(DLRover061Checkpointer, DLRover061Engine)
        install_command = 'pip install git+https://github.com/intelligent-machine-learning/dlrover.git'
        self.assertIn(install_command, str(error.exception))

    def test_new_dlrover_api_is_accepted(self):
        _validate_dlrover_flash_checkpoint_api(DLRoverMasterCheckpointer, DLRoverMasterEngine)


if __name__ == '__main__':
    unittest.main()

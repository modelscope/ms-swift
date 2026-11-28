# Copyright (c) ModelScope Contributors. All rights reserved.
import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch


@unittest.skipUnless(importlib.util.find_spec('fsdp_turbo'), 'requires the optional FSDPTurbo package')
class TestFSDPTurboLogging(unittest.TestCase):

    def test_logging_when_swift_already_initialized_process_group(self):
        from fsdp_turbo.training.trainer import BaseTrainer

        from swift.fsdpturbo.trainer import FSDPTurboTrainer

        trainer = object.__new__(FSDPTurboTrainer)
        trainer.args = SimpleNamespace(validate_fsdpturbo=Mock(), offload_params=False)
        # Exercise the real BaseTrainer early return for an existing group.
        with patch('torch.distributed.is_initialized', return_value=True), \
                patch('torch.distributed.get_world_size', return_value=8), \
                patch('fsdp_turbo.utils.device.set_accelerator_compatible', return_value='gloo'), \
                patch('fsdp_turbo.utils.log.set_log_level') as configure_logging, \
                patch.object(BaseTrainer, '_init_distributed', autospec=True,
                             side_effect=BaseTrainer._init_distributed) as initialize:
            trainer._init_distributed()
        initialize.assert_called_once_with(trainer)
        configure_logging.assert_called_once_with('INFO')
        trainer.args.validate_fsdpturbo.assert_called_once_with(8)


if __name__ == '__main__':
    unittest.main()

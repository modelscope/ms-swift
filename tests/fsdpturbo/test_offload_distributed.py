# Copyright (c) ModelScope Contributors. All rights reserved.
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from swift.arguments import SftArguments
from swift.fsdpturbo.arguments import FSDPTurboSftArguments


class TestOffloadDistributed(unittest.TestCase):

    def initialize(self,
                   *,
                   offload=True,
                   distributed=True,
                   initialized=False,
                   device='npu',
                   backend='hccl',
                   config='cpu:gloo,npu:hccl',
                   requested=None):
        args = object.__new__(FSDPTurboSftArguments)
        args.offload_params = offload
        args.ddp_backend = requested
        args.ddp_timeout = 123
        with ExitStack() as stack:
            parent = stack.enter_context(patch.object(SftArguments, '_init_device'))
            stack.enter_context(patch('swift.utils.is_dist', return_value=distributed))
            stack.enter_context(patch('torch.distributed.is_initialized', return_value=initialized))
            stack.enter_context(patch('torch.distributed.get_backend_config', return_value=config))
            stack.enter_context(
                patch('torch.accelerator.current_accelerator', return_value=SimpleNamespace(type=device)))
            stack.enter_context(patch('torch.distributed.get_default_backend_for_device', return_value=backend))
            initialize = stack.enter_context(patch('swift.utils.init_process_group'))
            args._init_device()
        parent.assert_called_once()
        return args, initialize

    def test_offload_initializes_cpu_and_accelerator_before_hf_arguments(self):
        args, initialize = self.initialize()
        initialize.assert_called_once_with(backend='cpu:gloo,npu:hccl', timeout=123)
        self.assertIsNone(args.ddp_backend)

    def test_cuda_uses_its_registered_backend(self):
        _, initialize = self.initialize(device='cuda', backend='nccl')
        initialize.assert_called_once_with(backend='cpu:gloo,cuda:nccl', timeout=123)

    def test_non_offload_and_non_distributed_paths_do_not_create_groups(self):
        for kwargs in ({'offload': False}, {'distributed': False}):
            with self.subTest(kwargs=kwargs):
                _, initialize = self.initialize(**kwargs)
                initialize.assert_not_called()

    def test_existing_dual_backend_group_is_preserved(self):
        _, initialize = self.initialize(initialized=True)
        initialize.assert_not_called()

    def test_existing_accelerator_only_group_is_rejected_early(self):
        with self.assertRaisesRegex(ValueError, 'CPU.*process group'):
            self.initialize(initialized=True, config='npu:hccl')

    def test_conflicting_explicit_backend_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'ddp_backend'):
            self.initialize(requested='gloo')


if __name__ == '__main__':
    unittest.main()

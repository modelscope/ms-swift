import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from swift.cli import main as cli_main_module


class TestFSDPTurboCli(unittest.TestCase):

    def _run_shared_router(self, argv, route_mapping, **kwargs):
        completed = SimpleNamespace(returncode=0)
        with mock.patch.object(sys, 'argv', argv), \
                mock.patch.object(cli_main_module.importlib.util, 'find_spec',
                                  return_value=SimpleNamespace(origin='/virtual/target.py')), \
                mock.patch.object(cli_main_module.subprocess, 'run', return_value=completed) as run:
            cli_main_module.cli_main(route_mapping, **kwargs)
        return run.call_args.args[0]

    def test_primary_route_defers_torchrun_to_fsdpturbo_router(self):
        self.assertEqual(cli_main_module.ROUTE_MAPPING['fsdpturbo'], 'swift.cli.fsdpturbo')
        with mock.patch.dict(os.environ, {'NPROC_PER_NODE': '2'}, clear=True):
            command = self._run_shared_router(['swift', 'fsdpturbo', 'sft', '--model', 'tiny'],
                                              cli_main_module.ROUTE_MAPPING)

        self.assertNotIn('torch.distributed.run', command)
        self.assertEqual(command[1:], ['/virtual/target.py', 'sft', '--model', 'tiny'])

    def test_secondary_route_launches_torchrun_exactly_once(self):
        with mock.patch.dict(os.environ, {'NPROC_PER_NODE': '2'}, clear=True):
            command = self._run_shared_router(['fsdpturbo', 'sft', '--model', 'tiny'],
                                              {'sft': 'swift.cli._fsdpturbo.sft'},
                                              torchrun_all_methods=True)

        self.assertEqual(command.count('torch.distributed.run'), 1)
        self.assertEqual(command[-3:], ['/virtual/target.py', '--model', 'tiny'])

    def test_fsdpturbo_entrypoint_uses_secondary_route(self):
        from swift.cli import fsdpturbo

        with mock.patch.object(fsdpturbo, 'swift_cli_main') as shared_cli_main:
            fsdpturbo.cli_main()

        self.assertEqual(fsdpturbo.ROUTE_MAPPING, {'sft': 'swift.cli._fsdpturbo.sft'})
        shared_cli_main.assert_called_once_with(fsdpturbo.ROUTE_MAPPING, torchrun_all_methods=True)

    def test_worker_disables_swift_model_patches_before_import(self):
        from swift.cli._fsdpturbo.sft import ensure_npu_model_patch_disabled

        argv = ['sft.py', '--model', 'tiny']
        ensure_npu_model_patch_disabled(argv)

        self.assertEqual(argv[-2:], ['--enable_npu_model_patch', 'false'])

        explicit_argv = ['sft.py', '--enable_npu_model_patch', 'false']
        ensure_npu_model_patch_disabled(explicit_argv)
        self.assertEqual(explicit_argv, ['sft.py', '--enable_npu_model_patch', 'false'])

    def test_worker_rejects_enabling_swift_model_patches(self):
        from swift.cli._fsdpturbo.sft import ensure_npu_model_patch_disabled

        for argv in [
            ['sft.py', '--enable_npu_model_patch', 'true'],
            ['sft.py', '--enable-npu-model-patch=1'],
        ]:
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                ensure_npu_model_patch_disabled(argv)


if __name__ == '__main__':
    unittest.main()

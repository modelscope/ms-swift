# Copyright (c) ModelScope Contributors. All rights reserved.
import importlib.util
import json
import os
import subprocess
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
FLAG = 'RAY_ENABLE_ZERO_COPY_TORCH_TENSORS'

CLI_PROBE = '''
import importlib.abc
import importlib.util
import json
import os
import runpy
import sys

flag = 'RAY_ENABLE_ZERO_COPY_TORCH_TENSORS'
root = sys.argv[2]
first_ray_env = None

class Loader(importlib.abc.Loader):
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        if module.__name__.endswith('.pipeline'):
            import ray
            from ray._private import ray_constants
            def main():
                print('PROBE ' + json.dumps({'route': 'ray', 'value': os.environ.get(flag),
                    'first_ray_env': first_ray_env,
                    'effective': getattr(ray_constants, flag, None)}))
            module.main = main
        else:
            module.megatron_rlhf_main = lambda: print('PROBE ' + json.dumps({
                'route': 'native', 'value': os.environ.get(flag)}))

class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        global first_ray_env
        if fullname == 'swift':
            # Dependencies may import Ray during Swift package initialization.
            import ray
        if fullname == 'ray':
            first_ray_env = os.environ.get(flag)
        if fullname in ('swift.ray.megatron.pipeline', 'swift.megatron'):
            return importlib.util.spec_from_loader(fullname, Loader(), is_package=fullname == 'swift.megatron')

sys.meta_path.insert(0, Finder())
sys.argv = ['megatron rlhf', '--use_ray', sys.argv[1]]
runpy.run_path(root + '/swift/cli/_megatron/rlhf.py', run_name='__main__')
'''


@unittest.skipUnless(importlib.util.find_spec('ray'), 'Ray is required')
class TestRayTensorTransport(unittest.TestCase):

    def cli_probe(self, value=None, use_ray=True):
        env = os.environ.copy()
        env.pop(FLAG, None)
        if value is not None:
            env[FLAG] = value
        env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
        result = subprocess.run([sys.executable, '-c', CLI_PROBE,
                                 str(use_ray).lower(), str(ROOT)],
                                env=env,
                                capture_output=True,
                                text=True,
                                timeout=90,
                                check=True)
        return json.loads(next(line[6:] for line in result.stdout.splitlines() if line.startswith('PROBE ')))

    def test_cli_enables_before_ray_import(self):
        result = self.cli_probe()
        self.assertEqual(result['route'], 'ray')
        self.assertEqual(result['value'], '1')
        self.assertEqual(result['first_ray_env'], '1')
        if result['effective'] is not None:
            self.assertTrue(result['effective'])

    def test_cli_preserves_opt_out_before_ray_import(self):
        result = self.cli_probe('0')
        self.assertEqual(result['value'], '0')
        self.assertEqual(result['first_ray_env'], '0')
        if result['effective'] is not None:
            self.assertFalse(result['effective'])

    def test_non_ray_cli_does_not_set_default(self):
        self.assertEqual(self.cli_probe(use_ray=False), {'route': 'native', 'value': None})

    def test_cli_launcher_uses_script_path(self):
        from swift.cli.main import cli_main

        script = str(ROOT / 'swift/cli/_megatron/rlhf.py')
        with patch.object(sys, 'argv', ['megatron', 'rlhf', '--use_ray', 'true']), \
                patch('swift.cli.main.importlib.util.find_spec', return_value=SimpleNamespace(origin=script)), \
                patch('swift.cli.main.get_torchrun_args', return_value=None), \
                patch('swift.cli.main.subprocess.run', return_value=SimpleNamespace(returncode=0)) as run:
            cli_main({'rlhf': 'swift.cli._megatron.rlhf'}, is_megatron=True)
        run.assert_called_once_with([sys.executable, script, '--use_ray', 'true'])

    def test_pipeline_propagates_job_environment(self):
        from swift.ray.megatron.pipeline import MegatronRayPipeline

        for value in [None, '0', '1', '']:
            with self.subTest(value=value), patch.dict(os.environ):
                os.environ.pop(FLAG, None)
                if value is not None:
                    os.environ[FLAG] = value
                pipeline = MegatronRayPipeline.__new__(MegatronRayPipeline)
                for name in ('_build_dataset', '_compute_train_iters', '_create_pools', '_init_worker_groups',
                             '_init_rollout_replicas', '_init_teacher_replicas', '_create_trainer'):
                    setattr(pipeline, name, Mock())
                pipeline._colocate_offload_ctx = nullcontext
                with patch('swift.ray.megatron.pipeline.ray.init') as init:
                    pipeline.init()
                self.assertTrue(init.call_args.kwargs['ignore_reinit_error'])
                self.assertEqual(init.call_args.kwargs['runtime_env']['env_vars'][FLAG],
                                 value if value is not None else '0')
                if value is None:
                    self.assertNotIn(FLAG, os.environ)

    def test_worker_actor_environment_preserves_user_value(self):
        from swift.ray.megatron.worker_group import WorkerGroup

        pool = SimpleNamespace(pgs=[object()], process_on_nodes=[2], visible_devices=[0, 1], world_size=2)
        for value in [None, '0', '1', '']:
            with self.subTest(value=value), patch.dict(os.environ):
                os.environ.pop(FLAG, None)
                if value is not None:
                    os.environ[FLAG] = value
                worker_cls = Mock()
                with patch.object(WorkerGroup, '_discover_master', return_value=('127.0.0.1', 12345)):
                    group = WorkerGroup.from_pool('train', pool, worker_cls)
                self.assertEqual(group.world_size, 2)
                for call in worker_cls.options.call_args_list:
                    self.assertEqual(call.kwargs['runtime_env']['env_vars'][FLAG], value if value is not None else '0')
                    self.assertEqual(call.kwargs['num_gpus'], 0)

    def test_rollout_actor_environment_preserves_user_value(self):
        from swift.ray.megatron.rollout.replica import RolloutReplica

        for value in [None, '0', '1', '']:
            with self.subTest(value=value), patch.dict(os.environ):
                os.environ.pop(FLAG, None)
                if value is not None:
                    os.environ[FLAG] = value
                replica = RolloutReplica(SimpleNamespace())
                actor_cls = Mock()
                server_module = SimpleNamespace(VllmServer=Mock)
                with patch.dict(sys.modules, {'swift.ray.megatron.rollout.vllm_server': server_module}), \
                        patch('ray.remote', return_value=Mock(return_value=actor_cls)):
                    replica._spawn_actors([('a' * 56, '0')], gpus_per_node=1)
                env = actor_cls.options.call_args.kwargs['runtime_env']['env_vars']
                self.assertEqual(env[FLAG], value if value is not None else '0')
                self.assertEqual(len(replica.servers), 1)

    def test_real_actor_roundtrip_and_owned_copies(self):
        if sys.platform != 'linux':
            self.skipTest('Real Ray actor replay is qualified on Linux')
        from ray._private import ray_constants

        if not hasattr(ray_constants, FLAG):
            self.skipTest('Requires Ray >= 2.53')
        for value in ['0', '1']:
            with self.subTest(value=value):
                env = os.environ.copy()
                env[FLAG] = value
                env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
                result = subprocess.run([
                    sys.executable,
                    str(ROOT / 'scripts/benchmark/ray_tensor_transport.py'), '--payload-mib', '1', '--rounds', '1'
                ],
                                        env=env,
                                        capture_output=True,
                                        text=True,
                                        timeout=600)
                self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-5000:])
                data = json.loads(next(line[7:] for line in result.stdout.splitlines() if line.startswith('RESULT ')))
                self.assertEqual(data['enabled'], value == '1')
                self.assertTrue(data['native_scatter_isolated'])
                self.assertTrue(all(row['shared_cpu_storage'] == (value == '1') for row in data['ownership']))


if __name__ == '__main__':
    unittest.main()

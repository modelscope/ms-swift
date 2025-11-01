# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import functools
import inspect
import os
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Literal, Optional, TypeVar, Union

import json
import numpy as np

from swift.llm.argument.base_args.ray_args import RayArguments
from swift.ray.resource_manager import ResourceManager
from swift.utils import find_free_port
from swift.utils.utils import find_node_ip

T = TypeVar('T')


def get_args():
    parser = argparse.ArgumentParser()
    _, unknown = parser.parse_known_args()
    return json.dumps(unknown)


class RayMixin:

    def call_inner_function(self, func, *args, **kwargs):
        return getattr(self.trainer, func)(*args, **kwargs)


class RayHelper:
    resource_manager: Optional[ResourceManager] = None

    worker_cls: Dict = {}

    args: RayArguments = None

    worker_instance: Dict = {}

    initialized = False

    device_groups: Dict[str, Any] = None

    _registry = None

    @staticmethod
    def init_registry():
        if RayHelper._registry is not None:
            return

        import ray

        @ray.remote
        class WorkerRegistry:

            def __init__(self):
                self.workers = {}

            def register_workers(self, group: str, worker_handles: List):
                self.workers[group] = worker_handles

            def get_workers(self, group: Optional[str] = None):
                if group:
                    return self.workers.get(group, [])
                return self.workers

            def clear(self):
                self.workers.clear()

        try:
            RayHelper._registry = ray.get_actor('swift_worker_registry')
        except ValueError:
            try:
                RayHelper._registry = WorkerRegistry.options(
                    name='swift_worker_registry',
                    lifetime='detached',
                ).remote()
            except ValueError:
                RayHelper._registry = ray.get_actor('swift_worker_registry')
        assert RayHelper._registry is not None

    @staticmethod
    def initialize(device_groups: Dict[str, Any]):
        """Initialize RayHelper.

        Args:
            device_groups: The device groups to initialize.

        Returns:
            None
        """
        if RayHelper.ray_inited():
            return
        import ray
        RayHelper.device_groups = device_groups
        ray.init()
        if RayHelper.resource_manager is None:
            # Resource manager initialize only once in the pipeline process.
            RayHelper.resource_manager = ResourceManager(device_groups)
        RayHelper.init_registry()

    @staticmethod
    def teardown():
        if RayHelper.resource_manager is not None:
            RayHelper.resource_manager.destroy_placement_group()
            RayHelper.resource_manager = None

        if RayHelper._registry is not None:
            import ray
            try:
                ray.get(RayHelper._registry.clear.remote())
                ray.kill(RayHelper._registry)
            except:  # noqa
                pass
            RayHelper._registry = None

    @staticmethod
    @contextmanager
    def patch_init():
        if not RayHelper.is_default():
            from transformers import Trainer
            init_method = Trainer.__init__

            @functools.wraps(init_method)
            def new_init(self, *args, **kwargs):
                from accelerate import Accelerator
                self.processing_class = kwargs['processing_class']
                self.args = kwargs['args']
                self.accelerator = Accelerator(kwargs['args'])
                self.is_deepspeed_enabled = getattr(self.accelerator.state, 'deepspeed_plugin', None) is not None
                self.is_fsdp_enabled = getattr(self.accelerator.state, 'fsdp_plugin', None) is not None

            Trainer.__init__ = new_init
        yield
        if not RayHelper.is_default():
            Trainer.__init__ = init_method

    @staticmethod
    def is_called_from_init():
        """If some function called from __init__.

        Ray functions perform different behaviors depending on whether they are called from __init__.

        Returns:
            Boolean.
        """
        stack = inspect.stack()
        for frame_info in stack[1:]:
            if frame_info.function == '__init__':
                return True
        return False

    @staticmethod
    def ray_inited():
        try:
            import ray
        except ImportError:
            # not installed, not inited
            return False
        return ray.is_initialized()

    @staticmethod
    def is_default():
        return 'default' in os.environ.get('RAY_SWIFT_GROUP', '').split(',')

    @staticmethod
    def is_worker():
        import ray
        return RayHelper.ray_inited() and ray._private.worker.global_worker.mode == ray._private.worker.WORKER_MODE

    @staticmethod
    def worker(group: Union[str, List[str]]):

        def decorator(cls):
            if not RayHelper.ray_inited():
                return cls
            if RayHelper.is_worker():
                return cls
            cls.decorated = True
            groups = [group] if isinstance(group, str) else group
            import ray
            _cls = ray.remote(cls)
            for g in groups:
                RayHelper.worker_cls[g] = _cls

            init_method = cls.__init__

            @functools.wraps(init_method)
            def new_init(self, *args, **kwargs):
                if not RayHelper.is_worker():
                    # Create remote workers
                    RayHelper._create_workers(group, *args, **kwargs)
                init_method(self, *args, **kwargs)

            cls.__init__ = new_init

            return cls

        return decorator

    @staticmethod
    def collect_func(method: Union[Literal['none', 'flatten'], Callable], result):
        if isinstance(result[0], tuple):
            output = []
            for i in range(len(result[0])):
                _single_result = [r[i] for r in result]
                output.append(RayHelper.collect_func(method, _single_result))
            return output
        if method == 'none':
            return result
        elif method == 'flatten':
            flatten = [item for sublist in result for item in sublist]
            if isinstance(result[0], np.ndarray):
                return np.array(flatten)
            return type(result[0])(flatten)
        elif isinstance(method, Callable):
            # Callable
            return method(result)
        else:
            raise ValueError(f'Unsupported collect method: {method}')

    @staticmethod
    def function(group: str,
                 dispatch: Union[Literal['slice', 'all'], Callable] = 'all',
                 execute: Literal['first', 'peer', 'all'] = 'all',
                 collect: Union[Literal['none', 'flatten'], Callable] = 'none'):
        """Remote execution function.

        Args:
            group: The group to execute.
            dispatch: How to dispatch the arguments.
                'slice': load balance
                'all': all processes do the same thing
            execute: How to execute
                'first': Only first worker
                'all': All processes
            collect: How to collect the results.
                'none': Return as-is
                'flatten': Return a flattened list
        Returns:
            The execution result.
        """

        def decorator(func: Callable[..., T]) -> Callable[..., T]:

            @functools.wraps(func)
            def wrapper(self, *args, **kwargs) -> T:
                if not RayHelper.ray_inited():
                    return func(self, *args, **kwargs)
                RayHelper.init_registry()
                if RayHelper.is_worker():
                    if not hasattr(self, 'group'):
                        # pass through env
                        self.group = os.environ['RAY_SWIFT_GROUP'].split(',')
                    if group not in self.group:
                        if RayHelper.is_called_from_init():
                            # Functions in init of different group, do nothing
                            return None
                        else:
                            result = RayHelper.execute_all_sync(group, dispatch, execute, func.__name__, *args,
                                                                **kwargs)
                            return RayHelper.collect_func(collect, result)
                    else:
                        return func(self, *args, **kwargs)
                else:
                    if RayHelper.is_called_from_init():
                        # each worker do its own init
                        return None
                    result = RayHelper.execute_all_sync(group, dispatch, execute, func.__name__, *args, **kwargs)
                    return RayHelper.collect_func(collect, result)

            return wrapper

        return decorator

    @staticmethod
    def execute_all_sync(group, dispatch, execute, method_name: str, *args, **kwargs):
        import ray
        return ray.get(RayHelper.execute_all_async(group, dispatch, execute, method_name, *args, **kwargs))

    @staticmethod
    def get_workers(group, execute):
        import ray
        if group in RayHelper.worker_instance:
            workers = RayHelper.worker_instance[group]
        else:
            workers = ray.get(RayHelper._registry.get_workers.remote(group))
        if execute == 'first':
            return [workers[0]]
        elif execute == 'all':
            return workers
        elif execute == 'peer':
            return workers[RayHelper.get_peer_index(len(workers))]
        else:
            raise ValueError(f'Unsupported execute method: {execute}')

    @staticmethod
    def get_peer_index(target_size):
        rank = int(os.environ.get('RANK', '0'))
        world_size = int(os.environ.get('WORLD_SIZE', '1'))

        k, m = divmod(target_size, world_size)
        start_idx = rank * k + min(rank, m)
        end_idx = (rank + 1) * k + min(rank + 1, m)
        if target_size < world_size:
            start_idx = rank % target_size
            end_idx = start_idx + 1

        return slice(start_idx, end_idx)

    @staticmethod
    def execute_all_async(group, dispatch, execute, method_name: str, *args, **kwargs):
        import ray
        workers = RayHelper.get_workers(group, execute)
        length = len(workers)
        if execute == 'first':
            return getattr(workers[0], method_name).remote(*args, **kwargs)
        elif dispatch == 'all':
            return [getattr(worker, method_name).remote(*args, **kwargs) for worker in workers]
        elif dispatch == 'slice':
            result = []

            def dispatch_func(arg, n):
                if isinstance(arg, list):
                    k, m = divmod(len(arg), n)
                    return [arg[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]
                else:
                    return [arg] * n

            args = [dispatch_func(arg, length) for arg in args]
            kwargs = {k: dispatch_func(v, length) for k, v in kwargs.items()}
            for i in range(length):
                sliced_args = tuple(arg[i] for arg in args)
                sliced_kwargs = {k: v[i] for k, v in kwargs.items()}
                if (sliced_args and sliced_args[0]) or (kwargs and list(kwargs.values())):
                    # skip empty input
                    if hasattr(workers[i], method_name):
                        remote_call = getattr(workers[i], method_name)
                        result.append(remote_call.remote(*sliced_args, **sliced_kwargs))
                    else:
                        remote_call = getattr(workers[i], 'call_inner_function')
                        result.append(remote_call.remote(method_name, *sliced_args, **sliced_kwargs))

            return result
        elif isinstance(dispatch, Callable):
            # dispatch is Callable
            result = []
            for i in range(length):
                sliced_args, sliced_kwargs = dispatch(length, i, *args, **kwargs)
                remote_call = getattr(workers[i], method_name)
                result.append(remote_call.remote(*sliced_args, **sliced_kwargs))
            return result
        else:
            raise ValueError(f'Invalid dispatch method: {dispatch}')

    @staticmethod
    def _create_workers(group: Union[str, List[str]], *args, **kwargs):
        import ray
        from ray.runtime_env import RuntimeEnv
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
        exp_name = os.environ.get('RAY_SWIFT_EXP_NAME')
        if not exp_name:
            exp_name = ''
        else:
            exp_name += '-'

        if isinstance(group, str):
            group = [group]

        for _group in group:
            if _group in RayHelper.worker_instance:
                continue

            worker_cls = RayHelper.worker_cls[_group]

            _config = None
            for name, config in RayHelper.device_groups.items():
                if name in RayHelper.resource_manager.possible_keys:
                    continue

                if _group in config['workers']:
                    _config = config
                    break

            if _config is None:
                continue
            local_groups = _config['workers']

            VISIBLE_ENV_MAPPING = {
                'GPU': 'CUDA_VISIBLE_DEVICES',
                'NPU': 'ASCEND_VISIBLE_DEVICES',
            }

            if _config['device'].upper() != 'CPU':
                world_size = len(_config['ranks'])
                placement_groups: List[List[Dict]] = RayHelper.resource_manager.resource(_group)
                workers = []
                ip, port = None, None
                for rank, (deploy_pg, gpu) in enumerate(zip(placement_groups, _config['ranks'])):
                    deploy_pg: Dict
                    cluster_name = exp_name + '-'.join(local_groups)
                    worker_name = cluster_name + '-' + str(rank)
                    env_vars = os.environ.copy()
                    env_vars.update({
                        'WORLD_SIZE':
                        str(world_size),
                        'RANK':
                        str(rank),
                        'LOCAL_RANK':
                        str(0),
                        'CLUSTER_NAME':
                        cluster_name,
                        'WORKER_NAME':
                        worker_name,
                        VISIBLE_ENV_MAPPING[_config['device'].upper()]:
                        ','.join([str(r) for r in deploy_pg['gpu_rank']]),  # TODO npu
                        'RAY_SWIFT_ARGS':
                        get_args(),  # pass through env
                    })

                    @ray.remote
                    def get_node_address():
                        return find_node_ip(), find_free_port()

                    if rank == 0:
                        ip, port = ray.get(
                            get_node_address.options(placement_group=deploy_pg['placement_group']).remote())

                    env_vars['MASTER_ADDR'] = ip
                    env_vars['MASTER_PORT'] = str(port)
                    env_vars['RAY_SWIFT_GROUP'] = ','.join(local_groups)

                    runtime_env = RuntimeEnv(env_vars=env_vars)

                    worker_options = {
                        'scheduling_strategy':
                        PlacementGroupSchedulingStrategy(placement_group=deploy_pg['placement_group']),
                        'name':
                        worker_name,
                        'namespace':
                        'default',
                        'runtime_env':
                        runtime_env,
                        'num_cpus':
                        0.01,
                        'num_gpus':
                        0.01,
                    }

                    worker = worker_cls.options(**worker_options).remote(*args, **kwargs)
                    workers.append(worker)
            else:
                world_size = _config['ranks']
                placement_groups: List[List[Dict]] = RayHelper.resource_manager.resource(_group)
                workers = []
                for deploy_pg, index in zip(placement_groups, list(range(world_size))):
                    deploy_pg: Dict
                    cluster_name = '-'.join(local_groups)
                    worker_name = cluster_name + '-' + str(index)
                    env_vars = os.environ.copy()
                    env_vars.update({
                        'CLUSTER_NAME': cluster_name,
                        'WORKER_NAME': worker_name,
                        VISIBLE_ENV_MAPPING[_config['device'].upper()]: '',
                        'RAY_SWIFT_ARGS': get_args(),  # pass through env
                    })
                    env_vars['RAY_SWIFT_GROUP'] = ','.join(local_groups)

                    runtime_env = RuntimeEnv(env_vars=env_vars)

                    worker_options = {
                        'scheduling_strategy':
                        PlacementGroupSchedulingStrategy(placement_group=deploy_pg['placement_group']),
                        'name':
                        worker_name,
                        'namespace':
                        'default',
                        'runtime_env':
                        runtime_env,
                        'num_cpus':
                        0.01,
                    }

                    worker = worker_cls.options(**worker_options).remote(*args, **kwargs)
                    workers.append(worker)

            for g in local_groups:
                RayHelper.worker_instance[g] = workers
                ray.get(RayHelper._registry.register_workers.remote(g, workers))

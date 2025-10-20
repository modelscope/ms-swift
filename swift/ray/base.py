import functools
import os
from typing import Callable, TypeVar, List, Dict, Literal, Union, Any, Type
import ray
from ray.runtime_env import RuntimeEnv
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from swift.llm.argument.base_args.ray_args import RayArguments
from swift.ray.resource_manager import ResourceManager

T = TypeVar('T')


class RayHelper:

    resource_manager: ResourceManager = None

    worker_cls: Dict = {}

    args: RayArguments = None

    worker_instance: Dict = {}

    initialized = False

    device_groups: Dict[str, Any] = None

    @staticmethod
    def initialize(device_groups: Dict[str, Any]):
        RayHelper.device_groups = device_groups
        ray.init()
        if RayHelper.resource_manager is None:
            RayHelper.resource_manager = ResourceManager(device_groups)
        RayHelper.initialized = True

    @staticmethod
    def worker(group: Union[str, List[str]]):
        def decorator(cls):
            if not RayHelper.initialized:
                return cls
            cls.decorated = True
            groups = [group] if isinstance(group, str) else group
            _cls = ray.remote(cls)
            for g in groups:
                RayHelper.worker_cls[g] = _cls

            init_method = cls.__init__

            @functools.wraps(init_method)
            def new_init(self, *args, **kwargs):
                RayHelper._create_workers(group, *args, **kwargs)
                init_method(self, *args, **kwargs)

            cls.__init__ = new_init

            return cls

        return decorator

    @staticmethod
    def function(group: str, dispatch: Literal['slice', 'all'] = 'all', execute: Literal['first', 'all'] = 'all'):

        def decorator(func: Callable[..., T]) -> Callable[..., T]:

            @functools.wraps(func)
            def wrapper(self, *args, **kwargs) -> T:
                if not RayHelper.initialized:
                    return func(self, *args, **kwargs)
                if RayHelper.resource_manager is None:
                    if group not in self.group:
                        if func.__name__ == '__init__':
                            return None
                        else:
                            raise ValueError()
                    else:
                        return func(self, *args, **kwargs)
                else:
                    return RayHelper.execute_all_sync(group, func.__name__, *args, **kwargs)

            return wrapper

        return decorator

    @staticmethod
    def execute_all_sync(group, method_name: str, *args, **kwargs):
        return ray.get(RayHelper.execute_all_async(group, method_name, *args, **kwargs))

    @staticmethod
    def execute_all_async(group, method_name: str, *args, **kwargs):
        workers = RayHelper.worker_instance[group]
        length = len(workers)
        if all(isinstance(arg, list) for arg in args) and all(isinstance(kwarg, list) for kwarg in kwargs.values()):
            if all(len(arg) == length for arg in args) and all(len(kwarg) == length for kwarg in kwargs.values()):
                result = []
                for i in range(length):
                    sliced_args = tuple(arg[i] for arg in args)
                    sliced_kwargs = {k: v[i] for k, v in kwargs.items()}
                    remote_call = getattr(workers[i], method_name)
                    result.append(remote_call.remote(*sliced_args, **sliced_kwargs))
                return result

        return [getattr(worker, method_name).remote(*args, **kwargs) for worker in workers]

    @staticmethod
    def _create_workers(group: Union[str, List[str]], *args, **kwargs):
        nproc_per_node = int(RayHelper.device_groups['nproc_per_node'])
        ip, port = RayHelper.device_groups['master_addr']

        if isinstance(group, str):
            group = [group]

        worker_cls = RayHelper.worker_cls[group[0]]

        _config = None
        for name, config in RayHelper.device_groups.items():
            if name in RayHelper.resource_manager.possible_keys:
                continue

            if group[0] in config['workers']:
                _config = config
                break

        assert _config is not None

        world_size = len(_config['ranks']) // nproc_per_node
        placement_groups: List[List[Dict]] = RayHelper.resource_manager.resource(group[0])
        workers = []
        for rank, pgs in enumerate(placement_groups):
            deploy_pg = pgs
            worker_name = '-'.join(_config['workers']) + '-' + str(rank)
            env_vars = {
                "WORLD_SIZE": str(world_size),
                "RANK": str(rank),
                "LOCAL_RANK": str(0),
                "CLUSTER_NAME": '-'.join(_config['workers']),
                "WORKER_NAME": worker_name,
            }

            if rank != 0:
                env_vars["MASTER_ADDR"] = ip
                env_vars["MASTER_PORT"] = str(port)
            if "ROLL_LOG_DIR" in os.environ:
                env_vars["ROLL_LOG_DIR"] = os.environ["ROLL_LOG_DIR"]

            runtime_env = RuntimeEnv(env_vars=env_vars)

            worker_options = {
                "scheduling_strategy": PlacementGroupSchedulingStrategy(placement_group=deploy_pg["placement_group"]),
                "name": worker_name,
                "namespace": 'default',
                "runtime_env": runtime_env,
                "num_cpus": 0.01,
                "num_gpus": 0.01,
            }

            worker = worker_cls.options(**worker_options).remote(*args, **kwargs)
            workers.append(worker)

        for _group in group:
            RayHelper.worker_instance[_group] = workers

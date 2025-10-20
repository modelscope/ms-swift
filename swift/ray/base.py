import functools
import os
from typing import Callable, TypeVar, List, Dict, Literal, Union, Any
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
            cls.decorated = True

            if isinstance(group, str):
                group = [group]
            _cls = ray.remote(cls)
            for g in group:
                RayHelper.worker_cls[g] = _cls
            _cls.group = group
            return _cls

        return decorator

    @staticmethod
    def function(group: str, dispatch: Literal['slice', 'all'] = 'all', execute: Literal['first', 'all'] = 'all'):

        def decorator(func: Callable[..., T]) -> Callable[..., T]:

            @functools.wraps(func)
            def wrapper(self, *args, **kwargs) -> T:
                if not RayHelper.initialized:
                    return func(*args, **kwargs)
                if RayHelper.resource_manager is None:
                    if group not in self.group:
                        if func.__name__ == '__init__':
                            return None
                        else:
                            raise ValueError()
                    else:
                        return func(*args, **kwargs)
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
    def _create_workers():
        nproc_per_node = int(RayHelper.device_groups['groups']['nproc_per_node'])
        ip, port = RayHelper.device_groups['master_addr']
        for group_name, group in RayHelper.device_groups['groups'].items():
            if group_name == 'nproc_per_node':
                continue

            worker_cls = RayHelper.worker_cls[group_name]

            worker = group['worker'][0]
            world_size = len(group['ranks']) // nproc_per_node
            placement_groups: List[List[Dict]] = RayHelper.resource_manager.resource(worker)
            workers = []
            for rank, pgs in enumerate(placement_groups):
                deploy_pg = pgs[0]
                worker_name = '-'.join(group['worker'])
                env_vars = {
                    "WORLD_SIZE": str(world_size),
                    "RANK": str(rank),
                    "LOCAL_RANK": str(0),
                    "CLUSTER_NAME": worker_name,
                    "WORKER_NAME": worker_name,
                }

                if rank != 0:
                    env_vars["MASTER_ADDR"] = ip
                    env_vars["MASTER_PORT"] = port
                if "ROLL_LOG_DIR" in os.environ:
                    env_vars["ROLL_LOG_DIR"] = os.environ["ROLL_LOG_DIR"]

                runtime_env = RuntimeEnv(env_vars=env_vars)

                worker_options = {
                    "scheduling_strategy": PlacementGroupSchedulingStrategy(placement_group=deploy_pg["placement_group"]),
                    "name": worker_name,
                    "namespace": '',
                    "runtime_env": runtime_env,
                    "num_cpus": 0.01,
                    "num_gpus": 0.01,
                }

                worker = worker_cls.options(**worker_options).remote(args=worker_cls.args)
                workers.append(worker)
            RayHelper.worker_instance[group_name] = workers

import functools
import os
from typing import Callable, TypeVar, List, Dict, Literal, Union, Any, Type
import ray
import inspect
from ray.runtime_env import RuntimeEnv
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from swift.llm.argument.base_args.ray_args import RayArguments
from swift.ray.resource_manager import ResourceManager
from swift.utils import find_free_port
from swift.utils.utils import find_node_ip

T = TypeVar('T')


def is_called_from_init():
    stack = inspect.stack()
    for frame_info in stack[1:]:
        if frame_info.function == '__init__':
            return True
    return False


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

        is_worker = ray.is_initialized() and ray._private.worker.global_worker.mode == ray._private.worker.WORKER_MODE

        def decorator(cls):
            if is_worker:
                return cls
            cls.decorated = True
            groups = [group] if isinstance(group, str) else group
            _cls = ray.remote(cls)
            for g in groups:
                RayHelper.worker_cls[g] = _cls

            init_method = cls.__init__

            @functools.wraps(init_method)
            def new_init(self, *args, **kwargs):
                if not is_worker:
                    RayHelper._create_workers(group, *args, **kwargs)
                init_method(self, *args, **kwargs)

            cls.__init__ = new_init

            return cls

        return decorator
    
    @staticmethod
    def collect_func(method: Literal['none', 'flatten']):
        if method == 'none':
            return lambda x: x
        elif method == 'flatten':
            return lambda x: [item for sublist in x for item in sublist]

    @staticmethod
    def function(group: str, dispatch: Literal['slice', 'all'] = 'all', execute: Literal['first', 'all'] = 'all', collect: Literal['none', 'flatten'] = 'none'):

        def decorator(func: Callable[..., T]) -> Callable[..., T]:

            @functools.wraps(func)
            def wrapper(self, *args, **kwargs) -> T:
                is_worker = ray.is_initialized() and ray._private.worker.global_worker.mode == ray._private.worker.WORKER_MODE
                if is_worker:
                    if not hasattr(self, 'group'):
                        self.group = os.environ['RAY_SWIFT_GROUP'].split(',')
                    if group not in self.group:
                        if is_called_from_init():
                            return None
                        else:
                            raise ValueError()
                    else:
                        return func(self, *args, **kwargs)
                else:
                    if is_called_from_init():
                        return None
                    result = RayHelper.execute_all_sync(group, dispatch, execute, func.__name__, *args, **kwargs)
                    return RayHelper.collect_func(collect)(result)
            return wrapper

        return decorator

    @staticmethod
    def execute_all_sync(group, dispatch, execute, method_name: str, *args, **kwargs):
        return ray.get(RayHelper.execute_all_async(group, dispatch, execute, method_name, *args, **kwargs))

    @staticmethod
    def execute_all_async(group, dispatch, execute, method_name: str, *args, **kwargs):
        workers = RayHelper.worker_instance[group]
        length = len(workers)
        if execute == 'first':
            return getattr(workers[0], method_name).remote(*args, **kwargs)
        elif dispatch == 'all':
            return [getattr(worker, method_name).remote(*args, **kwargs) for worker in workers]
        elif dispatch == 'slice':
            result = []
            for i in range(length):
                sliced_args = tuple(arg[i] for arg in args)
                sliced_kwargs = {k: v[i] for k, v in kwargs.items()}
                remote_call = getattr(workers[i], method_name)
                result.append(remote_call.remote(*sliced_args, **sliced_kwargs))
        else:
            result = []
            for i in range(length):
                sliced_args, sliced_kwargs = dispatch(length, i, *args, **kwargs)
                remote_call = getattr(workers[i], method_name)
                result.append(remote_call.remote(*sliced_args, **sliced_kwargs))
            return result

    @staticmethod
    def _create_workers(group: Union[str, List[str]], *args, **kwargs):
        nproc_per_node = int(RayHelper.device_groups['nproc_per_node'])

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

            assert _config is not None
            local_groups = _config['workers']
            world_size = len(_config['ranks']) // nproc_per_node
            placement_groups: List[List[Dict]] = RayHelper.resource_manager.resource(_group)
            workers = []
            ip, port = None, None
            for rank, (pgs, gpu) in enumerate(zip(placement_groups, _config['ranks'])):
                deploy_pg = pgs
                node_idx = gpu // nproc_per_node
                cluster_name = '-'.join(local_groups)
                worker_name = cluster_name + '-' + str(rank)
                env_vars = os.environ.copy()
                env_vars.update({
                    "WORLD_SIZE": str(world_size),
                    "RANK": str(rank),
                    "LOCAL_RANK": str(0),
                    "CLUSTER_NAME": cluster_name,
                    "WORKER_NAME": worker_name,
                    "CUDA_VISIBLE_DEVICES": str(deploy_pg["gpu_rank"]),
                })

                node_id = RayHelper.resource_manager.nodes[node_idx]['NodeID']

                @ray.remote
                def get_node_address():
                    return find_node_ip(), find_free_port()
                
                if rank == 0:
                    ip, port = ray.get(get_node_address.remote())

                env_vars["MASTER_ADDR"] = ip
                env_vars["MASTER_PORT"] = str(port)
                env_vars["RAY_SWIFT_GROUP"] = ','.join(local_groups)
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

            for g in local_groups:
                RayHelper.worker_instance[g] = workers

import functools
import os
from typing import Callable, TypeVar, List, Dict, Literal
import ray
from ray.runtime_env import RuntimeEnv
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from swift.llm.argument.base_args.ray_args import RayArguments
from swift.ray.resource_manager import ResourceManager

T = TypeVar('T')


class RayUtil:

    resource_manager: ResourceManager = None

    def __init__(self, args: RayArguments):
        self.role = ''
        self.group = ''
        self.workers = []

    def initialize(self, args: RayArguments):
        if RayUtil.resource_manager is None:
            RayUtil.resource_manager = ResourceManager()

    @classmethod
    def worker(cls, group: str, dispatch: Literal['slice', 'all'], execute: Literal['first', 'all']):

        def decorator(func: Callable[..., T]) -> Callable[..., T]:

            @functools.wraps(func)
            def wrapper(self, *args, **kwargs) -> T:
                if ray:

                if self.group == 'worker':
                    if group != self.group:
                        return None
                    else:
                        return func(*args, **kwargs)
                else:
                    return self.execute_all_sync(func.__name__, *args, **kwargs)

            return wrapper

        return decorator

    def execute_all_sync(self, method_name: str, *args, **kwargs):
        return ray.get(self.execute_all_async(method_name, *args, **kwargs))

    def execute_all_async(self, method_name: str, *args, **kwargs):
        length = len(self.workers)
        if all(isinstance(arg, list) for arg in args) and all(isinstance(kwarg, list) for kwarg in kwargs.values()):
            if all(len(arg) == length for arg in args) and all(len(kwarg) == length for kwarg in kwargs.values()):
                result = []
                for i in range(length):
                    sliced_args = tuple(arg[i] for arg in args)
                    sliced_kwargs = {k: v[i] for k, v in kwargs.items()}
                    remote_call = getattr(self.workers[i], method_name)
                    result.append(remote_call.remote(*sliced_args, **sliced_kwargs))
                return result

        return [getattr(worker, method_name).remote(*args, **kwargs) for worker in self.workers]

    def _create_workers(self):
        placement_groups: List[List[Dict]] = self.resource_manager.allocate_placement_group(
            device_mapping=self.worker_config.device_mapping, world_size=self.worker_config.world_size
        )

        for rank, pgs in enumerate(placement_groups):
            deploy_pg = pgs[0]
            pg_zero_gpu_ranks = sorted([pg["gpu_rank"] for pg in pgs if pg["node_rank"] == deploy_pg["node_rank"]])
            worker_name = f"{self.group_name}-{rank}"
            env_vars = {
                "WORLD_SIZE": str(self.world_size),
                "RANK": str(rank),
                "LOCAL_RANK": str(0),
                "CLUSTER_NAME": self.cluster_name,
                "WORKER_NAME": worker_name,
            }

            if rank != 0:
                env_vars["MASTER_ADDR"] = self.master_addr
                env_vars["MASTER_PORT"] = str(self.master_port)
            if deploy_pg["gpu_rank"] is not None:
                current_platform.update_env_vars_for_visible_devices(env_vars=env_vars, gpu_ranks=pg_zero_gpu_ranks)
            if "ROLL_LOG_DIR" in os.environ:
                env_vars["ROLL_LOG_DIR"] = os.environ["ROLL_LOG_DIR"]
            env_vars.update(self.worker_config.system_envs)

            runtime_env = RuntimeEnv(env_vars=env_vars)
            self.worker_config.resource_placement_groups = pgs

            worker_options = {
                "scheduling_strategy": PlacementGroupSchedulingStrategy(placement_group=deploy_pg["placement_group"]),
                "name": worker_name,
                "namespace": '',
                "runtime_env": runtime_env,
                "num_cpus": 0.01,
            }

            if current_platform.ray_device_key == "GPU":
                worker_options.update({"num_gpus": 0.01 if self.worker_config.device_mapping else 0})
            elif current_platform.ray_device_key == "NPU":
                worker_options.update(
                    {
                        "num_gpus": 0,
                        "resources": {
                            current_platform.ray_device_key: 0.01 if self.worker_config.device_mapping else 0
                        },
                    }
                )

            worker = self.__class__.options(**worker_options).remote(worker_config=self.worker_config)
            self.workers.append(worker)
            if rank == 0:
                self.master_addr, self.master_port = ray.get(worker.get_master_addr_and_port.remote())

    @classmethod
    def func_generator(cls, method_name, dispatch_fn, collect_fn, execute_fn):
        def func(*args, blocking=True, **kwargs):

            args, kwargs = dispatch_fn(cls, *args, **kwargs)
            output = execute_fn(method_name, *args, **kwargs)
            if blocking:
                timeout = None
                output = ray.get(output, timeout=timeout)
            output = collect_fn(cls, output)
            return output

        return func

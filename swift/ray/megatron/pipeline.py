# Copyright (c) ModelScope Contributors. All rights reserved.
import importlib
import os
import ray
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from swift.utils import get_logger
from .driver_utils import (build_dataset_from_dict, compute_iter_params, estimate_dp_size, merge_group_dict,
                           parse_ray_yaml)

logger = get_logger()

_TRAINER_REGISTRY: Dict[str, Dict[str, Any]] = {
    'grpo': {
        'trainer': 'swift.ray.megatron.grpo_trainer.GRPOTrainer',
        'loss': 'swift.ray.megatron.loss.grpo.GRPOLoss',
    },
}

_KNOWN_GROUPS = frozenset(('train', 'rollout'))


def register_ray_trainer(
    rlhf_type: str,
    trainer: str,
    loss: Optional[str] = None,
):
    """Register a custom algorithm for the Ray pipeline.

    Args:
        rlhf_type: Algorithm identifier (e.g. ``'grpo'``).
        trainer: Dotted path to the driver-side trainer class.
            The class must accept ``(worker_groups, rollout_replicas)``
            and expose ``set_data_info()`` / ``train()`` methods.
            Example: ``'swift.ray.megatron.grpo_trainer.GRPOTrainer'``
        loss: Dotted path to a ``Loss`` subclass that defines
            ``forward_step`` + ``loss_func``.
            Pass ``None`` to use the internal trainer's forward_step.
    """
    _TRAINER_REGISTRY[rlhf_type] = {'trainer': trainer, 'loss': loss}


class MegatronRayPipeline:

    def __init__(self, config_path: str):
        self.ray_config, group_configs, shared_config = parse_ray_yaml(config_path)
        shared_config['use_ray'] = True

        self.rlhf_type = self.ray_config.rlhf_type
        if self.rlhf_type not in _TRAINER_REGISTRY:
            raise ValueError('Unknown rlhf_type %r. Available: %s' % (self.rlhf_type, list(_TRAINER_REGISTRY)))

        self.group_cfgs: Dict[str,
                              Dict[str,
                                   Any]] = {g: merge_group_dict(shared_config, gd)
                                            for g, gd in group_configs.items()}
        self.shared_cfg = {k: v for k, v in shared_config.items() if v is not None}

        self._entry = _TRAINER_REGISTRY[self.rlhf_type]
        self.resource_pool_manager = None
        self.worker_groups: Dict[str, Any] = {}
        self.rollout_replicas: List[Any] = []

    def init(self) -> None:
        # Initialize Ray, create resource pools, spawn workers and replicas.
        self._data_info = self._build_dataset()
        self._compute_train_iters()
        ray.init(ignore_reinit_error=True)
        self._create_pools()
        self._init_worker_groups()
        with self._colocate_offload_ctx():
            self._init_rollout_replicas()
        self._driver_trainer = self._create_trainer()
        self._driver_trainer.set_data_info(self._data_info)

    def train(self) -> Any:
        """Run the training loop.  Requires ``init()`` to have been called."""
        if not hasattr(self, '_driver_trainer'):
            raise RuntimeError('MegatronRayPipeline.train(): call init() first')
        return self._driver_trainer.train()

    def run(self) -> Any:
        """Convenience: ``init()`` + ``train()`` + ``shutdown()``."""
        self.init()
        try:
            return self.train()
        finally:
            self._shutdown()

    def _build_dataset(self) -> Dict[str, Any]:
        cfg = dict(self.shared_cfg)
        return build_dataset_from_dict(cfg)

    def _compute_train_iters(self):
        train_cfg = self.group_cfgs.get('train')
        gpus = self.group_gpus.get('train', 0)
        assert train_cfg is not None and gpus > 0
        dp_size = estimate_dp_size(train_cfg, gpus)
        iter_params = compute_iter_params(self._data_info, dp_size)
        train_iters = iter_params.get('train_iters')
        assert train_iters is not None and train_iters > 0

        self.group_cfgs['train']['train_iters'] = train_iters

    def _create_pools(self):
        from .resource_pool import ResourcePool, ResourcePoolManager

        colocated_sets = {frozenset(g) for g in self.colocate_groups}
        pool_mapping: Dict[str, ResourcePool] = {}
        assigned: set = set()

        for colocated in colocated_sets:
            gpus = self.group_gpus.get(next(iter(colocated)), 0)
            if gpus <= 0:
                continue
            pon = self.ray_config.gpus_as_process_on_nodes(gpus)
            shared = ResourcePool(pon, max_colocate_count=len(colocated))
            for g in colocated:
                pool_mapping[g] = shared
                assigned.add(g)

        for name, gpus in self.group_gpus.items():
            if name in assigned or gpus <= 0:
                continue
            pon = self.ray_config.gpus_as_process_on_nodes(gpus)
            pool_mapping[name] = ResourcePool(pon)

        self.resource_pool_manager = ResourcePoolManager(pool_mapping)
        self.resource_pool_manager.create_all()

    def _is_rollout_hybrid(self) -> bool:
        """True if rollout shares its pool with train (HYBRID mode)."""
        return any('rollout' in cg and 'train' in cg for cg in self.colocate_groups)

    def _init_worker_groups(self):
        self._validate_grpo_train_batch_params()

        self._spawn_train_group('train')

        train_wg = self.worker_groups['train']
        padding_vals = train_wg.broadcast('get_padding_to')
        self._data_info['_padding_to'] = next((v for v in padding_vals if v is not None), None)
        logger.debug('padding_to from worker: %s', self._data_info.get('_padding_to'))

    # TODO: reuse
    def _validate_grpo_train_batch_params(self) -> None:
        if self.rlhf_type != 'grpo':
            return
        train_cfg = self.group_cfgs.get('train')
        train_gpus = self.group_gpus.get('train', 0)
        if not train_cfg or train_gpus <= 0:
            return

        cfg = dict(train_cfg)
        global_batch_size = int(cfg.get('global_batch_size', 0) or 0)
        micro_batch_size = int(cfg.get('micro_batch_size', 1) or 1)
        num_generations = int(cfg.get('num_generations', 8) or 8)
        generation_batch_size = cfg.get('generation_batch_size')
        steps_per_generation = cfg.get('steps_per_generation')

        if global_batch_size <= 0:
            return
        if generation_batch_size is not None and steps_per_generation is not None:
            raise ValueError("'generation_batch_size' and 'steps_per_generation' cannot be both configured")

        if generation_batch_size is None:
            if steps_per_generation is None:
                generation_batch_size = global_batch_size
            else:
                generation_batch_size = global_batch_size * int(steps_per_generation)
        else:
            generation_batch_size = int(generation_batch_size)
            if generation_batch_size % global_batch_size != 0:
                raise ValueError(f'generation_batch_size ({generation_batch_size}) must be divisible by '
                                 f'global_batch_size ({global_batch_size})')

        dp_size = estimate_dp_size(cfg, train_gpus)
        num_rollout_prompt = generation_batch_size // num_generations
        if num_rollout_prompt % dp_size != 0:
            raise ValueError(f'Invalid GRPO batch config: '
                             f'(generation_batch_size={generation_batch_size} // num_generations={num_generations}) '
                             f'= {num_rollout_prompt} must be divisible by dp_size={dp_size}.')

        per_device_num_rollout_prompt = num_rollout_prompt // dp_size
        if per_device_num_rollout_prompt < 1:
            raise ValueError(f'Invalid GRPO batch config: per_device_num_rollout_prompt='
                             f'((generation_batch_size={generation_batch_size} // num_generations={num_generations}) '
                             f'// dp_size={dp_size}) = {per_device_num_rollout_prompt} < 1. '
                             f'Increase generation_batch_size/steps_per_generation, or reduce num_generations.')
        if per_device_num_rollout_prompt % micro_batch_size != 0:
            raise ValueError(
                f'Invalid GRPO batch config: per_device_num_rollout_prompt='
                f'{per_device_num_rollout_prompt} must be divisible by micro_batch_size={micro_batch_size}.')

    def _spawn_train_group(self, role: str) -> None:
        from .megatron_worker import MegatronWorker
        from .worker_group import WorkerGroup

        pool = self.resource_pool_manager.get_pool(role)
        cfg = dict(self.group_cfgs.get(role, {}))
        cfg.setdefault('rlhf_type', self.ray_config.rlhf_type)
        worker_cls = ray.remote(num_gpus=1.0)(MegatronWorker)
        wg = WorkerGroup.from_pool(role, pool, worker_cls=worker_cls)

        loss_cls = self._entry.get('loss')
        rollout_config = self._build_rollout_config_for_workers()
        wg.broadcast('init_model', cfg, loss_cls_path=loss_cls, rollout_config=rollout_config)
        wg.build_dispatch_info(worker_cls=MegatronWorker)

        self.worker_groups[role] = wg
        logger.info('MegatronWorker group [%s] on %d GPUs', role, pool.world_size)

    @contextmanager
    def _colocate_offload_ctx(self):
        """Offload colocated train workers while vLLM replicas initialize.

        In colocate mode the training model must free GPU memory so that
        vLLM can allocate KV cache.  After initialization the model is
        reloaded.  In separate mode this is a no-op.
        """
        need = self._is_rollout_hybrid() and bool(self.shared_cfg.get('offload_model', True))
        colocated_wgs = [
            wg for role, wg in self.worker_groups.items() if need and any(role in g for g in self.colocate_groups)
        ]
        for wg in colocated_wgs:
            wg.broadcast('offload_to_cpu')
        try:
            yield
        finally:
            for wg in colocated_wgs:
                wg.broadcast('reload_to_gpu')

    def _init_rollout_replicas(self) -> None:
        rollout_gpus = self.group_gpus.get('rollout', 0)
        if rollout_gpus <= 0:
            self.rollout_replicas = []
            return

        from .rollout.replica import RolloutReplica

        rollout_cfg = self.group_cfgs.get('rollout', {})
        is_hybrid = self._is_rollout_hybrid()
        pool = self.resource_pool_manager.get_pool('train' if is_hybrid else 'rollout')

        self.rollout_replicas = RolloutReplica.create_replicas(
            rollout_cfg=rollout_cfg,
            rollout_gpus=rollout_gpus,
            pool=pool,
            is_hybrid=is_hybrid,
            sleep_level=self.ray_config.sleep_level,
        )

    def _build_rollout_config_for_workers(self) -> Optional[Dict[str, Any]]:
        """Build rollout config dict for MegatronWorker._init_rollout_adapter.

        Returns None if no rollout GPUs are configured.
        """
        rollout_gpus = self.group_gpus.get('rollout', 0)
        if rollout_gpus <= 0:
            return None
        rollout_cfg = self.group_cfgs.get('rollout', {})
        bucket_mb = int(os.environ.get('SWIFT_RAY_WEIGHT_BUCKET_MB', '2048'))
        return {
            'rollout_tp_size': rollout_cfg.get('vllm_tensor_parallel_size', 1),
            'rollout_dp_size': rollout_cfg.get('vllm_data_parallel_size', 1),
            'bucket_size_mb': bucket_mb,
        }

    def _create_trainer(self):
        cls_path = self._entry['trainer']
        mod_path, cls_name = cls_path.rsplit('.', 1)
        mod = importlib.import_module(mod_path)
        trainer_cls = getattr(mod, cls_name)
        weight_sync_mode = self._get_weight_sync_mode()
        sleep_level = self._resolve_sleep_level()
        return trainer_cls(
            self.worker_groups, self.rollout_replicas, weight_sync_mode=weight_sync_mode, sleep_level=sleep_level)

    def _resolve_sleep_level(self) -> int:
        """Determine sleep_level based on deployment mode.

        - colocate: honor user config (1 or 2), default 1.
        - separate: always 0 (vLLM stays resident), warn if user set otherwise.
        """
        user_level = self.ray_config.sleep_level
        if self._is_rollout_hybrid():
            return user_level
        if user_level != 0:
            logger.warning('sleep_level=%d ignored in separate mode (vLLM stays resident). '
                           'Overriding to 0.', user_level)
        return 0

    def _get_weight_sync_mode(self) -> str:
        """Determine weight sync mode based on deployment topology.

        - colocate: naive (IPC/ZMQ) — vLLM shares the node with trainer
        - separate: nccl (broadcast) — vLLM on dedicated GPUs
        """
        if self._is_rollout_hybrid():
            return 'naive'
        return 'nccl'

    @property
    def group_gpus(self) -> Dict[str, int]:
        return self.ray_config.group_gpus

    @property
    def colocate_groups(self) -> List[List[str]]:
        return self.ray_config.colocate_groups

    def _shutdown(self):
        """Best-effort teardown — each step swallows exceptions so a
        failure in one stage does not skip the remaining cleanup."""
        for replica in self.rollout_replicas:
            try:
                replica.shutdown()
            except Exception as e:  # noqa: BLE001
                logger.warning('RolloutReplica shutdown failed: %s', e)
        self.rollout_replicas = []

        seen: set = set()
        for wg in self.worker_groups.values():
            if id(wg) not in seen:
                seen.add(id(wg))
                try:
                    wg.shutdown()
                except Exception as e:  # noqa: BLE001
                    logger.warning('WorkerGroup shutdown failed: %s', e)
        self.worker_groups.clear()

        if self.resource_pool_manager is not None:
            try:
                self.resource_pool_manager.destroy_all()
            except Exception as e:  # noqa: BLE001
                logger.warning('destroy_all placement groups failed: %s', e)
        ray.shutdown()


def main():
    import sys
    argv = sys.argv[1:]
    config_path = None
    for i, arg in enumerate(argv):
        if arg == '--config' and i + 1 < len(argv):
            config_path = argv[i + 1]
            break
    if config_path is None:
        raise ValueError('Usage: python -m swift.ray.megatron.pipeline --config <yaml>')

    return MegatronRayPipeline(config_path).run()


if __name__ == '__main__':
    main()

# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import ray
import yaml
from typing import Any, Dict, List, Optional

from swift.utils import get_logger
from .ray_trainer import RayTrainer

logger = get_logger()

# ======================================================================
# Trainer registry
# ======================================================================

_TRAINER_REGISTRY: Dict[str, Dict[str, Any]] = {
    'dpo': {
        'trainer': 'swift.ray.megatron.ray_trainer.DPORayTrainer',
        'groups': {
            'train': {
                'trainable': True
            },
            'ref': {
                'trainable': False
            },
        },
    },
    'grpo': {
        'trainer': 'swift.ray.megatron.ray_trainer.GRPORayTrainer',
        'groups': {
            'train': {
                'trainable': True
            },
            'rollout': {
                'trainable': False,
                'worker_cls': 'swift.ray.megatron.vllm_worker.VllmWorker'
            },
        },
    },
}


def register_ray_trainer(rlhf_type: str, trainer_cls_path: str, groups: Dict[str, bool]):
    _TRAINER_REGISTRY[rlhf_type] = {
        'trainer': trainer_cls_path,
        'groups': groups,
    }


def _build_group_argv(shared: Dict[str, Any], group: Dict[str, Any]) -> List[str]:
    """Merge shared + group-specific config into a CLI argv list."""
    merged = {**shared, **group}
    skip = {'gpus', 'colocate_groups', 'rlhf_type'}
    argv: List[str] = []
    for key, val in merged.items():
        if key in skip or val is None:
            continue
        argv.append('--%s' % key)
        if isinstance(val, bool):
            argv.append('true' if val else 'false')
        elif isinstance(val, (list, tuple)):
            argv.extend(str(v) for v in val)
        elif isinstance(val, dict):
            argv.append(json.dumps(val))
        else:
            argv.append(str(val))
    return argv


def parse_ray_config(config_path: str) -> Dict[str, Any]:
    """Parse a Ray YAML config file.

    Returns a dict with ``rlhf_type``, ``group_argv``, ``group_gpus``,
    ``colocate_groups``.
    """
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    rlhf_type = raw.pop('rlhf_type', 'dpo')
    entry = _TRAINER_REGISTRY.get(rlhf_type)
    if entry is None:
        raise ValueError('Unknown rlhf_type %r. Available: %s' % (rlhf_type, list(_TRAINER_REGISTRY)))

    colocate_groups = raw.pop('colocate_groups', [])
    groups_def = entry['groups']
    group_names = list(groups_def.keys())

    group_sections: Dict[str, dict] = {}
    for g in group_names:
        group_sections[g] = raw.pop(g, {})
    shared = dict(raw)

    group_argv: Dict[str, List[str]] = {}
    group_gpus: Dict[str, int] = {}
    for g, cfg in group_sections.items():
        group_gpus[g] = cfg.pop('gpus', 0)
        group_argv[g] = _build_group_argv(shared, cfg)

    shared_argv = _build_group_argv({}, shared)

    return {
        'rlhf_type': rlhf_type,
        'group_argv': group_argv,
        'group_gpus': group_gpus,
        'colocate_groups': colocate_groups,
        'shared_argv': shared_argv,
    }


_GROUP_TO_ROLE = {
    'train': 'model',
    'rollout': 'rollout',
    'ref': 'ref',
    'teacher': 'teacher',
}


def _compute_hybrid_role(group_names: List[str]) -> str:
    """Build the HybridWorker role string from group names.

    E.g. ['train', 'rollout'] → 'model_rollout'
    """
    roles = []
    for g in sorted(group_names, key=lambda x: list(_GROUP_TO_ROLE.keys()).index(x) if x in _GROUP_TO_ROLE else 99):
        role = _GROUP_TO_ROLE.get(g)
        if role:
            roles.append(role)
    return '_'.join(roles)


# ======================================================================
# Pipeline
# ======================================================================


class MegatronRayPipeline:
    """Algorithm-agnostic Ray Megatron training pipeline.

    Supports two deployment modes:
      - **Co-located (hybrid)**: Multiple roles share the same GPU via
        HybridWorker. Configured via ``colocate_groups``.
      - **Separated**: Each role runs as an independent worker
        (MegatronWorker or VllmWorker) on dedicated GPUs.
    """

    def __init__(self, config_path: str):
        parsed = parse_ray_config(config_path)
        self.rlhf_type = parsed['rlhf_type']
        self.group_argv = parsed['group_argv']
        self.group_gpus = parsed['group_gpus']
        self.colocate_groups = parsed['colocate_groups']

        self.shared_argv = parsed['shared_argv']
        self._entry = _TRAINER_REGISTRY[self.rlhf_type]
        self.resource_pool_manager = None
        self.worker_groups: Dict[str, Any] = {}

    def run(self) -> Any:
        """Full lifecycle: build dataset → ray init → train → shutdown."""
        self._data_info = self._build_dataset()
        self._inject_train_iters()
        ray.init(ignore_reinit_error=True)
        try:
            self._create_pools()
            self._init_worker_groups()
            ray_trainer = self._create_trainer()
            ray_trainer.set_data_info(self._data_info)
            return ray_trainer.fit()
        finally:
            self._shutdown()

    def _inject_train_iters(self):
        """Pre-compute train_iters on driver and inject into train group argv.

        This ensures workers build their lr schedulers with the correct total
        steps instead of the placeholder value ``1``.
        """
        from .ray_trainer import compute_iter_params

        train_argv = self.group_argv.get('train')
        if train_argv is None:
            return

        dp_size = self._estimate_dp_size('train')
        if dp_size <= 0:
            return

        iter_params = compute_iter_params(self._data_info, dp_size)
        train_iters = iter_params.get('train_iters')
        if train_iters is None or train_iters <= 0:
            return

        self._set_argv_value(train_argv, 'train_iters', str(train_iters))
        logger.info('Injected --train_iters=%d into train argv (dp_size=%d)', train_iters, dp_size)

    def _estimate_dp_size(self, group_name: str) -> int:
        """Estimate DP size from GPU count and parallel config in argv."""
        gpus = self.group_gpus.get(group_name, 0)
        if gpus <= 0:
            return 0
        argv = self.group_argv.get(group_name, [])
        tp = 1
        pp = 1
        cp = 1
        for i, arg in enumerate(argv):
            if arg == '--tensor_model_parallel_size' and i + 1 < len(argv):
                tp = int(argv[i + 1])
            elif arg == '--pipeline_model_parallel_size' and i + 1 < len(argv):
                pp = int(argv[i + 1])
            elif arg == '--context_parallel_size' and i + 1 < len(argv):
                cp = int(argv[i + 1])
        model_parallel = tp * pp * cp
        return max(gpus // model_parallel, 1)

    @staticmethod
    def _set_argv_value(argv: List[str], key: str, value: str):
        """Set --key value in argv list, replacing if exists."""
        flag = f'--{key}'
        for i, arg in enumerate(argv):
            if arg == flag and i + 1 < len(argv):
                argv[i + 1] = value
                return
        argv.extend([flag, value])

    # ------------------------------------------------------------------
    # Infrastructure
    # ------------------------------------------------------------------

    def _build_dataset(self) -> Dict[str, Any]:
        from functools import partial

        from .ray_trainer import build_dataset_from_argv

        data_info = build_dataset_from_argv(self.shared_argv, rlhf_type=self.rlhf_type)

        padding_to = self._compute_max_padding_to()
        if padding_to is not None:
            collator_fn = data_info['_collator_fn']
            data_info['data_collator'] = partial(collator_fn, padding_to=padding_to)

        return data_info

    def _compute_max_padding_to(self) -> Optional[int]:
        max_val = None
        for _name, argv in self.group_argv.items():
            tp = 1
            sp = False
            cp = 1
            for i, arg in enumerate(argv):
                if arg == '--tensor_model_parallel_size' and i + 1 < len(argv):
                    tp = int(argv[i + 1])
                elif arg == '--sequence_parallel' and i + 1 < len(argv):
                    sp = argv[i + 1].lower() == 'true'
                elif arg == '--context_parallel_size' and i + 1 < len(argv):
                    cp = int(argv[i + 1])
            val = None
            if tp > 1 and sp:
                val = tp
            if cp > 1:
                val = (val or 1) * cp
            if val is not None:
                max_val = max(max_val or 1, val)
        return max_val

    def _create_pools(self):
        from .resource_pool import ResourcePool, ResourcePoolManager

        colocated_sets = {frozenset(g) for g in self.colocate_groups}
        pool_mapping: Dict[str, ResourcePool] = {}
        assigned: set = set()

        for colocated in colocated_sets:
            max_gpus = max(self.group_gpus.get(g, 0) for g in colocated)
            if max_gpus <= 0:
                continue
            shared = ResourcePool([max_gpus], max_colocate_count=1)
            for g in colocated:
                pool_mapping[g] = shared
                assigned.add(g)

        for name, gpus in self.group_gpus.items():
            if name in assigned or gpus <= 0:
                continue
            pool_mapping[name] = ResourcePool([gpus])

        self.resource_pool_manager = ResourcePoolManager(pool_mapping)
        self.resource_pool_manager.create_all()

    def _get_extra_train_kwargs(self, group_names) -> dict:
        """Compute extra init kwargs for training workers."""
        rlhf_type = self.rlhf_type
        has_ref = 'ref' in group_names
        if rlhf_type == 'dpo' and has_ref:
            colocate_set = self._get_colocate_set('ref')
            is_ref_colocated = colocate_set is not None and 'train' in colocate_set
            if not is_ref_colocated:
                return {'trainer_cls_path': 'swift.ray.megatron.ray_megatron_trainer.RayMegatronDPOTrainer'}
        return {}

    def _get_colocate_set(self, group_name: str) -> Optional[frozenset]:
        """Return the colocate set containing group_name, or None."""
        for cg in self.colocate_groups:
            if group_name in cg:
                return frozenset(cg)
        return None

    def _init_worker_groups(self):
        """Initialize worker groups for all roles.

        Co-located groups use HybridWorker (single process, multiple roles).
        Separated groups use MegatronWorker or VllmWorker (independent processes).
        """
        from .worker_group import WorkerGroup

        group_defs = self._entry['groups']
        group_names = set(g for g, gpus in self.group_gpus.items() if gpus > 0)
        extra_train_kwargs = self._get_extra_train_kwargs(group_names)

        colocated_done: set = set()

        for group_name in self.group_argv:
            gpus = self.group_gpus.get(group_name, 0)
            if gpus <= 0 or group_name in colocated_done:
                continue

            colocate_set = self._get_colocate_set(group_name)

            if colocate_set is not None:
                active_members = [g for g in colocate_set if g in group_names]
                if len(active_members) > 1:
                    self._init_colocated_group(active_members, group_defs, extra_train_kwargs)
                    colocated_done.update(active_members)
                    continue

            self._init_separated_group(group_name, group_defs, extra_train_kwargs)

    def _init_colocated_group(
        self,
        members: List[str],
        group_defs: Dict[str, Any],
        extra_train_kwargs: Dict,
    ):
        """Create a single HybridWorker group for co-located roles."""
        from .hybrid_worker import HybridWorker
        from .worker_group import WorkerGroup

        primary = members[0]
        pool = self.resource_pool_manager.get_pool(primary)
        argv = self.group_argv[primary]
        role = _compute_hybrid_role(members)

        worker_cls = ray.remote(num_gpus=1.0)(HybridWorker)
        wg = WorkerGroup.from_pool(primary, pool, worker_cls=worker_cls)

        init_kwargs = dict(role=role)
        init_kwargs.update(extra_train_kwargs)
        wg.broadcast('init_model', argv, **init_kwargs)
        wg.build_dispatch_info(worker_cls=HybridWorker)

        for g in members:
            self.worker_groups[g] = wg

        logger.info('Co-located group [%s] as HybridWorker(role=%s) on %d GPUs', ', '.join(members), role,
                    pool.world_size)

    def _init_separated_group(
        self,
        group_name: str,
        group_defs: Dict[str, Any],
        extra_train_kwargs: Dict,
    ):
        """Create an independent worker group for a separated role."""
        from .worker_group import WorkerGroup

        pool = self.resource_pool_manager.get_pool(group_name)
        argv = self.group_argv[group_name]
        gdef = group_defs.get(group_name, {})
        if isinstance(gdef, bool):
            gdef = {'trainable': gdef}
        trainable = gdef.get('trainable', False)
        custom_worker_cls = gdef.get('worker_cls')

        worker_cls_obj = None
        build_cls = None

        if custom_worker_cls:
            import importlib
            mod_path, cls_name = custom_worker_cls.rsplit('.', 1)
            mod = importlib.import_module(mod_path)
            raw_cls = getattr(mod, cls_name)
            worker_cls_obj = ray.remote(num_gpus=1.0)(raw_cls)
            build_cls = raw_cls

        wg = WorkerGroup.from_pool(group_name, pool, worker_cls=worker_cls_obj)

        init_kwargs = dict(trainable=trainable)
        if trainable:
            init_kwargs.update(extra_train_kwargs)
        if custom_worker_cls and 'vllm' in custom_worker_cls.lower():
            init_kwargs.pop('trainable', None)
            init_kwargs.pop('trainer_cls_path', None)
        wg.broadcast('init_model', argv, **init_kwargs)

        if build_cls is None:
            from .megatron_worker import MegatronWorker
            build_cls = MegatronWorker
        wg.build_dispatch_info(worker_cls=build_cls)

        self.worker_groups[group_name] = wg

        logger.info('Separated group [%s] (%s) on %d GPUs', group_name, custom_worker_cls or 'MegatronWorker',
                    pool.world_size)

    def _create_trainer(self):
        cls_path = self._entry['trainer']
        mod_path, cls_name = cls_path.rsplit('.', 1)
        import importlib
        mod = importlib.import_module(mod_path)
        trainer_cls = getattr(mod, cls_name)
        return trainer_cls(self.worker_groups)

    def _shutdown(self):
        if self.resource_pool_manager is not None:
            self.resource_pool_manager.destroy_all()
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

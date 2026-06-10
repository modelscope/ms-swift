"""Driver-side helpers shared by the Ray Megatron pipeline.

This module owns the "plain Python" helpers that do not belong to any
particular trainer class:

* YAML → dict config parsing and merging
* structured config parsing from YAML group dicts
* driver-side dataset building (via dict, no argv round-trip)
* train/eval iteration bookkeeping
* extracting the canonical iteration from worker results
"""
import json
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from swift.utils.logger import get_logger

logger = get_logger()

_RAY_ONLY_KEYS = frozenset({'gpus', 'colocate_groups', 'nnodes'})

_PARALLEL_DEFAULTS: Dict[str, Any] = {
    'tensor_model_parallel_size': 1,
    'pipeline_model_parallel_size': 1,
    'context_parallel_size': 1,
}


def parse_args_from_dict(class_type, cfg: Dict[str, Any]):
    """Construct a dataclass from a config dict via HfArgumentParser."""
    from swift.utils import parse_args

    argv = _dict_to_argv(cfg)
    args, remaining_args = parse_args(class_type, argv)
    if remaining_args:
        logger.warning('parse_args_from_dict: unrecognised args: %s', remaining_args)
    return args


def _dict_to_argv(cfg: Dict[str, Any]) -> List[str]:
    argv: List[str] = []
    for k, v in cfg.items():
        if k in _RAY_ONLY_KEYS or v is None:
            continue
        flag = f'--{k}'
        if isinstance(v, bool):
            argv += [flag, str(v).lower()]
        elif isinstance(v, (list, tuple)):
            argv.append(flag)
            argv += [str(item) for item in v]
        elif isinstance(v, dict):
            argv += [flag, json.dumps(v)]
        else:
            argv += [flag, str(v)]
    return argv


@dataclass
class RayConfig:
    rlhf_type: str = 'grpo'
    colocate_groups: List[List[str]] = field(default_factory=list)
    train_gpus: int = 0
    rollout_gpus: int = 0
    teacher_gpus: int = 0
    sleep_level: int = 1
    nnodes: int = 1

    @property
    def group_gpus(self) -> Dict[str, int]:
        return {
            'train': self.train_gpus,
            'rollout': self.rollout_gpus,
            'teacher': self.teacher_gpus,
        }

    def gpus_as_process_on_nodes(self, total_gpus: int) -> List[int]:
        """Split ``total_gpus`` evenly across ``nnodes`` for ResourcePool."""
        if self.nnodes <= 1:
            return [total_gpus]
        per_node, remainder = divmod(total_gpus, self.nnodes)
        if remainder != 0:
            raise ValueError(f'total_gpus={total_gpus} is not evenly divisible by nnodes={self.nnodes}')
        return [per_node] * self.nnodes


def parse_ray_yaml(config_path: str) -> 'tuple[RayConfig, Dict[str, Dict[str, Any]], Dict[str, Any]]':
    """Parse a Ray YAML config into (ray_config, group_dicts, shared_dict)."""
    import yaml
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    rlhf_type = raw.get('rlhf_type')
    colocate_groups = raw.pop('colocate_groups', [])
    sleep_level = int(raw.pop('sleep_level', 1))
    nnodes = int(raw.pop('nnodes', 1))

    group_configs: Dict[str, dict] = {}
    for g in KNOWN_GROUPS:
        group_configs[g] = raw.pop(g, {}) or {}

    gpu_counts = {g: int(cfg.pop('gpus', 0)) for g, cfg in group_configs.items()}
    shared_config = dict(raw)
    for key, default in _PARALLEL_DEFAULTS.items():
        shared_config.setdefault(key, default)

    ray_config = RayConfig(
        rlhf_type=rlhf_type,
        colocate_groups=colocate_groups,
        train_gpus=gpu_counts.get('train', 0),
        rollout_gpus=gpu_counts.get('rollout', 0),
        teacher_gpus=gpu_counts.get('teacher', 0),
        sleep_level=sleep_level,
        nnodes=nnodes,
    )

    _validate_colocate_groups(colocate_groups, gpu_counts)

    return ray_config, group_configs, shared_config


KNOWN_GROUPS = frozenset(('train', 'rollout', 'teacher'))


def _validate_colocate_groups(
    colocate_groups: List[List[str]],
    gpu_counts: Dict[str, int],
) -> None:
    """Validate colocate_groups: ≥2 roles, known, non-overlapping, each with gpus > 0."""
    if not colocate_groups:
        return
    seen: set = set()
    for idx, group in enumerate(colocate_groups):
        if not isinstance(group, list) or len(group) < 2:
            raise ValueError(f'colocate_groups[{idx}] must be a list of ≥2 roles, '
                             f'got {group!r}')
        group_gpu_counts = set()
        for role in group:
            if role not in KNOWN_GROUPS:
                raise ValueError(f'colocate_groups[{idx}] contains unknown role {role!r}; '
                                 f'valid roles: {sorted(KNOWN_GROUPS)}')
            if role in seen:
                raise ValueError(f'Role {role!r} appears in multiple colocate groups')
            seen.add(role)
            n = gpu_counts.get(role, 0)
            if n <= 0:
                raise ValueError(f'Role {role!r} in colocate_groups[{idx}] has 0 GPUs; '
                                 f'colocated roles must each have gpus > 0')
            group_gpu_counts.add(n)
        if len(group_gpu_counts) > 1:
            raise ValueError(f'colocate_groups[{idx}] roles have different GPU counts '
                             f'{dict(zip(group, [gpu_counts[r] for r in group]))}; '
                             f'colocated roles must share the same GPU set')


def merge_group_dict(shared: Dict[str, Any], group: Dict[str, Any]) -> Dict[str, Any]:
    """Merge shared + group config, stripping Ray-only keys and None values."""
    merged = {**shared, **group}
    for k in _RAY_ONLY_KEYS:
        merged.pop(k, None)
    return {k: v for k, v in merged.items() if v is not None}


def estimate_dp_size(cfg: Dict[str, Any], gpus: int) -> int:
    """Estimate DP size from a merged group config dict."""
    tp = cfg.get('tensor_model_parallel_size', 1)
    pp = cfg.get('pipeline_model_parallel_size', 1)
    cp = cfg.get('context_parallel_size', 1)
    assert gpus % (tp * pp * cp) == 0
    return gpus // (tp * pp * cp)


def build_dataset_from_dict(cfg: Dict[str, Any]):
    """Build dataset on the driver without instantiating a Megatron pipeline.

    Uses only ``BaseArguments`` methods (``get_model_processor``,
    ``get_template``, ``get_dataset_kwargs``) so no distributed init
    or Megatron-specific ``__post_init__`` logic is triggered.
    """
    import os

    from swift.megatron.arguments import MegatronRLHFArguments
    from swift.rlhf_trainers.utils import identity_data_collator
    from swift.utils import seed_everything, to_abspath

    cfg = dict(cfg)
    cfg['skip_megatron_init'] = True
    args = parse_args_from_dict(MegatronRLHFArguments, cfg)

    if hasattr(args, 'seed'):
        seed_everything(args.seed)

    rlhf_type = args.rlhf_type
    if rlhf_type in ('grpo', 'gkd'):
        args.remove_unused_columns = False
    if args.output_dir is None:
        args.output_dir = f'megatron_output/{args.model_suffix}'
    args.output_dir = to_abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    with torch.device('meta'):
        _, processor = args.get_model_processor(load_model=False, download_model=args.mcore_model is None)

    template = _prepare_template(args, processor)
    train_dataset, val_dataset = _prepare_dataset(args, template)

    data_collator = identity_data_collator if rlhf_type in ('grpo', 'gkd') else template.data_collator

    # TODO: integrate val_dataset / eval_iters into the training loop
    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'data_collator': data_collator,
        'micro_batch_size': args.micro_batch_size,
        'global_batch_size': args.global_batch_size,
        'padding_free': args.padding_free,
        'num_train_epochs': args.num_train_epochs,
        'train_iters': args.train_iters,
        'save_strategy': args.save_strategy,
        'eval_iters': args.eval_iters,
        'num_generations': args.num_generations,
        'template': template,
        '_driver_args': args,
    }


def _prepare_template(args, processor):
    """Create template from args and processor — no pipeline object needed."""
    template = args.get_template(processor)
    mode_mapping = {'grpo': 'train', 'gkd': 'train', 'kto': 'kto'}
    template.set_mode(mode_mapping.get(args.rlhf_type, 'rlhf'))
    template.use_megatron = True
    return template


def _prepare_dataset(args, template):
    """Load and optionally encode dataset — no pipeline object needed."""
    from swift.dataset import DatasetLoader, load_dataset

    # Ray pipeline has no validation/eval loop yet
    if args.split_dataset_ratio and args.split_dataset_ratio > 0:
        logger.warning(
            'Ray pipeline has no validation loop yet; overriding split_dataset_ratio '
            '%s -> 0.0 (no validation split).', args.split_dataset_ratio)
        args.split_dataset_ratio = 0.0
    if args.val_dataset:
        logger.warning('Ray pipeline has no validation loop yet; ignoring val_dataset=%s.', args.val_dataset)
        args.val_dataset = []

    pre_process = args.rlhf_type not in ('grpo', 'gkd')
    train_datasets, val_datasets = [], []

    if args.dataset or args.val_dataset:
        dataset_kwargs = args.get_dataset_kwargs()
        train_dataset, val_dataset = None, None
        if args.dataset:
            train_dataset, val_dataset = load_dataset(
                args.dataset,
                split_dataset_ratio=args.split_dataset_ratio,
                shuffle=args.dataset_shuffle,
                **dataset_kwargs)
        if args.val_dataset:
            _, val_dataset = load_dataset(
                args.val_dataset, split_dataset_ratio=1.0, shuffle=args.val_dataset_shuffle, **dataset_kwargs)

        if not pre_process:
            return train_dataset, val_dataset

        from swift.dataset import AddLengthPreprocessor
        for i, ds in enumerate([train_dataset, val_dataset]):
            if ds is None:
                continue
            if not args.lazy_tokenize and not args.streaming:
                preprocessor = AddLengthPreprocessor(template=template)
                batch_size = 100 if args.model_meta.is_multimodal else 1000
                ds = preprocessor(
                    ds,
                    num_proc=args.dataset_num_proc,
                    load_from_cache_file=args.load_from_cache_file,
                    strict=args.strict,
                    batch_size=batch_size)
            if i == 0:
                train_datasets.append(ds)
            else:
                val_datasets.append(ds)

    train_dataset = DatasetLoader.concat_datasets(train_datasets) if train_datasets else None
    val_dataset = DatasetLoader.concat_datasets(val_datasets) if val_datasets else None
    return train_dataset, val_dataset


def compute_iter_params(data_info: Dict[str, Any], dp_size: int) -> Dict[str, Any]:
    """Compute train_iters / eval_iters / save_steps on the driver."""
    mbs = data_info['micro_batch_size']
    gbs = data_info['global_batch_size']
    step_batch_size = mbs * dp_size
    num_gen = data_info.get('num_generations', 1)
    train_ds = data_info.get('train_dataset')
    val_ds = data_info.get('val_dataset')
    train_len = len(train_ds) if train_ds is not None and hasattr(train_ds, '__len__') else 0
    val_len = len(val_ds) if val_ds is not None and hasattr(val_ds, '__len__') else 0

    result: Dict[str, Any] = {}

    if data_info.get('save_strategy') == 'epoch' and train_len > 0:
        ds_sample = train_len // step_batch_size * step_batch_size * num_gen
        result['save_steps'] = ds_sample // gbs
        result['eval_steps'] = result['save_steps']

    train_iters = data_info.get('train_iters')
    if data_info.get('num_train_epochs') is not None and train_len > 0:
        ds_sample = train_len // step_batch_size * step_batch_size * num_gen
        train_iters = ds_sample * data_info['num_train_epochs'] // gbs
    result['train_iters'] = train_iters

    eval_iters = data_info.get('eval_iters', -1)
    if eval_iters is not None and eval_iters < 0:
        if val_len == 0:
            eval_iters = 0
        else:
            ds_sample = val_len // step_batch_size * step_batch_size * num_gen
            eval_iters = max(ds_sample // gbs, 1)
    if val_len > 0 and val_len < step_batch_size:
        eval_iters = 0
    result['eval_iters'] = eval_iters or 0

    return result


def extract_iteration(step_results) -> int:
    """Read the canonical iteration off ``WorkerGroup.execute`` results."""
    if not step_results:
        return 0
    for r in step_results:
        if isinstance(r, dict) and 'iteration' in r:
            return int(r['iteration'])
    return 0

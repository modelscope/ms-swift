"""CLI-only run bootstrap: the side effects legacy ``Arguments.__post_init__`` used to own.

Config dataclasses stay pure data. Command-line entry points call :func:`bootstrap_run` after parsing
and before entering a recipe; programmatic recipe callers (including twinkle-cs) keep full control of
I/O and do not trigger downloads, imports, logins, or directory creation merely by constructing a
Config.

Distributed initialization is intentionally absent. ``TrainAssembly.initialize_twinkle`` is the single
owner; doing it here as legacy did would double-initialize the process group.
"""
from __future__ import annotations
import os
from typing import TYPE_CHECKING, Optional

import json

if TYPE_CHECKING:
    from swift.dev.config import CheckpointConfig, DatasetConfig, ModelConfig, TunerConfig


def _parse_mapping(value):
    if value is None or isinstance(value, dict):
        return value or {}
    if not isinstance(value, str):
        raise TypeError(f'Expected a dict or JSON string, got {type(value).__name__}.')
    if os.path.isfile(value):
        with open(value, encoding='utf-8') as f:
            return json.load(f)
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise TypeError(f'Expected a JSON object, got {type(parsed).__name__}.')
    return parsed


def _prepare_output_dir(checkpoint_config: 'CheckpointConfig', *, add_version: Optional[bool],
                        create_output_dir: bool) -> None:
    from swift.utils import add_version_to_work_dir

    checkpoint_config.output_dir = os.path.abspath(os.path.expanduser(checkpoint_config.output_dir))
    use_version = checkpoint_config.add_version if add_version is None else add_version
    if use_version:
        checkpoint_config.output_dir = add_version_to_work_dir(checkpoint_config.output_dir)
    if create_output_dir:
        os.makedirs(checkpoint_config.output_dir, exist_ok=True)

    if checkpoint_config.resume_from_checkpoint:
        resume = os.path.abspath(os.path.expanduser(checkpoint_config.resume_from_checkpoint))
        if not os.path.exists(resume):
            raise ValueError(f'resume_from_checkpoint does not exist: {resume}')
        checkpoint_config.resume_from_checkpoint = resume


def _import_plugins(model_config: 'ModelConfig') -> None:
    from swift.utils import import_external_file

    paths = list(model_config.external_plugins or []) + list(model_config.custom_register_path or [])
    for path in dict.fromkeys(paths):
        import_external_file(path)


def _export_model_kwargs(model_config: 'ModelConfig') -> None:
    model_config.model_kwargs = _parse_mapping(model_config.model_kwargs)
    for key, value in model_config.model_kwargs.items():
        os.environ[key.upper()] = str(value)


def _resolve_model(model_config: 'ModelConfig', dataset_config: Optional['DatasetConfig']) -> None:
    if not model_config.model:
        return
    from swift.dev.utils.hub import safe_snapshot_download

    use_hf = dataset_config.use_hf if dataset_config is not None else None
    hub_token = dataset_config.hub_token if dataset_config is not None else None
    model_config.model = safe_snapshot_download(
        model_config.model, revision=model_config.model_revision, use_hf=use_hf, hub_token=hub_token)


def _resolve_adapters(tuner_config: Optional['TunerConfig'], dataset_config: Optional['DatasetConfig']) -> None:
    if tuner_config is None or not tuner_config.adapters:
        return
    from swift.dev.utils.hub import safe_snapshot_download

    use_hf = dataset_config.use_hf if dataset_config is not None else None
    hub_token = dataset_config.hub_token if dataset_config is not None else None
    tuner_config.adapters = [
        safe_snapshot_download(adapter, use_hf=use_hf, hub_token=hub_token) for adapter in tuner_config.adapters
    ]


def _login_hub(dataset_config: Optional['DatasetConfig']) -> None:
    if dataset_config is None or not dataset_config.hub_token:
        return
    # Legacy logs in whenever a token is supplied (not only on push), so private model/adapter reads
    # and a later push share the same authenticated session.
    from swift.dev.utils.hub import get_hub
    hub = get_hub(dataset_config.use_hf)
    hub.try_login(dataset_config.hub_token)


def process_and_validate_configs(configs: dict) -> None:
    """Run the shared pure config lifecycle for a parsed CLI config mapping.

    Serving and artifact commands do not expose training knobs, but the shared process/validation
    passes still own JSON normalization and cross-config guards. Temporary defaults keep that common
    contract without adding irrelevant training flags to each command's public CLI.
    """
    from swift.dev.config import DistributedConfig, TrainConfig, process_configs, validate_configs

    train_config = configs.get('train_config') or TrainConfig()
    distributed_config = configs.get('distributed_config') or DistributedConfig()
    process_configs(
        configs['model_config'],
        configs['template_config'],
        configs['dataset_config'],
        train_config,
        distributed_config,
        configs.get('checkpoint_config'),
        configs.get('tuner_config'),
        rlhf_config=configs.get('rlhf_config'),
        megatron_config=configs.get('megatron_config'),
        quantize_config=configs.get('quantize_config'),
    )
    validate_configs(
        configs['model_config'],
        configs['template_config'],
        configs['dataset_config'],
        train_config,
        distributed_config,
        configs.get('checkpoint_config'),
        configs.get('tuner_config'),
        configs.get('rlhf_config'),
        configs.get('logging_config'),
    )


def bootstrap_run(
    model_config: 'ModelConfig',
    checkpoint_config: 'CheckpointConfig',
    dataset_config: Optional['DatasetConfig'] = None,
    tuner_config: Optional['TunerConfig'] = None,
    *,
    seed: Optional[int] = None,
    add_version: Optional[bool] = None,
    create_output_dir: bool = True,
    resolve_model: bool = True,
) -> None:
    """Apply CLI runtime side effects, without initializing distributed state.

    ``add_version=None`` follows ``checkpoint_config.add_version``. Export callers pass ``False``
    because their output path names the final artifact, not a versioned training work directory.
    """
    if seed is not None:
        from swift.utils import seed_everything
        rank = max(int(os.environ.get('RANK', '-1')), 0)
        seed_everything(seed + rank)
    if dataset_config is not None and dataset_config.use_hf:
        os.environ['USE_HF'] = '1'
    _prepare_output_dir(checkpoint_config, add_version=add_version, create_output_dir=create_output_dir)
    _import_plugins(model_config)
    _export_model_kwargs(model_config)
    _login_hub(dataset_config)
    if resolve_model:
        _resolve_model(model_config, dataset_config)
    _resolve_adapters(tuner_config, dataset_config)

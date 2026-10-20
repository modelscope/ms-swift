# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
import os
from typing import TYPE_CHECKING, Optional, Sequence

if TYPE_CHECKING:
    from swift.dev.config import DistributedConfig, ModelConfig, TrainConfig, TunerConfig


def peek_backend(argv: Sequence[str]) -> Optional[str]:
    """Read ``--backend`` straight from raw argv, before the Configs are parsed.

    The backend selects which legacy-flag contract applies (Megatron and Transformers expose
    different knobs), so it has to be known before ``reject_legacy_only_flags`` runs.
    """
    from swift.dev.cli.parser import _scan_flag

    for index, token in enumerate(argv):
        if not token.startswith('--'):
            continue
        option, value, _, _ = _scan_flag(argv, index)
        if option.replace('-', '_') != 'backend':
            continue
        return value
    return None


def is_megatron_argv(argv: Sequence[str]) -> bool:
    """Whether raw argv selects the Megatron backend via ``--backend megatron``."""
    return peek_backend(argv) == 'megatron'


def enter_megatron_backend(distributed_config: 'DistributedConfig') -> None:
    """Pin the runtime prerequisites every Megatron run needs, regardless of the command."""
    os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')
    distributed_config.backend = 'megatron'
    # WORLD_SIZE is set by the torchrun launcher; fall back to a single process when run directly.
    distributed_config.nproc_per_node = distributed_config.nproc_per_node or int(os.environ.get('WORLD_SIZE', '1'))


def _fix_mtp(model_config: 'ModelConfig') -> None:
    """Derive dev's MTP training switch from the legacy-compatible layer count."""
    if model_config.mtp_num_layers is None:
        model_config.mtp_loss_scaling_factor = None
        return
    model_config.enable_mtp_training = True


def _derive_megatron_ga(train_config: 'TrainConfig', distributed_config: 'DistributedConfig', world_size: int) -> None:
    """Run the shared GA derivation against the launcher's effective world size.

    ``_derive_gradient_accumulation_steps`` reads WORLD_SIZE first and falls back to nproc_per_node,
    but the fallback would under-count a multi-node run where nproc is per-node -- so the launcher
    value is handed in explicitly. nproc_per_node is swapped only for the call and restored after,
    keeping its per-node semantics intact.
    """
    if train_config.global_batch_size is None:
        return
    from swift.dev.config.process import _derive_gradient_accumulation_steps

    original_nproc = distributed_config.nproc_per_node
    distributed_config.nproc_per_node = world_size
    try:
        _derive_gradient_accumulation_steps(train_config, distributed_config, True)
    finally:
        distributed_config.nproc_per_node = original_nproc


def _select_megatron_tuner(tuner: 'TunerConfig') -> Optional['TunerConfig']:
    # Legacy's MegatronArguments defaults tuner_type to 'full' (swift/megatron/arguments/
    # megatron_args.py), overriding the shared base default of 'lora'. TunerConfig carries that base
    # default, so a Megatron argv with no explicit --tuner/--tuner_type must resolve to full-parameter
    # training: falling through to the base 'lora' here silently trains adapters instead of the whole
    # model, breaking argv parity with legacy (the cli-vs-legacy e2e gate caught it as a ~6x smaller
    # grad_norm and a loss that barely moved). An explicit --tuner lora is still honoured.
    if tuner.tuner == 'full':
        return None
    if tuner.tuner == 'lora':
        return tuner if 'tuner' in getattr(tuner, '_explicit_fields', ()) else None
    raise NotImplementedError(f'dev Megatron training supports tuner in {{full, lora}}, got {tuner.tuner!r}.')


def configure_megatron(model_config: 'ModelConfig', train_config: 'TrainConfig',
                       distributed_config: 'DistributedConfig',
                       tuner_config: 'TunerConfig') -> Optional['TunerConfig']:
    """Apply the Megatron training shim in place; return the selected tuner (None for full-param)."""
    enter_megatron_backend(distributed_config)
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    _fix_mtp(model_config)
    if train_config.weight_decay_incr_style == 'constant':
        train_config.start_weight_decay = None
        train_config.end_weight_decay = None
    _derive_megatron_ga(train_config, distributed_config, world_size)
    return _select_megatron_tuner(tuner_config)

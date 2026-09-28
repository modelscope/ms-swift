"""Shared Megatron shim for the unified dev CLI.

twinkle merges the Megatron and Transformers stacks, so the backend is chosen by a single parameter
(``DistributedConfig.backend``) rather than by a separate ``megatron`` command. Every training and
export entry point therefore parses the same Config surface and, when ``--backend megatron`` is set,
applies the small amount of Megatron-only fix-up collected here. Keeping it in one module means sft,
pt, rlhf and export all trigger Megatron identically instead of each carrying its own copy.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

if TYPE_CHECKING:
    from swift.dev.config import DistributedConfig, ModelConfig, TrainConfig, TunerConfig


@dataclass
class MegatronCliCompatConfig:
    """Legacy control flags that have no Config value because dev makes them unconditional."""

    # dev never initializes Megatron during parsing, so the safe behavior requested by this legacy
    # switch is already unconditional. Semantic aliases such as attention_backend/bf16/fp16 are not
    # duplicated here: parser.normalize_argv writes them directly to ModelConfig.
    skip_megatron_init: bool = True


def peek_backend(argv: Sequence[str]) -> Optional[str]:
    """Read ``--backend`` straight from raw argv, before the Configs are parsed.

    The backend selects which legacy-flag contract applies (Megatron and Transformers expose
    different knobs), so it has to be known before ``reject_legacy_only_flags`` runs -- earlier than
    the full parse that would otherwise populate ``DistributedConfig.backend``.
    """
    for index, token in enumerate(argv):
        if not token.startswith('--'):
            continue
        option, separator, inline_value = token[2:].partition('=')
        if option.replace('-', '_') != 'backend':
            continue
        if separator:
            return inline_value
        if index + 1 < len(argv) and not argv[index + 1].startswith('--'):
            return argv[index + 1]
        return None
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
    """Use the shared Config derivation while honoring the launcher's effective world size."""
    if train_config.global_batch_size is None:
        return
    from swift.dev.config.process import _derive_gradient_accumulation_steps

    original_nproc = distributed_config.nproc_per_node
    distributed_config.nproc_per_node = world_size
    try:
        _derive_gradient_accumulation_steps(train_config, distributed_config, True)
    finally:
        distributed_config.nproc_per_node = original_nproc or world_size


def _select_megatron_tuner(tuner: 'TunerConfig') -> Optional['TunerConfig']:
    if tuner.tuner_type == 'full':
        return None
    if tuner.tuner_type == 'lora':
        return tuner
    raise NotImplementedError(f'dev Megatron training supports tuner_type in {{full, lora}}, got {tuner.tuner_type!r}.')


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

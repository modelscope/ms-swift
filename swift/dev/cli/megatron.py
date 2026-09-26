"""Megatron CLI entry points backed directly by dev's atomic Configs."""
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        DistributedConfig,
        LoggingConfig,
        MegatronConfig,
        ModelConfig,
        MoEConfig,
        PluginConfig,
        QuantizeConfig,
        TemplateConfig,
        TrainConfig,
        TunerConfig,
    )


@dataclass
class MegatronCliCompatConfig:
    """Legacy control flags that have no Config value because dev makes them unconditional."""

    # dev never initializes Megatron during parsing, so the safe behavior requested by this legacy
    # switch is already unconditional. Semantic aliases such as attention_backend/bf16/fp16 are not
    # duplicated here: parser.normalize_argv writes them directly to ModelConfig.
    skip_megatron_init: bool = True


def _fix_mtp(model_config: 'ModelConfig') -> None:
    """Derive dev's MTP training switch from the legacy-compatible layer count."""
    if model_config.mtp_num_layers is None:
        model_config.mtp_loss_scaling_factor = None
        return
    model_config.enable_mtp_training = True


def _derive_megatron_ga(train_config, distributed_config, world_size: int) -> None:
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
    raise NotImplementedError(f'dev Megatron CLI supports tuner_type in {{full, lora}}, got {tuner.tuner_type!r}.')


def parse_megatron_configs(
    argv: Optional[List[str]] = None,
    *,
    world_size: Optional[int] = None,
    command: str = 'megatron_sft',
) -> Tuple['ModelConfig', 'PluginConfig', 'TemplateConfig', 'DatasetConfig', 'TrainConfig', 'DistributedConfig',
           'CheckpointConfig', 'LoggingConfig', Optional['TunerConfig'], 'QuantizeConfig', 'MegatronConfig',
           'MoEConfig']:
    """Parse Megatron argv directly into dev Configs, with a compatibility shim."""
    import os

    from swift.dev.cli.parser import parse_configs, resolve_argv
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        DistributedConfig,
        LoggingConfig,
        MegatronConfig,
        ModelConfig,
        MoEConfig,
        PluginConfig,
        QuantizeConfig,
        TemplateConfig,
        TrainConfig,
        TunerConfig,
    )

    effective_argv = resolve_argv(argv)
    from swift.dev.cli.legacy_coverage import reject_legacy_only_flags
    reject_legacy_only_flags(command, effective_argv)
    classes = [
        ModelConfig, PluginConfig, TemplateConfig, DatasetConfig, TrainConfig, DistributedConfig, CheckpointConfig,
        LoggingConfig, TunerConfig, QuantizeConfig, MegatronConfig, MoEConfig, MegatronCliCompatConfig
    ]
    configs, remaining = parse_configs(classes, effective_argv)
    if remaining:
        raise ValueError(f'Unrecognized arguments: {remaining}. This legacy Megatron flag has no dev Config '
                         'consumer and is refused rather than silently dropped.')
    (model_config, plugin_config, template_config, dataset_config, train_config, distributed_config,
     checkpoint_config, logging_config, tuner, quantize_config, megatron_config, moe_config, compat) = configs

    world_size = int(os.environ.get('WORLD_SIZE', '1')) if world_size is None else world_size
    distributed_config.backend = 'megatron'
    distributed_config.nproc_per_node = distributed_config.nproc_per_node or world_size

    # attention_backend/bf16/fp16 have already been normalized directly onto ModelConfig by
    # parser.normalize_argv; the compatibility object only carries skip_megatron_init.
    del compat
    _fix_mtp(model_config)
    if train_config.weight_decay_incr_style == 'constant':
        train_config.start_weight_decay = None
        train_config.end_weight_decay = None
    _derive_megatron_ga(train_config, distributed_config, world_size)

    return (model_config, plugin_config, template_config, dataset_config, train_config, distributed_config,
            checkpoint_config, logging_config, _select_megatron_tuner(tuner), quantize_config, megatron_config,
            moe_config)


def megatron_sft_main(argv: Optional[List[str]] = None) -> List[dict]:
    """dev Megatron SFT entry, self-parsed without constructing legacy MegatronSftArguments."""
    from swift.dev.config import process_and_validate_configs
    from swift.dev.recipe import run_sft

    (model_config, plugin_config, template_config, dataset_config, train_config, distributed_config,
     checkpoint_config, logging_config, tuner_config, quantize_config, megatron_config,
     moe_config) = parse_megatron_configs(argv)
    process_and_validate_configs({
        'model_config': model_config,
        'plugin_config': plugin_config,
        'template_config': template_config,
        'dataset_config': dataset_config,
        'train_config': train_config,
        'distributed_config': distributed_config,
        'checkpoint_config': checkpoint_config,
        'logging_config': logging_config,
        'tuner_config': tuner_config,
        'quantize_config': quantize_config,
        'megatron_config': megatron_config,
        'moe_config': moe_config,
    })

    return run_sft(
        model_config,
        template_config,
        dataset_config,
        train_config,
        distributed_config=distributed_config,
        checkpoint_config=checkpoint_config,
        tuner_config=tuner_config,
        logging_config=logging_config,
        quantize_config=quantize_config,
        megatron_config=megatron_config,
        moe_config=moe_config,
        output_dir=checkpoint_config.output_dir,
    )


if __name__ == '__main__':
    megatron_sft_main()

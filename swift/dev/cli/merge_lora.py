"""Dedicated LoRA merge CLI using the dev merge recipe."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class MergeLoraCliConfig:
    replace_if_exists: bool = False


def parse_merge_lora_configs(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    from swift.dev.cli.legacy_coverage import reject_legacy_only_flags
    from swift.dev.cli.parser import parse_configs_strict, resolve_argv
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        ModelConfig,
        RuntimeConfig,
        TemplateConfig,
        TunerConfig,
    )

    effective_argv = resolve_argv(argv)
    reject_legacy_only_flags('merge_lora', effective_argv)
    classes = [
        ModelConfig, TemplateConfig, DatasetConfig, CheckpointConfig, TunerConfig, MergeLoraCliConfig, RuntimeConfig
    ]
    configs = parse_configs_strict(
        classes, effective_argv, command='swift merge-lora', load_args_default=True)
    names = ('model_config', 'template_config', 'dataset_config', 'checkpoint_config', 'tuner_config', 'cli_config',
             'runtime_config')
    return dict(zip(names, configs))


def merge_lora_main(argv: Optional[List[str]] = None) -> str:
    from swift.dev.config import process_and_validate_configs
    from swift.dev.recipe import run_merge_lora

    configs = parse_merge_lora_configs(argv)
    process_and_validate_configs(configs, add_version=False, create_output_dir=False)
    return run_merge_lora(
        configs['model_config'],
        configs['tuner_config'],
        template_config=configs['template_config'],
        checkpoint_config=configs['checkpoint_config'],
        replace_if_exists=configs['cli_config'].replace_if_exists,
    )


if __name__ == '__main__':
    merge_lora_main()

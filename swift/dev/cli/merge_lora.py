"""Dedicated LoRA merge CLI using the dev merge recipe."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class MergeLoraCliConfig:
    replace_if_exists: bool = False


def parse_merge_lora_configs(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    from swift.dev.cli.parser import parse_configs_strict
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        ModelConfig,
        RuntimeConfig,
        TemplateConfig,
        TunerConfig,
    )

    classes = [
        ModelConfig, TemplateConfig, DatasetConfig, CheckpointConfig, TunerConfig, MergeLoraCliConfig, RuntimeConfig
    ]
    configs = parse_configs_strict(classes, argv, command='swift merge-lora')
    names = ('model_config', 'template_config', 'dataset_config', 'checkpoint_config', 'tuner_config', 'cli_config',
             'runtime_config')
    return dict(zip(names, configs))


def merge_lora_main(argv: Optional[List[str]] = None) -> str:
    from swift.dev.cli.runtime import bootstrap_run, process_and_validate_configs
    from swift.dev.recipe import run_merge_lora

    configs = parse_merge_lora_configs(argv)
    process_and_validate_configs(configs)
    bootstrap_run(
        configs['model_config'], configs['checkpoint_config'], configs['dataset_config'], configs['tuner_config'],
        seed=configs['runtime_config'].seed, add_version=False, create_output_dir=False)
    return run_merge_lora(
        configs['model_config'],
        configs['tuner_config'],
        template_config=configs['template_config'],
        checkpoint_config=configs['checkpoint_config'],
        replace_if_exists=configs['cli_config'].replace_if_exists,
    )


if __name__ == '__main__':
    merge_lora_main()

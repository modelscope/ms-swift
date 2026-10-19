# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class MergeCliConfig:
    replace_if_exists: bool = False


def parse_merge_configs(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    from swift.dev2.cli.legacy_coverage import reject_legacy_only_flags
    from swift.dev2.cli.parser import parse_configs_strict, resolve_argv
    from swift.dev2.config import (
        CheckpointConfig,
        DatasetConfig,
        ModelConfig,
        PluginConfig,
        RuntimeConfig,
        TemplateConfig,
        TunerConfig,
    )

    effective_argv = resolve_argv(argv)
    reject_legacy_only_flags('merge', effective_argv)
    classes = [
        ModelConfig, PluginConfig, TemplateConfig, DatasetConfig, CheckpointConfig, TunerConfig, MergeCliConfig,
        RuntimeConfig
    ]
    configs = parse_configs_strict(classes, effective_argv, command='swift merge', load_args_default=True)
    names = ('model_config', 'plugin_config', 'template_config', 'dataset_config', 'checkpoint_config',
             'tuner_config', 'cli_config', 'runtime_config')
    return dict(zip(names, configs))


def merge_main(argv: Optional[List[str]] = None) -> str:
    from swift.dev2.config import process_and_validate_configs
    from swift.dev2.recipe.export import run_merge_lora

    configs = parse_merge_configs(argv)
    process_and_validate_configs(configs, add_version=False, create_output_dir=False)
    return run_merge_lora(
        configs['model_config'],
        configs['tuner_config'],
        template_config=configs['template_config'],
        checkpoint_config=configs['checkpoint_config'],
        replace_if_exists=configs['cli_config'].replace_if_exists,
        parallel_size=configs['runtime_config'].weight_transform_parallel_size,
    )


if __name__ == '__main__':
    merge_main()

# Copyright (c) ModelScope Contributors. All rights reserved.
"""Runner for one capability combination: apply config overrides, train, save, report.

Launched by ``capability/test_capability.py`` under ``torchrun``; never imported. Overrides arrive as
one JSON blob per Config so the test side stays declarative and this side stays dumb.

Usage: capability.py '<json spec>' where the spec holds one dict per Config plus ``output_dir`` and
``result_path``.
"""
import json
import sys


def main(spec: dict) -> None:
    from swift.dev.config import (CheckpointConfig, DatasetConfig, DistributedConfig, ModelConfig, TemplateConfig,
                                  TrainConfig)
    from swift.dev.config.process import process_configs
    from swift.dev.recipe import run_sft

    model_config = ModelConfig(**spec['model'])
    template_config = TemplateConfig(**spec['template'])
    dataset_config = DatasetConfig(**spec['dataset'])
    train_config = TrainConfig(**spec['train'])
    distributed_config = DistributedConfig(**spec['distributed'])
    checkpoint_config = CheckpointConfig()
    process_configs(model_config, template_config, dataset_config, train_config, distributed_config,
                    checkpoint_config)

    history = run_sft(model_config, template_config, dataset_config, train_config, distributed_config,
                      checkpoint_config, output_dir=spec['output_dir'])

    import torch.distributed as dist
    if not dist.is_initialized() or dist.get_rank() == 0:
        # rank 0 owns the verdict file, so the caller reads exactly one result.
        with open(spec['result_path'], 'w') as f:
            json.dump({'losses': [row['loss'] for row in history if 'loss' in row]}, f)


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))

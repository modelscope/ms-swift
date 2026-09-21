"""Megatron pretraining CLI over the shared dev Config parser."""
from __future__ import annotations
import os
from typing import List, Optional


def megatron_pt_main(argv: Optional[List[str]] = None):
    from swift.dev.cli.megatron import parse_megatron_configs
    from swift.dev.cli.runtime import bootstrap_run
    from swift.dev.config import process_configs, validate_configs
    from swift.dev.recipe import run_pt

    os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')
    (model_config, template_config, dataset_config, train_config, distributed_config, checkpoint_config,
     logging_config, tuner_config) = parse_megatron_configs(argv)
    if model_config.task_type not in (None, 'causal_lm'):
        raise ValueError(f'PT requires task_type="causal_lm", got {model_config.task_type!r}.')
    if template_config.use_chat_template not in (None, False):
        raise ValueError('PT requires use_chat_template=false.')
    if template_config.loss_scale not in ('default', 'all'):
        raise ValueError('PT requires loss_scale="all".')
    model_config.task_type = 'causal_lm'
    template_config.use_chat_template = False
    template_config.loss_scale = 'all'
    process_configs(
        model_config, template_config, dataset_config, train_config, distributed_config, checkpoint_config,
        tuner_config)
    validate_configs(
        model_config, template_config, dataset_config, train_config, distributed_config, checkpoint_config,
        tuner_config, logging_config=logging_config)
    bootstrap_run(model_config, checkpoint_config, dataset_config, tuner_config, seed=train_config.seed)
    return run_pt(
        model_config,
        template_config,
        dataset_config,
        train_config,
        distributed_config,
        checkpoint_config,
        tuner_config,
        logging_config,
        output_dir=checkpoint_config.output_dir)


if __name__ == '__main__':
    megatron_pt_main()

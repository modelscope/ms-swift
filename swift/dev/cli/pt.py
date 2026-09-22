"""PT CLI: the SFT Config surface with non-chat causal-LM defaults."""
from __future__ import annotations
from typing import List, Optional


def pt_main(argv: Optional[List[str]] = None) -> List[dict]:
    from swift.dev.cli.sft import parse_sft_configs
    from swift.dev.config import process_and_validate_configs
    from swift.dev.recipe import run_pt

    (model_config, template_config, dataset_config, train_config, distributed_config, checkpoint_config,
     logging_config, tuner_config, quantize_config) = parse_sft_configs(argv, command='pt')
    if model_config.task_type not in (None, 'causal_lm'):
        raise ValueError(f'PT requires task_type="causal_lm", got {model_config.task_type!r}.')
    if template_config.use_chat_template not in (None, False):
        raise ValueError('PT requires use_chat_template=false.')
    if template_config.loss_scale not in ('default', 'all'):
        raise ValueError('PT requires loss_scale="all".')
    model_config.task_type = 'causal_lm'
    template_config.use_chat_template = False
    template_config.loss_scale = 'all'
    process_and_validate_configs({
        'model_config': model_config,
        'template_config': template_config,
        'dataset_config': dataset_config,
        'train_config': train_config,
        'distributed_config': distributed_config,
        'checkpoint_config': checkpoint_config,
        'logging_config': logging_config,
        'tuner_config': tuner_config,
        'quantize_config': quantize_config,
    })
    return run_pt(
        model_config,
        template_config,
        dataset_config,
        train_config,
        distributed_config,
        checkpoint_config,
        tuner_config,
        logging_config,
        quantize_config,
        output_dir=checkpoint_config.output_dir,
    )


if __name__ == '__main__':
    pt_main()

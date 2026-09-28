"""PT CLI: the SFT Config surface with non-chat causal-LM defaults."""
from __future__ import annotations
from typing import List, Optional


def pt_main(argv: Optional[List[str]] = None) -> List[dict]:
    from swift.dev.cli.sft import parse_sft_configs
    from swift.dev.config import process_and_validate_configs
    from swift.dev.recipe import run_pt

    configs = parse_sft_configs(argv, command='pt')
    model_config = configs['model_config']
    template_config = configs['template_config']
    if model_config.task_type not in (None, 'causal_lm'):
        raise ValueError(f'PT requires task_type="causal_lm", got {model_config.task_type!r}.')
    if template_config.use_chat_template not in (None, False):
        raise ValueError('PT requires use_chat_template=false.')
    if template_config.loss_scale not in ('default', 'all'):
        raise ValueError('PT requires loss_scale="all".')
    model_config.task_type = 'causal_lm'
    template_config.use_chat_template = False
    template_config.loss_scale = 'all'
    process_and_validate_configs(configs)
    is_megatron = configs['distributed_config'].backend == 'megatron'
    megatron_kwargs = ({
        'megatron_config': configs['megatron_config'],
        'moe_config': configs['moe_config']
    } if is_megatron else {})
    return run_pt(
        model_config,
        template_config,
        configs['dataset_config'],
        configs['train_config'],
        configs['distributed_config'],
        configs['checkpoint_config'],
        configs['tuner_config'],
        configs['logging_config'],
        quantize_config=configs['quantize_config'],
        output_dir=configs['checkpoint_config'].output_dir,
        **megatron_kwargs,
    )


if __name__ == '__main__':
    pt_main()

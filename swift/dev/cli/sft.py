from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from swift.arguments import SftArguments
    from swift.dev.configs import (CheckpointConfig, DatasetConfig, DistributedConfig, ModelConfig, TemplateConfig,
                                   TrainConfig, TunerConfig)


def _dtype_name(value: object) -> Optional[str]:
    if value is None:
        return None
    name = str(value)
    return name.split('.')[-1] if name.startswith('torch.') else name


def _fill_from_args(config, args: 'SftArguments'):
    for f in dataclasses.fields(config):
        if not hasattr(args, f.name):
            continue
        value = getattr(args, f.name)
        if value is not None:
            setattr(config, f.name, value)
    return config


def args_to_configs(
    args: 'SftArguments',
) -> Tuple['ModelConfig', 'TemplateConfig', 'DatasetConfig', 'TrainConfig', 'DistributedConfig', 'CheckpointConfig',
           Optional['TunerConfig']]:
    from swift.dev.configs import (CheckpointConfig, DatasetConfig, DistributedConfig, ModelConfig, TemplateConfig,
                                   TrainConfig, TunerConfig)

    # This mapping is HF-surface only. MegatronSftArguments is a SEPARATE hierarchy (not an
    # SftArguments subclass) that names the same hyperparameters differently -- lr, train_iters,
    # micro_batch_size, lr_decay_style, lr_warmup_fraction, adam_eps -- so the name-based copy below
    # would silently leave 34 of TrainConfig's 58 fields at their dev defaults, and DistributedConfig
    # has no `backend` on that surface either, so the run would build a TransformersModel. Fail here
    # instead: a Megatron CLI needs its own mapping with those renames spelled out.
    if hasattr(args, 'train_iters'):
        raise NotImplementedError(
            'args_to_configs maps the HF-surface SftArguments only; got what looks like Megatron '
            'arguments (has train_iters). The Megatron CLI is not wired yet -- drive the Megatron '
            'backend through run_sft with explicit Configs (DistributedConfig(backend="megatron")).')

    model_config = _fill_from_args(ModelConfig(), args)
    model_config.torch_dtype = _dtype_name(args.torch_dtype)

    template_config = _fill_from_args(TemplateConfig(), args)
    dataset_config = _fill_from_args(DatasetConfig(), args)
    train_config = _fill_from_args(TrainConfig(), args)
    distributed_config = _fill_from_args(DistributedConfig(), args)
    checkpoint_config = _fill_from_args(CheckpointConfig(), args)

    checkpoint_config.save_steps = int(checkpoint_config.save_steps)
    if train_config.eval_steps is not None:
        train_config.eval_steps = int(train_config.eval_steps)

    # `--optimizer` selects one of legacy's optimizer plugins (galore / lorap / muon / muonclip /
    # multimodal) through optimizers_map, and those callbacks are built on HfTrainer's
    # create_optimizer, which dev does not use. It stays on legacy SftArguments, so it would parse
    # fine and then be dropped on the floor; refuse it instead.
    if getattr(args, 'optimizer', None) is not None:
        raise NotImplementedError(f'--optimizer {args.optimizer!r} is not supported by the dev SFT pipeline: legacy '
                                  'dispatches it to an HfTrainer-based optimizer callback, which dev has no equivalent '
                                  'of. Use --optim for the optimizer itself, and drop --optimizer.')

    tuner_type = args.tuner_type
    if tuner_type == 'full':
        tuner_config = None
    elif tuner_type == 'lora':
        tuner_config = _fill_from_args(TunerConfig(), args)
        tuner_config.tuner_type = 'lora'
    else:
        raise NotImplementedError(f'dev SFT CLI supports tuner_type in {{full, lora}}, got {tuner_type!r}.')

    return (model_config, template_config, dataset_config, train_config, distributed_config, checkpoint_config,
            tuner_config)


def sft_main(args: Optional[List[str]] = None) -> List[dict]:
    from swift import SftArguments
    from swift.dev.recipes import run_sft
    from swift.utils import parse_args

    if isinstance(args, SftArguments):
        sft_args = args
    else:
        sft_args, remaining = parse_args(SftArguments, args)
        if remaining:
            raise ValueError(f'Unrecognized arguments: {remaining}')

    (model_config, template_config, dataset_config, train_config, distributed_config, checkpoint_config,
     tuner_config) = args_to_configs(sft_args)

    return run_sft(
        model_config,
        template_config,
        dataset_config,
        train_config,
        distributed_config=distributed_config,
        checkpoint_config=checkpoint_config,
        tuner_config=tuner_config,
        output_dir=checkpoint_config.output_dir,
    )


if __name__ == '__main__':
    sft_main()

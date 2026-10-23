# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
import os
from typing import Any, Dict, List, Optional

from swift.dev.utils.logger import get_logger

logger = get_logger()


def parse_export_configs(argv: Optional[List[str]] = None, *, command: str = 'export') -> Dict[str, Any]:
    """Parse argv directly into the Configs consumed by export recipes."""
    from swift.dev.cli.legacy_coverage import reject_legacy_only_flags
    from swift.dev.cli.parser import flag_names, parse_configs, resolve_argv
    from swift.dev.config import (
        CheckpointConfig,
        ConvertConfig,
        DatasetConfig,
        DistributedConfig,
        MegatronConfig,
        ModelConfig,
        MoEConfig,
        PluginConfig,
        QuantizeConfig,
        RuntimeConfig,
        TemplateConfig,
        TunerConfig,
    )

    effective_argv = resolve_argv(argv)
    reject_legacy_only_flags(command, effective_argv)
    classes = [ModelConfig, PluginConfig, TemplateConfig, DatasetConfig, DistributedConfig, CheckpointConfig,
               QuantizeConfig, ConvertConfig, TunerConfig, RuntimeConfig, MegatronConfig, MoEConfig]
    configs, remaining = parse_configs(classes, effective_argv, load_args_default=True)
    if remaining:
        raise ValueError(f'Unrecognized arguments: {remaining}. The dev export CLI parses the Config surface '
                         'directly; a flag with no matching Config field is refused rather than dropped.')
    names = ('model_config', 'plugin_config', 'template_config', 'dataset_config', 'distributed_config',
             'checkpoint_config', 'quantize_config', 'convert_config', 'tuner_config', 'runtime_config',
             'megatron_config', 'moe_config')
    result = dict(zip(names, configs))
    # `swift export --backend megatron` is the single Megatron export entry (there is no separate
    # `megatron export` command): pin the runtime prerequisites the mcore convert path needs.
    if result['distributed_config'].backend == 'megatron':
        from swift.dev.cli._megatron_compat import enter_megatron_backend
        enter_megatron_backend(result['distributed_config'])
    result['checkpoint_config']._output_dir_explicit = 'output_dir' in flag_names(effective_argv)
    return result


def _derive_output_dir(configs: Dict[str, Any]) -> None:  # noqa: C901
    """Derive the default output as ``{source}-{operation}``; the dir belongs to the final step only,
    so a merge chained into quantization writes to its own ``{adapter}-merged`` directory."""
    checkpoint = configs['checkpoint_config']
    if checkpoint._output_dir_explicit:
        return
    convert = configs['convert_config']
    quantize = configs['quantize_config']
    tuner = configs['tuner_config']
    source = tuner.adapters[0] if tuner.adapters else configs['model_config'].model
    if not source:
        return
    suffix = None
    if convert.to_peft_format:
        suffix = 'peft'
    elif quantize.quant_method:
        suffix = quantize.quant_method
        if quantize.quant_bits is not None:
            suffix += f'-int{quantize.quant_bits}'
    elif convert.merge_lora:
        suffix = 'merged'
    elif convert.to_mcore:
        suffix = 'mcore'
    elif convert.to_hf:
        suffix = 'hf'
    elif convert.to_cached_dataset:
        suffix = 'cached_dataset'
    if suffix:
        parent, name = os.path.split(source.rstrip('/'))
        checkpoint.output_dir = os.path.join(parent, f'{name}-{suffix}')


def _force_eager_encoding_for_cached_dataset(configs: Dict[str, Any]) -> None:
    """Pin ``lazy_tokenize=False`` for ``--to_cached_dataset``: a cached dataset must be materialised
    and saved, which lazy tokenisation cannot do; warn when overriding an explicit True."""
    convert_config = configs['convert_config']
    dataset_config = configs['dataset_config']
    if not convert_config.to_cached_dataset or dataset_config.lazy_tokenize is False:
        return
    if dataset_config.lazy_tokenize:
        logger.warning('--to_cached_dataset overrides lazy_tokenize=True to False: a cached dataset must be '
                       'encoded eagerly to be materialised and saved.')
    dataset_config.lazy_tokenize = False


def _validate_export(configs: Dict[str, Any]) -> None:
    quantize_config = configs['quantize_config']
    convert_config = configs['convert_config']
    dataset_config = configs['dataset_config']
    checkpoint_config = configs['checkpoint_config']
    distributed_config = configs['distributed_config']

    if quantize_config.quant_bits is not None and quantize_config.quant_method is None:
        raise ValueError('Please specify the quantization method using `--quant_method`.')
    if (quantize_config.quant_method and quantize_config.quant_method != 'fp8'
            and quantize_config.quant_bits is None):
        raise ValueError('Please specify `--quant_bits`.')
    if quantize_config.quant_method in {'gptq', 'gptq_v2', 'awq'} and not dataset_config.dataset:
        raise ValueError(f'quant_method={quantize_config.quant_method!r} needs a calibration dataset.')
    if (convert_config.to_mcore or convert_config.to_hf) and convert_config.merge_lora:
        raise ValueError('`--merge_lora` cannot be combined with `--to_mcore`/`--to_hf` in the dev export CLI. '
                         'Run the merge and conversion as separate commands.')
    if convert_config.to_mcore and convert_config.to_hf:
        raise ValueError('Choose exactly one conversion direction: --to_mcore or --to_hf.')

    requested_save_options = {'safe_serialization', 'max_shard_size'}.intersection(
        getattr(checkpoint_config, '_explicit_fields', ()))
    chained_merge = bool(convert_config.merge_lora
                         and (quantize_config.quant_method or checkpoint_config.push_to_hub))
    if requested_save_options and not ((convert_config.merge_lora and not chained_merge) or convert_config.to_hf
                                       or quantize_config.quant_method):
        raise NotImplementedError(
            f'{sorted(requested_save_options)} do not configure the selected export operation. They are supported '
            'for a standalone LoRA merge, for quantization, or for mcore-to-HF conversion where applicable.')
    if convert_config.to_hf and not checkpoint_config.safe_serialization:
        raise NotImplementedError('mcore-to-HF conversion always writes safetensors; use --safe_serialization true.')
    if (convert_config.to_hf and distributed_config.bridge_backend == 'megatron-bridge'
            and 'max_shard_size' in requested_save_options):
        raise NotImplementedError(
            'megatron-bridge AutoBridge does not expose max_shard_size. Use --bridge_backend mcore-bridge or '
            'remove the option.')


def run_export_configs(configs: Dict[str, Any]) -> Optional[str]:
    """Execute an already parsed export Config set."""
    from swift.dev.config import process_and_validate_configs
    from swift.dev.recipe.export import (
        export_cached_dataset,
        run_convert,
        run_merge_lora,
        run_push_to_hub,
        run_quantize,
        run_to_peft_format,
    )

    _validate_export(configs)
    _derive_output_dir(configs)
    # Pin lazy_tokenize before the shared process/validate pass: _check_lazy_tokenize and
    # builders/dataset._encode_mode would otherwise resolve it to lazy (the multimodal default) and refuse.
    _force_eager_encoding_for_cached_dataset(configs)
    # Export output is a final artifact, not a training work directory. Recipes own directory creation
    # and their replace/overwrite policy, so run initialization only normalizes the path.
    process_and_validate_configs(configs, add_version=False, create_output_dir=False)

    model_config = configs['model_config']
    template_config = configs['template_config']
    dataset_config = configs['dataset_config']
    distributed_config = configs['distributed_config']
    checkpoint_config = configs['checkpoint_config']
    quantize_config = configs['quantize_config']
    convert_config = configs['convert_config']
    tuner_config = configs['tuner_config']

    output_dir = checkpoint_config.output_dir
    result: Optional[str] = None

    if convert_config.to_peft_format:
        if not tuner_config.adapters:
            raise ValueError('--to_peft_format requires at least one --adapters path.')
        converted = run_to_peft_format(tuner_config.adapters[0], output_dir)
        tuner_config.adapters[0] = converted
        result = converted

    if convert_config.merge_lora:
        # to_peft_format chains like quant/push: the derived `{adapter}-peft` dir is where to_peft
        # itself just wrote, so merge must fall back to its own `{converted}-merged` default
        # (legacy clears output_dir for `to_peft_format or quant_method or to_ollama or push_to_hub`).
        chains = bool(quantize_config.quant_method or checkpoint_config.push_to_hub
                      or convert_config.to_peft_format)
        merged = run_merge_lora(
            model_config,
            tuner_config,
            template_config=template_config,
            checkpoint_config=None if chains else checkpoint_config,
            output_dir=None if chains else output_dir,
            replace_if_exists=convert_config.exist_ok,
            parallel_size=configs['runtime_config'].weight_transform_parallel_size)
        model_config.model = merged
        tuner_config.adapters = []
        result = merged

    # Keep legacy's if/elif contract: at most one post-merge action per invocation.
    if quantize_config.quant_method:
        result = run_quantize(
            model_config,
            template_config,
            quantize_config,
            dataset_config,
            output_dir=output_dir,
            safe_serialization=checkpoint_config.safe_serialization,
            max_shard_size=checkpoint_config.max_shard_size,
            exist_ok=convert_config.exist_ok)
    elif convert_config.to_cached_dataset:
        train_dir, _ = export_cached_dataset(
            model_config, template_config, dataset_config, output_dir=output_dir,
            template_mode=convert_config.template_mode, store_format=convert_config.store_format,
            store_encoded=convert_config.store_encoded, store_fields=convert_config.store_fields)
        result = train_dir
    elif convert_config.to_hf or convert_config.to_mcore:
        result = run_convert(
            model_config,
            convert_config,
            template_config=template_config,
            distributed_config=distributed_config,
            checkpoint_config=checkpoint_config,
            megatron_config=configs['megatron_config'],
            moe_config=configs['moe_config'],
            output_dir=output_dir)
    if checkpoint_config.push_to_hub:
        source = result or (tuner_config.adapters[0] if tuner_config.adapters else model_config.model)
        result = run_push_to_hub(source, checkpoint_config, dataset_config, convert_config.commit_message)
    return result


def export_main(argv: Optional[List[str]] = None) -> Optional[str]:
    """dev entry point for ``swift export``; return the output path of the step that ran."""
    return run_export_configs(parse_export_configs(argv))


if __name__ == '__main__':
    export_main()

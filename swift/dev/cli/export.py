"""export CLI: legacy ExportArguments -> dev Configs -> the export recipes.

dev counterpart of legacy ``swift export`` (swift/pipelines/export/export.py::SwiftExport). Same
shape as ``swift.dev.cli.sft``: parse the legacy argument surface, translate it into dev's atomic
Configs, then hand off to a recipe. No export logic lives here -- the work is in
``swift.dev.recipes`` (run_merge_lora / run_quantize / run_convert / export_cached_dataset).

Two behaviours of legacy's ``run()`` are load-bearing and reproduced deliberately:

1. ORDER. merge_lora runs FIRST and then CHAINS: it rewrites args.model to the merged directory and
   clears args.adapters, so a following --quant_method operates on the merged weights rather than on
   the base model plus a dangling adapter. Everything after it is an if/elif chain, so at most one of
   quantize / ollama / cached_dataset / convert / push_to_hub runs.
2. output_dir OWNERSHIP. When merge_lora is combined with a later step, the merged weights must NOT
   land in the user's --output_dir (that belongs to the final artifact); legacy passes output_dir=None
   for the merge so it falls back to '{adapter}-merged'.
"""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from swift.arguments import ExportArguments


def _dtype_name(value: object) -> Optional[str]:
    """torch.bfloat16 -> 'bfloat16'; dev Configs carry the string form."""
    if value is None:
        return None
    name = str(value)
    return name.split('.')[-1] if name.startswith('torch.') else name


def _fill_from_args(config, args: 'ExportArguments'):
    """Copy same-named, non-None attributes off the legacy args onto a dev Config."""
    for f in dataclasses.fields(config):
        if not hasattr(args, f.name):
            continue
        value = getattr(args, f.name)
        if value is not None:
            setattr(config, f.name, value)
    return config


def export_args_to_configs(args: 'ExportArguments') -> dict:
    """ExportArguments -> the dev Configs the export recipes consume.

    Returned as a dict rather than a fixed tuple because each export path needs a different subset
    (quantize wants QuantizeConfig + DatasetConfig, convert wants ConvertConfig + DistributedConfig,
    merge_lora wants TunerConfig), and a 8-tuple at every call site would be unreadable.
    """
    from swift.dev.configs import (CheckpointConfig, ConvertConfig, DatasetConfig, DistributedConfig, ModelConfig,
                                   QuantizeConfig, TemplateConfig, TunerConfig)

    model_config = _fill_from_args(ModelConfig(), args)
    model_config.torch_dtype = _dtype_name(args.torch_dtype)

    template_config = _fill_from_args(TemplateConfig(), args)
    dataset_config = _fill_from_args(DatasetConfig(), args)
    distributed_config = _fill_from_args(DistributedConfig(), args)
    checkpoint_config = _fill_from_args(CheckpointConfig(), args)
    quantize_config = _fill_from_args(QuantizeConfig(), args)
    convert_config = _fill_from_args(ConvertConfig(), args)
    convert_config.test_convert_dtype = _dtype_name(args.test_convert_dtype) or 'float32'

    tuner_config = _fill_from_args(TunerConfig(), args)
    tuner_config.adapters = list(args.adapters or [])

    return {
        'model_config': model_config,
        'template_config': template_config,
        'dataset_config': dataset_config,
        'distributed_config': distributed_config,
        'checkpoint_config': checkpoint_config,
        'quantize_config': quantize_config,
        'convert_config': convert_config,
        'tuner_config': tuner_config,
    }


def _reject_unwired(args: 'ExportArguments') -> None:
    """Refuse the legacy export paths that have no dev recipe yet.

    These would otherwise parse fine and then silently do nothing, which is worse than failing: the
    user gets an exit code 0 and an empty output directory.
    """
    if getattr(args, 'to_ollama', False):
        raise NotImplementedError('`--to_ollama` is not wired into the dev export pipeline: there is no ollama recipe '
                                  'in swift.dev.recipes yet. Use legacy `swift export --to_ollama`.')
    if getattr(args, 'push_to_hub', False):
        raise NotImplementedError('`--push_to_hub` is not wired into the dev export pipeline: pushing is a hub '
                                  'operation with no dev recipe. Use legacy `swift export --push_to_hub`.')
    if getattr(args, 'to_peft_format', False):
        raise NotImplementedError('`--to_peft_format` is not wired into the dev export pipeline. Use legacy '
                                  '`swift export --to_peft_format`.')


def export_main(args: Optional[Any] = None) -> Optional[str]:
    """dev entry point for ``swift export``; returns the output path of the step that ran.

    Returns None when no export flag was given (legacy's run() likewise falls through silently).
    """
    from swift.arguments import ExportArguments
    from swift.dev.recipes import export_cached_dataset, run_convert, run_merge_lora, run_quantize
    from swift.utils import parse_args

    if isinstance(args, ExportArguments):
        export_args = args
    else:
        export_args, remaining = parse_args(ExportArguments, args)
        if remaining:
            raise ValueError(f'Unrecognized arguments: {remaining}')

    _reject_unwired(export_args)
    configs = export_args_to_configs(export_args)
    output_dir = configs['checkpoint_config'].output_dir
    result: Optional[str] = None

    if export_args.merge_lora:
        # Only --quant_method can actually follow a merge: ExportArguments.__post_init__ force-clears
        # merge_lora when to_mcore/to_hf is set (an mcore LoRA is merged in Megatron format instead,
        # see ConvertConfig.mcore_adapter), so that combination never reaches here.
        chains = bool(export_args.quant_method)
        merged = run_merge_lora(
            configs['model_config'],
            configs['tuner_config'],
            template_config=configs['template_config'],
            # When chaining, checkpoint_config is withheld too: run_merge_lora resolves
            # output_dir -> checkpoint_config.output_dir -> '{adapter}-merged', so passing it would
            # still drop the intermediate merge into the FINAL artifact's directory.
            checkpoint_config=None if chains else configs['checkpoint_config'],
            output_dir=None if chains else output_dir,
            # legacy's --exist_ok governs whether an existing output dir may be overwritten; the
            # recipe spells the same thing as replace_if_exists.
            replace_if_exists=export_args.exist_ok)
        # Chain: the merged weights become the input model, and the adapters are spent.
        configs['model_config'].model = merged
        configs['tuner_config'].adapters = []
        result = merged

    # if/elif: legacy runs at most ONE of these per invocation.
    if export_args.quant_method:
        result = run_quantize(
            configs['model_config'],
            configs['template_config'],
            configs['quantize_config'],
            configs['dataset_config'],
            output_dir=output_dir,
            quant_n_samples=export_args.quant_n_samples,
            batch_size=export_args.quant_batch_size or 1,
            group_size=export_args.group_size)
    elif export_args.to_cached_dataset:
        train_dir, _ = export_cached_dataset(
            configs['model_config'], configs['template_config'], configs['dataset_config'], output_dir=output_dir)
        result = train_dir
    elif export_args.to_hf or export_args.to_mcore:
        result = run_convert(
            configs['model_config'],
            configs['convert_config'],
            template_config=configs['template_config'],
            distributed_config=configs['distributed_config'],
            checkpoint_config=configs['checkpoint_config'],
            output_dir=output_dir)
    return result


if __name__ == '__main__':
    export_main()

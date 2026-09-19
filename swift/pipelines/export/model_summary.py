# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import os
from typing import List, Optional, Union

from swift.arguments import ExportArguments
from swift.utils import get_logger

logger = get_logger()


def _human_readable(num: int) -> str:
    for unit in ['', 'K', 'M', 'B']:
        if abs(num) < 1000:
            return f'{num:.2f}{unit}' if unit else str(num)
        num /= 1000
    return f'{num:.2f}T'


def _count_parameters(model):
    """Count parameters without touching the tensor data (safe on the meta device)."""
    total, by_dtype = 0, {}
    for _, param in model.named_parameters():
        numel = param.numel()
        total += numel
        key = str(param.dtype).replace('torch.', '')
        by_dtype[key] = by_dtype.get(key, 0) + numel
    return total, by_dtype


def export_model_summary(args: ExportArguments) -> None:
    """Print the model architecture / parameter statistics without loading any pretrained weights.

    This is driven by `--load_model` and `--return_dummy_model`:
      - `--return_dummy_model true`: the full architecture is built from `config.json`, no weights are read.
      - `--load_model false`: only the tokenizer/template are prepared, so no architecture is printed.
    """
    model, processor = args.get_model_processor()
    template = args.get_template(processor)
    if model is not None and template.use_model:
        template.model = model

    logger.info(f'model_type: {args.model_type}')
    logger.info(f'model_dir: {args.model_dir}')
    logger.info(f'task_type: {args.task_type}')
    logger.info(f'torch_dtype: {args.torch_dtype}')
    logger.info(f'template: {template.template_meta.template_type}')
    logger.info(f'max_length: {template.max_length}')
    logger.info(f'tokenizer: {type(template.tokenizer).__name__}, vocab_size: {len(template.tokenizer)}')

    summary = {
        'model': args.model,
        'model_type': args.model_type,
        'model_dir': args.model_dir,
        'task_type': args.task_type,
        'torch_dtype': str(args.torch_dtype),
        'template': template.template_meta.template_type,
        'max_length': template.max_length,
        'tokenizer_class': type(template.tokenizer).__name__,
        'vocab_size': len(template.tokenizer),
        'load_model': args.load_model,
        'return_dummy_model': args.return_dummy_model,
    }

    if model is None:
        logger.info('The model was not instantiated because `--load_model false` was set. '
                    'Set `--return_dummy_model true` if you want to inspect the architecture '
                    'without loading the weights.')
    else:
        n_params, by_dtype = _count_parameters(model)
        devices = sorted({str(p.device) for p in model.parameters()})
        logger.info(f'model architecture:\n{model}')
        logger.info(f'model_class: {type(model).__name__}')
        logger.info(f'num_parameters: {n_params} ({_human_readable(n_params)})')
        logger.info(f'parameters_by_dtype: {by_dtype}')
        logger.info(f'parameter devices: {devices}')
        if 'meta' in devices:
            logger.info('The parameters live on the meta device, so no real memory is allocated. '
                        'They cannot be used for a forward pass; call `model.to_empty(device="cpu")` first.')
        summary.update({
            'model_class': type(model).__name__,
            'num_parameters': n_params,
            'num_parameters_readable': _human_readable(n_params),
            'parameters_by_dtype': by_dtype,
            'parameter_devices': devices,
        })

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        summary_path = os.path.join(args.output_dir, 'model_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f'The model summary has been saved to: `{summary_path}`')
        if model is not None:
            arch_path = os.path.join(args.output_dir, 'model_architecture.txt')
            with open(arch_path, 'w', encoding='utf-8') as f:
                f.write(str(model))
            logger.info(f'The model architecture has been saved to: `{arch_path}`')


def model_summary_main(args: Optional[Union[List[str], ExportArguments]] = None):
    return export_model_summary(args)
